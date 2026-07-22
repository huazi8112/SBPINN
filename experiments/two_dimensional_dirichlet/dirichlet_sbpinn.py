from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import trange

from dirichlet_common import (
    BenchmarkConfig,
    cell_center_grid,
    conditional_density_metrics,
    drift_numpy,
    sliced_wasserstein_2d,
    time_key,
    unnormalized_density_metrics,
    write_json,
)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class DirichletEmbedding(nn.Module):
    def __init__(self, harmonics: Tuple[int, ...]):
        super().__init__()
        values = torch.tensor(harmonics, dtype=torch.float64)
        self.register_buffer("harmonics", values)

    @property
    def output_dim(self) -> int:
        return 3 + 4 * len(self.harmonics)

    def forward(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        x = points[:, 0:1]
        y = points[:, 1:2]
        angle_x = 2.0 * math.pi * x * self.harmonics[None, :]
        angle_y = 2.0 * math.pi * y * self.harmonics[None, :]
        return torch.cat(
            [
                x,
                y,
                times,
                torch.sin(angle_x),
                torch.cos(angle_x),
                torch.sin(angle_y),
                torch.cos(angle_y),
            ],
            dim=1,
        )


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous = input_dim
    for _ in range(hidden_layers):
        linear = nn.Linear(previous, hidden_dim)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, nn.Tanh()])
        previous = hidden_dim
    final = nn.Linear(previous, 1)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class DirichletLogDensityNetwork(nn.Module):
    def __init__(self, cfg: BenchmarkConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = DirichletEmbedding(cfg.harmonics)
        self.network = make_mlp(
            self.embedding.output_dim,
            cfg.hidden_dim,
            cfg.hidden_layers,
        )

    def raw_output(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(self.embedding(points, times))

    def log_unnormalized(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.raw_output(points, times)
        x = points[:, 0:1]
        y = points[:, 1:2]
        base = torch.clamp(
            x * (1.0 - x) * y * (1.0 - y),
            min=1.0e-14,
        )
        return raw + self.cfg.boundary_power * torch.log(base)


def split_samples(
    payload: np.lib.npyio.NpzFile,
    cfg: BenchmarkConfig,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, np.ndarray],
    np.ndarray,
]:
    training: Dict[float, np.ndarray] = {}
    validation: Dict[float, np.ndarray] = {}
    rng = np.random.default_rng(cfg.seed + 20000)
    for time_value in cfg.snapshot_times:
        samples = payload[time_key(time_value)].astype(
            np.float64,
            copy=False,
        )
        indices = rng.permutation(len(samples))
        validation_count = max(
            64,
            min(
                cfg.stage1_validation_batch_size,
                len(samples) // 5,
            ),
        )
        validation[time_value] = samples[indices[:validation_count]]
        training[time_value] = samples[indices[validation_count:]]
    return training, validation, payload["survival"].astype(np.float64)


def linear_survival_mass(
    times: torch.Tensor,
    snapshot_times: torch.Tensor,
    survival_values: torch.Tensor,
) -> torch.Tensor:
    times_flat = times.reshape(-1)
    indices = torch.searchsorted(
        snapshot_times,
        times_flat,
        right=True,
    )
    indices = torch.clamp(indices, 1, len(snapshot_times) - 1)
    left_index = indices - 1
    right_index = indices
    left_time = snapshot_times[left_index]
    right_time = snapshot_times[right_index]
    left_value = survival_values[left_index]
    right_value = survival_values[right_index]
    weight = (times_flat - left_time) / torch.clamp(
        right_time - left_time,
        min=1.0e-12,
    )
    values = left_value + weight * (right_value - left_value)
    return values.reshape(times.shape)


def fixed_uniform_points(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    side = max(4, int(round(math.sqrt(count))))
    axis = (
        torch.arange(side, device=device, dtype=dtype) + 0.5
    ) / side
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)


def normalized_nll(
    network: DirichletLogDensityNetwork,
    data_points: torch.Tensor,
    time_value: float,
    uniform_points: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    data_times = torch.full(
        (len(data_points), 1),
        time_value,
        device=data_points.device,
        dtype=data_points.dtype,
    )
    uniform_times = torch.full(
        (len(uniform_points), 1),
        time_value,
        device=data_points.device,
        dtype=data_points.dtype,
    )
    log_data = network.log_unnormalized(data_points, data_times)
    log_uniform = network.log_unnormalized(
        uniform_points,
        uniform_times,
    )
    log_normalizer = (
        torch.logsumexp(log_uniform.reshape(-1), dim=0)
        - math.log(log_uniform.numel())
    )
    return -torch.mean(log_data) + log_normalizer


def score_from_network(
    network: DirichletLogDensityNetwork,
    points: torch.Tensor,
    time_value: float,
    create_graph: bool,
) -> torch.Tensor:
    score_points = points.detach().clone()
    score_points.requires_grad_(True)
    times = torch.full(
        (len(score_points), 1),
        time_value,
        device=score_points.device,
        dtype=score_points.dtype,
    )
    log_density = network.log_unnormalized(score_points, times)
    return torch.autograd.grad(
        log_density.sum(),
        score_points,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]


def dsm_loss(
    network: DirichletLogDensityNetwork,
    clean_points: torch.Tensor,
    time_value: float,
    cfg: BenchmarkConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    if len(clean_points) == 0:
        return torch.zeros(
            (),
            device=clean_points.device,
            dtype=clean_points.dtype,
        )
    scale_values = torch.tensor(
        cfg.dsm_noise_scales,
        device=clean_points.device,
        dtype=clean_points.dtype,
    )
    scale_indices = torch.randint(
        0,
        len(scale_values),
        (len(clean_points),),
        device=clean_points.device,
        generator=generator,
    )
    sigma = scale_values[scale_indices].reshape(-1, 1)
    margin = 3.0 * sigma.reshape(-1)
    interior = torch.all(
        (clean_points > margin[:, None])
        & (clean_points < 1.0 - margin[:, None]),
        dim=1,
    )
    clean = clean_points[interior]
    sigma = sigma[interior]
    if len(clean) < 8:
        return torch.zeros(
            (),
            device=clean_points.device,
            dtype=clean_points.dtype,
        )
    noise = sigma * torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    noisy = clean + noise
    target = (clean - noisy) / (sigma * sigma)
    predicted = score_from_network(
        network,
        noisy,
        time_value,
        create_graph=True,
    )
    return 0.5 * torch.mean(
        sigma * sigma * (predicted - target).pow(2)
    )


def validation_nll(
    network: DirichletLogDensityNetwork,
    validation: Dict[float, torch.Tensor],
    uniform_points: torch.Tensor,
    cfg: BenchmarkConfig,
) -> float:
    values = []
    with torch.no_grad():
        for time_value in cfg.snapshot_times:
            values.append(
                normalized_nll(
                    network,
                    validation[time_value],
                    time_value,
                    uniform_points,
                    cfg,
                )
            )
    return float(torch.stack(values).mean().detach().cpu())


def train_stage1(
    network: DirichletLogDensityNetwork,
    training: Dict[float, torch.Tensor],
    validation: Dict[float, torch.Tensor],
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg.stage1_learning_rate,
        weight_decay=cfg.stage1_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.stage1_iterations, 1),
        eta_min=0.1 * cfg.stage1_learning_rate,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 30000)
    uniform_points = fixed_uniform_points(
        cfg.stage1_uniform_batch_size,
        device,
        dtype,
    )
    history = []
    best_state = copy.deepcopy(network.state_dict())
    best_validation = float("inf")
    best_iteration = 0
    stale = 0
    stopped_early = False

    for iteration in trange(
        cfg.stage1_iterations,
        desc="Stage I killed-SDE density/score initialization",
    ):
        time_index = int(
            torch.randint(
                0,
                len(cfg.snapshot_times),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        time_value = cfg.snapshot_times[time_index]
        pool = training[time_value]
        indices = torch.randint(
            0,
            len(pool),
            (cfg.stage1_batch_size,),
            device=device,
            generator=generator,
        )
        data_points = pool[indices]
        nll = normalized_nll(
            network,
            data_points,
            time_value,
            uniform_points,
            cfg,
        )
        dsm = dsm_loss(
            network,
            data_points,
            time_value,
            cfg,
            generator,
        )
        total = (
            cfg.stage1_nll_weight * nll
            + cfg.stage1_dsm_weight * dsm
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(),
            cfg.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        validation_value = float("nan")
        if (
            (iteration + 1) % cfg.stage1_validation_every == 0
            or iteration == 0
            or iteration + 1 == cfg.stage1_iterations
        ):
            network.eval()
            validation_value = validation_nll(
                network,
                validation,
                uniform_points,
                cfg,
            )
            network.train()
            if validation_value < best_validation - 1.0e-7:
                best_validation = validation_value
                best_iteration = iteration + 1
                best_state = copy.deepcopy(network.state_dict())
                stale = 0
            else:
                stale += 1

        history.append(
            {
                "iteration": iteration + 1,
                "time": time_value,
                "total_loss": float(total.detach().cpu()),
                "nll_loss": float(nll.detach().cpu()),
                "dsm_loss": float(dsm.detach().cpu()),
                "validation_nll": validation_value,
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if (
            iteration + 1 >= cfg.stage1_min_iterations
            and stale >= cfg.stage1_patience_evaluations
        ):
            stopped_early = True
            break

    network.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(
        output_dir / "stage1_history.csv",
        index=False,
    )
    torch.save(
        {
            "state_dict": best_state,
            "best_iteration": best_iteration,
            "best_validation_nll": best_validation,
        },
        output_dir / "stage1_best_checkpoint.pt",
    )
    return {
        "best_iteration": best_iteration,
        "best_validation_nll": best_validation,
        "completed_iterations": len(history),
        "stopped_early": stopped_early,
    }


def conditional_grid_density(
    network: DirichletLogDensityNetwork,
    time_value: float,
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    _, _, points_np = cell_center_grid(grid_size)
    points = torch.tensor(
        points_np,
        device=device,
        dtype=dtype,
    )
    times = torch.full(
        (len(points), 1),
        time_value,
        device=device,
        dtype=dtype,
    )
    log_unnormalized = network.log_unnormalized(
        points,
        times,
    ).reshape(grid_size, grid_size)
    shifted = log_unnormalized - torch.max(log_unnormalized)
    density = torch.exp(shifted)
    return density / torch.mean(density)


def fractional_laplacian_zero_extension(
    density: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    n = density.shape[0]
    extended_size = cfg.pde_padding_factor * n
    start = (extended_size - n) // 2
    extended = torch.zeros(
        (extended_size, extended_size),
        device=density.device,
        dtype=density.dtype,
    )
    extended[start : start + n, start : start + n] = density
    dx = 1.0 / n
    angular = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            extended_size,
            d=dx,
            device=density.device,
            dtype=density.dtype,
        )
    )
    kx, ky = torch.meshgrid(angular, angular, indexing="xy")
    symbol = (
        torch.abs(kx) ** cfg.alpha
        + torch.abs(ky) ** cfg.alpha
    )
    operated = torch.fft.ifft2(
        symbol * torch.fft.fft2(extended)
    ).real
    return operated[start : start + n, start : start + n]


def advection_divergence(
    density: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    n = density.shape[0]
    _, _, points_np = cell_center_grid(n)
    drift = torch.tensor(
        drift_numpy(points_np, cfg).reshape(n, n, 2),
        device=density.device,
        dtype=density.dtype,
    )
    flux_x = drift[:, :, 0] * density
    flux_y = drift[:, :, 1] * density
    dx = 1.0 / n
    padded_x = torch.nn.functional.pad(
        flux_x,
        (1, 1, 0, 0),
        mode="constant",
        value=0.0,
    )
    padded_y = torch.nn.functional.pad(
        flux_y,
        (0, 0, 1, 1),
        mode="constant",
        value=0.0,
    )
    return (
        (padded_x[:, 2:] - padded_x[:, :-2]) / (2.0 * dx)
        + (padded_y[2:, :] - padded_y[:-2, :]) / (2.0 * dx)
    )


def pde_residual_loss(
    network: DirichletLogDensityNetwork,
    time_value: float,
    snapshot_times: torch.Tensor,
    survival_values: torch.Tensor,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    delta = min(
        cfg.pde_time_delta,
        0.45 * time_value,
        0.45 * (cfg.final_time - time_value),
    )
    if delta <= 1.0e-8:
        delta = min(cfg.pde_time_delta, 0.25 * cfg.final_time)
    t_minus = max(0.0, time_value - delta)
    t_plus = min(cfg.final_time, time_value + delta)

    q_minus = conditional_grid_density(
        network,
        t_minus,
        cfg.pde_grid_size,
        device,
        dtype,
    )
    q_center = conditional_grid_density(
        network,
        time_value,
        cfg.pde_grid_size,
        device,
        dtype,
    )
    q_plus = conditional_grid_density(
        network,
        t_plus,
        cfg.pde_grid_size,
        device,
        dtype,
    )
    time_tensor = torch.tensor(
        [[t_minus], [time_value], [t_plus]],
        device=device,
        dtype=dtype,
    )
    mass = linear_survival_mass(
        time_tensor,
        snapshot_times,
        survival_values,
    ).reshape(-1)
    p_minus = mass[0] * q_minus
    p_center = mass[1] * q_center
    p_plus = mass[2] * q_plus
    time_derivative = (p_plus - p_minus) / (t_plus - t_minus)

    advection = advection_divergence(p_center, cfg)
    fractional = fractional_laplacian_zero_extension(
        p_center,
        cfg,
    )
    diffusion = cfg.fractional_coefficient * fractional
    residual = time_derivative + advection + diffusion
    scale = (
        torch.sqrt(
            torch.mean(time_derivative.detach() ** 2)
            + torch.mean(advection.detach() ** 2)
            + torch.mean(diffusion.detach() ** 2)
        )
        + 1.0e-12
    )
    normalized = residual / scale
    return torch.mean(normalized**2), {
        "normalized_residual_RMSE": torch.sqrt(
            torch.mean(normalized**2)
        ),
        "raw_residual_RMSE": torch.sqrt(
            torch.mean(residual**2)
        ),
        "mass": mass[1],
    }


def teacher_consistency_loss(
    student: DirichletLogDensityNetwork,
    teacher: DirichletLogDensityNetwork,
    points: torch.Tensor,
    time_value: float,
) -> torch.Tensor:
    student_score = score_from_network(
        student,
        points,
        time_value,
        create_graph=True,
    )
    teacher_score = score_from_network(
        teacher,
        points,
        time_value,
        create_graph=False,
    )
    return torch.mean((student_score - teacher_score.detach()) ** 2)


def train_stage2(
    network: DirichletLogDensityNetwork,
    teacher: DirichletLogDensityNetwork,
    training: Dict[float, torch.Tensor],
    validation: Dict[float, torch.Tensor],
    survival: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg.stage2_learning_rate,
        weight_decay=cfg.stage2_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.stage2_iterations, 1),
        eta_min=0.05 * cfg.stage2_learning_rate,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 40000)
    uniform_points = fixed_uniform_points(
        cfg.stage2_uniform_batch_size,
        device,
        dtype,
    )
    snapshot_times = torch.tensor(
        (0.0,) + cfg.snapshot_times,
        device=device,
        dtype=dtype,
    )
    survival_values = torch.tensor(
        np.concatenate([[1.0], survival]),
        device=device,
        dtype=dtype,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    history = []
    selection_rows = []
    checkpoint_dir = output_dir / "stage2_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for iteration in trange(
        cfg.stage2_iterations,
        desc="Stage II transient exterior-Dirichlet SBPINN",
    ):
        progress = iteration / max(cfg.stage2_iterations - 1, 1)
        teacher_weight = (
            cfg.stage2_teacher_weight_start
            + progress
            * (
                cfg.stage2_teacher_weight_end
                - cfg.stage2_teacher_weight_start
            )
        )
        pde_weight = (
            cfg.stage2_pde_weight_start
            * (
                cfg.stage2_pde_weight_end
                / cfg.stage2_pde_weight_start
            )
            ** progress
        )

        discrete_index = int(
            torch.randint(
                0,
                len(cfg.snapshot_times),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        data_time = cfg.snapshot_times[discrete_index]
        pool = training[data_time]
        indices = torch.randint(
            0,
            len(pool),
            (cfg.stage2_batch_size,),
            device=device,
            generator=generator,
        )
        data_points = pool[indices]
        nll = normalized_nll(
            network,
            data_points,
            data_time,
            uniform_points,
            cfg,
        )
        consistency = teacher_consistency_loss(
            network,
            teacher,
            data_points[: min(512, len(data_points))],
            data_time,
        )

        interior_low = max(
            cfg.pde_time_delta * 1.2,
            min(cfg.snapshot_times),
        )
        interior_high = cfg.final_time - cfg.pde_time_delta * 1.2
        random_value = torch.rand(
            (),
            device=device,
            generator=generator,
        ).item()
        pde_time = interior_low + random_value * (
            interior_high - interior_low
        )
        pde_loss, pde_diagnostics = pde_residual_loss(
            network,
            pde_time,
            snapshot_times,
            survival_values,
            cfg,
            device,
            dtype,
        )
        raw = network.raw_output(
            uniform_points,
            torch.full(
                (len(uniform_points), 1),
                data_time,
                device=device,
                dtype=dtype,
            ),
        )
        smoothness = torch.mean(raw**2)
        total = (
            cfg.stage2_nll_weight * nll
            + teacher_weight * consistency
            + pde_weight * pde_loss
            + cfg.stage2_smoothness_weight * smoothness
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(),
            cfg.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        validation_value = float("nan")
        if (
            (iteration + 1) % cfg.stage2_validation_every == 0
            or iteration == 0
            or iteration + 1 == cfg.stage2_iterations
        ):
            network.eval()
            validation_value = validation_nll(
                network,
                validation,
                uniform_points,
                cfg,
            )
            evaluation_losses = []
            for evaluation_time in cfg.evaluation_times:
                loss_value, diagnostics = pde_residual_loss(
                    network,
                    evaluation_time,
                    snapshot_times,
                    survival_values,
                    cfg,
                    device,
                    dtype,
                )
                evaluation_losses.append(
                    float(
                        diagnostics[
                            "normalized_residual_RMSE"
                        ].detach().cpu()
                    )
                )
            mean_residual = float(np.mean(evaluation_losses))
            network.train()
            checkpoint_path = (
                checkpoint_dir
                / f"stage2_iter_{iteration + 1:05d}.pt"
            )
            torch.save(
                {
                    "state_dict": network.state_dict(),
                    "iteration": iteration + 1,
                    "validation_nll": validation_value,
                    "mean_normalized_pde_residual": mean_residual,
                },
                checkpoint_path,
            )
            selection_rows.append(
                {
                    "iteration": iteration + 1,
                    "validation_nll": validation_value,
                    "mean_normalized_pde_residual": mean_residual,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

        history.append(
            {
                "iteration": iteration + 1,
                "data_time": data_time,
                "pde_time": pde_time,
                "total_loss": float(total.detach().cpu()),
                "nll_loss": float(nll.detach().cpu()),
                "consistency_loss": float(
                    consistency.detach().cpu()
                ),
                "pde_loss": float(pde_loss.detach().cpu()),
                "normalized_pde_residual_RMSE": float(
                    pde_diagnostics[
                        "normalized_residual_RMSE"
                    ].detach().cpu()
                ),
                "teacher_weight": teacher_weight,
                "pde_weight": pde_weight,
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
                "validation_nll": validation_value,
            }
        )

    pd.DataFrame(history).to_csv(
        output_dir / "stage2_history.csv",
        index=False,
    )
    selection = pd.DataFrame(selection_rows)
    best_nll = float(selection["validation_nll"].min())
    selection["nll_excess"] = np.maximum(
        selection["validation_nll"]
        - best_nll
        - cfg.stage2_selection_nll_tolerance,
        0.0,
    )
    selection["selection_metric"] = (
        selection["mean_normalized_pde_residual"]
        + 5.0 * selection["nll_excess"]
    )
    eligible = selection[
        selection["iteration"] >= cfg.stage2_min_selection_iteration
    ]
    if eligible.empty:
        eligible = selection
    selected = eligible.sort_values(
        ["selection_metric", "iteration"]
    ).iloc[0]
    checkpoint = torch.load(
        selected["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    network.load_state_dict(checkpoint["state_dict"])
    selection["selected"] = (
        selection["iteration"] == int(selected["iteration"])
    )
    selection.to_csv(
        output_dir / "stage2_selection.csv",
        index=False,
    )
    torch.save(
        checkpoint,
        output_dir / "stage2_selected_checkpoint.pt",
    )
    return {
        "selected_iteration": int(selected["iteration"]),
        "selected_validation_nll": float(
            selected["validation_nll"]
        ),
        "selected_mean_normalized_pde_residual": float(
            selected["mean_normalized_pde_residual"]
        ),
    }


def evaluate_model(
    network: DirichletLogDensityNetwork,
    cfg: BenchmarkConfig,
    train_survival: np.ndarray,
    reference_dir: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    reference_payload = np.load(
        reference_dir / "reference_pde_finest.npz"
    )
    reference_summary = __import__("json").loads(
        (reference_dir / "reference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    reference_times = reference_payload["times"]
    reference_densities = reference_payload["densities"]
    grid_size = reference_densities.shape[1]
    _, _, points = cell_center_grid(grid_size)
    snapshot_time_tensor = torch.tensor(
        (0.0,) + cfg.snapshot_times,
        device=device,
        dtype=dtype,
    )
    survival_tensor = torch.tensor(
        np.concatenate([[1.0], train_survival]),
        device=device,
        dtype=dtype,
    )

    per_time = []
    model_densities = []
    conditional_densities = []
    for index, time_value in enumerate(reference_times):
        conditional = conditional_grid_density(
            network,
            float(time_value),
            grid_size,
            device,
            dtype,
        ).detach().cpu().numpy()
        mass = float(
            linear_survival_mass(
                torch.tensor(
                    [[float(time_value)]],
                    device=device,
                    dtype=dtype,
                ),
                snapshot_time_tensor,
                survival_tensor,
            ).item()
        )
        model = mass * conditional
        reference = reference_densities[index]
        reference_mass = float(np.mean(reference))
        reference_conditional = reference / max(
            reference_mass,
            1.0e-300,
        )
        metrics = {
            "time": float(time_value),
            **unnormalized_density_metrics(model, reference),
            **conditional_density_metrics(
                conditional,
                reference_conditional,
            ),
            **sliced_wasserstein_2d(
                points,
                conditional.reshape(-1),
                reference_conditional.reshape(-1),
                cfg.wasserstein_samples,
                cfg.wasserstein_directions,
                cfg.seed + index * 100,
            ),
            "model_survival_mass": mass,
            "reference_survival_mass": reference_mass,
            "survival_mass_absolute_error": abs(
                mass - reference_mass
            ),
        }
        per_time.append(metrics)
        model_densities.append(model)
        conditional_densities.append(conditional)

    frame = pd.DataFrame(per_time)
    frame.to_csv(
        output_dir / "sbpinn_metrics_by_time.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "sbpinn_evaluation.npz",
        times=reference_times,
        model_densities=np.stack(model_densities),
        conditional_densities=np.stack(conditional_densities),
        reference_densities=reference_densities,
    )
    aggregate = {
        "mean_relative_L2_error": float(
            frame["relative_L2_error"].mean()
        ),
        "mean_Hellinger_distance": float(
            frame["Hellinger_distance"].mean()
        ),
        "mean_survival_mass_absolute_error": float(
            frame["survival_mass_absolute_error"].mean()
        ),
        "maximum_negative_mass": float(
            frame["negative_mass"].max()
        ),
        "mean_sliced_W": float(frame["mean_sliced_W"].mean()),
        "reference_high_fidelity": bool(
            reference_summary["high_fidelity_reference"]
        ),
        "per_time": per_time,
    }
    return aggregate


def run_sbpinn(
    cfg: BenchmarkConfig,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg.device)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    payload = np.load(reference_dir / "train_killed_samples.npz")
    training_np, validation_np, survival = split_samples(
        payload,
        cfg,
    )
    training = {
        time_value: torch.tensor(
            samples,
            device=device,
            dtype=dtype,
        )
        for time_value, samples in training_np.items()
    }
    validation = {
        time_value: torch.tensor(
            samples,
            device=device,
            dtype=dtype,
        )
        for time_value, samples in validation_np.items()
    }

    teacher = DirichletLogDensityNetwork(cfg).to(
        device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    stage1 = train_stage1(
        teacher,
        training,
        validation,
        cfg,
        device,
        dtype,
        output_dir,
    )
    stage1_time = time.perf_counter() - start

    student = DirichletLogDensityNetwork(cfg).to(
        device=device,
        dtype=dtype,
    )
    student.load_state_dict(copy.deepcopy(teacher.state_dict()))
    stage2_start = time.perf_counter()
    stage2 = train_stage2(
        student,
        teacher,
        training,
        validation,
        survival,
        cfg,
        device,
        dtype,
        output_dir,
    )
    stage2_time = time.perf_counter() - stage2_start

    metrics = evaluate_model(
        student,
        cfg,
        survival,
        reference_dir,
        output_dir,
        device,
        dtype,
    )
    peak_cuda = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    metrics.update(
        {
            "device": str(device),
            "stage1": stage1,
            "stage2": stage2,
            "stage1_training_time_s": stage1_time,
            "stage2_training_time_s": stage2_time,
            "total_training_time_s": stage1_time + stage2_time,
            "peak_cuda_memory_mb": peak_cuda,
            "parameter_count": sum(
                parameter.numel()
                for parameter in student.parameters()
            ),
            "boundary_parameterization": (
                "[x(1-x)y(1-y)]^(alpha/2) * exp(f_theta)"
            ),
        }
    )
    torch.save(
        {
            "teacher": teacher.state_dict(),
            "student": student.state_dict(),
            "config": cfg.__dict__,
            "stage1": stage1,
            "stage2": stage2,
        },
        output_dir / "sbpinn_checkpoint.pt",
    )
    write_json(output_dir / "sbpinn_summary.json", metrics)
    return metrics
