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

from nonlinear_torus_common import (
    BenchmarkConfig,
    density_metrics,
    drift_numpy,
    fractional_symbol,
    periodic_grid,
    set_seeds,
    torus_sliced_wasserstein,
    write_json,
)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class PeriodicEmbedding(nn.Module):
    def __init__(self, harmonics: Tuple[int, ...]):
        super().__init__()
        values = torch.tensor(harmonics, dtype=torch.float64)
        self.register_buffer("harmonics", values)

    @property
    def output_dim(self) -> int:
        return 4 * len(self.harmonics)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        angle_x = (
            2.0
            * math.pi
            * points[:, 0:1]
            * self.harmonics[None, :]
        )
        angle_y = (
            2.0
            * math.pi
            * points[:, 1:2]
            * self.harmonics[None, :]
        )
        return torch.cat(
            [
                torch.sin(angle_x),
                torch.cos(angle_x),
                torch.sin(angle_y),
                torch.cos(angle_y),
            ],
            dim=1,
        )


def make_mlp(
    input_dim: int,
    output_dim: int,
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
    final = nn.Linear(previous, output_dim)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


class ConservativeLogDensityNetwork(nn.Module):
    """Scalar periodic log-density; its gradient is exactly conservative."""

    def __init__(self, cfg: BenchmarkConfig):
        super().__init__()
        self.embedding = PeriodicEmbedding(cfg.harmonics)
        self.network = make_mlp(
            self.embedding.output_dim,
            1,
            cfg.hidden_dim,
            cfg.hidden_layers,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(points))


def periodic_displacement(
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> torch.Tensor:
    return torch.remainder(clean - noisy + 0.5, 1.0) - 0.5


def scaled_score_from_log_density(
    network: ConservativeLogDensityNetwork,
    points: torch.Tensor,
    score_scale: float,
    create_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = network(points)
    gradient = torch.autograd.grad(
        q.sum(),
        points,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return gradient / score_scale, q


def split_train_validation(
    samples: np.ndarray,
    seed: int,
    validation_fraction: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    validation_count = max(
        1024,
        int(round(validation_fraction * len(samples))),
    )
    validation_count = min(validation_count, len(samples) // 3)
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    return samples[training_indices], samples[validation_indices]


def make_fixed_dsm_validation_batch(
    validation_samples: torch.Tensor,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 31001)
    count = min(
        cfg.stage1_validation_batch_size,
        validation_samples.shape[0],
    )
    indices = torch.randperm(
        validation_samples.shape[0],
        device=device,
        generator=generator,
    )[:count]
    clean = validation_samples[indices].detach().clone()
    scale_indices = torch.randint(
        0,
        len(cfg.dsm_noise_scales),
        (count,),
        device=device,
        generator=generator,
    )
    scale_values = torch.tensor(
        cfg.dsm_noise_scales,
        device=device,
        dtype=dtype,
    )
    sigma = scale_values[scale_indices].reshape(-1, 1)
    noise = torch.randn(
        clean.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * sigma
    noisy = torch.remainder(clean + noise, 1.0)
    displacement = periodic_displacement(clean, noisy)
    target_scaled = (
        displacement
        / (sigma * sigma * cfg.score_scale)
    )
    return {
        "clean": clean,
        "noisy": noisy,
        "sigma": sigma,
        "target_scaled": target_scaled,
    }


def dsm_validation_loss(
    network: ConservativeLogDensityNetwork,
    validation_batch: Dict[str, torch.Tensor],
    cfg: BenchmarkConfig,
) -> float:
    noisy = validation_batch["noisy"].detach().clone()
    noisy.requires_grad_(True)
    predicted, _ = scaled_score_from_log_density(
        network,
        noisy,
        cfg.score_scale,
        create_graph=False,
    )
    sigma = validation_batch["sigma"]
    target = validation_batch["target_scaled"]
    loss = 0.5 * torch.mean(
        sigma * sigma * (predicted - target).pow(2).sum(dim=1, keepdim=True)
    )
    return float(loss.detach().cpu())


def train_stage1(
    network: ConservativeLogDensityNetwork,
    training_samples: torch.Tensor,
    validation_samples: torch.Tensor,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg.learning_rate_stage1,
        weight_decay=cfg.stage1_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.stage1_iterations, 1),
        eta_min=0.1 * cfg.learning_rate_stage1,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 30000)
    scale_values = torch.tensor(
        cfg.dsm_noise_scales,
        device=device,
        dtype=dtype,
    )
    validation_batch = make_fixed_dsm_validation_batch(
        validation_samples,
        cfg,
        device,
        dtype,
    )

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(network.state_dict())
    best_validation = float("inf")
    best_iteration = 0
    evaluations_without_improvement = 0
    stopped_early = False

    network.train()
    for iteration in trange(
        cfg.stage1_iterations,
        desc="Stage I conservative torus DSM",
    ):
        indices = torch.randint(
            0,
            training_samples.shape[0],
            (cfg.stage1_batch_size,),
            device=device,
            generator=generator,
        )
        clean = training_samples[indices].detach().clone()
        scale_indices = torch.randint(
            0,
            len(cfg.dsm_noise_scales),
            (cfg.stage1_batch_size,),
            device=device,
            generator=generator,
        )
        sigma = scale_values[scale_indices].reshape(-1, 1)
        noise = torch.randn(
            clean.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        ) * sigma
        noisy = torch.remainder(clean + noise, 1.0)
        displacement = periodic_displacement(clean, noisy)
        target_scaled = (
            displacement
            / (sigma * sigma * cfg.score_scale)
        )

        noisy.requires_grad_(True)
        predicted_scaled, q = scaled_score_from_log_density(
            network,
            noisy,
            cfg.score_scale,
            create_graph=True,
        )
        dsm_loss = 0.5 * torch.mean(
            sigma
            * sigma
            * (predicted_scaled - target_scaled)
            .pow(2)
            .sum(dim=1, keepdim=True)
        )
        gauge_loss = q.mean().pow(2)
        total = (
            dsm_loss
            + cfg.stage1_output_gauge_weight * gauge_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            network.parameters(),
            cfg.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        validation_loss = float("nan")
        is_evaluation = (
            (iteration + 1) % cfg.stage1_validation_every == 0
            or iteration == 0
            or iteration + 1 == cfg.stage1_iterations
        )
        if is_evaluation:
            network.eval()
            validation_loss = dsm_validation_loss(
                network,
                validation_batch,
                cfg,
            )
            network.train()
            if validation_loss < best_validation - 1.0e-8:
                best_validation = validation_loss
                best_iteration = iteration + 1
                best_state = copy.deepcopy(network.state_dict())
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1

        history.append(
            {
                "iteration": iteration + 1,
                "total_loss": float(total.detach().cpu()),
                "dsm_loss": float(dsm_loss.detach().cpu()),
                "gauge_loss": float(gauge_loss.detach().cpu()),
                "validation_dsm_loss": validation_loss,
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )

        if (
            iteration + 1 >= cfg.stage1_min_iterations
            and evaluations_without_improvement
            >= cfg.stage1_patience_evaluations
        ):
            stopped_early = True
            break

    network.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "best_iteration": best_iteration,
            "best_validation_dsm_loss": best_validation,
        },
        output_dir / "stage1_best_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(
        output_dir / "stage1_history.csv",
        index=False,
    )
    return {
        "best_iteration": best_iteration,
        "best_validation_dsm_loss": best_validation,
        "stopped_early": stopped_early,
        "completed_iterations": len(history),
    }


def build_pde_grid(
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    grid_size = cfg.pde_grid_size
    _, _, points_np = periodic_grid(grid_size)
    points = torch.tensor(
        points_np,
        device=device,
        dtype=dtype,
    )
    drift = torch.tensor(
        drift_numpy(points_np, cfg).reshape(
            grid_size,
            grid_size,
            2,
        ),
        device=device,
        dtype=dtype,
    )
    angular = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            grid_size,
            d=1.0 / grid_size,
            device=device,
            dtype=dtype,
        )
    )
    kx, ky = torch.meshgrid(
        angular,
        angular,
        indexing="xy",
    )
    symbol = (
        torch.abs(kx) ** cfg.alpha
        + torch.abs(ky) ** cfg.alpha
    )
    cycles = torch.fft.fftfreq(
        grid_size,
        d=1.0 / grid_size,
        device=device,
        dtype=dtype,
    )
    fx, fy = torch.meshgrid(cycles, cycles, indexing="xy")
    radial_frequency = torch.sqrt(fx * fx + fy * fy)
    tail_mask = radial_frequency > cfg.stage2_spectral_cutoff
    return points, drift, kx, ky, symbol, tail_mask


def normalized_density_on_grid(
    network: ConservativeLogDensityNetwork,
    points: torch.Tensor,
    grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = network(points).reshape(grid_size, grid_size)
    shifted = q - torch.max(q)
    unnormalized = torch.exp(shifted)
    density = unnormalized / torch.mean(unnormalized)
    return density, q


def fractional_pde_loss(
    network: ConservativeLogDensityNetwork,
    grid_points: torch.Tensor,
    drift: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    symbol: torch.Tensor,
    tail_mask: torch.Tensor,
    cfg: BenchmarkConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    density, q = normalized_density_on_grid(
        network,
        grid_points,
        cfg.pde_grid_size,
    )
    density_fft = torch.fft.fft2(density)
    flux_x = drift[:, :, 0] * density
    flux_y = drift[:, :, 1] * density
    div_flux = torch.fft.ifft2(
        1j * kx * torch.fft.fft2(flux_x)
        + 1j * ky * torch.fft.fft2(flux_y)
    ).real
    fractional = torch.fft.ifft2(
        symbol * density_fft
    ).real
    advection_term = -div_flux
    diffusion_term = (
        -cfg.fractional_coefficient * fractional
    )
    residual = advection_term + diffusion_term
    scale = (
        torch.sqrt(
            torch.mean(advection_term.detach() ** 2)
            + torch.mean(diffusion_term.detach() ** 2)
        )
        + 1.0e-12
    )
    normalized_residual = residual / scale
    pde_loss = torch.mean(normalized_residual**2)

    centered_fft = torch.fft.fft2(density - torch.mean(density))
    power = torch.abs(centered_fft) ** 2
    spectral_tail_fraction = (
        torch.sum(power[tail_mask])
        / (torch.sum(power) + 1.0e-12)
    )
    return pde_loss, spectral_tail_fraction, {
        "pde_residual_RMSE": torch.sqrt(
            torch.mean(residual**2)
        ),
        "normalized_pde_residual_RMSE": torch.sqrt(
            torch.mean(normalized_residual**2)
        ),
        "q_mean": torch.mean(q),
        "density_mass": torch.mean(density),
    }


def normalized_sample_nll(
    network: ConservativeLogDensityNetwork,
    data_points: torch.Tensor,
    uniform_points: torch.Tensor,
) -> torch.Tensor:
    q_data = network(data_points)
    q_uniform = network(uniform_points)
    log_normalizer = (
        torch.logsumexp(q_uniform.reshape(-1), dim=0)
        - math.log(q_uniform.numel())
    )
    return -torch.mean(q_data) + log_normalizer


def fixed_stage2_validation(
    validation_samples: torch.Tensor,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 41001)
    validation_count = min(
        cfg.stage2_validation_batch_size,
        validation_samples.shape[0],
    )
    indices = torch.randperm(
        validation_samples.shape[0],
        device=device,
        generator=generator,
    )[:validation_count]
    data = validation_samples[indices].detach().clone()
    side = max(
        16,
        int(round(math.sqrt(cfg.stage2_uniform_batch_size))),
    )
    axis = torch.arange(
        side,
        device=device,
        dtype=dtype,
    ) / side
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    uniform = torch.stack(
        [x.reshape(-1), y.reshape(-1)],
        dim=1,
    )
    return {"data": data, "uniform": uniform}


def evaluate_validation_nll(
    network: ConservativeLogDensityNetwork,
    validation_batch: Dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        value = normalized_sample_nll(
            network,
            validation_batch["data"],
            validation_batch["uniform"],
        )
    return float(value.detach().cpu())


def teacher_consistency_loss(
    student: ConservativeLogDensityNetwork,
    teacher: ConservativeLogDensityNetwork,
    points: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    student_points = points.detach().clone()
    student_points.requires_grad_(True)
    student_score, _ = scaled_score_from_log_density(
        student,
        student_points,
        cfg.score_scale,
        create_graph=True,
    )

    teacher_points = points.detach().clone()
    teacher_points.requires_grad_(True)
    teacher_score, _ = scaled_score_from_log_density(
        teacher,
        teacher_points,
        cfg.score_scale,
        create_graph=False,
    )
    return torch.mean((student_score - teacher_score.detach()) ** 2)


def train_stage2(
    density_network: ConservativeLogDensityNetwork,
    teacher_network: ConservativeLogDensityNetwork,
    training_samples: torch.Tensor,
    validation_samples: torch.Tensor,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(
        density_network.parameters(),
        lr=cfg.learning_rate_stage2,
        weight_decay=cfg.stage2_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.stage2_iterations, 1),
        eta_min=0.05 * cfg.learning_rate_stage2,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + 40000)
    (
        grid_points,
        drift,
        kx,
        ky,
        symbol,
        tail_mask,
    ) = build_pde_grid(cfg, device, dtype)
    validation_batch = fixed_stage2_validation(
        validation_samples,
        cfg,
        device,
        dtype,
    )

    teacher_network.eval()
    for parameter in teacher_network.parameters():
        parameter.requires_grad_(False)

    checkpoint_dir = output_dir / "stage2_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    selection_rows: List[Dict[str, float | str]] = []

    last_pde_loss = torch.zeros((), device=device, dtype=dtype)
    last_spectral = torch.zeros((), device=device, dtype=dtype)
    last_diagnostics = {
        "pde_residual_RMSE": torch.zeros(
            (),
            device=device,
            dtype=dtype,
        ),
        "normalized_pde_residual_RMSE": torch.ones(
            (),
            device=device,
            dtype=dtype,
        ),
    }

    for iteration in trange(
        cfg.stage2_iterations,
        desc="Stage II likelihood + score + fractional PDE",
    ):
        progress = (
            iteration / max(cfg.stage2_iterations - 1, 1)
        )
        teacher_weight = (
            cfg.stage2_teacher_weight_start
            + progress
            * (
                cfg.stage2_teacher_weight_end
                - cfg.stage2_teacher_weight_start
            )
        )
        if cfg.stage2_pde_weight_start > 0:
            pde_weight = (
                cfg.stage2_pde_weight_start
                * (
                    cfg.stage2_pde_weight_end
                    / cfg.stage2_pde_weight_start
                )
                ** progress
            )
        else:
            pde_weight = (
                progress * cfg.stage2_pde_weight_end
            )

        data_indices = torch.randint(
            0,
            training_samples.shape[0],
            (cfg.stage2_batch_size,),
            device=device,
            generator=generator,
        )
        data_points = training_samples[data_indices]
        uniform_points = torch.rand(
            (cfg.stage2_uniform_batch_size, 2),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        nll = normalized_sample_nll(
            density_network,
            data_points,
            uniform_points,
        )

        consistency_count = min(1024, cfg.stage2_batch_size)
        consistency = teacher_consistency_loss(
            density_network,
            teacher_network,
            data_points[:consistency_count],
            cfg,
        )

        if iteration % cfg.pde_every == 0:
            (
                pde_loss,
                spectral_tail,
                diagnostics,
            ) = fractional_pde_loss(
                density_network,
                grid_points,
                drift,
                kx,
                ky,
                symbol,
                tail_mask,
                cfg,
            )
            last_pde_loss = pde_loss
            last_spectral = spectral_tail
            last_diagnostics = diagnostics
            active_pde_loss = pde_loss
            active_spectral = spectral_tail
        else:
            active_pde_loss = last_pde_loss.detach()
            active_spectral = last_spectral.detach()

        gauge = density_network(uniform_points).mean().pow(2)
        total = (
            cfg.stage2_nll_weight * nll
            + teacher_weight * consistency
            + pde_weight * active_pde_loss
            + cfg.stage2_spectral_tail_weight * active_spectral
            + cfg.gauge_weight * gauge
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            density_network.parameters(),
            cfg.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        validation_nll = float("nan")
        is_selection_step = (
            (iteration + 1) % cfg.stage2_validation_every == 0
            or iteration == 0
            or iteration + 1 == cfg.stage2_iterations
        )
        if is_selection_step:
            validation_nll = evaluate_validation_nll(
                density_network,
                validation_batch,
            )
            if iteration % cfg.pde_every != 0:
                with torch.enable_grad():
                    (
                        evaluation_pde,
                        evaluation_spectral,
                        evaluation_diagnostics,
                    ) = fractional_pde_loss(
                        density_network,
                        grid_points,
                        drift,
                        kx,
                        ky,
                        symbol,
                        tail_mask,
                        cfg,
                    )
                pde_value = float(
                    evaluation_diagnostics[
                        "normalized_pde_residual_RMSE"
                    ].detach().cpu()
                )
                spectral_value = float(
                    evaluation_spectral.detach().cpu()
                )
            else:
                pde_value = float(
                    last_diagnostics[
                        "normalized_pde_residual_RMSE"
                    ].detach().cpu()
                )
                spectral_value = float(
                    last_spectral.detach().cpu()
                )

            checkpoint_path = (
                checkpoint_dir
                / f"stage2_iter_{iteration + 1:05d}.pt"
            )
            torch.save(
                {
                    "state_dict": density_network.state_dict(),
                    "iteration": iteration + 1,
                    "validation_nll": validation_nll,
                    "normalized_pde_residual": pde_value,
                    "spectral_tail_fraction": spectral_value,
                },
                checkpoint_path,
            )
            selection_rows.append(
                {
                    "iteration": iteration + 1,
                    "validation_nll": validation_nll,
                    "normalized_pde_residual": pde_value,
                    "spectral_tail_fraction": spectral_value,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

        history.append(
            {
                "iteration": iteration + 1,
                "total_loss": float(total.detach().cpu()),
                "data_nll": float(nll.detach().cpu()),
                "consistency_loss": float(
                    consistency.detach().cpu()
                ),
                "pde_loss": float(
                    active_pde_loss.detach().cpu()
                ),
                "spectral_tail_fraction": float(
                    active_spectral.detach().cpu()
                ),
                "gauge_loss": float(gauge.detach().cpu()),
                "teacher_weight": float(teacher_weight),
                "pde_weight": float(pde_weight),
                "pde_residual_RMSE": float(
                    last_diagnostics[
                        "pde_residual_RMSE"
                    ].detach().cpu()
                ),
                "normalized_pde_residual_RMSE": float(
                    last_diagnostics[
                        "normalized_pde_residual_RMSE"
                    ].detach().cpu()
                ),
                "validation_nll": validation_nll,
                "gradient_norm_before_clip": float(
                    gradient_norm.detach().cpu()
                ),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(
        output_dir / "stage2_history.csv",
        index=False,
    )
    selection_frame = pd.DataFrame(selection_rows)
    global_best_nll = float(
        selection_frame["validation_nll"].min()
    )
    selection_frame["nll_excess"] = np.maximum(
        selection_frame["validation_nll"]
        - global_best_nll
        - cfg.stage2_nll_selection_tolerance,
        0.0,
    )
    selection_frame["selection_metric"] = (
        selection_frame["normalized_pde_residual"]
        + 5.0 * selection_frame["nll_excess"]
        + 0.05 * selection_frame["spectral_tail_fraction"]
    )
    eligible = selection_frame[
        selection_frame["iteration"]
        >= cfg.stage2_min_selection_iteration
    ].copy()
    if eligible.empty:
        eligible = selection_frame.copy()
    selected_row = eligible.sort_values(
        ["selection_metric", "iteration"],
        ascending=[True, True],
    ).iloc[0]
    selected_checkpoint = torch.load(
        selected_row["checkpoint_path"],
        map_location=device,
        weights_only=False,
    )
    density_network.load_state_dict(
        selected_checkpoint["state_dict"]
    )
    selection_frame["selected"] = (
        selection_frame["iteration"]
        == int(selected_row["iteration"])
    )
    selection_frame.to_csv(
        output_dir / "stage2_selection.csv",
        index=False,
    )
    torch.save(
        selected_checkpoint,
        output_dir / "stage2_selected_checkpoint.pt",
    )
    return {
        "selected_iteration": int(selected_row["iteration"]),
        "selected_validation_nll": float(
            selected_row["validation_nll"]
        ),
        "selected_normalized_pde_residual": float(
            selected_row["normalized_pde_residual"]
        ),
        "selected_spectral_tail_fraction": float(
            selected_row["spectral_tail_fraction"]
        ),
        "global_best_validation_nll": global_best_nll,
    }


def evaluate_density_network(
    teacher_network: ConservativeLogDensityNetwork,
    density_network: ConservativeLogDensityNetwork,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, float]:
    reference_payload = np.load(
        reference_dir / "reference_density.npz"
    )
    reference = reference_payload["density"]
    grid_size = cfg.evaluation_grid_size
    if reference.shape != (grid_size, grid_size):
        raise ValueError(
            "Reference grid and evaluation grid must match."
        )
    _, _, points_np = periodic_grid(grid_size)

    q_blocks: List[np.ndarray] = []
    score_blocks: List[np.ndarray] = []
    block_size = 32768
    for start in range(0, len(points_np), block_size):
        block = torch.tensor(
            points_np[start : start + block_size],
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            q_blocks.append(
                density_network(block)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        score_points = block.detach().clone()
        score_points.requires_grad_(True)
        teacher_score, _ = scaled_score_from_log_density(
            teacher_network,
            score_points,
            cfg.score_scale,
            create_graph=False,
        )
        score_blocks.append(
            teacher_score.detach().cpu().numpy()
        )

    q_values = np.concatenate(q_blocks).reshape(
        grid_size,
        grid_size,
    )
    shifted = q_values - np.max(q_values)
    model = np.exp(shifted)
    model /= np.mean(model)

    metrics = density_metrics(model, reference)
    metrics.update(
        torus_sliced_wasserstein(
            points_np,
            model.reshape(-1),
            reference.reshape(-1),
            cfg,
        )
    )

    reference_log = np.log(
        np.maximum(reference, 1.0e-10)
    )
    kx, ky, _ = fractional_symbol(grid_size, cfg)
    reference_score_x = np.fft.ifft2(
        1j * kx * np.fft.fft2(reference_log)
    ).real
    reference_score_y = np.fft.ifft2(
        1j * ky * np.fft.fft2(reference_log)
    ).real
    reference_scaled_score = np.column_stack(
        [
            reference_score_x.reshape(-1),
            reference_score_y.reshape(-1),
        ]
    ) / cfg.score_scale
    learned_scaled_score = np.vstack(score_blocks)
    probability_weights = reference.reshape(-1).copy()
    probability_weights /= probability_weights.sum()
    score_difference = (
        learned_scaled_score - reference_scaled_score
    )
    metrics["reference_weighted_scaled_score_RMSE"] = float(
        np.sqrt(
            np.sum(
                probability_weights[:, None]
                * score_difference**2
            )
        )
    )

    audit_size = cfg.residual_audit_grid_size
    _, _, audit_points = periodic_grid(audit_size)
    q_audit_blocks: List[np.ndarray] = []
    for start in range(0, len(audit_points), block_size):
        block = torch.tensor(
            audit_points[start : start + block_size],
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            q_audit_blocks.append(
                density_network(block)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
    q_audit = np.concatenate(q_audit_blocks).reshape(
        audit_size,
        audit_size,
    )
    audit_density = np.exp(q_audit - np.max(q_audit))
    audit_density /= np.mean(audit_density)
    audit_drift = drift_numpy(
        audit_points,
        cfg,
    ).reshape(audit_size, audit_size, 2)
    audit_kx, audit_ky, audit_symbol = fractional_symbol(
        audit_size,
        cfg,
    )
    flux_x = audit_drift[:, :, 0] * audit_density
    flux_y = audit_drift[:, :, 1] * audit_density
    div_flux = np.fft.ifft2(
        1j * audit_kx * np.fft.fft2(flux_x)
        + 1j * audit_ky * np.fft.fft2(flux_y)
    ).real
    fractional = np.fft.ifft2(
        audit_symbol * np.fft.fft2(audit_density)
    ).real
    advection_term = -div_flux
    diffusion_term = (
        -cfg.fractional_coefficient * fractional
    )
    residual = advection_term + diffusion_term
    term_scale = np.sqrt(
        np.mean(advection_term**2)
        + np.mean(diffusion_term**2)
    )
    metrics["evaluation_pde_residual_RMSE"] = float(
        np.sqrt(np.mean(residual**2))
    )
    metrics["evaluation_pde_residual_normalized_RMSE"] = float(
        metrics["evaluation_pde_residual_RMSE"]
        / max(float(term_scale), 1.0e-15)
    )

    centered_fft = np.fft.fft2(model - np.mean(model))
    cycles = np.fft.fftfreq(
        grid_size,
        d=1.0 / grid_size,
    )
    fx, fy = np.meshgrid(cycles, cycles, indexing="xy")
    tail_mask = (
        np.sqrt(fx * fx + fy * fy)
        > cfg.stage2_spectral_cutoff
    )
    power = np.abs(centered_fft) ** 2
    metrics["evaluation_spectral_tail_fraction"] = float(
        np.sum(power[tail_mask])
        / max(float(np.sum(power)), 1.0e-300)
    )

    np.savez_compressed(
        output_dir / "sbpinn_evaluation.npz",
        model_density=model,
        reference_density=reference,
        log_density=q_values,
        audit_density=audit_density,
    )
    return metrics


def run_sbpinn(
    cfg: BenchmarkConfig,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.validate()
    set_seeds(cfg.seed)
    device = choose_device(cfg.device)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    train_payload = np.load(
        reference_dir / "train_samples.npz"
    )
    train_samples_np = train_payload["samples"].astype(
        np.float64,
        copy=False,
    )
    training_np, validation_np = split_train_validation(
        train_samples_np,
        cfg.seed + 29000,
    )
    training_samples = torch.tensor(
        training_np,
        device=device,
        dtype=dtype,
    )
    validation_samples = torch.tensor(
        validation_np,
        device=device,
        dtype=dtype,
    )

    teacher_network = ConservativeLogDensityNetwork(cfg).to(
        device=device,
        dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    stage1_summary = train_stage1(
        teacher_network,
        training_samples,
        validation_samples,
        cfg,
        device,
        dtype,
        output_dir,
    )
    stage1_time = time.perf_counter() - start

    density_network = ConservativeLogDensityNetwork(cfg).to(
        device=device,
        dtype=dtype,
    )
    density_network.load_state_dict(
        copy.deepcopy(teacher_network.state_dict())
    )

    stage2_start = time.perf_counter()
    stage2_summary = train_stage2(
        density_network,
        teacher_network,
        training_samples,
        validation_samples,
        cfg,
        device,
        dtype,
        output_dir,
    )
    stage2_time = time.perf_counter() - stage2_start

    metrics = evaluate_density_network(
        teacher_network,
        density_network,
        cfg,
        device,
        dtype,
        reference_dir,
        output_dir,
    )
    peak_cuda = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in teacher_network.parameters()
    ) + sum(
        parameter.numel()
        for parameter in density_network.parameters()
    )
    metrics.update(
        {
            "alpha": cfg.alpha,
            "noise_scale": cfg.noise_scale,
            "fractional_coefficient": (
                cfg.fractional_coefficient
            ),
            "device": str(device),
            "parameter_count": parameter_count,
            "train_sample_count": int(len(training_np)),
            "validation_sample_count": int(len(validation_np)),
            "stage1_training_time_s": stage1_time,
            "stage2_training_time_s": stage2_time,
            "total_training_time_s": stage1_time + stage2_time,
            "peak_cuda_memory_mb": peak_cuda,
            "stage1": stage1_summary,
            "stage2": stage2_summary,
            "conservative_score_parameterization": True,
            "stage1_objective": "multi-noise torus denoising score matching",
            "stage2_objective": (
                "held-out likelihood + conservative-score consistency "
                "+ continuation fractional PDE residual"
            ),
        }
    )
    torch.save(
        {
            "teacher_network": teacher_network.state_dict(),
            "density_network": density_network.state_dict(),
            "config": cfg.__dict__,
            "stage1": stage1_summary,
            "stage2": stage2_summary,
        },
        output_dir / "sbpinn_checkpoint.pt",
    )
    write_json(output_dir / "sbpinn_summary.json", metrics)
    return metrics
