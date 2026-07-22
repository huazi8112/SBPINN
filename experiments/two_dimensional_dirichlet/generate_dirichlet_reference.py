from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from dirichlet_common import (
    BenchmarkConfig,
    cell_center_grid,
    cic_density,
    config_to_dict,
    drift_numpy,
    initial_density_numpy,
    interpolate_density,
    normalize_grid_density,
    relative_l2,
    set_seeds,
    time_key,
    write_json,
    zero_padded_gaussian_smooth,
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


def drift_torch(
    points: torch.Tensor,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    two_pi = 2.0 * math.pi
    four_pi = 4.0 * math.pi
    dx_potential = (
        cfg.potential_strength
        * four_pi
        * torch.sin(four_pi * (x - cfg.x_shift))
        + cfg.coupling_strength
        * two_pi
        * torch.sin(two_pi * (x - y - cfg.coupling_phase))
        + cfg.asymmetry_strength
        * two_pi
        * torch.cos(two_pi * x + 0.31)
        * torch.sin(four_pi * y - 0.27)
    )
    dy_potential = (
        cfg.potential_strength
        * four_pi
        * torch.sin(four_pi * (y - cfg.y_shift))
        - cfg.coupling_strength
        * two_pi
        * torch.sin(two_pi * (x - y - cfg.coupling_phase))
        + cfg.asymmetry_strength
        * four_pi
        * torch.sin(two_pi * x + 0.31)
        * torch.cos(four_pi * y - 0.27)
    )
    stream_x = two_pi * cfg.rotation_strength * (
        torch.sin(two_pi * (x + cfg.stream_x_shift))
        * torch.cos(two_pi * (y + cfg.stream_y_shift))
    )
    stream_y = -two_pi * cfg.rotation_strength * (
        torch.cos(two_pi * (x + cfg.stream_x_shift))
        * torch.sin(two_pi * (y + cfg.stream_y_shift))
    )
    return torch.stack(
        [-dx_potential + stream_x, -dy_potential + stream_y],
        dim=1,
    )


def symmetric_alpha_stable(
    shape: Tuple[int, ...],
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    epsilon = torch.finfo(dtype).eps
    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    uniform = torch.clamp(uniform, epsilon, 1.0 - epsilon)
    angle = math.pi * (uniform - 0.5)
    exponential_uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    exponential_uniform = torch.clamp(
        exponential_uniform,
        epsilon,
        1.0 - epsilon,
    )
    exponential = -torch.log(exponential_uniform)
    return (
        torch.sin(alpha * angle)
        / torch.cos(angle).pow(1.0 / alpha)
        * (
            torch.cos((1.0 - alpha) * angle) / exponential
        ).pow((1.0 - alpha) / alpha)
    )


def sample_initial_points(
    count: int,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    points = torch.empty((count, 2), device=device, dtype=dtype)
    remaining = torch.arange(count, device=device)
    while len(remaining) > 0:
        mixture = torch.rand(
            (len(remaining),),
            device=device,
            dtype=dtype,
            generator=generator,
        ) < cfg.initial_weight_1
        candidate = torch.empty(
            (len(remaining), 2),
            device=device,
            dtype=dtype,
        )
        if torch.any(mixture):
            candidate[mixture] = (
                torch.tensor(
                    cfg.initial_mean_1,
                    device=device,
                    dtype=dtype,
                )
                + cfg.initial_std_1
                * torch.randn(
                    (int(mixture.sum()), 2),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        if torch.any(~mixture):
            candidate[~mixture] = (
                torch.tensor(
                    cfg.initial_mean_2,
                    device=device,
                    dtype=dtype,
                )
                + cfg.initial_std_2
                * torch.randn(
                    (int((~mixture).sum()), 2),
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        accepted = torch.all(
            (candidate > 0.0) & (candidate < 1.0),
            dim=1,
        )
        points[remaining[accepted]] = candidate[accepted]
        remaining = remaining[~accepted]
    return points


def simulate_killed_ensemble(
    cfg: BenchmarkConfig,
    count: int,
    dt: float,
    seed: int,
    device: torch.device,
    description: str,
) -> Tuple[Dict[float, np.ndarray], Dict[float, float], Dict[str, object]]:
    dtype = torch.float64
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    points = sample_initial_points(
        count,
        cfg,
        device,
        dtype,
        generator,
    )
    alive = torch.ones(count, device=device, dtype=torch.bool)
    snapshot_steps = {
        int(round(time_value / dt)): time_value
        for time_value in cfg.snapshot_times
    }
    total_steps = int(round(cfg.final_time / dt))
    increment_scale = cfg.noise_scale * dt ** (1.0 / cfg.alpha)
    snapshots: Dict[float, np.ndarray] = {}
    survival: Dict[float, float] = {}
    start = time.perf_counter()

    for step in trange(total_steps, desc=description):
        active_indices = torch.nonzero(
            alive,
            as_tuple=False,
        ).reshape(-1)
        if len(active_indices) > 0:
            active_points = points[active_indices]
            stable_noise = symmetric_alpha_stable(
                active_points.shape,
                cfg.alpha,
                device,
                dtype,
                generator,
            )
            updated = (
                active_points
                + drift_torch(active_points, cfg) * dt
                + increment_scale * stable_noise
            )
            remains_inside = torch.all(
                (updated > 0.0) & (updated < 1.0),
                dim=1,
            )
            points[active_indices[remains_inside]] = updated[
                remains_inside
            ]
            alive[active_indices[~remains_inside]] = False

        completed_step = step + 1
        if completed_step in snapshot_steps:
            time_value = snapshot_steps[completed_step]
            surviving_points = points[alive]
            snapshots[time_value] = (
                surviving_points.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            survival[time_value] = float(
                alive.double().mean().detach().cpu()
            )

    return snapshots, survival, {
        "particle_count": count,
        "dt": dt,
        "seed": seed,
        "device": str(device),
        "runtime_s": time.perf_counter() - start,
        "final_survival": float(alive.double().mean().cpu()),
    }


def build_extended_symbol(
    interior_size: int,
    padding_factor: int,
    cfg: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, int]:
    extended_size = padding_factor * interior_size
    dx = 1.0 / interior_size
    angular = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            extended_size,
            d=dx,
            device=device,
            dtype=dtype,
        )
    )
    kx, ky = torch.meshgrid(angular, angular, indexing="xy")
    symbol = (
        torch.abs(kx) ** cfg.alpha
        + torch.abs(ky) ** cfg.alpha
    )
    return symbol, extended_size


def diffusion_step(
    density: torch.Tensor,
    dt: float,
    symbol: torch.Tensor,
    padding_factor: int,
    cfg: BenchmarkConfig,
) -> Tuple[torch.Tensor, float]:
    n = density.shape[0]
    extended_size = padding_factor * n
    start = (extended_size - n) // 2
    extended = torch.zeros(
        (extended_size, extended_size),
        device=density.device,
        dtype=density.dtype,
    )
    extended[start : start + n, start : start + n] = density
    multiplier = torch.exp(
        -cfg.fractional_coefficient * symbol * dt
    )
    evolved = torch.fft.ifft2(
        multiplier * torch.fft.fft2(extended)
    ).real
    cropped = evolved[start : start + n, start : start + n]
    negative_before = float(
        torch.mean(torch.clamp(-cropped, min=0.0)).detach().cpu()
    )
    return torch.clamp(cropped, min=0.0), negative_before


def advection_upwind_step(
    density: torch.Tensor,
    dt: float,
    cfg: BenchmarkConfig,
) -> torch.Tensor:
    n = density.shape[0]
    dx = 1.0 / n
    device = density.device
    dtype = density.dtype

    x_faces = torch.arange(
        n + 1,
        device=device,
        dtype=dtype,
    ) / n
    y_centers = (
        torch.arange(n, device=device, dtype=dtype) + 0.5
    ) / n
    xf, yc = torch.meshgrid(x_faces, y_centers, indexing="xy")
    face_x_points = torch.stack(
        [xf.reshape(-1), yc.reshape(-1)],
        dim=1,
    )
    velocity_x = drift_torch(
        face_x_points,
        cfg,
    )[:, 0].reshape(n, n + 1)

    x_centers = (
        torch.arange(n, device=device, dtype=dtype) + 0.5
    ) / n
    y_faces = torch.arange(
        n + 1,
        device=device,
        dtype=dtype,
    ) / n
    xc, yf = torch.meshgrid(x_centers, y_faces, indexing="xy")
    face_y_points = torch.stack(
        [xc.reshape(-1), yf.reshape(-1)],
        dim=1,
    )
    velocity_y = drift_torch(
        face_y_points,
        cfg,
    )[:, 1].reshape(n + 1, n)

    padded_x = torch.nn.functional.pad(
        density,
        (1, 1, 0, 0),
        mode="constant",
        value=0.0,
    )
    left = padded_x[:, :-1]
    right = padded_x[:, 1:]
    flux_x = torch.where(
        velocity_x >= 0.0,
        velocity_x * left,
        velocity_x * right,
    )

    padded_y = torch.nn.functional.pad(
        density,
        (0, 0, 1, 1),
        mode="constant",
        value=0.0,
    )
    bottom = padded_y[:-1, :]
    top = padded_y[1:, :]
    flux_y = torch.where(
        velocity_y >= 0.0,
        velocity_y * bottom,
        velocity_y * top,
    )

    divergence = (
        (flux_x[:, 1:] - flux_x[:, :-1]) / dx
        + (flux_y[1:, :] - flux_y[:-1, :]) / dx
    )
    return torch.clamp(density - dt * divergence, min=0.0)


def solve_reference_pde(
    cfg: BenchmarkConfig,
    grid_size: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    dtype = torch.float64
    _, _, points = cell_center_grid(grid_size)
    initial = initial_density_numpy(points, cfg).reshape(
        grid_size,
        grid_size,
    )
    initial = normalize_grid_density(initial)
    density = torch.tensor(initial, device=device, dtype=dtype)
    symbol, _ = build_extended_symbol(
        grid_size,
        cfg.reference_padding_factor,
        cfg,
        device,
        dtype,
    )
    dt = cfg.reference_time_step
    total_steps = int(round(cfg.final_time / dt))
    evaluation_steps = {
        int(round(time_value / dt)): time_value
        for time_value in cfg.evaluation_times
    }
    snapshots: Dict[float, np.ndarray] = {}
    negative_correction = 0.0
    start = time.perf_counter()

    for step in trange(
        total_steps,
        desc=f"Zero-extension PDE reference N={grid_size}",
    ):
        density, correction_1 = diffusion_step(
            density,
            0.5 * dt,
            symbol,
            cfg.reference_padding_factor,
            cfg,
        )
        density = advection_upwind_step(density, dt, cfg)
        density, correction_2 = diffusion_step(
            density,
            0.5 * dt,
            symbol,
            cfg.reference_padding_factor,
            cfg,
        )
        negative_correction += correction_1 + correction_2

        completed_step = step + 1
        if completed_step in evaluation_steps:
            time_value = evaluation_steps[completed_step]
            snapshots[time_value] = density.detach().cpu().numpy()

    payload = {
        "times": np.array(cfg.evaluation_times),
        "densities": np.stack(
            [snapshots[t] for t in cfg.evaluation_times],
            axis=0,
        ),
    }
    np.savez_compressed(
        output_dir / f"pde_reference_n{grid_size}.npz",
        **payload,
    )
    return {
        "grid_size": grid_size,
        "runtime_s": time.perf_counter() - start,
        "negative_correction_accumulator": negative_correction,
        "masses": {
            str(t): float(np.mean(snapshots[t]))
            for t in cfg.evaluation_times
        },
    }


def generate_reference_data(
    cfg: BenchmarkConfig,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.validate()
    set_seeds(cfg.seed)
    device = choose_device(cfg.device)

    train_snapshots, train_survival, train_meta = (
        simulate_killed_ensemble(
            cfg,
            cfg.train_particles,
            cfg.simulation_dt,
            cfg.seed + 1000,
            device,
            "Killed-SDE training ensemble",
        )
    )
    train_payload: Dict[str, np.ndarray] = {
        "times": np.array(cfg.snapshot_times),
        "survival": np.array(
            [train_survival[t] for t in cfg.snapshot_times]
        ),
    }
    for time_value, samples in train_snapshots.items():
        train_payload[time_key(time_value)] = samples
    np.savez_compressed(
        output_dir / "train_killed_samples.npz",
        **train_payload,
    )

    replicate_snapshots: List[Dict[float, np.ndarray]] = []
    replicate_survival: List[Dict[float, float]] = []
    replicate_meta: List[Dict[str, object]] = []
    for replicate in range(cfg.reference_replicates):
        snapshots, survival, metadata = simulate_killed_ensemble(
            cfg,
            cfg.reference_particles,
            cfg.simulation_dt,
            cfg.seed + 3000 + 1000 * replicate,
            device,
            f"Killed-SDE reference replicate {replicate + 1}",
        )
        replicate_snapshots.append(snapshots)
        replicate_survival.append(survival)
        replicate_meta.append(metadata)
        payload = {
            "times": np.array(cfg.snapshot_times),
            "survival": np.array(
                [survival[t] for t in cfg.snapshot_times]
            ),
        }
        for time_value, samples in snapshots.items():
            payload[time_key(time_value)] = samples
        np.savez_compressed(
            output_dir
            / f"reference_killed_replicate_{replicate + 1}.npz",
            **payload,
        )

    refined_snapshots, refined_survival, refined_meta = (
        simulate_killed_ensemble(
            cfg,
            cfg.refined_reference_particles,
            cfg.simulation_dt * cfg.refined_dt_factor,
            cfg.seed + 9000,
            device,
            "Killed-SDE dt-refined ensemble",
        )
    )
    refined_payload = {
        "times": np.array(cfg.snapshot_times),
        "survival": np.array(
            [refined_survival[t] for t in cfg.snapshot_times]
        ),
    }
    for time_value, samples in refined_snapshots.items():
        refined_payload[time_key(time_value)] = samples
    np.savez_compressed(
        output_dir / "reference_killed_refined_dt.npz",
        **refined_payload,
    )

    reference_mc_survival = {
        time_value: float(
            np.mean(
                [
                    survival[time_value]
                    for survival in replicate_survival
                ]
            )
        )
        for time_value in cfg.snapshot_times
    }
    reference_mc_conditional: Dict[float, np.ndarray] = {}
    reference_mc_replicate_densities: Dict[
        float, List[np.ndarray]
    ] = {}
    finest_size = max(cfg.reference_grid_sizes)
    for time_value in cfg.evaluation_times:
        replicate_densities = []
        for snapshots in replicate_snapshots:
            density = cic_density(
                snapshots[time_value],
                finest_size,
            )
            density = zero_padded_gaussian_smooth(
                density,
                cfg.reference_kde_bandwidth,
            )
            replicate_densities.append(density)
        reference_mc_replicate_densities[time_value] = (
            replicate_densities
        )
        reference_mc_conditional[time_value] = normalize_grid_density(
            np.mean(np.stack(replicate_densities), axis=0)
        )

    pde_runs = [
        solve_reference_pde(cfg, size, device, output_dir)
        for size in cfg.reference_grid_sizes
    ]
    pde_payloads = {
        size: np.load(
            output_dir / f"pde_reference_n{size}.npz"
        )
        for size in cfg.reference_grid_sizes
    }
    finest_size = max(cfg.reference_grid_sizes)
    finest_payload = pde_payloads[finest_size]
    finest_densities = {
        time_value: finest_payload["densities"][index]
        for index, time_value in enumerate(cfg.evaluation_times)
    }
    np.savez_compressed(
        output_dir / "reference_pde_finest.npz",
        times=np.array(cfg.evaluation_times),
        densities=np.stack(
            [finest_densities[t] for t in cfg.evaluation_times]
        ),
    )

    convergence_rows = []
    sorted_sizes = sorted(cfg.reference_grid_sizes)
    for smaller, larger in zip(sorted_sizes[:-1], sorted_sizes[1:]):
        smaller_payload = pde_payloads[smaller]
        larger_payload = pde_payloads[larger]
        for index, time_value in enumerate(cfg.evaluation_times):
            smaller_on_large = interpolate_density(
                smaller_payload["densities"][index],
                larger,
            )
            convergence_rows.append(
                {
                    "smaller_grid": smaller,
                    "larger_grid": larger,
                    "time": time_value,
                    "relative_L2": relative_l2(
                        smaller_on_large,
                        larger_payload["densities"][index],
                    ),
                }
            )
    convergence_frame = pd.DataFrame(convergence_rows)
    convergence_frame.to_csv(
        output_dir / "pde_reference_grid_convergence.csv",
        index=False,
    )

    replicate_shape_errors = []
    mass_errors = []
    mc_shape_errors = []
    dt_mass_errors = []
    for time_value in cfg.evaluation_times:
        mc_mean = reference_mc_conditional[time_value]
        for density in reference_mc_replicate_densities[time_value]:
            replicate_shape_errors.append(
                relative_l2(density, mc_mean)
            )
        pde_density = finest_densities[time_value]
        pde_mass = float(np.mean(pde_density))
        mc_mass = reference_mc_survival[time_value]
        mass_errors.append(abs(pde_mass - mc_mass))
        pde_conditional = pde_density / max(pde_mass, 1.0e-300)
        mc_shape_errors.append(
            relative_l2(pde_conditional, mc_mean)
        )
        dt_mass_errors.append(
            abs(
                refined_survival[time_value]
                - reference_mc_survival[time_value]
            )
        )

    maximum_grid_difference = float(
        convergence_frame[
            convergence_frame["larger_grid"] == finest_size
        ]["relative_L2"].max()
    )
    maximum_mass_error = float(max(mass_errors))
    maximum_mc_shape_error = float(max(mc_shape_errors))
    high_fidelity_reference = bool(
        maximum_grid_difference
        < cfg.reference_grid_convergence_threshold
        and maximum_mass_error < cfg.reference_mass_threshold
        and maximum_mc_shape_error
        < cfg.reference_mc_shape_threshold
    )

    summary = {
        "config": config_to_dict(cfg),
        "device": str(device),
        "train": train_meta,
        "reference_replicates": replicate_meta,
        "refined_reference": refined_meta,
        "pde_runs": pde_runs,
        "reference_mc_survival": {
            str(k): v for k, v in reference_mc_survival.items()
        },
        "maximum_pde_grid_relative_L2": maximum_grid_difference,
        "maximum_pde_vs_mc_mass_error": maximum_mass_error,
        "maximum_pde_vs_mc_conditional_relative_L2": (
            maximum_mc_shape_error
        ),
        "maximum_mc_replicate_relative_L2": float(
            max(replicate_shape_errors)
        ),
        "maximum_dt_refined_survival_error": float(
            max(dt_mass_errors)
        ),
        "high_fidelity_reference": high_fidelity_reference,
        "boundary_condition": (
            "homogeneous exterior Dirichlet; particles are killed "
            "after leaving Omega=(0,1)^2"
        ),
    }
    write_json(output_dir / "reference_summary.json", summary)
    return summary
