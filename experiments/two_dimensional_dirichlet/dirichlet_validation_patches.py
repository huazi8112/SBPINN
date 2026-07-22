from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import trange

from dirichlet_common import (
    BenchmarkConfig,
    cell_center_grid,
    cic_density,
    conditional_density_metrics,
    config_to_dict,
    drift_numpy,
    initial_density_numpy,
    interpolate_density,
    normalize_grid_density,
    relative_l2,
    set_seeds,
    sliced_wasserstein_2d,
    time_key,
    unnormalized_density_metrics,
    write_json,
    zero_padded_gaussian_smooth,
)
from generate_dirichlet_reference import (
    advection_upwind_step,
    build_extended_symbol,
    choose_device,
    diffusion_step,
)
from dirichlet_p1_fem import resource_estimate


def config_from_json(path: Path) -> BenchmarkConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = BenchmarkConfig()
    for key, value in payload.items():
        if key in ("fractional_coefficient", "fractional_constant"):
            continue
        if hasattr(cfg, key):
            current = getattr(cfg, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    cfg.boundary_power = 0.5 * cfg.alpha
    cfg.validate()
    return cfg


def audit_times(cfg: BenchmarkConfig) -> Tuple[float, ...]:
    values = set(float(t) for t in cfg.evaluation_times)
    for time_value in cfg.evaluation_times:
        delta = min(
            cfg.pde_time_delta,
            0.45 * time_value,
            0.45 * (cfg.final_time - time_value),
        )
        if delta <= 1.0e-8:
            delta = min(
                cfg.pde_time_delta,
                0.25 * cfg.final_time,
            )
        values.add(max(0.0, time_value - delta))
        values.add(min(cfg.final_time, time_value + delta))
    return tuple(sorted(values))


def solve_reference_pde_enhanced(
    cfg: BenchmarkConfig,
    grid_size: int,
    padding_factor: int,
    time_step: float,
    device: torch.device,
    output_path: Path,
    description: str,
) -> Dict[str, object]:
    """Run the zero-extension PDE reference and retain residual-audit times."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        padding_factor,
        cfg,
        device,
        dtype,
    )

    requested_times = audit_times(cfg)
    requested_steps: Dict[int, float] = {}
    for time_value in requested_times:
        step = int(round(time_value / time_step))
        represented = step * time_step
        if abs(represented - time_value) > 5.0e-11:
            raise ValueError(
                f"Requested time {time_value} is not represented by "
                f"dt={time_step}."
            )
        if step > 0:
            requested_steps[step] = time_value

    total_steps = int(round(cfg.final_time / time_step))
    snapshots: Dict[float, np.ndarray] = {}
    negative_correction = 0.0
    start_time = time.perf_counter()

    for step in trange(total_steps, desc=description):
        density, correction_1 = diffusion_step(
            density,
            0.5 * time_step,
            symbol,
            padding_factor,
            cfg,
        )
        density = advection_upwind_step(
            density,
            time_step,
            cfg,
        )
        density, correction_2 = diffusion_step(
            density,
            0.5 * time_step,
            symbol,
            padding_factor,
            cfg,
        )
        negative_correction += correction_1 + correction_2
        completed_step = step + 1
        if completed_step in requested_steps:
            time_value = requested_steps[completed_step]
            snapshots[time_value] = (
                density.detach().cpu().numpy()
            )

    evaluation_densities = np.stack(
        [snapshots[float(t)] for t in cfg.evaluation_times],
        axis=0,
    )
    all_densities = np.stack(
        [snapshots[float(t)] for t in requested_times],
        axis=0,
    )
    np.savez_compressed(
        output_path,
        times=np.asarray(cfg.evaluation_times, dtype=np.float64),
        densities=evaluation_densities,
        audit_times=np.asarray(requested_times, dtype=np.float64),
        audit_densities=all_densities,
        grid_size=np.asarray(grid_size),
        padding_factor=np.asarray(padding_factor),
        time_step=np.asarray(time_step),
    )
    return {
        "grid_size": grid_size,
        "padding_factor": padding_factor,
        "time_step": time_step,
        "runtime_s": time.perf_counter() - start_time,
        "negative_correction_accumulator": negative_correction,
        "masses": {
            str(t): float(np.mean(snapshots[float(t)]))
            for t in cfg.evaluation_times
        },
        "output": str(output_path),
    }


def load_existing_mc_reference(
    base_reference_dir: Path,
    cfg: BenchmarkConfig,
    grid_size: int,
) -> Tuple[
    Dict[float, np.ndarray],
    Dict[float, List[np.ndarray]],
    Dict[float, float],
    float,
]:
    replicates = []
    for index in range(1, cfg.reference_replicates + 1):
        path = (
            base_reference_dir
            / f"reference_killed_replicate_{index}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        replicates.append(np.load(path))

    means: Dict[float, np.ndarray] = {}
    replicate_densities: Dict[float, List[np.ndarray]] = {}
    survival: Dict[float, float] = {}
    replicate_errors = []

    for time_value in cfg.evaluation_times:
        densities = []
        survival_values = []
        for replicate in replicates:
            density = cic_density(
                replicate[time_key(time_value)],
                grid_size,
            )
            density = zero_padded_gaussian_smooth(
                density,
                cfg.reference_kde_bandwidth,
            )
            densities.append(density)
            time_index = int(
                np.where(
                    np.isclose(
                        replicate["times"],
                        time_value,
                    )
                )[0][0]
            )
            survival_values.append(
                float(replicate["survival"][time_index])
            )
        mean_density = normalize_grid_density(
            np.mean(np.stack(densities), axis=0)
        )
        means[time_value] = mean_density
        replicate_densities[time_value] = densities
        survival[time_value] = float(np.mean(survival_values))
        replicate_errors.extend(
            relative_l2(density, mean_density)
            for density in densities
        )
    return (
        means,
        replicate_densities,
        survival,
        float(max(replicate_errors)),
    )


def copy_or_interpolate_base_reference(
    base_reference_dir: Path,
    grid_size: int,
    target_path: Path,
    cfg: BenchmarkConfig,
) -> Dict[str, object]:
    """Reuse the already completed baseline reference when available."""
    direct = base_reference_dir / f"pde_reference_n{grid_size}.npz"
    if direct.exists():
        payload = np.load(direct)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_path,
            times=payload["times"],
            densities=payload["densities"],
            audit_times=payload["times"],
            audit_densities=payload["densities"],
            grid_size=np.asarray(grid_size),
            padding_factor=np.asarray(cfg.reference_padding_factor),
            time_step=np.asarray(cfg.reference_time_step),
        )
        return {
            "grid_size": grid_size,
            "padding_factor": cfg.reference_padding_factor,
            "time_step": cfg.reference_time_step,
            "runtime_s": 0.0,
            "reused_from": str(direct),
            "output": str(target_path),
        }
    raise FileNotFoundError(
        f"Baseline reference {direct} was not found."
    )


def central_advection_divergence_numpy(
    density: np.ndarray,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    n = density.shape[0]
    _, _, points = cell_center_grid(n)
    drift = drift_numpy(points, cfg).reshape(n, n, 2)
    flux_x = drift[:, :, 0] * density
    flux_y = drift[:, :, 1] * density
    dx = 1.0 / n
    padded_x = np.pad(
        flux_x,
        ((0, 0), (1, 1)),
        mode="constant",
    )
    padded_y = np.pad(
        flux_y,
        ((1, 1), (0, 0)),
        mode="constant",
    )
    return (
        (padded_x[:, 2:] - padded_x[:, :-2]) / (2.0 * dx)
        + (padded_y[2:, :] - padded_y[:-2, :]) / (2.0 * dx)
    )


def fractional_laplacian_zero_extension_numpy(
    density: np.ndarray,
    cfg: BenchmarkConfig,
    padding_factor: int,
) -> np.ndarray:
    n = density.shape[0]
    extended_size = padding_factor * n
    start = (extended_size - n) // 2
    extended = np.zeros(
        (extended_size, extended_size),
        dtype=np.float64,
    )
    extended[start : start + n, start : start + n] = density
    dx = 1.0 / n
    angular = (
        2.0
        * math.pi
        * np.fft.fftfreq(extended_size, d=dx)
    )
    kx, ky = np.meshgrid(angular, angular, indexing="xy")
    symbol = np.abs(kx) ** cfg.alpha + np.abs(ky) ** cfg.alpha
    operated = np.fft.ifft2(
        symbol * np.fft.fft2(extended)
    ).real
    return operated[start : start + n, start : start + n]


def density_at_time(
    payload: np.lib.npyio.NpzFile,
    time_value: float,
) -> np.ndarray:
    times = payload["audit_times"]
    index = int(np.where(np.isclose(times, time_value))[0][0])
    return payload["audit_densities"][index]


def reference_residual_audit(
    reference_payload_path: Path,
    cfg: BenchmarkConfig,
    audit_grid_size: int,
    padding_factor: int,
    output_path: Path,
) -> Dict[str, object]:
    payload = np.load(reference_payload_path)
    rows = []
    for time_value in cfg.evaluation_times:
        delta = min(
            cfg.pde_time_delta,
            0.45 * time_value,
            0.45 * (cfg.final_time - time_value),
        )
        if delta <= 1.0e-8:
            delta = min(
                cfg.pde_time_delta,
                0.25 * cfg.final_time,
            )
        t_minus = max(0.0, time_value - delta)
        t_plus = min(cfg.final_time, time_value + delta)
        p_minus = density_at_time(payload, t_minus)
        p_center = density_at_time(payload, time_value)
        p_plus = density_at_time(payload, t_plus)

        if p_center.shape[0] != audit_grid_size:
            p_minus = interpolate_density(p_minus, audit_grid_size)
            p_center = interpolate_density(p_center, audit_grid_size)
            p_plus = interpolate_density(p_plus, audit_grid_size)

        time_derivative = (p_plus - p_minus) / (
            t_plus - t_minus
        )
        advection = central_advection_divergence_numpy(
            p_center,
            cfg,
        )
        fractional = fractional_laplacian_zero_extension_numpy(
            p_center,
            cfg,
            padding_factor,
        )
        diffusion = cfg.fractional_coefficient * fractional
        residual = time_derivative + advection + diffusion
        scale = math.sqrt(
            float(np.mean(time_derivative**2))
            + float(np.mean(advection**2))
            + float(np.mean(diffusion**2))
        )
        rows.append(
            {
                "time": time_value,
                "audit_grid_size": audit_grid_size,
                "raw_residual_RMSE": float(
                    np.sqrt(np.mean(residual**2))
                ),
                "normalized_residual_RMSE": float(
                    np.sqrt(np.mean(residual**2))
                    / max(scale, 1.0e-15)
                ),
                "time_derivative_RMS": float(
                    np.sqrt(np.mean(time_derivative**2))
                ),
                "advection_RMS": float(
                    np.sqrt(np.mean(advection**2))
                ),
                "diffusion_RMS": float(
                    np.sqrt(np.mean(diffusion**2))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return {
        "audit_grid_size": audit_grid_size,
        "mean_normalized_residual_RMSE": float(
            frame["normalized_residual_RMSE"].mean()
        ),
        "maximum_normalized_residual_RMSE": float(
            frame["normalized_residual_RMSE"].max()
        ),
        "per_time": rows,
    }


def generate_validation_reference(
    cfg: BenchmarkConfig,
    base_results_dir: Path,
    patch_reference_dir: Path,
    device: torch.device,
    resume: bool,
    quick: bool,
) -> Dict[str, object]:
    patch_reference_dir.mkdir(parents=True, exist_ok=True)
    base_reference_dir = base_results_dir / "reference"
    base_grid = int(cfg.validation_reference_base_grid)
    validation_grids = tuple(
        int(value) for value in cfg.validation_reference_grids
    )
    baseline_grids = (base_grid,) + validation_grids

    runs = []
    payload_paths: Dict[int, Path] = {}
    for grid_size in baseline_grids:
        path = (
            patch_reference_dir
            / (
                f"pde_reference_n{grid_size}_"
                f"p{cfg.reference_padding_factor}_"
                f"dt{cfg.reference_time_step:.8f}.npz"
            )
        )
        payload_paths[grid_size] = path
        if resume and path.exists():
            runs.append(
                {
                    "grid_size": grid_size,
                    "padding_factor": cfg.reference_padding_factor,
                    "time_step": cfg.reference_time_step,
                    "runtime_s": 0.0,
                    "resumed": True,
                    "output": str(path),
                }
            )
            continue
        if grid_size == base_grid:
            runs.append(
                copy_or_interpolate_base_reference(
                    base_reference_dir,
                    grid_size,
                    path,
                    cfg,
                )
            )
        else:
            runs.append(
                solve_reference_pde_enhanced(
                    cfg,
                    grid_size,
                    cfg.reference_padding_factor,
                    cfg.reference_time_step,
                    device,
                    path,
                    (
                        f"Validation reference N={grid_size}, "
                        f"padding={cfg.reference_padding_factor}, "
                        f"dt={cfg.reference_time_step:g}"
                    ),
                )
            )

    finest_grid = max(baseline_grids)
    baseline_finest_path = payload_paths[finest_grid]
    sensitivity_specs = [
        (
            "padding",
            cfg.validation_padding_factor,
            cfg.reference_time_step,
        ),
        (
            "time_step",
            cfg.reference_padding_factor,
            cfg.validation_time_step,
        ),
    ]
    sensitivity_paths: Dict[str, Path] = {}
    for name, padding, time_step in sensitivity_specs:
        path = (
            patch_reference_dir
            / (
                f"pde_reference_n{finest_grid}_"
                f"p{padding}_dt{time_step:.8f}_{name}.npz"
            )
        )
        sensitivity_paths[name] = path
        if not (resume and path.exists()):
            solve_reference_pde_enhanced(
                cfg,
                finest_grid,
                padding,
                time_step,
                device,
                path,
                (
                    f"Sensitivity {name}, N={finest_grid}, "
                    f"padding={padding}, dt={time_step:g}"
                ),
            )

    baseline_payloads = {
        grid: np.load(path)
        for grid, path in payload_paths.items()
    }
    convergence_rows = []
    sorted_grids = sorted(baseline_grids)
    for smaller, larger in zip(
        sorted_grids[:-1],
        sorted_grids[1:],
    ):
        small_payload = baseline_payloads[smaller]
        large_payload = baseline_payloads[larger]
        for index, time_value in enumerate(cfg.evaluation_times):
            small_on_large = interpolate_density(
                small_payload["densities"][index],
                larger,
            )
            convergence_rows.append(
                {
                    "smaller_grid": smaller,
                    "larger_grid": larger,
                    "time": time_value,
                    "relative_L2": relative_l2(
                        small_on_large,
                        large_payload["densities"][index],
                    ),
                }
            )
    convergence = pd.DataFrame(convergence_rows)
    convergence.to_csv(
        patch_reference_dir
        / "pde_reference_grid_convergence.csv",
        index=False,
    )

    baseline_finest = np.load(baseline_finest_path)
    sensitivity_rows = []
    for name, path in sensitivity_paths.items():
        payload = np.load(path)
        for index, time_value in enumerate(cfg.evaluation_times):
            sensitivity_rows.append(
                {
                    "sensitivity": name,
                    "time": time_value,
                    "relative_L2": relative_l2(
                        payload["densities"][index],
                        baseline_finest["densities"][index],
                    ),
                    "mass_absolute_difference": abs(
                        float(np.mean(payload["densities"][index]))
                        - float(
                            np.mean(
                                baseline_finest["densities"][index]
                            )
                        )
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        patch_reference_dir
        / "pde_reference_sensitivity.csv",
        index=False,
    )

    (
        mc_conditional,
        mc_replicates,
        mc_survival,
        max_mc_replicate,
    ) = load_existing_mc_reference(
        base_reference_dir,
        cfg,
        finest_grid,
    )
    mass_errors = []
    shape_errors = []
    for index, time_value in enumerate(cfg.evaluation_times):
        pde_density = baseline_finest["densities"][index]
        pde_mass = float(np.mean(pde_density))
        mass_errors.append(
            abs(pde_mass - mc_survival[time_value])
        )
        shape_errors.append(
            relative_l2(
                pde_density / max(pde_mass, 1.0e-300),
                mc_conditional[time_value],
            )
        )

    np.savez_compressed(
        patch_reference_dir / "reference_pde_finest.npz",
        times=baseline_finest["times"],
        densities=baseline_finest["densities"],
        audit_times=baseline_finest["audit_times"],
        audit_densities=baseline_finest["audit_densities"],
    )

    residual_same_grid = reference_residual_audit(
        baseline_finest_path,
        cfg,
        int(cfg.reference_residual_audit_grid),
        cfg.pde_padding_factor,
        patch_reference_dir
        / "reference_residual_audit_stage2_grid.csv",
    )
    residual_native = reference_residual_audit(
        baseline_finest_path,
        cfg,
        finest_grid,
        cfg.reference_padding_factor,
        patch_reference_dir
        / "reference_residual_audit_native_grid.csv",
    )

    maximum_grid_difference = float(
        convergence[
            convergence["larger_grid"] == finest_grid
        ]["relative_L2"].max()
    )
    maximum_sensitivity = float(
        sensitivity["relative_L2"].max()
    )
    maximum_mass_error = float(max(mass_errors))
    maximum_shape_error = float(max(shape_errors))
    high_fidelity = bool(
        maximum_grid_difference
        < cfg.reference_grid_convergence_threshold
        and maximum_sensitivity
        < cfg.validation_sensitivity_threshold
        and maximum_mass_error < cfg.reference_mass_threshold
        and maximum_shape_error
        < cfg.reference_mc_shape_threshold
    )

    summary = {
        "config": config_to_dict(cfg),
        "device": str(device),
        "baseline_reference_runs": runs,
        "baseline_grid_sequence": baseline_grids,
        "finest_grid": finest_grid,
        "padding_sensitivity_factor": (
            cfg.validation_padding_factor
        ),
        "time_step_sensitivity": cfg.validation_time_step,
        "maximum_pde_grid_relative_L2": (
            maximum_grid_difference
        ),
        "maximum_padding_or_dt_relative_L2": (
            maximum_sensitivity
        ),
        "maximum_pde_vs_mc_mass_error": maximum_mass_error,
        "maximum_pde_vs_mc_conditional_relative_L2": (
            maximum_shape_error
        ),
        "maximum_mc_replicate_relative_L2": max_mc_replicate,
        "high_fidelity_reference": high_fidelity,
        "reference_residual_audit_stage2_grid": (
            residual_same_grid
        ),
        "reference_residual_audit_native_grid": residual_native,
        "boundary_condition": (
            "homogeneous exterior Dirichlet; killed alpha-stable "
            "process on Omega=(0,1)^2"
        ),
    }
    write_json(
        patch_reference_dir / "reference_summary.json",
        summary,
    )
    return summary


def reevaluate_sbpinn_checkpoint(
    cfg: BenchmarkConfig,
    base_results_dir: Path,
    patch_reference_dir: Path,
    patch_sbpinn_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    from dirichlet_sbpinn import (
        DirichletLogDensityNetwork,
        evaluate_model,
    )

    patch_sbpinn_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        base_results_dir / "sbpinn/sbpinn_checkpoint.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    network = DirichletLogDensityNetwork(cfg).to(
        device=device,
        dtype=dtype,
    )
    network.load_state_dict(checkpoint["student"])
    network.eval()

    train_payload = np.load(
        base_results_dir
        / "reference/train_killed_samples.npz"
    )
    survival = train_payload["survival"].astype(np.float64)
    metrics = evaluate_model(
        network,
        cfg,
        survival,
        patch_reference_dir,
        patch_sbpinn_dir,
        device,
        dtype,
    )
    base_summary = json.loads(
        (
            base_results_dir / "sbpinn/sbpinn_summary.json"
        ).read_text(encoding="utf-8")
    )
    metrics.update(
        {
            "reevaluated_from_checkpoint": str(
                checkpoint_path
            ),
            "training_was_not_repeated": True,
            "device": str(device),
            "parameter_count": base_summary["parameter_count"],
            "stage1": base_summary["stage1"],
            "stage2": base_summary["stage2"],
            "stage1_training_time_s": base_summary[
                "stage1_training_time_s"
            ],
            "stage2_training_time_s": base_summary[
                "stage2_training_time_s"
            ],
            "total_training_time_s": base_summary[
                "total_training_time_s"
            ],
            "peak_cuda_memory_mb": base_summary[
                "peak_cuda_memory_mb"
            ],
        }
    )
    write_json(
        patch_sbpinn_dir / "sbpinn_summary.json",
        metrics,
    )
    shutil.copy2(
        checkpoint_path,
        patch_sbpinn_dir / "sbpinn_checkpoint_reused.pt",
    )
    return metrics


def reevaluate_existing_fem(
    cfg: BenchmarkConfig,
    base_results_dir: Path,
    patch_reference_dir: Path,
    patch_fem_dir: Path,
) -> List[Dict[str, object]]:
    patch_fem_dir.mkdir(parents=True, exist_ok=True)
    reference = np.load(
        patch_reference_dir / "reference_pde_finest.npz"
    )
    target_size = reference["densities"].shape[1]
    _, _, evaluation_points = cell_center_grid(target_size)
    rows = []

    base_frame_path = base_results_dir / "fem/fem_results.csv"
    base_frame = pd.read_csv(base_frame_path)
    for _, base_row in base_frame.iterrows():
        if base_row.get("status") != "completed":
            continue
        cells = int(base_row["cells_per_axis"])
        solution_path = (
            base_results_dir / f"fem/fem_n{cells}_solution.npz"
        )
        if not solution_path.exists():
            continue
        payload = np.load(solution_path)
        old_densities = payload["densities"]
        per_time = []
        new_densities = []
        for index, time_value in enumerate(reference["times"]):
            density = old_densities[index]
            if density.shape[0] != target_size:
                density = interpolate_density(
                    density,
                    target_size,
                )
            reference_density = reference["densities"][index]
            reference_mass = float(
                np.mean(reference_density)
            )
            numerical_mass = float(np.mean(density))
            reference_conditional = (
                reference_density
                / max(reference_mass, 1.0e-300)
            )
            positive = np.maximum(density, 0.0)
            numerical_conditional = (
                positive
                / max(float(np.mean(positive)), 1.0e-300)
            )
            metrics = {
                "time": float(time_value),
                **unnormalized_density_metrics(
                    density,
                    reference_density,
                ),
                **conditional_density_metrics(
                    numerical_conditional,
                    reference_conditional,
                ),
                **sliced_wasserstein_2d(
                    evaluation_points,
                    numerical_conditional.reshape(-1),
                    reference_conditional.reshape(-1),
                    cfg.wasserstein_samples,
                    cfg.wasserstein_directions,
                    cfg.seed + cells * 10 + index,
                ),
                "survival_mass_absolute_error": abs(
                    numerical_mass - reference_mass
                ),
            }
            per_time.append(metrics)
            new_densities.append(density)

        frame = pd.DataFrame(per_time)
        frame.to_csv(
            patch_fem_dir
            / f"fem_n{cells}_metrics_by_time.csv",
            index=False,
        )
        np.savez_compressed(
            patch_fem_dir / f"fem_n{cells}_solution.npz",
            times=reference["times"],
            densities=np.stack(new_densities),
        )
        result = {
            "method": base_row["method"],
            "cells_per_axis": cells,
            "status": "completed_reused_and_reevaluated",
            "unknown_count": base_row.get(
                "unknown_count",
                (cells - 1) ** 2,
            ),
            "estimated_peak_memory_GB": base_row.get(
                "estimated_peak_memory_GB",
                resource_estimate(cells)[
                    "estimated_peak_memory_GB"
                ],
            ),
            "total_time_s": base_row.get("total_time_s", np.nan),
            "mean_relative_L2_error": float(
                frame["relative_L2_error"].mean()
            ),
            "mean_Hellinger_distance": float(
                frame["Hellinger_distance"].mean()
            ),
            "mean_survival_mass_absolute_error": float(
                frame[
                    "survival_mass_absolute_error"
                ].mean()
            ),
            "maximum_negative_mass": float(
                frame["negative_mass"].max()
            ),
            "mean_sliced_W": float(
                frame["mean_sliced_W"].mean()
            ),
            "per_time": per_time,
            "base_result_reused": True,
        }
        write_json(
            patch_fem_dir / f"fem_n{cells}_result.json",
            result,
        )
        rows.append(result)
    return rows


def aggregate_fem_results(
    patch_fem_dir: Path,
    meshes: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for cells in meshes:
        path = patch_fem_dir / f"fem_n{cells}_result.json"
        if path.exists():
            rows.append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        else:
            rows.append(
                {
                    "method": (
                        "mass-lumped exterior-Dirichlet P1 FEM "
                        "with dense nonlocal quadrature"
                    ),
                    "cells_per_axis": cells,
                    "status": "not_run",
                    **resource_estimate(cells),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        patch_fem_dir / "fem_results.csv",
        index=False,
    )
    return frame


def build_validation_report(
    cfg: BenchmarkConfig,
    patch_output_dir: Path,
    all_meshes: Sequence[int],
) -> Dict[str, object]:
    reference = json.loads(
        (
            patch_output_dir
            / "reference/reference_summary.json"
        ).read_text(encoding="utf-8")
    )
    sbpinn = json.loads(
        (
            patch_output_dir / "sbpinn/sbpinn_summary.json"
        ).read_text(encoding="utf-8")
    )
    fem_frame = aggregate_fem_results(
        patch_output_dir / "fem",
        all_meshes,
    )
    completed = fem_frame[
        fem_frame["status"].isin(
            ["completed", "completed_reused_and_reevaluated"]
        )
    ].copy()
    valid = completed[
        completed["maximum_negative_mass"] < 0.01
    ].copy()
    best_fem = (
        valid.sort_values("mean_relative_L2_error").iloc[0]
        if not valid.empty
        else None
    )

    sb_time = pd.DataFrame(sbpinn["per_time"])
    per_time_wins = 0
    fem_time = None
    if best_fem is not None:
        fem_time = pd.DataFrame(best_fem["per_time"])
        merged = sb_time.merge(
            fem_time,
            on="time",
            suffixes=("_SBPINN", "_FEM"),
        )
        per_time_wins = int(
            np.sum(
                merged["relative_L2_error_SBPINN"]
                < merged["relative_L2_error_FEM"]
            )
        )
        merged.to_csv(
            patch_output_dir / "per_time_comparison.csv",
            index=False,
        )
    core_trend = bool(
        best_fem is not None
        and sbpinn["mean_relative_L2_error"]
        < best_fem["mean_relative_L2_error"]
        and sbpinn["mean_Hellinger_distance"]
        < best_fem["mean_Hellinger_distance"]
        and sbpinn["maximum_negative_mass"]
        <= best_fem["maximum_negative_mass"] + 1.0e-14
        and per_time_wins >= max(
            1,
            math.ceil(0.75 * len(cfg.evaluation_times)),
        )
    )
    improvement = (
        1.0
        - sbpinn["mean_relative_L2_error"]
        / best_fem["mean_relative_L2_error"]
        if best_fem is not None
        else float("nan")
    )
    robust_trend = bool(core_trend and improvement >= 0.15)
    verdict = (
        "VALIDATION_PATCHES_PASS"
        if reference["high_fidelity_reference"]
        and robust_trend
        else "VALIDATION_PATCHES_NOT_YET_PASS"
    )

    residual_floor = reference[
        "reference_residual_audit_stage2_grid"
    ]["mean_normalized_residual_RMSE"]
    sb_residual = sbpinn["stage2"][
        "selected_mean_normalized_pde_residual"
    ]
    residual_ratio = (
        sb_residual / max(residual_floor, 1.0e-15)
    )

    best_fem_payload = (
        best_fem.to_dict() if best_fem is not None else None
    )
    decision = {
        "verdict": verdict,
        "reference_high_fidelity": reference[
            "high_fidelity_reference"
        ],
        "core_trend_pass": core_trend,
        "robust_trend_pass": robust_trend,
        "per_time_relative_L2_wins": per_time_wins,
        "required_per_time_wins": math.ceil(
            0.75 * len(cfg.evaluation_times)
        ),
        "relative_L2_improvement_fraction": improvement,
        "reference_residual_floor_stage2_grid": residual_floor,
        "SBPINN_selected_residual": sb_residual,
        "SBPINN_to_reference_residual_ratio": residual_ratio,
        "best_fem": best_fem_payload,
    }
    write_json(
        patch_output_dir / "automatic_verdict.json",
        decision,
    )

    rows = [
        {
            "method": "SBPINN (frozen checkpoint)",
            "configuration": (
                f"{sbpinn['parameter_count']} parameters"
            ),
            "mean_relative_L2_error": sbpinn[
                "mean_relative_L2_error"
            ],
            "mean_Hellinger_distance": sbpinn[
                "mean_Hellinger_distance"
            ],
            "maximum_negative_mass": sbpinn[
                "maximum_negative_mass"
            ],
            "runtime_s": sbpinn["total_training_time_s"],
        }
    ]
    if best_fem is not None:
        rows.append(
            {
                "method": "Best validated P1 FEM",
                "configuration": (
                    f"{int(best_fem['cells_per_axis'])}x"
                    f"{int(best_fem['cells_per_axis'])}"
                ),
                "mean_relative_L2_error": best_fem[
                    "mean_relative_L2_error"
                ],
                "mean_Hellinger_distance": best_fem[
                    "mean_Hellinger_distance"
                ],
                "maximum_negative_mass": best_fem[
                    "maximum_negative_mass"
                ],
                "runtime_s": best_fem["total_time_s"],
            }
        )
    pd.DataFrame(rows).to_csv(
        patch_output_dir / "comparison_table.csv",
        index=False,
    )

    best_text = (
        "No completed probability-valid FEM result."
        if best_fem is None
        else (
            f"{int(best_fem['cells_per_axis'])}x"
            f"{int(best_fem['cells_per_axis'])}, "
            f"mean Relative L2="
            f"{best_fem['mean_relative_L2_error']:.6f}"
        )
    )
    report = f"""# Dirichlet benchmark validation-patch report

## Frozen model rule

The SBPINN checkpoint, SDE parameters, network, losses, iterations and
evaluation times were not changed. Only the reference, FEM refinement and
residual-floor audit were updated.

## Reference validation

- finest baseline grid: {reference['finest_grid']}x{reference['finest_grid']}
- maximum last-pair grid difference:
  {reference['maximum_pde_grid_relative_L2']:.6f}
- maximum padding/dt sensitivity:
  {reference['maximum_padding_or_dt_relative_L2']:.6f}
- maximum PDE-MC mass error:
  {reference['maximum_pde_vs_mc_mass_error']:.6f}
- maximum PDE-MC conditional-density error:
  {reference['maximum_pde_vs_mc_conditional_relative_L2']:.6f}
- high-fidelity gate:
  {reference['high_fidelity_reference']}

## Re-evaluated SBPINN

- mean Relative L2:
  {sbpinn['mean_relative_L2_error']:.6f}
- mean Hellinger:
  {sbpinn['mean_Hellinger_distance']:.6f}
- maximum negative mass:
  {sbpinn['maximum_negative_mass']:.3E}

## Finest validated FEM

{best_text}

## Residual-floor audit

- reference residual on the Stage-II grid:
  {residual_floor:.6f}
- frozen SBPINN selected residual:
  {sb_residual:.6f}
- ratio:
  {residual_ratio:.3f}

## Decision

```text
{verdict}
```

- core trend pass: {core_trend}
- robust trend pass (at least 15% Relative-L2 improvement):
  {robust_trend}
- per-time Relative-L2 wins:
  {per_time_wins}/{len(cfg.evaluation_times)}
"""
    (
        patch_output_dir / "VALIDATION_PATCH_REPORT.md"
    ).write_text(report, encoding="utf-8")
    return decision
