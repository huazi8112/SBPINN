from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from nonlinear_torus_common import (
    BenchmarkConfig,
    config_to_dict,
    reference_density_from_samples,
    stationary_residual_metrics,
    relative_l2_between,
    set_seeds,
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
        * torch.sin(
            two_pi * (x - y - cfg.coupling_phase)
        )
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
        * torch.sin(
            two_pi * (x - y - cfg.coupling_phase)
        )
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
        [
            -dx_potential + stream_x,
            -dy_potential + stream_y,
        ],
        dim=1,
    )


def symmetric_alpha_stable(
    shape: tuple[int, ...],
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    eps = torch.finfo(dtype).eps
    uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    uniform = torch.clamp(uniform, eps, 1.0 - eps)
    angle = math.pi * (uniform - 0.5)

    exponential_uniform = torch.rand(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    exponential_uniform = torch.clamp(
        exponential_uniform,
        eps,
        1.0 - eps,
    )
    exponential = -torch.log(exponential_uniform)

    numerator = torch.sin(alpha * angle)
    denominator = torch.cos(angle).pow(1.0 / alpha)
    correction = (
        torch.cos((1.0 - alpha) * angle)
        / exponential
    ).pow((1.0 - alpha) / alpha)
    return numerator / denominator * correction



def simulate_coupled_dt_refinement(
    cfg: BenchmarkConfig,
    seed: int,
    chains: int,
    burnin_steps: int,
    snapshots: int,
    snapshot_stride: int,
    dt: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Couple dt and dt/2 simulations with shared stable increments.

    The sum of two independent symmetric alpha-stable increments at dt/2
    has the exact dt increment distribution. The coupling therefore
    suppresses Monte Carlo noise in the time-step refinement comparison.
    """
    dtype = torch.float64
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    coarse = torch.rand(
        (chains, 2),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    fine = coarse.clone()
    half_dt = 0.5 * dt
    half_scale = (
        cfg.noise_scale * half_dt ** (1.0 / cfg.alpha)
    )
    total_steps = burnin_steps + snapshots * snapshot_stride
    coarse_collected: List[np.ndarray] = []
    fine_collected: List[np.ndarray] = []
    start_time = time.perf_counter()

    for step in trange(
        total_steps,
        desc="Coupled dt versus dt/2 reference",
    ):
        first_noise = symmetric_alpha_stable(
            (chains, 2),
            cfg.alpha,
            device,
            dtype,
            generator,
        )
        second_noise = symmetric_alpha_stable(
            (chains, 2),
            cfg.alpha,
            device,
            dtype,
            generator,
        )

        coarse = torch.remainder(
            coarse
            + drift_torch(coarse, cfg) * dt
            + half_scale * (first_noise + second_noise),
            1.0,
        )
        fine = torch.remainder(
            fine
            + drift_torch(fine, cfg) * half_dt
            + half_scale * first_noise,
            1.0,
        )
        fine = torch.remainder(
            fine
            + drift_torch(fine, cfg) * half_dt
            + half_scale * second_noise,
            1.0,
        )

        if (
            step >= burnin_steps
            and (step - burnin_steps + 1) % snapshot_stride == 0
        ):
            coarse_collected.append(
                coarse.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )
            fine_collected.append(
                fine.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

    coarse_samples = np.concatenate(coarse_collected, axis=0)
    fine_samples = np.concatenate(fine_collected, axis=0)
    return coarse_samples, fine_samples, {
        "seed": seed,
        "chains": chains,
        "burnin_steps": burnin_steps,
        "snapshots": snapshots,
        "snapshot_stride": snapshot_stride,
        "coarse_dt": dt,
        "fine_dt": half_dt,
        "sample_count_per_level": int(len(coarse_samples)),
        "runtime_s": time.perf_counter() - start_time,
        "device": str(device),
    }


def simulate_stationary_samples(
    cfg: BenchmarkConfig,
    seed: int,
    chains: int,
    burnin_steps: int,
    snapshots: int,
    snapshot_stride: int,
    dt: float,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, Dict[str, float]]:
    dtype = torch.float64
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    points = torch.rand(
        (chains, 2),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    increment_scale = cfg.noise_scale * dt ** (1.0 / cfg.alpha)
    total_steps = burnin_steps + snapshots * snapshot_stride
    collected: List[np.ndarray] = []

    start_time = time.perf_counter()
    iterator = trange(total_steps, desc=description)
    for step in iterator:
        stable_noise = symmetric_alpha_stable(
            (chains, 2),
            cfg.alpha,
            device,
            dtype,
            generator,
        )
        points = torch.remainder(
            points
            + drift_torch(points, cfg) * dt
            + increment_scale * stable_noise,
            1.0,
        )
        if (
            step >= burnin_steps
            and (step - burnin_steps + 1) % snapshot_stride == 0
        ):
            collected.append(
                points.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

    samples = np.concatenate(collected, axis=0)
    elapsed = time.perf_counter() - start_time
    return samples, {
        "seed": seed,
        "chains": chains,
        "burnin_steps": burnin_steps,
        "snapshots": snapshots,
        "snapshot_stride": snapshot_stride,
        "dt": dt,
        "sample_count": int(len(samples)),
        "runtime_s": elapsed,
        "device": str(device),
    }



def generate_reference_data(
    cfg: BenchmarkConfig,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.validate()
    set_seeds(cfg.seed)
    device = choose_device(cfg.device)

    train_samples, train_meta = simulate_stationary_samples(
        cfg,
        cfg.seed + 1000,
        cfg.train_chains,
        cfg.train_burnin_steps,
        cfg.train_snapshots,
        cfg.train_snapshot_stride,
        cfg.simulation_dt,
        device,
        "Independent SDE train sample pool",
    )
    np.savez_compressed(
        output_dir / "train_samples.npz",
        samples=train_samples,
    )

    replicate_samples: List[np.ndarray] = []
    replicate_metadata: List[Dict[str, float]] = []
    for replicate in range(cfg.reference_replicates):
        samples, metadata = simulate_stationary_samples(
            cfg,
            cfg.seed + 2000 + 1000 * replicate,
            cfg.reference_chains,
            cfg.reference_burnin_steps,
            cfg.reference_snapshots,
            cfg.reference_snapshot_stride,
            cfg.simulation_dt,
            device,
            f"Reference replicate {replicate + 1}",
        )
        replicate_samples.append(samples)
        replicate_metadata.append(metadata)
        np.savez_compressed(
            output_dir / f"reference_replicate_{replicate + 1}.npz",
            samples=samples,
        )

    (
        coupled_coarse_samples,
        coupled_fine_samples,
        refined_meta,
    ) = simulate_coupled_dt_refinement(
        cfg,
        cfg.seed + 9000,
        cfg.refined_reference_chains,
        cfg.reference_burnin_steps,
        cfg.refined_reference_snapshots,
        cfg.reference_snapshot_stride,
        cfg.simulation_dt,
        device,
    )
    np.savez_compressed(
        output_dir / "reference_refined_dt_samples.npz",
        coarse_samples=coupled_coarse_samples,
        fine_samples=coupled_fine_samples,
    )

    candidate_rows: List[Dict[str, float | bool]] = []
    candidate_payload: Dict[float, Dict[str, object]] = {}
    for bandwidth in cfg.reference_bandwidth_candidates:
        replicate_densities = [
            reference_density_from_samples(
                samples,
                cfg,
                bandwidth=bandwidth,
            )
            for samples in replicate_samples
        ]
        reference_density = np.mean(
            np.stack(replicate_densities, axis=0),
            axis=0,
        )
        reference_density /= np.mean(reference_density)

        coupled_coarse_density = reference_density_from_samples(
            coupled_coarse_samples,
            cfg,
            bandwidth=bandwidth,
        )
        coupled_fine_density = reference_density_from_samples(
            coupled_fine_samples,
            cfg,
            bandwidth=bandwidth,
        )

        replicate_to_mean = [
            relative_l2_between(density, reference_density)
            for density in replicate_densities
        ]
        pairwise = []
        for first in range(len(replicate_densities)):
            for second in range(first + 1, len(replicate_densities)):
                pairwise.append(
                    relative_l2_between(
                        replicate_densities[first],
                        replicate_densities[second],
                    )
                )
        dt_discrepancy = relative_l2_between(
            coupled_fine_density,
            coupled_coarse_density,
        )
        residual = stationary_residual_metrics(
            reference_density,
            cfg,
        )
        replicate_max = float(max(replicate_to_mean))
        admissible = bool(
            replicate_max < cfg.reference_replicate_threshold
            and dt_discrepancy < cfg.reference_dt_threshold
        )
        candidate_rows.append(
            {
                "bandwidth": float(bandwidth),
                "replicate_max_relative_L2": replicate_max,
                "replicate_median_relative_L2": float(
                    np.median(replicate_to_mean)
                ),
                "pairwise_median_relative_L2": float(
                    np.median(pairwise)
                ),
                "coupled_dt_relative_L2": float(dt_discrepancy),
                "normalized_PDE_residual": residual[
                    "normalized_residual_RMSE"
                ],
                "reference_minimum": float(
                    np.min(reference_density)
                ),
                "reference_maximum": float(
                    np.max(reference_density)
                ),
                "admissible_MC_stability": admissible,
            }
        )
        candidate_payload[float(bandwidth)] = {
            "reference_density": reference_density,
            "replicate_densities": np.stack(
                replicate_densities,
                axis=0,
            ),
            "coarse_density": coupled_coarse_density,
            "fine_density": coupled_fine_density,
            "replicate_to_mean": replicate_to_mean,
            "pairwise": pairwise,
            "dt_discrepancy": dt_discrepancy,
            "residual": residual,
        }

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(
        output_dir / "reference_bandwidth_candidates.csv",
        index=False,
    )

    admissible_frame = candidate_frame[
        candidate_frame["admissible_MC_stability"]
    ].copy()
    if not admissible_frame.empty:
        selected_row = admissible_frame.sort_values(
            ["normalized_PDE_residual", "bandwidth"],
            ascending=[True, True],
        ).iloc[0]
        selection_reason = (
            "minimum stationary PDE residual among MC-stable candidates"
        )
    else:
        candidate_frame["fallback_score"] = (
            candidate_frame["normalized_PDE_residual"]
            + 2.0 * candidate_frame["replicate_max_relative_L2"]
            + candidate_frame["coupled_dt_relative_L2"]
        )
        selected_row = candidate_frame.sort_values(
            ["fallback_score", "bandwidth"],
            ascending=[True, True],
        ).iloc[0]
        selection_reason = (
            "fallback composite because no candidate passed MC stability"
        )

    selected_bandwidth = float(selected_row["bandwidth"])
    selected = candidate_payload[selected_bandwidth]
    reference_density = selected["reference_density"]
    replicate_densities = selected["replicate_densities"]
    refined_density = selected["fine_density"]

    selected_index = list(
        cfg.reference_bandwidth_candidates
    ).index(selected_bandwidth)
    neighboring_differences = []
    for neighbor_index in (selected_index - 1, selected_index + 1):
        if 0 <= neighbor_index < len(cfg.reference_bandwidth_candidates):
            neighbor_bandwidth = float(
                cfg.reference_bandwidth_candidates[neighbor_index]
            )
            neighboring_differences.append(
                relative_l2_between(
                    candidate_payload[neighbor_bandwidth][
                        "reference_density"
                    ],
                    reference_density,
                )
            )
    bandwidth_sensitivity = float(
        max(neighboring_differences)
        if neighboring_differences
        else 0.0
    )

    high_fidelity_reference = bool(
        float(selected_row["replicate_max_relative_L2"])
        < cfg.reference_replicate_threshold
        and float(selected_row["coupled_dt_relative_L2"])
        < cfg.reference_dt_threshold
        and float(selected_row["normalized_PDE_residual"])
        < cfg.reference_pde_residual_threshold
    )

    np.savez_compressed(
        output_dir / "reference_density.npz",
        density=reference_density,
        replicate_densities=replicate_densities,
        refined_density=refined_density,
        selected_bandwidth=np.array(selected_bandwidth),
    )

    summary = {
        "config": config_to_dict(cfg),
        "device": str(device),
        "train": train_meta,
        "replicates": replicate_metadata,
        "refined": refined_meta,
        "selected_bandwidth": selected_bandwidth,
        "selection_reason": selection_reason,
        "replicate_to_mean_relative_L2": selected[
            "replicate_to_mean"
        ],
        "pairwise_replicate_relative_L2": selected["pairwise"],
        "reference_noise_floor_relative_L2": float(
            np.median(selected["replicate_to_mean"])
        ),
        "refined_dt_relative_L2": float(
            selected["dt_discrepancy"]
        ),
        "bandwidth_sensitivity_relative_L2": (
            bandwidth_sensitivity
        ),
        "stationary_PDE_residual": selected["residual"],
        "reference_PDE_residual_pass": bool(
            selected["residual"]["normalized_residual_RMSE"]
            < cfg.reference_pde_residual_threshold
        ),
        "high_fidelity_reference": high_fidelity_reference,
        "reference_minimum": float(np.min(reference_density)),
        "reference_maximum": float(np.max(reference_density)),
    }
    write_json(output_dir / "reference_summary.json", summary)
    return summary
