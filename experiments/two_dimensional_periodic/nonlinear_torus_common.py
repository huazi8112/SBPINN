from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class BenchmarkConfig:
    # Fractional SDE on the two-dimensional torus.
    alpha: float = 1.5
    noise_scale: float = 0.34
    seed: int = 42

    # Non-gradient coupled four-well drift.
    potential_strength: float = 0.16
    coupling_strength: float = 0.18
    asymmetry_strength: float = 0.055
    rotation_strength: float = 0.50
    x_shift: float = 0.073
    y_shift: float = 0.119
    coupling_phase: float = 0.137
    stream_x_shift: float = 0.061
    stream_y_shift: float = -0.093

    # Generic Fourier feature bank. It does not encode an exact density mode.
    harmonics: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    score_scale: float = 4.0 * math.pi
    hidden_dim: int = 128
    hidden_layers: int = 4

    # Two-stage SBPINN: bounded numerical stabilization patch.
    # Stage I learns a conservative score as the gradient of a scalar
    # log-density network using torus denoising score matching.
    stage1_iterations: int = 3200
    stage1_batch_size: int = 2048
    stage1_validation_batch_size: int = 4096
    stage1_validation_every: int = 50
    stage1_patience_evaluations: int = 12
    stage1_min_iterations: int = 800
    dsm_noise_scales: Tuple[float, ...] = (0.008, 0.016, 0.032)
    learning_rate_stage1: float = 6.0e-4
    stage1_weight_decay: float = 1.0e-6
    stage1_output_gauge_weight: float = 1.0e-4

    # Stage II starts from Stage I and combines held-out data likelihood,
    # conservative-score consistency and a continuation in PDE weight.
    stage2_iterations: int = 3500
    stage2_batch_size: int = 2048
    stage2_uniform_batch_size: int = 4096
    stage2_validation_batch_size: int = 8192
    stage2_validation_every: int = 100
    stage2_min_selection_iteration: int = 800
    learning_rate_stage2: float = 3.0e-4
    stage2_weight_decay: float = 1.0e-6
    stage2_nll_weight: float = 1.0
    stage2_teacher_weight_start: float = 0.50
    stage2_teacher_weight_end: float = 0.05
    stage2_pde_weight_start: float = 0.05
    stage2_pde_weight_end: float = 1.00
    stage2_spectral_tail_weight: float = 0.01
    stage2_spectral_cutoff: float = 16.0
    stage2_nll_selection_tolerance: float = 0.03
    gauge_weight: float = 1.0e-4

    gradient_clip_norm: float = 10.0
    pde_grid_size: int = 256
    pde_every: int = 2

    # Independent SDE reference generation.
    simulation_dt: float = 1.0e-3
    train_chains: int = 4096
    train_burnin_steps: int = 4000
    train_snapshots: int = 64
    train_snapshot_stride: int = 25

    reference_replicates: int = 3
    reference_chains: int = 4096
    reference_burnin_steps: int = 4000
    reference_snapshots: int = 64
    reference_snapshot_stride: int = 25

    refined_reference_chains: int = 2048
    refined_reference_snapshots: int = 48

    reference_grid_size: int = 256
    # Candidate selection is based on replicate stability, coupled dt/dt/2
    # stability and the independent stationary PDE residual.
    reference_bandwidth: float = 0.008
    reference_bandwidth_candidates: Tuple[float, ...] = (
        0.004, 0.006, 0.008, 0.010, 0.012, 0.015
    )
    reference_replicate_threshold: float = 0.03
    reference_dt_threshold: float = 0.03
    reference_pde_residual_threshold: float = 0.05
    evaluation_grid_size: int = 256
    residual_audit_grid_size: int = 512
    plot_grid_size: int = 256
    wasserstein_samples: int = 50000
    wasserstein_directions: int = 64

    # Conventional dense tensor-product periodic Q1 FEM.
    fem_cells_per_axis: Tuple[int, ...] = (
        12, 16, 24, 32, 40, 48, 64, 80, 96
    )
    fem_precheck_cells: Tuple[int, ...] = (128, 160)
    fem_memory_budget_gb: float = 16.0
    fem_time_budget_s: float = 3600.0

    output_dir: str = "results_nonlinear_torus_levy"
    device: str = "auto"
    quick: bool = False

    @property
    def fractional_coefficient(self) -> float:
        return float(self.noise_scale ** self.alpha)

    def validate(self) -> None:
        if not (0.0 < self.alpha < 2.0):
            raise ValueError("alpha must lie strictly between 0 and 2.")
        if self.noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive.")
        if self.simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive.")
        if self.reference_replicates < 2:
            raise ValueError(
                "At least two independent reference replicates are required."
            )

    def apply_quick_mode(self) -> None:
        self.harmonics = (1, 2, 3, 4)
        self.hidden_dim = 24
        self.hidden_layers = 2
        self.stage1_iterations = 10
        self.stage1_batch_size = 128
        self.stage1_validation_batch_size = 128
        self.stage1_validation_every = 2
        self.stage1_patience_evaluations = 20
        self.stage1_min_iterations = 2
        self.dsm_noise_scales = (0.02, 0.04)

        self.stage2_iterations = 12
        self.stage2_batch_size = 128
        self.stage2_uniform_batch_size = 256
        self.stage2_validation_batch_size = 256
        self.stage2_validation_every = 2
        self.stage2_min_selection_iteration = 2
        self.pde_grid_size = 16
        self.pde_every = 2

        self.train_chains = 256
        self.train_burnin_steps = 30
        self.train_snapshots = 3
        self.train_snapshot_stride = 3
        self.reference_replicates = 2
        self.reference_chains = 256
        self.reference_burnin_steps = 30
        self.reference_snapshots = 3
        self.reference_snapshot_stride = 3
        self.refined_reference_chains = 128
        self.refined_reference_snapshots = 2

        self.reference_grid_size = 32
        self.reference_bandwidth = 0.03
        self.reference_bandwidth_candidates = (0.02, 0.03, 0.04)
        self.reference_replicate_threshold = 1.0
        self.reference_dt_threshold = 1.0
        self.reference_pde_residual_threshold = 2.0
        self.evaluation_grid_size = 32
        self.residual_audit_grid_size = 32
        self.plot_grid_size = 32
        self.wasserstein_samples = 500
        self.wasserstein_directions = 8

        self.fem_cells_per_axis = (4, 6, 8)
        self.fem_precheck_cells = (12,)
        self.fem_memory_budget_gb = 2.0
        self.fem_time_budget_s = 120.0


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def drift_numpy(
    points: np.ndarray,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    two_pi = 2.0 * math.pi
    four_pi = 4.0 * math.pi

    dx_potential = (
        cfg.potential_strength
        * four_pi
        * np.sin(four_pi * (x - cfg.x_shift))
        + cfg.coupling_strength
        * two_pi
        * np.sin(
            two_pi * (x - y - cfg.coupling_phase)
        )
        + cfg.asymmetry_strength
        * two_pi
        * np.cos(two_pi * x + 0.31)
        * np.sin(four_pi * y - 0.27)
    )
    dy_potential = (
        cfg.potential_strength
        * four_pi
        * np.sin(four_pi * (y - cfg.y_shift))
        - cfg.coupling_strength
        * two_pi
        * np.sin(
            two_pi * (x - y - cfg.coupling_phase)
        )
        + cfg.asymmetry_strength
        * four_pi
        * np.sin(two_pi * x + 0.31)
        * np.cos(four_pi * y - 0.27)
    )

    stream_x = two_pi * cfg.rotation_strength * (
        np.sin(two_pi * (x + cfg.stream_x_shift))
        * np.cos(two_pi * (y + cfg.stream_y_shift))
    )
    stream_y = -two_pi * cfg.rotation_strength * (
        np.cos(two_pi * (x + cfg.stream_x_shift))
        * np.sin(two_pi * (y + cfg.stream_y_shift))
    )

    return np.column_stack(
        [
            -dx_potential + stream_x,
            -dy_potential + stream_y,
        ]
    )


def periodic_grid(
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(grid_size, dtype=np.float64) / grid_size
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack([x.reshape(-1), y.reshape(-1)])
    return x, y, points


def fractional_symbol(
    grid_size: int,
    cfg: BenchmarkConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    angular = 2.0 * math.pi * np.fft.fftfreq(
        grid_size,
        d=1.0 / grid_size,
    )
    kx, ky = np.meshgrid(angular, angular, indexing="xy")
    symbol = (
        np.abs(kx) ** cfg.alpha
        + np.abs(ky) ** cfg.alpha
    )
    return kx, ky, symbol


def cic_density(
    samples: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    samples = np.mod(np.asarray(samples, dtype=np.float64), 1.0)
    scaled = samples * grid_size
    i0 = np.floor(scaled[:, 0]).astype(np.int64) % grid_size
    j0 = np.floor(scaled[:, 1]).astype(np.int64) % grid_size
    tx = scaled[:, 0] - np.floor(scaled[:, 0])
    ty = scaled[:, 1] - np.floor(scaled[:, 1])
    i1 = (i0 + 1) % grid_size
    j1 = (j0 + 1) % grid_size

    indices = np.concatenate(
        [
            j0 * grid_size + i0,
            j0 * grid_size + i1,
            j1 * grid_size + i0,
            j1 * grid_size + i1,
        ]
    )
    weights = np.concatenate(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ]
    )
    counts = np.bincount(
        indices,
        weights=weights,
        minlength=grid_size * grid_size,
    ).reshape(grid_size, grid_size)
    density = counts / max(float(np.mean(counts)), 1.0e-300)
    return density


def periodic_gaussian_smooth(
    density: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    grid_size = density.shape[0]
    cycles = np.fft.fftfreq(
        grid_size,
        d=1.0 / grid_size,
    )
    kx, ky = np.meshgrid(cycles, cycles, indexing="xy")
    multiplier = np.exp(
        -0.5
        * (2.0 * math.pi * bandwidth) ** 2
        * (kx**2 + ky**2)
    )
    smoothed = np.fft.ifft2(
        multiplier * np.fft.fft2(density)
    ).real
    smoothed = np.maximum(smoothed, 0.0)
    smoothed /= max(float(np.mean(smoothed)), 1.0e-300)
    return smoothed


def reference_density_from_samples(
    samples: np.ndarray,
    cfg: BenchmarkConfig,
    bandwidth: float | None = None,
) -> np.ndarray:
    density = cic_density(samples, cfg.reference_grid_size)
    return periodic_gaussian_smooth(
        density,
        cfg.reference_bandwidth
        if bandwidth is None
        else bandwidth,
    )


def density_metrics(
    numerical: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    numerical = np.asarray(numerical, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    numerical_mass = float(np.mean(numerical))
    reference_mass = float(np.mean(reference))
    negative_mass = float(np.mean(np.maximum(-numerical, 0.0)))
    l1 = float(np.mean(np.abs(numerical - reference)))
    l2 = float(np.sqrt(np.mean((numerical - reference) ** 2)))
    reference_l2 = float(np.sqrt(np.mean(reference**2)))
    relative_l2 = l2 / max(reference_l2, 1.0e-15)

    positive = np.maximum(numerical, 0.0)
    positive /= max(float(np.mean(positive)), 1.0e-15)
    reference_normalized = (
        reference / max(reference_mass, 1.0e-15)
    )
    hellinger = float(
        np.sqrt(
            0.5
            * np.mean(
                (
                    np.sqrt(positive)
                    - np.sqrt(reference_normalized)
                )
                ** 2
            )
        )
    )
    midpoint = 0.5 * (positive + reference_normalized)
    kl_ref_num = float(
        np.mean(
            reference_normalized
            * (
                np.log(
                    np.maximum(reference_normalized, 1.0e-300)
                )
                - np.log(np.maximum(positive, 1.0e-300))
            )
        )
    )
    js = float(
        0.5
        * np.mean(
            reference_normalized
            * (
                np.log(
                    np.maximum(reference_normalized, 1.0e-300)
                )
                - np.log(np.maximum(midpoint, 1.0e-300))
            )
        )
        + 0.5
        * np.mean(
            positive
            * (
                np.log(np.maximum(positive, 1.0e-300))
                - np.log(np.maximum(midpoint, 1.0e-300))
            )
        )
    )
    return {
        "mass": numerical_mass,
        "mass_deviation": abs(numerical_mass - 1.0),
        "negative_mass": negative_mass,
        "L1_error": l1,
        "L2_error": l2,
        "relative_L2_error": relative_l2,
        "Hellinger_distance": hellinger,
        "KL_reference_to_numerical": kl_ref_num,
        "Jensen_Shannon_divergence": js,
        "minimum_density": float(np.min(numerical)),
        "maximum_density": float(np.max(numerical)),
    }


def relative_l2_between(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    numerator = np.sqrt(np.mean((first - second) ** 2))
    denominator = np.sqrt(np.mean(second**2))
    return float(numerator / max(denominator, 1.0e-15))


def systematic_resample(
    points: np.ndarray,
    probabilities: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    probabilities = np.maximum(
        np.asarray(probabilities, dtype=np.float64),
        0.0,
    )
    probabilities /= max(float(probabilities.sum()), 1.0e-300)
    cumulative = np.cumsum(probabilities)
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.0, 1.0 / count)
    positions = start + np.arange(count) / count
    indices = np.searchsorted(cumulative, positions, side="right")
    return points[np.minimum(indices, len(points) - 1)]


def torus_embedding(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    angles = 2.0 * math.pi * points
    return np.column_stack(
        [
            np.cos(angles[:, 0]),
            np.sin(angles[:, 0]),
            np.cos(angles[:, 1]),
            np.sin(angles[:, 1]),
        ]
    )


def torus_sliced_wasserstein(
    points: np.ndarray,
    numerical: np.ndarray,
    reference: np.ndarray,
    cfg: BenchmarkConfig,
) -> Dict[str, float]:
    numerical_samples = systematic_resample(
        points,
        numerical,
        cfg.wasserstein_samples,
        cfg.seed + 50000,
    )
    reference_samples = systematic_resample(
        points,
        reference,
        cfg.wasserstein_samples,
        cfg.seed + 51000,
    )
    numerical_embedding = torus_embedding(numerical_samples)
    reference_embedding = torus_embedding(reference_samples)

    rng = np.random.default_rng(cfg.seed + 52000)
    directions = rng.normal(
        size=(cfg.wasserstein_directions, 4)
    )
    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )
    values = np.array(
        [
            float(
                np.mean(
                    np.abs(
                        np.sort(numerical_embedding @ direction)
                        - np.sort(reference_embedding @ direction)
                    )
                )
            )
            for direction in directions
        ]
    )
    return {
        "mean_torus_sliced_W": float(np.mean(values)),
        "median_torus_sliced_W": float(np.median(values)),
        "p90_torus_sliced_W": float(np.quantile(values, 0.90)),
        "max_torus_sliced_W": float(np.max(values)),
    }



def stationary_residual_metrics(
    density: np.ndarray,
    cfg: BenchmarkConfig,
) -> Dict[str, float]:
    """Evaluate the stationary fractional Fokker--Planck residual.

    The density is interpreted on a uniform periodic grid. This function is
    independent of neural-network training and is used both for reference
    selection and final auditing.
    """
    density = np.asarray(density, dtype=np.float64)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density must be a square two-dimensional array.")
    grid_size = density.shape[0]
    _, _, points = periodic_grid(grid_size)
    drift = drift_numpy(points, cfg).reshape(
        grid_size,
        grid_size,
        2,
    )
    kx, ky, symbol = fractional_symbol(grid_size, cfg)
    flux_x = drift[:, :, 0] * density
    flux_y = drift[:, :, 1] * density
    divergence = np.fft.ifft2(
        1j * kx * np.fft.fft2(flux_x)
        + 1j * ky * np.fft.fft2(flux_y)
    ).real
    fractional = np.fft.ifft2(
        symbol * np.fft.fft2(density)
    ).real
    advection = -divergence
    diffusion = -cfg.fractional_coefficient * fractional
    residual = advection + diffusion
    scale = np.sqrt(
        np.mean(advection**2)
        + np.mean(diffusion**2)
    )
    return {
        "residual_RMSE": float(np.sqrt(np.mean(residual**2))),
        "normalized_residual_RMSE": float(
            np.sqrt(np.mean(residual**2))
            / max(float(scale), 1.0e-15)
        ),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "advection_RMS": float(np.sqrt(np.mean(advection**2))),
        "diffusion_RMS": float(np.sqrt(np.mean(diffusion**2))),
    }

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def config_to_dict(cfg: BenchmarkConfig) -> Dict[str, object]:
    payload = asdict(cfg)
    payload["fractional_coefficient"] = cfg.fractional_coefficient
    return payload
