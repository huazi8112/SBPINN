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
    # Killed alpha-stable SDE on Omega=(0,1)^2.
    alpha: float = 1.5
    noise_scale: float = 0.34
    seed: int = 42
    final_time: float = 0.30
    snapshot_times: Tuple[float, ...] = (
        0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300
    )
    evaluation_times: Tuple[float, ...] = (0.050, 0.100, 0.200, 0.300)

    # Same nonlinear non-gradient drift as the periodic benchmark.
    potential_strength: float = 0.16
    coupling_strength: float = 0.18
    asymmetry_strength: float = 0.055
    rotation_strength: float = 0.50
    x_shift: float = 0.073
    y_shift: float = 0.119
    coupling_phase: float = 0.137
    stream_x_shift: float = 0.061
    stream_y_shift: float = -0.093

    # Initial two-component Gaussian mixture, truncated to Omega.
    initial_weight_1: float = 0.55
    initial_mean_1: Tuple[float, float] = (0.25, 0.28)
    initial_std_1: float = 0.055
    initial_mean_2: Tuple[float, float] = (0.70, 0.67)
    initial_std_2: float = 0.060

    # Killed-SDE data.
    simulation_dt: float = 5.0e-4
    train_particles: int = 32768
    reference_replicates: int = 3
    reference_particles: int = 65536
    refined_reference_particles: int = 32768
    refined_dt_factor: float = 0.5

    # Padded zero-extension PDE reference.
    reference_grid_sizes: Tuple[int, ...] = (64, 96, 128, 192, 256)
    reference_padding_factor: int = 3
    reference_time_step: float = 5.0e-4
    validation_reference_base_grid: int = 128
    validation_reference_grids: Tuple[int, ...] = (192, 256)
    validation_padding_factor: int = 4
    validation_time_step: float = 2.5e-4
    validation_sensitivity_threshold: float = 0.03
    reference_residual_audit_grid: int = 64
    reference_grid_convergence_threshold: float = 0.03
    reference_mass_threshold: float = 0.03
    reference_mc_shape_threshold: float = 0.06
    reference_kde_bandwidth: float = 0.010

    # Time-dependent SBPINN.
    harmonics: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    hidden_dim: int = 128
    hidden_layers: int = 4
    boundary_power: float = 0.75  # alpha / 2 for alpha=1.5
    score_scale: float = 4.0 * math.pi

    stage1_iterations: int = 3000
    stage1_batch_size: int = 2048
    stage1_uniform_batch_size: int = 4096
    stage1_validation_batch_size: int = 2048
    stage1_validation_every: int = 50
    stage1_patience_evaluations: int = 12
    stage1_min_iterations: int = 800
    stage1_learning_rate: float = 6.0e-4
    stage1_weight_decay: float = 1.0e-6
    stage1_nll_weight: float = 1.0
    stage1_dsm_weight: float = 0.20
    dsm_noise_scales: Tuple[float, ...] = (0.006, 0.012, 0.024)

    stage2_iterations: int = 3500
    stage2_batch_size: int = 2048
    stage2_uniform_batch_size: int = 4096
    stage2_validation_every: int = 100
    stage2_min_selection_iteration: int = 800
    stage2_learning_rate: float = 3.0e-4
    stage2_weight_decay: float = 1.0e-6
    stage2_nll_weight: float = 1.0
    stage2_teacher_weight_start: float = 0.40
    stage2_teacher_weight_end: float = 0.05
    stage2_pde_weight_start: float = 0.05
    stage2_pde_weight_end: float = 1.00
    stage2_selection_nll_tolerance: float = 0.04
    stage2_smoothness_weight: float = 1.0e-4
    pde_grid_size: int = 64
    pde_padding_factor: int = 3
    pde_time_delta: float = 2.5e-3
    gradient_clip_norm: float = 10.0

    # Dense mass-lumped exterior-Dirichlet P1 FEM baseline.
    fem_cells_per_axis: Tuple[int, ...] = (
        12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112
    )
    fem_precheck_cells: Tuple[int, ...] = (128,)
    fem_kernel_block_size: int = 256
    fem_time_step: float = 1.0e-3
    fem_exterior_padding: float = 2.0
    fem_memory_budget_gb: float = 16.0
    fem_time_budget_s: float = 3600.0

    wasserstein_samples: int = 30000
    wasserstein_directions: int = 48
    output_dir: str = "results_dirichlet_killed_levy"
    device: str = "auto"
    quick: bool = False

    @property
    def fractional_coefficient(self) -> float:
        return float(self.noise_scale ** self.alpha)

    @property
    def fractional_constant(self) -> float:
        # Fourier convention: F[(-Delta)^(alpha/2)u]=|xi|^alpha F[u].
        numerator = (
            2.0 ** self.alpha
            * math.gamma((2.0 + self.alpha) / 2.0)
        )
        denominator = (
            math.pi
            * abs(math.gamma(-self.alpha / 2.0))
        )
        return float(numerator / denominator)

    def validate(self) -> None:
        if not (0.0 < self.alpha < 2.0):
            raise ValueError("alpha must lie in (0,2).")
        if self.noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive.")
        if self.simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive.")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive.")
        if max(self.snapshot_times) > self.final_time + 1.0e-12:
            raise ValueError("snapshot_times exceed final_time.")
        if not set(self.evaluation_times).issubset(set(self.snapshot_times)):
            raise ValueError(
                "Every evaluation time must also be a snapshot time."
            )
        if self.reference_replicates < 2:
            raise ValueError("At least two reference replicates are required.")

    def apply_quick_mode(self) -> None:
        self.final_time = 0.04
        self.snapshot_times = (0.01, 0.02, 0.03, 0.04)
        self.evaluation_times = (0.02, 0.04)
        self.simulation_dt = 2.0e-3
        self.train_particles = 512
        self.reference_replicates = 2
        self.reference_particles = 768
        self.refined_reference_particles = 512

        self.reference_grid_sizes = (16, 24, 32, 40)
        self.reference_padding_factor = 2
        self.reference_time_step = 2.0e-3
        self.validation_reference_base_grid = 24
        self.validation_reference_grids = (32, 40)
        self.validation_padding_factor = 3
        self.validation_time_step = 1.0e-3
        self.validation_sensitivity_threshold = 2.0
        self.reference_residual_audit_grid = 16
        self.reference_grid_convergence_threshold = 2.0
        self.reference_mass_threshold = 2.0
        self.reference_mc_shape_threshold = 2.0
        self.reference_kde_bandwidth = 0.04

        self.harmonics = (1, 2, 3)
        self.hidden_dim = 24
        self.hidden_layers = 2
        self.stage1_iterations = 8
        self.stage1_batch_size = 96
        self.stage1_uniform_batch_size = 144
        self.stage1_validation_batch_size = 96
        self.stage1_validation_every = 2
        self.stage1_patience_evaluations = 20
        self.stage1_min_iterations = 2
        self.dsm_noise_scales = (0.02,)

        self.stage2_iterations = 10
        self.stage2_batch_size = 96
        self.stage2_uniform_batch_size = 144
        self.stage2_validation_every = 2
        self.stage2_min_selection_iteration = 2
        self.pde_grid_size = 12
        self.pde_padding_factor = 2
        self.pde_time_delta = 2.0e-3

        self.fem_cells_per_axis = (6, 8, 10, 12)
        self.fem_precheck_cells = (14,)
        self.fem_kernel_block_size = 32
        self.fem_time_step = 2.0e-3
        self.fem_exterior_padding = 0.5
        self.fem_memory_budget_gb = 2.0
        self.fem_time_budget_s = 120.0
        self.wasserstein_samples = 500
        self.wasserstein_directions = 8


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


def time_key(time_value: float) -> str:
    return f"t_{int(round(100000 * time_value)):06d}"


def cell_center_grid(
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = (np.arange(grid_size, dtype=np.float64) + 0.5) / grid_size
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack([x.reshape(-1), y.reshape(-1)])
    return x, y, points


def node_grid(
    cells_per_axis: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack([x.reshape(-1), y.reshape(-1)])
    return x, y, points


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
        * np.sin(two_pi * (x - y - cfg.coupling_phase))
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
        * np.sin(two_pi * (x - y - cfg.coupling_phase))
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


def initial_density_numpy(
    points: np.ndarray,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)

    def gaussian(mean: Tuple[float, float], std: float) -> np.ndarray:
        difference = points - np.asarray(mean)[None, :]
        exponent = -0.5 * np.sum(difference**2, axis=1) / (std**2)
        return np.exp(exponent) / (2.0 * math.pi * std**2)

    density = (
        cfg.initial_weight_1
        * gaussian(cfg.initial_mean_1, cfg.initial_std_1)
        + (1.0 - cfg.initial_weight_1)
        * gaussian(cfg.initial_mean_2, cfg.initial_std_2)
    )
    return density


def normalize_grid_density(density: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    mass = float(np.mean(density))
    if mass <= 0.0:
        raise ValueError("Cannot normalize a nonpositive grid density.")
    return density / mass


def boundary_envelope_numpy(
    points: np.ndarray,
    power: float,
    epsilon: float = 0.0,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    base = np.maximum(x * (1.0 - x) * y * (1.0 - y), 0.0)
    return (base + epsilon) ** power


def cic_density(
    samples: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[
        np.all((samples > 0.0) & (samples < 1.0), axis=1)
    ]
    if len(samples) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float64)
    scaled = samples * grid_size - 0.5
    i0_floor = np.floor(scaled[:, 0])
    j0_floor = np.floor(scaled[:, 1])
    tx = scaled[:, 0] - i0_floor
    ty = scaled[:, 1] - j0_floor
    i0 = i0_floor.astype(np.int64)
    j0 = j0_floor.astype(np.int64)

    counts = np.zeros(grid_size * grid_size, dtype=np.float64)
    for di, dj, weight in (
        (0, 0, (1.0 - tx) * (1.0 - ty)),
        (1, 0, tx * (1.0 - ty)),
        (0, 1, (1.0 - tx) * ty),
        (1, 1, tx * ty),
    ):
        ii = i0 + di
        jj = j0 + dj
        valid = (
            (ii >= 0)
            & (ii < grid_size)
            & (jj >= 0)
            & (jj < grid_size)
        )
        indices = jj[valid] * grid_size + ii[valid]
        counts += np.bincount(
            indices,
            weights=weight[valid],
            minlength=grid_size * grid_size,
        )
    density = counts.reshape(grid_size, grid_size)
    density /= max(float(np.mean(density)), 1.0e-300)
    return density


def zero_padded_gaussian_smooth(
    density: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    n = density.shape[0]
    padded_size = 2 * n
    padded = np.zeros((padded_size, padded_size), dtype=np.float64)
    start = n // 2
    padded[start : start + n, start : start + n] = density
    cycles = np.fft.fftfreq(padded_size, d=1.0 / n)
    kx, ky = np.meshgrid(cycles, cycles, indexing="xy")
    multiplier = np.exp(
        -0.5
        * (2.0 * math.pi * bandwidth) ** 2
        * (kx**2 + ky**2)
    )
    smoothed = np.fft.ifft2(
        multiplier * np.fft.fft2(padded)
    ).real
    cropped = smoothed[start : start + n, start : start + n]
    cropped = np.maximum(cropped, 0.0)
    cropped /= max(float(np.mean(cropped)), 1.0e-300)
    return cropped


def conditional_density_metrics(
    numerical: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    numerical = np.asarray(numerical, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    numerical_positive = np.maximum(numerical, 0.0)
    numerical_positive /= max(
        float(np.mean(numerical_positive)),
        1.0e-300,
    )
    reference_positive = np.maximum(reference, 0.0)
    reference_positive /= max(
        float(np.mean(reference_positive)),
        1.0e-300,
    )

    difference = numerical_positive - reference_positive
    l1 = float(np.mean(np.abs(difference)))
    l2 = float(np.sqrt(np.mean(difference**2)))
    rel_l2 = l2 / max(
        float(np.sqrt(np.mean(reference_positive**2))),
        1.0e-15,
    )
    hellinger = float(
        np.sqrt(
            0.5
            * np.mean(
                (
                    np.sqrt(numerical_positive)
                    - np.sqrt(reference_positive)
                )
                ** 2
            )
        )
    )
    midpoint = 0.5 * (numerical_positive + reference_positive)
    js = float(
        0.5
        * np.mean(
            reference_positive
            * (
                np.log(np.maximum(reference_positive, 1.0e-300))
                - np.log(np.maximum(midpoint, 1.0e-300))
            )
        )
        + 0.5
        * np.mean(
            numerical_positive
            * (
                np.log(np.maximum(numerical_positive, 1.0e-300))
                - np.log(np.maximum(midpoint, 1.0e-300))
            )
        )
    )
    return {
        "conditional_L1_error": l1,
        "conditional_L2_error": l2,
        "conditional_relative_L2_error": rel_l2,
        "Hellinger_distance": hellinger,
        "Jensen_Shannon_divergence": js,
    }


def unnormalized_density_metrics(
    numerical: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    numerical = np.asarray(numerical, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = numerical - reference
    l1 = float(np.mean(np.abs(difference)))
    l2 = float(np.sqrt(np.mean(difference**2)))
    relative_l2 = l2 / max(
        float(np.sqrt(np.mean(reference**2))),
        1.0e-15,
    )
    negative_mass = float(np.mean(np.maximum(-numerical, 0.0)))
    return {
        "L1_error": l1,
        "L2_error": l2,
        "relative_L2_error": relative_l2,
        "mass": float(np.mean(numerical)),
        "mass_deviation_from_reference": abs(
            float(np.mean(numerical))
            - float(np.mean(reference))
        ),
        "negative_mass": negative_mass,
        "minimum_density": float(np.min(numerical)),
        "maximum_density": float(np.max(numerical)),
    }


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((first - second) ** 2))
        / max(np.sqrt(np.mean(second**2)), 1.0e-15)
    )


def interpolate_density(
    density: np.ndarray,
    target_size: int,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    source_size = density.shape[0]
    source_axis = (
        np.arange(source_size, dtype=np.float64) + 0.5
    ) / source_size
    target_axis = (
        np.arange(target_size, dtype=np.float64) + 0.5
    ) / target_size
    interpolator = RegularGridInterpolator(
        (source_axis, source_axis),
        density,
        bounds_error=False,
        fill_value=0.0,
    )
    x, y = np.meshgrid(target_axis, target_axis, indexing="xy")
    points = np.column_stack([y.reshape(-1), x.reshape(-1)])
    return interpolator(points).reshape(target_size, target_size)


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


def sliced_wasserstein_2d(
    points: np.ndarray,
    numerical: np.ndarray,
    reference: np.ndarray,
    count: int,
    directions: int,
    seed: int,
) -> Dict[str, float]:
    numerical_samples = systematic_resample(
        points,
        numerical,
        count,
        seed + 1,
    )
    reference_samples = systematic_resample(
        points,
        reference,
        count,
        seed + 2,
    )
    rng = np.random.default_rng(seed + 3)
    direction_array = rng.normal(size=(directions, 2))
    direction_array /= np.linalg.norm(
        direction_array,
        axis=1,
        keepdims=True,
    )
    values = np.array(
        [
            np.mean(
                np.abs(
                    np.sort(numerical_samples @ direction)
                    - np.sort(reference_samples @ direction)
                )
            )
            for direction in direction_array
        ]
    )
    return {
        "mean_sliced_W": float(np.mean(values)),
        "median_sliced_W": float(np.median(values)),
        "max_sliced_W": float(np.max(values)),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def config_to_dict(cfg: BenchmarkConfig) -> Dict[str, object]:
    payload = asdict(cfg)
    payload["fractional_coefficient"] = cfg.fractional_coefficient
    payload["fractional_constant"] = cfg.fractional_constant
    return payload
