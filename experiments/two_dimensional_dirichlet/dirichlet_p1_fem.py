from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

from dirichlet_common import (
    BenchmarkConfig,
    cell_center_grid,
    conditional_density_metrics,
    drift_numpy,
    initial_density_numpy,
    sliced_wasserstein_2d,
    unnormalized_density_metrics,
    write_json,
)


def structured_triangles(cells: int) -> np.ndarray:
    triangles = []
    nodes_per_axis = cells + 1
    for j in range(cells):
        for i in range(cells):
            n00 = j * nodes_per_axis + i
            n10 = n00 + 1
            n01 = (j + 1) * nodes_per_axis + i
            n11 = n01 + 1
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    return np.asarray(triangles, dtype=np.int64)


def node_coordinates(cells: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, cells + 1)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([x.reshape(-1), y.reshape(-1)])


def interior_mapping(cells: int) -> Tuple[np.ndarray, Dict[int, int]]:
    nodes_per_axis = cells + 1
    indices = []
    for j in range(1, cells):
        for i in range(1, cells):
            indices.append(j * nodes_per_axis + i)
    array = np.asarray(indices, dtype=np.int64)
    return array, {int(global_id): local for local, global_id in enumerate(array)}


def assemble_mass_and_advection(
    cfg: BenchmarkConfig,
    cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = node_coordinates(cells)
    triangles = structured_triangles(cells)
    interior, mapping = interior_mapping(cells)
    count = len(interior)
    mass = np.zeros(count, dtype=np.float64)
    advection = np.zeros((count, count), dtype=np.float64)

    for triangle in triangles:
        vertices = coordinates[triangle]
        matrix = np.array(
            [
                [1.0, vertices[0, 0], vertices[0, 1]],
                [1.0, vertices[1, 0], vertices[1, 1]],
                [1.0, vertices[2, 0], vertices[2, 1]],
            ]
        )
        determinant = np.linalg.det(
            np.column_stack(
                [
                    vertices[1] - vertices[0],
                    vertices[2] - vertices[0],
                ]
            )
        )
        area = 0.5 * abs(determinant)
        inverse = np.linalg.inv(matrix)
        gradients = inverse[1:, :].T
        centroid = np.mean(vertices, axis=0)
        velocity = drift_numpy(
            centroid[None, :],
            cfg,
        )[0]

        for local_i, global_i in enumerate(triangle):
            if int(global_i) not in mapping:
                continue
            row = mapping[int(global_i)]
            mass[row] += area / 3.0
            for local_j, global_j in enumerate(triangle):
                if int(global_j) not in mapping:
                    continue
                column = mapping[int(global_j)]
                advection[row, column] += (
                    area
                    * float(velocity @ gradients[local_i])
                    / 3.0
                )
    return coordinates[interior], mass, advection


def exterior_integral(
    points: np.ndarray,
    cells: int,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    h = 1.0 / cells
    padding = cfg.fem_exterior_padding
    axis = np.arange(
        -padding + 0.5 * h,
        1.0 + padding,
        h,
    )
    x, y = np.meshgrid(axis, axis, indexing="xy")
    outside = ~(
        (x > 0.0)
        & (x < 1.0)
        & (y > 0.0)
        & (y < 1.0)
    )
    outside_points = np.column_stack(
        [x[outside], y[outside]]
    )
    values = np.zeros(len(points), dtype=np.float64)
    chunk = 128
    exponent = 2.0 + cfg.alpha
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        difference = (
            block[:, None, :] - outside_points[None, :, :]
        )
        radius_squared = np.sum(difference**2, axis=2)
        values[start : start + len(block)] = (
            h * h
            * np.sum(
                radius_squared ** (-0.5 * exponent),
                axis=1,
            )
        )
    far_tail = (
        2.0
        * math.pi
        / cfg.alpha
        * max(padding, h) ** (-cfg.alpha)
    )
    return values + far_tail


def assemble_fractional_matrix(
    points: np.ndarray,
    mass: np.ndarray,
    cells: int,
    cfg: BenchmarkConfig,
) -> np.ndarray:
    """Assemble the dense nonlocal matrix in row blocks.

    The mathematical discretization is unchanged. Blocking avoids forming a
    full (N,N,2) displacement tensor, which is the dominant avoidable memory
    cost on the 80--112 cell validation meshes.
    """
    count = len(points)
    matrix = np.empty((count, count), dtype=np.float64)
    row_sum = np.zeros(count, dtype=np.float64)
    exponent = -0.5 * (2.0 + cfg.alpha)
    block_size = max(8, int(cfg.fem_kernel_block_size))

    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        difference = (
            points[start:stop, None, :]
            - points[None, :, :]
        )
        radius_squared = np.sum(difference**2, axis=2)
        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        radius_squared[local_rows, global_rows] = np.inf
        kernel = radius_squared**exponent
        weighted = (
            cfg.fractional_constant
            * mass[start:stop, None]
            * mass[None, :]
            * kernel
        )
        matrix[start:stop, :] = -weighted
        row_sum[start:stop] = np.sum(weighted, axis=1)

    exterior = exterior_integral(points, cells, cfg)
    diagonal = row_sum + (
        cfg.fractional_constant * mass * exterior
    )
    np.fill_diagonal(matrix, diagonal)
    return matrix


def resource_estimate(cells: int) -> Dict[str, float]:
    unknowns = (cells - 1) ** 2
    matrix_gb = 8.0 * unknowns * unknowns / 1024**3
    return {
        "unknown_count": unknowns,
        "single_dense_matrix_GB": matrix_gb,
        "estimated_peak_memory_GB": 8.0 * matrix_gb,
        "estimated_factorization_flops": (
            2.0 / 3.0 * unknowns**3
        ),
    }


def interpolate_nodal_solution(
    values: np.ndarray,
    cells: int,
    target_size: int,
) -> np.ndarray:
    full = np.zeros((cells + 1, cells + 1), dtype=np.float64)
    full[1:-1, 1:-1] = values.reshape(cells - 1, cells - 1)
    node_axis = np.linspace(0.0, 1.0, cells + 1)
    interpolator = RegularGridInterpolator(
        (node_axis, node_axis),
        full,
        bounds_error=False,
        fill_value=0.0,
    )
    target_axis = (
        np.arange(target_size, dtype=np.float64) + 0.5
    ) / target_size
    x, y = np.meshgrid(target_axis, target_axis, indexing="xy")
    points = np.column_stack([y.reshape(-1), x.reshape(-1)])
    return interpolator(points).reshape(target_size, target_size)


def solve_fem(
    cfg: BenchmarkConfig,
    cells: int,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate = resource_estimate(cells)
    if estimate["estimated_peak_memory_GB"] > cfg.fem_memory_budget_gb:
        row = {
            "method": (
                "mass-lumped exterior-Dirichlet P1 FEM "
                "with dense nonlocal quadrature"
            ),
            "cells_per_axis": cells,
            "status": "memory_precheck_only",
            **estimate,
        }
        write_json(
            output_dir / f"fem_n{cells}_result.json",
            row,
        )
        return row

    reference_payload = np.load(
        reference_dir / "reference_pde_finest.npz"
    )
    reference_times = reference_payload["times"]
    reference_densities = reference_payload["densities"]
    target_size = reference_densities.shape[1]

    start = time.perf_counter()
    points, mass, advection = assemble_mass_and_advection(
        cfg,
        cells,
    )
    fractional_start = time.perf_counter()
    fractional = assemble_fractional_matrix(
        points,
        mass,
        cells,
        cfg,
    )
    fractional_time = time.perf_counter() - fractional_start

    system = (
        np.diag(mass)
        - cfg.fem_time_step * advection
        + cfg.fem_time_step
        * cfg.fractional_coefficient
        * fractional
    )
    factor_start = time.perf_counter()
    factor = scipy.linalg.lu_factor(
        system,
        check_finite=False,
    )
    factor_time = time.perf_counter() - factor_start

    initial_values = initial_density_numpy(points, cfg)
    initial_values /= max(
        float(np.dot(mass, initial_values)),
        1.0e-300,
    )
    solution = initial_values.copy()
    total_steps = int(round(cfg.final_time / cfg.fem_time_step))
    evaluation_steps = {
        int(round(t / cfg.fem_time_step)): (index, float(t))
        for index, t in enumerate(reference_times)
    }
    rows = []
    densities = []

    for step in range(total_steps):
        right_hand_side = mass * solution
        solution = scipy.linalg.lu_solve(
            factor,
            right_hand_side,
            check_finite=False,
        )
        completed_step = step + 1
        if completed_step in evaluation_steps:
            reference_index, time_value = evaluation_steps[completed_step]
            density = interpolate_nodal_solution(
                solution,
                cells,
                target_size,
            )
            reference = reference_densities[reference_index]
            reference_mass = float(np.mean(reference))
            numerical_mass = float(np.mean(density))
            reference_conditional = reference / max(
                reference_mass,
                1.0e-300,
            )
            numerical_conditional = np.maximum(density, 0.0)
            numerical_conditional /= max(
                float(np.mean(numerical_conditional)),
                1.0e-300,
            )
            _, _, evaluation_points = cell_center_grid(target_size)
            metrics = {
                "time": time_value,
                **unnormalized_density_metrics(density, reference),
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
                    cfg.seed + cells * 10 + reference_index,
                ),
                "survival_mass_absolute_error": abs(
                    numerical_mass - reference_mass
                ),
            }
            rows.append(metrics)
            densities.append(density)

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / f"fem_n{cells}_metrics_by_time.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / f"fem_n{cells}_solution.npz",
        times=reference_times,
        densities=np.stack(densities),
        final_nodal_values=solution,
    )
    row = {
        "method": (
            "mass-lumped exterior-Dirichlet P1 FEM "
            "with dense nonlocal quadrature"
        ),
        "cells_per_axis": cells,
        "status": "completed",
        **estimate,
        "fractional_assembly_time_s": fractional_time,
        "factorization_time_s": factor_time,
        "total_time_s": time.perf_counter() - start,
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
        "per_time": rows,
    }
    write_json(
        output_dir / f"fem_n{cells}_result.json",
        row,
    )
    return row
