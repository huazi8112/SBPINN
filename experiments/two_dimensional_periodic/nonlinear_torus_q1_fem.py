from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from nonlinear_torus_common import (
    BenchmarkConfig,
    density_metrics,
    drift_numpy,
    periodic_grid,
    torus_sliced_wasserstein,
    write_json,
)


def periodic_1d_matrices(
    cells_per_axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(cells_per_axis)
    h = 1.0 / n
    mass = np.zeros((n, n), dtype=np.float64)
    stiffness = np.zeros((n, n), dtype=np.float64)
    local_mass = h / 6.0 * np.array(
        [[2.0, 1.0], [1.0, 2.0]]
    )
    local_stiffness = 1.0 / h * np.array(
        [[1.0, -1.0], [-1.0, 1.0]]
    )
    for index in range(n):
        next_index = (index + 1) % n
        nodes = (index, next_index)
        for local_i, global_i in enumerate(nodes):
            for local_j, global_j in enumerate(nodes):
                mass[global_i, global_j] += local_mass[
                    local_i,
                    local_j,
                ]
                stiffness[global_i, global_j] += (
                    local_stiffness[local_i, local_j]
                )
    return mass, stiffness


def assemble_periodic_q1_advection(
    cfg: BenchmarkConfig,
    cells_per_axis: int,
) -> sp.csr_matrix:
    n = int(cells_per_axis)
    h = 1.0 / n
    gauss = np.array(
        [
            0.5 - 0.5 / math.sqrt(3.0),
            0.5 + 0.5 / math.sqrt(3.0),
        ]
    )
    weights = np.array([0.5, 0.5])
    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []

    for cell_y in range(n):
        for cell_x in range(n):
            nodes = np.array(
                [
                    cell_y * n + cell_x,
                    cell_y * n + (cell_x + 1) % n,
                    ((cell_y + 1) % n) * n + cell_x,
                    ((cell_y + 1) % n) * n
                    + (cell_x + 1) % n,
                ],
                dtype=int,
            )
            local = np.zeros((4, 4), dtype=np.float64)

            for qy, wy in zip(gauss, weights):
                for qx, wx in zip(gauss, weights):
                    x = (cell_x + qx) / n
                    y = (cell_y + qy) / n
                    drift = drift_numpy(
                        np.array([[x, y]]),
                        cfg,
                    )[0]
                    shape = np.array(
                        [
                            (1.0 - qx) * (1.0 - qy),
                            qx * (1.0 - qy),
                            (1.0 - qx) * qy,
                            qx * qy,
                        ]
                    )
                    grad_reference = np.array(
                        [
                            [-(1.0 - qy), -(1.0 - qx)],
                            [(1.0 - qy), -qx],
                            [-qy, (1.0 - qx)],
                            [qy, qx],
                        ]
                    )
                    grad_physical = grad_reference / h
                    weight = wx * wy * h * h

                    for test in range(4):
                        drift_dot_grad = float(
                            drift @ grad_physical[test]
                        )
                        for trial in range(4):
                            local[test, trial] += (
                                weight
                                * drift_dot_grad
                                * shape[trial]
                            )

            for local_i, global_i in enumerate(nodes):
                for local_j, global_j in enumerate(nodes):
                    rows.append(int(global_i))
                    columns.append(int(global_j))
                    values.append(float(local[local_i, local_j]))

    matrix = sp.coo_matrix(
        (values, (rows, columns)),
        shape=(n * n, n * n),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def direct_fractional_q1_matrices(
    cfg: BenchmarkConfig,
    cells_per_axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mass_1d, stiffness_1d = periodic_1d_matrices(
        cells_per_axis
    )
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        stiffness_1d,
        mass_1d,
        check_finite=False,
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    mass_2d = np.kron(mass_1d, mass_1d)
    eigenvectors_2d = np.kron(eigenvectors, eigenvectors)

    lambda_x, lambda_y = np.meshgrid(
        eigenvalues,
        eigenvalues,
        indexing="xy",
    )
    fractional_eigenvalues = (
        lambda_x ** (cfg.alpha / 2.0)
        + lambda_y ** (cfg.alpha / 2.0)
    ).reshape(-1)

    transformed = mass_2d @ eigenvectors_2d
    fractional_stiffness = (
        transformed
        * fractional_eigenvalues[None, :]
    ) @ transformed.T
    mass_vector = mass_2d @ np.ones(cells_per_axis**2)
    return fractional_stiffness, mass_vector


def resource_estimate(
    cells_per_axis: int,
) -> Dict[str, float]:
    nodes = cells_per_axis**2
    one_dense_gb = 8.0 * nodes * nodes / 1024**3
    estimated_peak_gb = 7.5 * one_dense_gb
    estimated_flops = (2.0 / 3.0) * nodes**3
    return {
        "node_count": nodes,
        "single_dense_matrix_GB": one_dense_gb,
        "estimated_peak_memory_GB": estimated_peak_gb,
        "estimated_dense_solve_flops": estimated_flops,
    }


def solve_single_mesh(
    cfg: BenchmarkConfig,
    cells_per_axis: int,
    reference_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate = resource_estimate(cells_per_axis)
    if estimate["estimated_peak_memory_GB"] > cfg.fem_memory_budget_gb:
        row: Dict[str, object] = {
            "method": "Dense spectral periodic Q1 FEM",
            "cells_per_axis": cells_per_axis,
            "status": "memory_precheck_only",
            **estimate,
        }
        write_json(
            output_dir / f"fem_n{cells_per_axis}_result.json",
            row,
        )
        return row

    reference = np.load(
        reference_dir / "reference_density.npz"
    )["density"]
    _, _, evaluation_points = periodic_grid(
        cfg.evaluation_grid_size
    )

    start = time.perf_counter()
    fractional_start = time.perf_counter()
    fractional_stiffness, mass_vector = (
        direct_fractional_q1_matrices(
            cfg,
            cells_per_axis,
        )
    )
    fractional_time = time.perf_counter() - fractional_start

    advection_start = time.perf_counter()
    advection = assemble_periodic_q1_advection(
        cfg,
        cells_per_axis,
    ).toarray()
    advection_time = time.perf_counter() - advection_start

    operator = (
        advection
        - cfg.fractional_coefficient
        * fractional_stiffness
    )
    node_count = cells_per_axis**2
    saddle = np.zeros(
        (node_count + 1, node_count + 1),
        dtype=np.float64,
    )
    saddle[:node_count, :node_count] = operator
    saddle[:node_count, node_count] = mass_vector
    saddle[node_count, :node_count] = mass_vector
    right_hand_side = np.zeros(node_count + 1)
    right_hand_side[node_count] = 1.0

    solve_start = time.perf_counter()
    solution = scipy.linalg.solve(
        saddle,
        right_hand_side,
        assume_a="gen",
        check_finite=False,
    )
    solve_time = time.perf_counter() - solve_start
    nodal_values = solution[:node_count]

    n = cells_per_axis
    values = nodal_values.reshape(n, n)
    scaled_x = np.mod(evaluation_points[:, 0], 1.0) * n
    scaled_y = np.mod(evaluation_points[:, 1], 1.0) * n
    i = np.floor(scaled_x).astype(int) % n
    j = np.floor(scaled_y).astype(int) % n
    xi = scaled_x - np.floor(scaled_x)
    eta = scaled_y - np.floor(scaled_y)
    i1 = (i + 1) % n
    j1 = (j + 1) % n
    numerical = (
        (1.0 - xi) * (1.0 - eta) * values[j, i]
        + xi * (1.0 - eta) * values[j, i1]
        + (1.0 - xi) * eta * values[j1, i]
        + xi * eta * values[j1, i1]
    ).reshape(reference.shape)

    metrics = density_metrics(numerical, reference)
    metrics.update(
        torus_sliced_wasserstein(
            evaluation_points,
            numerical.reshape(-1),
            reference.reshape(-1),
            cfg,
        )
    )
    row = {
        "method": "Dense spectral periodic Q1 FEM",
        "cells_per_axis": cells_per_axis,
        "status": "completed",
        **estimate,
        "fractional_matrix_time_s": fractional_time,
        "advection_assembly_time_s": advection_time,
        "dense_solve_time_s": solve_time,
        "total_time_s": time.perf_counter() - start,
        **metrics,
    }
    np.savez_compressed(
        output_dir / f"fem_n{cells_per_axis}_solution.npz",
        nodal_values=nodal_values,
        evaluation_density=numerical,
    )
    write_json(
        output_dir / f"fem_n{cells_per_axis}_result.json",
        row,
    )
    return row
