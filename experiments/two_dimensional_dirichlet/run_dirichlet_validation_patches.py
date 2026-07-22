from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dirichlet_common import write_json
from dirichlet_validation_patches import (
    build_validation_report,
    config_from_json,
    generate_validation_reference,
    reevaluate_existing_fem,
    reevaluate_sbpinn_checkpoint,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run the three frozen validation patches: high-resolution "
            "reference, fair FEM refinement and residual-floor audit."
        )
    )
    result.add_argument(
        "--base_results_dir",
        required=True,
        help="Existing completed V1 result directory.",
    )
    result.add_argument(
        "--output_dir",
        default="results_dirichlet_validation_v2",
    )
    result.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    result.add_argument(
        "--reference_grids",
        default="192,256",
    )
    result.add_argument(
        "--reference_base_grid",
        type=int,
        default=128,
    )
    result.add_argument(
        "--reference_padding",
        type=int,
        default=3,
    )
    result.add_argument(
        "--padding_sensitivity",
        type=int,
        default=4,
    )
    result.add_argument(
        "--reference_dt",
        type=float,
        default=5.0e-4,
    )
    result.add_argument(
        "--dt_sensitivity",
        type=float,
        default=2.5e-4,
    )
    result.add_argument(
        "--fem_meshes",
        default="80,96,112",
    )
    result.add_argument(
        "--fem_precheck",
        default="128",
    )
    result.add_argument(
        "--run_precheck_meshes",
        action="store_true",
        help="Actually run the precheck meshes after resource inspection.",
    )
    result.add_argument(
        "--fem_memory_budget_gb",
        type=float,
        default=16.0,
    )
    result.add_argument(
        "--fem_time_budget_s",
        type=float,
        default=3600.0,
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--quick", action="store_true")
    return result


def choose_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def run_fem_process(
    package_root: Path,
    cfg,
    output_dir: Path,
    mesh: int,
    timeout: float,
    env: dict[str, str],
) -> int:
    result_path = output_dir / f"fem/fem_n{mesh}_result.json"
    log_path = output_dir / f"logs/fem_{mesh}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(package_root / "run_validation_fem_single.py"),
        "--config_json",
        str(output_dir / "validation_config.json"),
        "--reference_dir",
        str(output_dir / "reference"),
        "--output_dir",
        str(output_dir / "fem"),
        "--mesh",
        str(mesh),
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout + 180.0,
            )
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = 124
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nTIMEOUT after {timeout} seconds.\n")

    if code != 0 and not result_path.exists():
        from dirichlet_p1_fem import resource_estimate

        write_json(
            result_path,
            {
                "method": (
                    "mass-lumped exterior-Dirichlet P1 FEM "
                    "with dense nonlocal quadrature"
                ),
                "cells_per_axis": mesh,
                "status": (
                    "timeout" if code == 124 else "failed_process"
                ),
                "process_exit_code": code,
                **resource_estimate(mesh),
            },
        )
    return code


def main() -> None:
    args = parser().parse_args()
    package_root = Path(__file__).resolve().parent
    base_results = Path(args.base_results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    cfg = config_from_json(
        base_results / "benchmark_config.json"
    )
    cfg.output_dir = str(output_dir)
    cfg.device = args.device
    cfg.validation_reference_base_grid = args.reference_base_grid
    cfg.validation_reference_grids = parse_int_tuple(
        args.reference_grids
    )
    cfg.reference_padding_factor = args.reference_padding
    cfg.validation_padding_factor = args.padding_sensitivity
    cfg.reference_time_step = args.reference_dt
    cfg.validation_time_step = args.dt_sensitivity
    cfg.fem_cells_per_axis = parse_int_tuple(args.fem_meshes)
    cfg.fem_precheck_cells = parse_int_tuple(args.fem_precheck)
    cfg.fem_memory_budget_gb = args.fem_memory_budget_gb
    cfg.fem_time_budget_s = args.fem_time_budget_s

    if args.quick:
        # Keep the trained model architecture, SDE, final time and evaluation
        # times unchanged. Only shrink the validation grids and diagnostics.
        cfg.validation_reference_base_grid = 64
        cfg.validation_reference_grids = (72,)
        cfg.reference_padding_factor = 3
        cfg.validation_padding_factor = 4
        cfg.reference_time_step = 5.0e-4
        cfg.validation_time_step = 2.5e-4
        cfg.reference_grid_convergence_threshold = 2.0
        cfg.reference_mass_threshold = 2.0
        cfg.reference_mc_shape_threshold = 2.0
        cfg.validation_sensitivity_threshold = 2.0
        cfg.reference_residual_audit_grid = 32
        cfg.wasserstein_samples = 500
        cfg.wasserstein_directions = 8
        cfg.fem_cells_per_axis = (10,)
        cfg.fem_precheck_cells = ()
        cfg.fem_memory_budget_gb = 2.0
        cfg.fem_time_budget_s = 180.0
        cfg.output_dir = str(output_dir)
        cfg.device = args.device

    write_json(
        output_dir / "validation_config.json",
        {
            **cfg.__dict__,
            "base_results_dir": str(base_results),
            "SBPINN_frozen": True,
        },
    )

    device = choose_device(args.device)
    reference_summary_path = (
        output_dir / "reference/reference_summary.json"
    )
    if not (args.resume and reference_summary_path.exists()):
        generate_validation_reference(
            cfg,
            base_results,
            output_dir / "reference",
            device,
            args.resume,
            args.quick,
        )

    sbpinn_summary_path = (
        output_dir / "sbpinn/sbpinn_summary.json"
    )
    if not (args.resume and sbpinn_summary_path.exists()):
        reevaluate_sbpinn_checkpoint(
            cfg,
            base_results,
            output_dir / "reference",
            output_dir / "sbpinn",
            device,
        )

    # Reuse and re-evaluate all already completed meshes against the
    # high-resolution reference.
    reevaluate_existing_fem(
        cfg,
        base_results,
        output_dir / "reference",
        output_dir / "fem",
    )

    if args.quick:
        new_meshes = (10,)
        precheck_meshes = ()
    else:
        new_meshes = parse_int_tuple(args.fem_meshes)
        precheck_meshes = parse_int_tuple(args.fem_precheck)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for mesh in new_meshes:
        result_path = output_dir / f"fem/fem_n{mesh}_result.json"
        if args.resume and result_path.exists():
            continue
        run_fem_process(
            package_root,
            cfg,
            output_dir,
            mesh,
            cfg.fem_time_budget_s,
            env,
        )

    from dirichlet_p1_fem import resource_estimate

    for mesh in precheck_meshes:
        result_path = output_dir / f"fem/fem_n{mesh}_result.json"
        if args.resume and result_path.exists():
            continue
        estimate = resource_estimate(mesh)
        if args.run_precheck_meshes:
            run_fem_process(
                package_root,
                cfg,
                output_dir,
                mesh,
                cfg.fem_time_budget_s,
                env,
            )
        else:
            write_json(
                result_path,
                {
                    "method": (
                        "mass-lumped exterior-Dirichlet P1 FEM "
                        "with dense nonlocal quadrature"
                    ),
                    "cells_per_axis": mesh,
                    "status": "resource_precheck_only",
                    **estimate,
                },
            )

    existing_frame = __import__("pandas").read_csv(
        base_results / "fem/fem_results.csv"
    )
    old_meshes = tuple(
        int(value)
        for value in existing_frame[
            existing_frame["status"] == "completed"
        ]["cells_per_axis"]
    )
    all_meshes = tuple(
        sorted(
            set(
                old_meshes
                + tuple(new_meshes)
                + tuple(precheck_meshes)
            )
        )
    )
    decision = build_validation_report(
        cfg,
        output_dir,
        all_meshes,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Validation output: {output_dir}")


if __name__ == "__main__":
    main()
