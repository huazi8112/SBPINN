from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all Dirichlet benchmark stages in isolated processes."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--noise_scale", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fem_meshes",
        default="12,16,24,32,40,48",
    )
    parser.add_argument(
        "--fem_precheck",
        default="56,64",
    )
    parser.add_argument(
        "--fem_memory_budget_gb",
        type=float,
        default=16.0,
    )
    parser.add_argument(
        "--fem_time_budget_s",
        type=float,
        default=3600.0,
    )
    parser.add_argument(
        "--output_dir",
        default="results_dirichlet_killed_levy",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_stage(
    name: str,
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    timeout: float | None = None,
    allow_failure: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(" ".join(command))
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nTIMEOUT after {timeout} seconds.\n")
        returncode = 124
    if returncode != 0 and not allow_failure:
        raise SystemExit(
            f"{name} failed with exit code {returncode}. "
            f"See {log_path}"
        )
    return returncode


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    runner = root / "run_dirichlet_benchmark.py"
    base_output = Path(args.output_dir).resolve()
    actual_output = (
        Path(str(base_output) + "_quick")
        if args.quick
        else base_output
    )
    logs = actual_output / "logs"
    common = [
        sys.executable,
        str(runner),
        "--output_dir",
        str(base_output),
        "--device",
        args.device,
        "--alpha",
        str(args.alpha),
        "--noise_scale",
        str(args.noise_scale),
        "--seed",
        str(args.seed),
        "--fem_meshes",
        args.fem_meshes,
        "--fem_precheck",
        args.fem_precheck,
        "--fem_memory_budget_gb",
        str(args.fem_memory_budget_gb),
        "--fem_time_budget_s",
        str(args.fem_time_budget_s),
    ]
    if args.quick:
        common.append("--quick")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    reference_done = (
        actual_output / "reference/reference_summary.json"
    ).exists()
    sbpinn_done = (
        actual_output / "sbpinn/sbpinn_summary.json"
    ).exists()

    if not (args.resume and reference_done):
        run_stage(
            "REFERENCE",
            common + ["--mode", "reference"],
            logs / "reference.log",
            env,
        )
    if not (args.resume and sbpinn_done):
        run_stage(
            "SBPINN",
            common + ["--mode", "sbpinn"],
            logs / "sbpinn.log",
            env,
        )

    meshes = parse_int_tuple(args.fem_meshes)
    precheck = parse_int_tuple(args.fem_precheck)
    if args.quick:
        meshes = (6, 8)
        precheck = (10,)
    for mesh in meshes + precheck:
        result_path = (
            actual_output / f"fem/fem_n{mesh}_result.json"
        )
        if args.resume and result_path.exists():
            continue
        returncode = run_stage(
            f"FEM_{mesh}",
            common
            + [
                "--mode",
                "fem_single",
                "--device",
                "cpu",
                "--fem_single_mesh",
                str(mesh),
            ],
            logs / f"fem_{mesh}.log",
            env,
            timeout=args.fem_time_budget_s + 180.0,
            allow_failure=True,
        )
        if returncode != 0 and not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "method": (
                            "mass-lumped exterior-Dirichlet P1 FEM "
                            "with dense nonlocal quadrature"
                        ),
                        "cells_per_axis": mesh,
                        "status": (
                            "timeout"
                            if returncode == 124
                            else "failed_process"
                        ),
                        "process_exit_code": returncode,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    run_stage(
        "REPORT",
        common + ["--mode", "report", "--device", "cpu"],
        logs / "report.log",
        env,
    )
    print(f"\nAll stages completed: {actual_output}")


if __name__ == "__main__":
    main()
