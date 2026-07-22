from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dirichlet_common import (
    BenchmarkConfig,
    config_to_dict,
    write_json,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transient killed-Levy exterior-Dirichlet "
            "SBPINN/FEM benchmark."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["reference", "sbpinn", "fem_single", "report"],
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
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
    parser.add_argument("--fem_single_mesh", type=int)
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
    return parser


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        alpha=args.alpha,
        noise_scale=args.noise_scale,
        seed=args.seed,
        fem_cells_per_axis=parse_int_tuple(args.fem_meshes),
        fem_precheck_cells=parse_int_tuple(args.fem_precheck),
        fem_memory_budget_gb=args.fem_memory_budget_gb,
        fem_time_budget_s=args.fem_time_budget_s,
        output_dir=args.output_dir,
        device=args.device,
        quick=args.quick,
    )
    cfg.boundary_power = 0.5 * cfg.alpha
    if args.quick:
        cfg.apply_quick_mode()
    cfg.validate()
    return cfg


def aggregate_fem(
    cfg: BenchmarkConfig,
    fem_dir: Path,
) -> pd.DataFrame:
    rows = []
    for cells in (
        tuple(cfg.fem_cells_per_axis)
        + tuple(cfg.fem_precheck_cells)
    ):
        path = fem_dir / f"fem_n{cells}_result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            from dirichlet_p1_fem import resource_estimate

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
    frame.to_csv(fem_dir / "fem_results.csv", index=False)
    return frame


def make_figures(
    output_dir: Path,
    cfg: BenchmarkConfig,
    fem_frame: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reference = np.load(
        output_dir / "reference/reference_pde_finest.npz"
    )
    sbpinn = np.load(
        output_dir / "sbpinn/sbpinn_evaluation.npz"
    )
    times = reference["times"]
    for index, time_value in enumerate(times):
        for name, density, title in (
            (
                "reference",
                reference["densities"][index],
                f"Reference density, t={time_value:.3f}",
            ),
            (
                "sbpinn",
                sbpinn["model_densities"][index],
                f"SBPINN density, t={time_value:.3f}",
            ),
            (
                "absolute_error",
                np.abs(
                    sbpinn["model_densities"][index]
                    - reference["densities"][index]
                ),
                f"SBPINN absolute error, t={time_value:.3f}",
            ),
        ):
            fig, ax = plt.subplots(
                figsize=(6.7, 5.5),
                constrained_layout=True,
            )
            image = ax.imshow(
                density,
                origin="lower",
                extent=[0.0, 1.0, 0.0, 1.0],
                aspect="equal",
            )
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(title)
            fig.colorbar(image, ax=ax)
            token = int(round(1000 * time_value))
            fig.savefig(
                figure_dir / f"{name}_t{token:03d}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    completed = fem_frame[
        fem_frame["status"] == "completed"
    ].copy()
    if not completed.empty:
        fig, ax = plt.subplots(
            figsize=(7.8, 5.0),
            constrained_layout=True,
        )
        ax.plot(
            completed["cells_per_axis"],
            completed["mean_relative_L2_error"],
            marker="o",
            label="P1 FEM",
        )
        sb_summary = json.loads(
            (
                output_dir / "sbpinn/sbpinn_summary.json"
            ).read_text(encoding="utf-8")
        )
        ax.axhline(
            sb_summary["mean_relative_L2_error"],
            linestyle="--",
            label="SBPINN",
        )
        ax.set_xlabel("Cells per axis")
        ax.set_ylabel("Mean relative L2 error")
        ax.set_title("Exterior-Dirichlet accuracy comparison")
        ax.legend()
        fig.savefig(
            figure_dir / "method_accuracy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def build_report(
    output_dir: Path,
    cfg: BenchmarkConfig,
) -> None:
    reference = json.loads(
        (
            output_dir / "reference/reference_summary.json"
        ).read_text(encoding="utf-8")
    )
    sbpinn = json.loads(
        (
            output_dir / "sbpinn/sbpinn_summary.json"
        ).read_text(encoding="utf-8")
    )
    fem_frame = aggregate_fem(cfg, output_dir / "fem")
    completed = fem_frame[
        fem_frame["status"] == "completed"
    ].copy()
    valid = completed[
        completed["maximum_negative_mass"] < 0.01
    ].copy()
    best_fem = (
        valid.sort_values("mean_relative_L2_error").iloc[0]
        if not valid.empty
        else None
    )

    trend_pass = False
    if best_fem is not None:
        trend_pass = bool(
            sbpinn["mean_relative_L2_error"]
            < best_fem["mean_relative_L2_error"]
            and sbpinn["mean_Hellinger_distance"]
            < best_fem["mean_Hellinger_distance"]
            and sbpinn["maximum_negative_mass"]
            <= best_fem["maximum_negative_mass"] + 1.0e-14
            and sbpinn["mean_survival_mass_absolute_error"]
            <= best_fem["mean_survival_mass_absolute_error"]
        )

    verdict = (
        "SAME_TREND_OBTAINED"
        if trend_pass and reference["high_fidelity_reference"]
        else "SAME_TREND_NOT_YET_OBTAINED"
    )
    best_fem_payload = (
        best_fem.to_dict() if best_fem is not None else None
    )
    write_json(
        output_dir / "automatic_verdict.json",
        {
            "verdict": verdict,
            "reference_high_fidelity": reference[
                "high_fidelity_reference"
            ],
            "trend_pass": trend_pass,
            "sbpinn": sbpinn,
            "best_fem": best_fem_payload,
        },
    )

    comparison_rows = [
        {
            "method": "SBPINN",
            "configuration": (
                f"{sbpinn['parameter_count']} parameters"
            ),
            "mean_relative_L2_error": sbpinn[
                "mean_relative_L2_error"
            ],
            "mean_Hellinger_distance": sbpinn[
                "mean_Hellinger_distance"
            ],
            "mean_survival_mass_absolute_error": sbpinn[
                "mean_survival_mass_absolute_error"
            ],
            "maximum_negative_mass": sbpinn[
                "maximum_negative_mass"
            ],
            "runtime_s": sbpinn["total_training_time_s"],
        }
    ]
    if best_fem is not None:
        comparison_rows.append(
            {
                "method": "Best probability-valid P1 FEM",
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
                "mean_survival_mass_absolute_error": best_fem[
                    "mean_survival_mass_absolute_error"
                ],
                "maximum_negative_mass": best_fem[
                    "maximum_negative_mass"
                ],
                "runtime_s": best_fem["total_time_s"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "comparison_table.csv",
        index=False,
    )

    best_text = (
        "No completed probability-valid FEM configuration."
        if best_fem is None
        else (
            f"Best FEM: {int(best_fem['cells_per_axis'])}x"
            f"{int(best_fem['cells_per_axis'])}; mean relative L2="
            f"{best_fem['mean_relative_L2_error']:.6f}."
        )
    )
    report = f"""# Exterior-Dirichlet killed-Lévy benchmark

## Problem

- Domain: Omega=(0,1)^2
- Boundary: homogeneous exterior Dirichlet
- SDE interpretation: particle is killed after leaving Omega
- alpha: {cfg.alpha}
- noise scale: {cfg.noise_scale}
- evaluation times: {cfg.evaluation_times}

## Reference audit

- high-fidelity gate: {reference['high_fidelity_reference']}
- maximum PDE grid difference:
  {reference['maximum_pde_grid_relative_L2']:.6f}
- maximum PDE-versus-MC survival-mass error:
  {reference['maximum_pde_vs_mc_mass_error']:.6f}
- maximum PDE-versus-MC conditional-density error:
  {reference['maximum_pde_vs_mc_conditional_relative_L2']:.6f}

## SBPINN

- mean relative L2:
  {sbpinn['mean_relative_L2_error']:.6f}
- mean Hellinger:
  {sbpinn['mean_Hellinger_distance']:.6f}
- mean survival-mass absolute error:
  {sbpinn['mean_survival_mass_absolute_error']:.6f}
- maximum negative mass:
  {sbpinn['maximum_negative_mass']:.3E}
- training time:
  {sbpinn['total_training_time_s']:.2f} s

## FEM

{best_text}

The FEM statement is restricted to the implemented mass-lumped structured
P1 method with dense nonlocal quadrature. It is not a statement about every
advanced adaptive or fast fractional FEM.

## Decision

```text
{verdict}
```
"""
    (output_dir / "EXPERIMENT_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    make_figures(output_dir, cfg, fem_frame)


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    base_output = Path(cfg.output_dir).resolve()
    output_dir = (
        Path(str(base_output) + "_quick")
        if cfg.quick
        else base_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "reference":
        from generate_dirichlet_reference import generate_reference_data

        write_json(
            output_dir / "reference/reference_config.json",
            config_to_dict(cfg),
        )
        generate_reference_data(cfg, output_dir / "reference")
    elif args.mode == "sbpinn":
        from dirichlet_sbpinn import run_sbpinn

        write_json(
            output_dir / "sbpinn/sbpinn_config.json",
            config_to_dict(cfg),
        )
        run_sbpinn(
            cfg,
            output_dir / "reference",
            output_dir / "sbpinn",
        )
    elif args.mode == "fem_single":
        if args.fem_single_mesh is None:
            raise ValueError("--fem_single_mesh is required.")
        from dirichlet_p1_fem import solve_fem

        solve_fem(
            cfg,
            int(args.fem_single_mesh),
            output_dir / "reference",
            output_dir / "fem",
        )
    else:
        write_json(
            output_dir / "benchmark_config.json",
            config_to_dict(cfg),
        )
        build_report(output_dir, cfg)


if __name__ == "__main__":
    main()
