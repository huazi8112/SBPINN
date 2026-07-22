from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nonlinear_torus_common import (
    BenchmarkConfig,
    config_to_dict,
    periodic_grid,
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
            "Nonlinear non-equilibrium torus alpha-stable SDE benchmark."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["reference", "sbpinn", "fem_single", "report"],
        required=True,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--noise_scale", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fem_meshes", default="12,16,24,32,40,48,64,80,96")
    parser.add_argument("--fem_precheck", default="128,160")
    parser.add_argument("--fem_single_mesh", type=int)
    parser.add_argument("--fem_memory_budget_gb", type=float, default=16.0)
    parser.add_argument("--fem_time_budget_s", type=float, default=3600.0)
    parser.add_argument(
        "--output_dir",
        default="results_nonlinear_torus_levy",
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
    if args.quick:
        cfg.apply_quick_mode()
    cfg.validate()
    return cfg


def aggregate_fem(
    cfg: BenchmarkConfig,
    fem_dir: Path,
) -> pd.DataFrame:
    rows = []
    requested = (
        tuple(cfg.fem_cells_per_axis)
        + tuple(cfg.fem_precheck_cells)
    )
    for cells in requested:
        path = fem_dir / f"fem_n{cells}_result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            from nonlinear_torus_q1_fem import resource_estimate

            rows.append(
                {
                    "method": "Dense spectral periodic Q1 FEM",
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
        output_dir / "reference/reference_density.npz"
    )["density"]
    model = np.load(
        output_dir / "sbpinn/sbpinn_evaluation.npz"
    )["model_density"]
    extent = [0.0, 1.0, 0.0, 1.0]

    for name, array, title in (
        ("reference_density", reference, "Independent SDE reference density"),
        ("sbpinn_density", model, "SBPINN stationary density"),
        (
            "sbpinn_absolute_error",
            np.abs(model - reference),
            "SBPINN absolute error",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
        image = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    completed = fem_frame[fem_frame["status"] == "completed"].copy()
    if not completed.empty:
        fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
        ax.plot(
            completed["cells_per_axis"],
            completed["relative_L2_error"],
            marker="o",
        )
        ax.axhline(0.05, linestyle="--", label="5% FEM target")
        ax.set_xlabel("Cells per axis")
        ax.set_ylabel("Relative L2 error")
        ax.set_yscale("log")
        ax.set_title("Dense periodic Q1 FEM accuracy")
        ax.legend()
        fig.savefig(
            figure_dir / "fem_error_vs_mesh.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figure_dir / "fem_error_vs_mesh.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        valid = completed[
            (completed["mass_deviation"] < 0.01)
            & (completed["negative_mass"] < 0.01)
        ]
        if not valid.empty:
            best = valid.sort_values("relative_L2_error").iloc[0]
            best_n = int(best["cells_per_axis"])
            fem_density = np.load(
                output_dir
                / f"fem/fem_n{best_n}_solution.npz"
            )["evaluation_density"]
            axis = np.arange(cfg.evaluation_grid_size) / cfg.evaluation_grid_size
            fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
            ax.plot(axis, np.diag(reference), label="Reference", linewidth=2.0)
            ax.plot(axis, np.diag(model), label="SBPINN")
            ax.plot(axis, np.diag(fem_density), label=f"FEM {best_n}x{best_n}")
            ax.set_xlabel(r"$x=y$")
            ax.set_ylabel("Density")
            ax.set_title("Diagonal stationary-density profile")
            ax.legend()
            fig.savefig(
                figure_dir / "diagonal_profiles.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                figure_dir / "diagonal_profiles.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def build_report(
    output_dir: Path,
    cfg: BenchmarkConfig,
) -> None:
    reference_summary = json.loads(
        (
            output_dir
            / "reference/reference_summary.json"
        ).read_text(encoding="utf-8")
    )
    sbpinn = json.loads(
        (
            output_dir
            / "sbpinn/sbpinn_summary.json"
        ).read_text(encoding="utf-8")
    )
    fem_frame = aggregate_fem(cfg, output_dir / "fem")
    completed = fem_frame[fem_frame["status"] == "completed"].copy()
    valid_fem = completed[
        (completed["mass_deviation"] < 0.01)
        & (completed["negative_mass"] < 0.01)
    ]
    best_valid_fem = (
        valid_fem.sort_values("relative_L2_error").iloc[0]
        if not valid_fem.empty
        else None
    )

    reference_gate = bool(
        reference_summary["high_fidelity_reference"]
    )
    sbpinn_high_accuracy = bool(
        sbpinn["relative_L2_error"] < 0.03
        and sbpinn["Hellinger_distance"] < 0.03
        and sbpinn["mass_deviation"] < 1.0e-6
        and sbpinn["negative_mass"] < 1.0e-12
        and sbpinn[
            "evaluation_pde_residual_normalized_RMSE"
        ] < 0.05
    )
    fem_no_high_accuracy = bool(
        valid_fem.empty
        or float(valid_fem["relative_L2_error"].min()) >= 0.05
    )
    target = bool(
        reference_gate
        and sbpinn_high_accuracy
        and fem_no_high_accuracy
    )
    verdict = (
        "TARGET_RESULT_OBTAINED"
        if target
        else "TARGET_RESULT_NOT_YET_OBTAINED"
    )

    comparison_rows = [
        {
            "method": "SBPINN",
            "configuration": (
                f"{sbpinn['parameter_count']} parameters"
            ),
            **{
                key: sbpinn[key]
                for key in (
                    "relative_L2_error",
                    "L1_error",
                    "Hellinger_distance",
                    "Jensen_Shannon_divergence",
                    "mass_deviation",
                    "negative_mass",
                    "minimum_density",
                    "maximum_density",
                    "total_training_time_s",
                    "peak_cuda_memory_mb",
                )
            },
        }
    ]
    for _, row in completed.iterrows():
        comparison_rows.append(
            {
                "method": "Dense spectral periodic Q1 FEM",
                "configuration": (
                    f"{int(row['cells_per_axis'])}x"
                    f"{int(row['cells_per_axis'])}"
                ),
                **{
                    key: row.get(key, np.nan)
                    for key in (
                        "relative_L2_error",
                        "L1_error",
                        "Hellinger_distance",
                        "Jensen_Shannon_divergence",
                        "mass_deviation",
                        "negative_mass",
                        "minimum_density",
                        "maximum_density",
                        "total_time_s",
                        "estimated_peak_memory_GB",
                    )
                },
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "comparison_table.csv",
        index=False,
    )

    verdict_payload = {
        "verdict": verdict,
        "criteria": {
            "high_fidelity_reference": reference_gate,
            "sbpinn_relative_L2_below_0.03": (
                sbpinn["relative_L2_error"] < 0.03
            ),
            "sbpinn_Hellinger_below_0.03": (
                sbpinn["Hellinger_distance"] < 0.03
            ),
            "sbpinn_normalized_PDE_residual_below_0.05": (
                sbpinn[
                    "evaluation_pde_residual_normalized_RMSE"
                ] < 0.05
            ),
            "sbpinn_probability_valid": (
                sbpinn["mass_deviation"] < 1.0e-6
                and sbpinn["negative_mass"] < 1.0e-12
            ),
            "no_completed_valid_FEM_below_0.05": (
                fem_no_high_accuracy
            ),
        },
        "reference_noise_floor_relative_L2": (
            reference_summary[
                "reference_noise_floor_relative_L2"
            ]
        ),
        "sbpinn": sbpinn,
        "best_valid_fem": (
            best_valid_fem.to_dict()
            if best_valid_fem is not None
            else None
        ),
    }
    write_json(output_dir / "automatic_verdict.json", verdict_payload)

    best_text = (
        "No completed probability-valid FEM configuration."
        if best_valid_fem is None
        else (
            f"Best completed probability-valid FEM: "
            f"{int(best_valid_fem['cells_per_axis'])}x"
            f"{int(best_valid_fem['cells_per_axis'])}, "
            f"relative L2="
            f"{best_valid_fem['relative_L2_error']:.6f}."
        )
    )
    report = f"""# Nonlinear torus Lévy benchmark report

## Model

The experiment studies a two-dimensional non-gradient, coupled, four-well
system on the torus driven by independent symmetric alpha-stable motions.

- alpha: {cfg.alpha}
- noise scale: {cfg.noise_scale}
- fractional coefficient sigma^alpha: {cfg.fractional_coefficient:.8f}
- no closed-form stationary density is used
- Stage-I samples and reference samples are generated from independent SDE runs

## Independent reference

- high-fidelity reference gate: {reference_gate}
- selected KDE bandwidth:
  {reference_summary['selected_bandwidth']:.6f}
- reference noise floor (relative L2):
  {reference_summary['reference_noise_floor_relative_L2']:.6f}
- refined-dt discrepancy:
  {reference_summary['refined_dt_relative_L2']:.6f}
- bandwidth sensitivity:
  {reference_summary['bandwidth_sensitivity_relative_L2']:.6f}
- normalized stationary PDE residual:
  {reference_summary['stationary_PDE_residual']['normalized_residual_RMSE']:.6f}

## SBPINN

- relative L2: {sbpinn['relative_L2_error']:.6f}
- L1: {sbpinn['L1_error']:.6f}
- Hellinger: {sbpinn['Hellinger_distance']:.6f}
- normalized evaluation PDE residual:
  {sbpinn['evaluation_pde_residual_normalized_RMSE']:.6f}
- mass deviation: {sbpinn['mass_deviation']:.3E}
- negative mass: {sbpinn['negative_mass']:.3E}
- total training time: {sbpinn['total_training_time_s']:.2f} s
- peak CUDA memory: {sbpinn['peak_cuda_memory_mb']:.2f} MB
- Stage-I selected iteration:
  {sbpinn['stage1']['best_iteration']}
- Stage-I held-out DSM loss:
  {sbpinn['stage1']['best_validation_dsm_loss']:.6E}
- Stage-II selected iteration:
  {sbpinn['stage2']['selected_iteration']}
- Stage-II selected validation NLL:
  {sbpinn['stage2']['selected_validation_nll']:.6f}

## FEM

{best_text}

The FEM statement is restricted to the implemented dense tensor-product
periodic Q1 method and the prescribed memory/time budget. It is not a claim
that every finite-element or spectral method fails.

## Automatic decision

```text
{verdict}
```

Success requires all of the following:

1. independent reference convergence passes;
2. SBPINN relative L2 < 0.03;
3. SBPINN Hellinger < 0.03;
4. normalized PDE residual < 0.05;
5. exact mass/nonnegativity constraints pass;
6. no completed probability-valid dense Q1 FEM reaches relative L2 < 0.05.
"""
    (output_dir / "EXPERIMENT_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    make_figures(output_dir, cfg, fem_frame)


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    output_dir = Path(cfg.output_dir).resolve()
    if cfg.quick:
        output_dir = Path(str(output_dir) + "_quick")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "reference":
        from generate_reference import generate_reference_data

        write_json(
            output_dir / "reference/reference_config.json",
            config_to_dict(cfg),
        )
        generate_reference_data(cfg, output_dir / "reference")
    elif args.mode == "sbpinn":
        from nonlinear_torus_sbpinn import run_sbpinn

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
            raise ValueError(
                "--fem_single_mesh is required for fem_single mode."
            )
        from nonlinear_torus_q1_fem import solve_single_mesh

        write_json(
            output_dir / "fem/fem_config.json",
            config_to_dict(cfg),
        )
        solve_single_mesh(
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
