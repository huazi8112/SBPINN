"""
Table-oriented exact-reference evaluation for the stable one-dimensional fOU code.

Place this file in the same directory as
``train_linear_fou.py``.  It uses the identical
training pipeline and produces the manuscript error table without generating
the six-panel figures.
"""

from pathlib import Path
import csv

from train_linear_fou import (
    Config,
    get_device,
    run_experiment,
)


def main() -> None:
    cfg = Config(
        output_dir="results_linear_fou_exact_stable_table",
        save_model=False,
    )
    device = get_device(cfg.device)
    alphas = [1.5, 1.6, 1.7, 1.8]
    all_results = {}

    for alpha in alphas:
        rows = run_experiment(alpha, cfg, device, make_figure=False)
        all_results[alpha] = {
            row["time"]: row for row in rows
        }

    metric_specs = (
        ("L2 Error", "L2_error"),
        ("Linf Error", "Linf_error"),
        ("Wasserstein", "Wasserstein"),
    )
    table_rows = []
    for display_name, key in metric_specs:
        for index, time in enumerate(cfg.eval_times):
            row = {
                "Metric": display_name if index == 0 else "",
                "Time": f"t={time:.2f}",
            }
            for alpha in alphas:
                row[f"alpha={alpha}"] = (
                    f"{all_results[alpha][float(time)][key]:.4E}"
                )
            table_rows.append(row)

    output_path = Path("experimental_errors_exact_stable_table.csv")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(table_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"Saved stable exact-reference table to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
