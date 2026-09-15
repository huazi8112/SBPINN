"""Standardized checkpoint evaluator for the SBPINN side of the Score-fPINN comparison.

This script does NOT retrain SBPINN. It loads a checkpoint produced by
experiments/linear_fou/train_linear_fou.py and evaluates exactly the same
accuracy metrics used by run_score_fpinn_fou.py:

- normalized full-domain L2 and relative L2
- Linf
- Wasserstein-1
- raw in-domain mass error
- tail relative L2 on |x| >= tail_start
- normalized tail-mass error

It intentionally does not report training wall-clock time or peak CUDA memory,
because those quantities cannot be reconstructed reliably from a saved model.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance

from experiments.linear_fou import train_linear_fou as sb


def normalize_density(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    m = float(simpson(p, x=x))
    if not np.isfinite(m) or m <= 1e-14:
        raise RuntimeError("invalid density mass")
    return p / m


def raw_prediction(
    qnet: sb.LLNetwork,
    x: np.ndarray,
    t: float,
    device: torch.device,
) -> np.ndarray:
    xt = torch.as_tensor(x[:, None], dtype=torch.float32, device=device)
    tt = torch.full_like(xt, float(t))
    qnet.eval()
    with torch.no_grad():
        pred = sb.safe_exp(qnet(xt, tt)).cpu().numpy().reshape(-1)
    return pred.astype(np.float64)


def two_tail_integral(values: np.ndarray, x: np.ndarray, tail_start: float) -> float:
    left = x <= -tail_start
    right = x >= tail_start
    total = 0.0
    if np.count_nonzero(left) >= 2:
        total += float(simpson(values[left], x=x[left]))
    if np.count_nonzero(right) >= 2:
        total += float(simpson(values[right], x=x[right]))
    return total


def evaluate_time(
    qnet: sb.LLNetwork,
    x: np.ndarray,
    t: float,
    alpha: float,
    cfg: sb.Config,
    device: torch.device,
    tail_start: float,
) -> Dict[str, float]:
    exact_raw = sb.exact_fou_pdf(x, float(t), alpha, cfg)
    pred_raw = raw_prediction(qnet, x, float(t), device)

    exact_mass = float(simpson(exact_raw, x=x))
    pred_mass = float(simpson(pred_raw, x=x))

    exact = normalize_density(exact_raw, x)
    pred = normalize_density(pred_raw, x)
    diff = pred - exact

    abs_l2 = float(math.sqrt(simpson(diff * diff, x=x)))
    ref_l2 = float(math.sqrt(simpson(exact * exact, x=x)))
    rel_l2 = abs_l2 / max(ref_l2, 1e-14)
    linf = float(np.max(np.abs(diff)))
    w1 = float(wasserstein_distance(x, x, u_weights=pred, v_weights=exact))

    tail_diff_sq = diff * diff
    tail_ref_sq = exact * exact
    tail_ref_norm = math.sqrt(max(two_tail_integral(tail_ref_sq, x, tail_start), 0.0))
    tail_rel_l2 = (
        math.sqrt(max(two_tail_integral(tail_diff_sq, x, tail_start), 0.0))
        / max(tail_ref_norm, 1e-14)
    )
    tail_mass_pred = two_tail_integral(pred, x, tail_start)
    tail_mass_exact = two_tail_integral(exact, x, tail_start)

    return {
        "alpha": float(alpha),
        "seed": int(cfg.seed),
        "time": float(t),
        "L2_error_normalized": abs_l2,
        "rel_L2_normalized": rel_l2,
        "Linf_error_normalized": linf,
        "Wasserstein_normalized": w1,
        "raw_mass_in_domain": pred_mass,
        "exact_mass_in_domain": exact_mass,
        "abs_in_domain_mass_error": abs(pred_mass - exact_mass),
        "tail_start": float(tail_start),
        "tail_rel_L2_normalized": float(tail_rel_l2),
        "tail_mass_normalized_pred": float(tail_mass_pred),
        "tail_mass_normalized_exact": float(tail_mass_exact),
        "tail_mass_abs_error_normalized": abs(tail_mass_pred - tail_mass_exact),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tail_start", type=float, default=4.0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg_dict = dict(ckpt["config"])
    cfg = sb.Config(**cfg_dict)
    alpha = float(ckpt["alpha"])

    qnet = sb.LLNetwork(cfg).to(device)
    qnet.load_state_dict(ckpt["ll_state_dict"])
    qnet.eval()

    x = np.linspace(-cfg.domain_bound, cfg.domain_bound, cfg.eval_grid_n)
    rows = [
        evaluate_time(qnet, x, float(t), alpha, cfg, device, args.tail_start)
        for t in cfg.eval_times
    ]
    positive = [r for r in rows if r["time"] > 1e-12]

    summary = {
        "method": "SBPINN",
        "alpha": alpha,
        "seed": int(cfg.seed),
        "mean_W1_positive_times": float(np.mean([r["Wasserstein_normalized"] for r in positive])),
        "max_W1_positive_times": float(np.max([r["Wasserstein_normalized"] for r in positive])),
        "mean_rel_L2_positive_times": float(np.mean([r["rel_L2_normalized"] for r in positive])),
        "mean_abs_in_domain_mass_error_positive_times": float(
            np.mean([r["abs_in_domain_mass_error"] for r in positive])
        ),
        "mean_tail_rel_L2_positive_times": float(
            np.mean([r["tail_rel_L2_normalized"] for r in positive])
        ),
        "ll_params": int(sum(p.numel() for p in qnet.parameters() if p.requires_grad)),
        "note": "Accuracy-only post-processing from saved checkpoint; training time and peak CUDA memory are not recoverable.",
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "sbpinn_fou_standardized_per_time.csv", rows)
    write_csv(out / "sbpinn_fou_standardized_summary.csv", [summary])

    print("SBPINN standardized evaluation")
    for r in rows:
        print(
            f"t={r['time']:.2f}: W={r['Wasserstein_normalized']:.5f}, "
            f"relL2={r['rel_L2_normalized']:.5f}, "
            f"tail-relL2={r['tail_rel_L2_normalized']:.5f}, "
            f"mass-abs={r['abs_in_domain_mass_error']:.5f}"
        )
    print(
        f"SUMMARY: mean W={summary['mean_W1_positive_times']:.5f}, "
        f"max W={summary['max_W1_positive_times']:.5f}, "
        f"mean relL2={summary['mean_rel_L2_positive_times']:.5f}, "
        f"tail relL2={summary['mean_tail_rel_L2_positive_times']:.5f}"
    )
    print(f"Results: {out.resolve()}")


if __name__ == "__main__":
    main()
