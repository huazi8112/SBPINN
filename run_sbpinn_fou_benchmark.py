"""Fair-comparison benchmark wrapper for revised SBPINN on 1D fOU.

This script deliberately reuses the frozen production implementation in
experiments.linear_fou.train_linear_fou. It does NOT modify the SBPINN
algorithm, loss, data construction, standard-GL stencil, network architecture,
or optimization hyperparameters.

It adds only:
  * synchronized Stage-I / Stage-II wall-clock timing,
  * CUDA peak allocated-memory measurement,
  * the same standardized evaluation metrics used by run_score_fpinn_fou.py:
      - normalized W1
      - normalized relative L2
      - normalized Linf
      - raw in-domain mass error
      - tail relative L2 on |x| >= tail_start
      - normalized tail-mass error

Use this for Reviewer #1 Comment 2 paired seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance

from experiments.linear_fou import train_linear_fou as sb


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_density(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    mass = float(simpson(p, x=x))
    if not np.isfinite(mass) or mass <= 1.0e-14:
        raise RuntimeError("invalid density mass")
    return p / mass


def two_tail_integral(values: np.ndarray, x: np.ndarray, tail_start: float) -> float:
    left = x <= -tail_start
    right = x >= tail_start
    total = 0.0
    if np.count_nonzero(left) >= 2:
        total += float(simpson(values[left], x=x[left]))
    if np.count_nonzero(right) >= 2:
        total += float(simpson(values[right], x=x[right]))
    return total


def predict_raw(
    ll_net: sb.LLNetwork,
    x: np.ndarray,
    t: float,
    device: torch.device,
) -> np.ndarray:
    xt = torch.as_tensor(x[:, None], dtype=torch.float32, device=device)
    tt = torch.full_like(xt, float(t))
    ll_net.eval()
    with torch.no_grad():
        pred = sb.safe_exp(ll_net(xt, tt)).cpu().numpy().reshape(-1)
    return pred.astype(np.float64)


def metrics_for_time(
    ll_net: sb.LLNetwork,
    x: np.ndarray,
    t: float,
    alpha: float,
    cfg: sb.Config,
    device: torch.device,
    tail_start: float,
) -> Dict[str, float]:
    exact_raw = sb.exact_fou_pdf(x, float(t), alpha, cfg)
    pred_raw = predict_raw(ll_net, x, float(t), device)

    exact_mass = float(simpson(exact_raw, x=x))
    pred_mass = float(simpson(pred_raw, x=x))

    exact = normalize_density(exact_raw, x)
    pred = normalize_density(pred_raw, x)
    diff = pred - exact

    abs_l2 = float(math.sqrt(simpson(diff * diff, x=x)))
    ref_l2 = float(math.sqrt(simpson(exact * exact, x=x)))
    rel_l2 = abs_l2 / max(ref_l2, 1.0e-14)
    linf = float(np.max(np.abs(diff)))
    w1 = float(wasserstein_distance(x, x, u_weights=pred, v_weights=exact))

    tail_diff_sq = diff * diff
    tail_ref_sq = exact * exact
    tail_ref_norm = math.sqrt(max(two_tail_integral(tail_ref_sq, x, tail_start), 0.0))
    tail_rel_l2 = (
        math.sqrt(max(two_tail_integral(tail_diff_sq, x, tail_start), 0.0))
        / max(tail_ref_norm, 1.0e-14)
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


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(alpha: float, seed: int, device_name: str, output_dir: str, tail_start: float) -> None:
    cfg = sb.Config(seed=seed, device=device_name, output_dir=output_dir)
    device = sb.get_device(device_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sb.set_seed(seed)

    print(f"Device: {device}")
    print(f"SBPINN direct fOU benchmark: alpha={alpha}, seed={seed}")
    print("Frozen revised route: Stage-I ordinary score -> Stage-II homotopy log-density + standard GL")
    print(
        f"Data: trajectories={cfg.train_trajectories}, "
        f"score points={cfg.score_train_points}, Stage-II points={cfg.ll_train_points}, "
        f"batch={cfg.batch_size}"
    )
    print(
        f"Budget: score={cfg.score_epochs}, "
        f"Stage-II warmup={cfg.ll_warmup_epochs}, physics={cfg.ll_physics_epochs}"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Same data generator and seed logic as the frozen production script.
    data_start = time.perf_counter()
    process = sb.FractionalOUProcess(cfg, alpha)
    score_data, ll_data = sb.generate_stage_datasets(process, cfg)
    data_time = time.perf_counter() - data_start

    score_net = sb.ScoreNetwork(cfg).to(device)
    ll_net = sb.LLNetwork(cfg).to(device)
    n_score = count_parameters(score_net)
    n_ll = count_parameters(ll_net)
    print(f"Parameters: score={n_score}, q={n_ll}, total={n_score+n_ll}")

    sync(device)
    t0 = time.perf_counter()
    score_losses = sb.train_score_network(score_net, score_data, cfg, device)
    sync(device)
    stage1_time = time.perf_counter() - t0
    score_net.eval()

    sync(device)
    t0 = time.perf_counter()
    ll_losses = sb.train_ll_network(ll_net, score_net, ll_data, process, alpha, cfg, device)
    sync(device)
    stage2_time = time.perf_counter() - t0
    ll_net.eval()

    peak_mb = float("nan")
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0**2

    grid = np.linspace(-cfg.domain_bound, cfg.domain_bound, cfg.eval_grid_n)
    rows = [
        metrics_for_time(ll_net, grid, float(t), alpha, cfg, device, tail_start)
        for t in cfg.eval_times
    ]
    positive = [r for r in rows if r["time"] > 1.0e-12]

    summary = {
        "method": "SBPINN",
        "alpha": float(alpha),
        "seed": int(seed),
        "mean_W1_positive_times": float(np.mean([r["Wasserstein_normalized"] for r in positive])),
        "max_W1_positive_times": float(np.max([r["Wasserstein_normalized"] for r in positive])),
        "mean_rel_L2_positive_times": float(np.mean([r["rel_L2_normalized"] for r in positive])),
        "mean_abs_in_domain_mass_error_positive_times": float(
            np.mean([r["abs_in_domain_mass_error"] for r in positive])
        ),
        "mean_tail_rel_L2_positive_times": float(
            np.mean([r["tail_rel_L2_normalized"] for r in positive])
        ),
        "data_time_s": float(data_time),
        "stage1_score_time_s": float(stage1_time),
        "stage2_time_s": float(stage2_time),
        "optimization_time_s": float(stage1_time + stage2_time),
        "total_time_s": float(data_time + stage1_time + stage2_time),
        "peak_cuda_memory_mb": float(peak_mb),
        "score_params": int(n_score),
        "ll_params": int(n_ll),
        "total_params": int(n_score + n_ll),
        "fractional_operator_in_training": "standard GL in Stage-II physics refinement",
    }

    write_csv(out / "sbpinn_fou_benchmark_per_time.csv", rows)
    write_csv(out / "sbpinn_fou_benchmark_summary.csv", [summary])

    protocol = {
        "method": "SBPINN",
        "alpha": float(alpha),
        "seed": int(seed),
        "tail_start": float(tail_start),
        "config": asdict(cfg),
        "algorithm_source": "experiments.linear_fou.train_linear_fou",
        "fairness_note": (
            "Wrapper only adds timing/memory/standardized evaluation; "
            "training functions and standard-GL protocol are unchanged."
        ),
    }
    with (out / "protocol_sbpinn_fou_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    if cfg.save_model:
        tag = sb.alpha_tag(alpha)
        torch.save(
            {
                "alpha": float(alpha),
                "config": asdict(cfg),
                "score_state_dict": score_net.state_dict(),
                "ll_state_dict": ll_net.state_dict(),
                "score_losses": list(score_losses),
                "ll_losses": list(ll_losses),
                "standardized_per_time": rows,
                "benchmark_summary": summary,
            },
            out / f"linear_fou_benchmark_model_alpha_{tag}.pt",
        )

    print("\nSBPINN standardized evaluation")
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
        f"tail relL2={summary['mean_tail_rel_L2_positive_times']:.5f}, "
        f"time={summary['optimization_time_s']:.2f}s, "
        f"peakCUDA={summary['peak_cuda_memory_mb']:.1f} MB"
    )
    print(f"Results: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tail_start", type=float, default=4.0)
    p.add_argument("--output_dir", required=True)
    return p


def main() -> None:
    a = build_parser().parse_args()
    if not (1.0 < a.alpha < 2.0):
        raise ValueError("alpha must lie in (1,2)")
    run(a.alpha, a.seed, a.device, a.output_dir, a.tail_start)


if __name__ == "__main__":
    main()
