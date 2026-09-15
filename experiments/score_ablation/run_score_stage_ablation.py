"""
Paired ablation study for the Stage-I score component of SBPINN.

Purpose
-------
This experiment directly addresses the reviewer request for a comparison against
"the same log-density GL-fPINN without Stage-I score learning" and, at the same
time, resolves the freeze-vs-replacement ambiguity by evaluating three explicit
ways of using the learned Stage-I score.

Default variants
----------------
1. homotopy
   Stage-I score learning -> score-consistency warm-up -> physics refinement
   with s_k=(1-lambda_k)s_theta + lambda_k grad(q_phi).
2. warmstart_only
   Stage-I score learning -> score-consistency warm-up -> physics refinement
   with grad(q_phi) only.  This isolates the value of score pretraining as an
   initialization/continuation device.
3. frozen_score
   Stage-I score learning -> score-consistency warm-up -> physics refinement
   with the frozen score in the drift term throughout.  This is the literal
   "frozen score" interpretation of the current Algorithm 1.
4. no_score
   No Stage I and no score-consistency loss.  The *same* LLNetwork, hard
   initial condition, collocation data, GL stencil, asymptotic padding, mass
   regularization, symmetry regularization, optimizer family, and PDE residual
   are retained.  The drift uses grad(q_phi) directly.

Fairness controls
-----------------
- Paired seeds: all variants use exactly the same Stage-II collocation set.
- Paired LL initialization: every variant starts from the same LLNetwork state.
- The exact fOU characteristic-function inversion is the common test reference.
- Final accuracy is measured at fixed evaluation times; convergence efficiency
  is additionally reported as time/GL evaluations to a configurable criterion.
- For no_score, the default budget is "same_stage2_updates": the number of full
  GL physics updates equals warm-up + physics updates of the score methods.  A
  stricter "same_gl" option is also available.

Run from the repository root, for example:
    python experiments/score_stage_ablation/run_score_stage_ablation.py \
        --device cuda --alpha 1.5 --seeds 42,123,2024,2025,2026

Quick smoke test:
    python experiments/score_stage_ablation/run_score_stage_ablation.py \
        --device cpu --quick --methods homotopy,no_score
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance
from torch.optim import Adam
from tqdm import tqdm

# Allow execution either from repo root or from this directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.linear_fou.train_linear_fou import (  # noqa: E402
    Config as BaseConfig,
    FractionalOUProcess,
    LLNetwork,
    ScoreNetwork,
    exact_fou_pdf,
    fractional_derivative_gl,
    generate_stage_datasets,
    get_device,
    global_mass_torch,
    normalize_density,
    safe_exp,
    set_seed,
    train_score_network,
)


VALID_METHODS = ("homotopy", "warmstart_only", "frozen_score", "no_score")
SCORE_METHODS = {"homotopy", "warmstart_only", "frozen_score"}


@dataclass
class AblationConfig(BaseConfig):
    alpha: float = 1.5
    output_dir: str = "results_score_stage_ablation"

    # Evaluation and convergence monitoring.
    ablation_eval_times: Tuple[float, ...] = (0.25, 0.50, 0.75)
    eval_every: int = 100
    target_w_max: float = 0.05
    min_physics_steps_before_target: int = 100

    # The no-score baseline can be given either the same number of *Stage-II
    # optimizer updates* (default; therefore more GL calls) or the same number
    # of GL physics updates as the score-guided methods.
    no_score_budget: str = "same_stage2_updates"  # or "same_gl"

    # Reproducibility / output.
    save_checkpoints: bool = True
    make_plots: bool = True


@dataclass
class EvalPoint:
    time: float
    l2: float
    linf: float
    wasserstein: float
    raw_mass: float
    mass_deviation: float


@dataclass
class VariantResult:
    method: str
    seed: int
    stage1_time_s: float
    stage2_optim_time_s: float
    total_optim_time_s: float
    warmup_steps: int
    physics_steps: int
    gl_residual_evals: int
    gl_stencil_points: int
    reached_target: bool
    target_physics_step: int
    target_total_stage2_step: int
    target_optim_time_s: float
    final_mean_w: float
    final_max_w: float
    final_mean_mass_deviation: float
    metrics: List[EvalPoint]
    history: List[Dict[str, float]]
    state_dict: Optional[Dict[str, torch.Tensor]] = None


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_device(device: torch.device) -> None:
    """Remove one-time backend/context startup from the timed comparison."""
    with torch.no_grad():
        for _ in range(3):
            a = torch.randn(64, 64, device=device)
            _ = a @ a
    sync_device(device)


def parse_int_list(text: str) -> List[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values


def parse_method_list(text: str) -> List[str]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    unknown = [v for v in values if v not in VALID_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods: {unknown}. Valid choices: {VALID_METHODS}"
        )
    if not values:
        raise argparse.ArgumentTypeError("At least one ablation method is required.")
    return values


def raw_density_on_grid(
    ll_net: LLNetwork,
    x_grid: np.ndarray,
    t_value: float,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)
    t = torch.full_like(x, float(t_value))
    ll_net.eval()
    with torch.no_grad():
        p = safe_exp(ll_net(x, t)).detach().cpu().numpy().reshape(-1)
    return np.maximum(p.astype(np.float64), 0.0)


def asymptotic_mass_from_grid(
    p_raw: np.ndarray,
    x_grid: np.ndarray,
    alpha: float,
    bound: float,
) -> float:
    interior = float(simpson(p_raw, x=x_grid))
    tail = float(bound / alpha * (p_raw[0] + p_raw[-1]))
    return interior + tail


def evaluate_variant(
    ll_net: LLNetwork,
    x_grid: np.ndarray,
    exact_refs: Dict[float, np.ndarray],
    cfg: AblationConfig,
    device: torch.device,
) -> List[EvalPoint]:
    rows: List[EvalPoint] = []
    for t_value in cfg.ablation_eval_times:
        raw = raw_density_on_grid(ll_net, x_grid, t_value, device)
        pred = normalize_density(raw, x_grid)
        exact = exact_refs[float(t_value)]
        diff = pred - exact
        l2 = float(math.sqrt(simpson(diff ** 2, x=x_grid)))
        linf = float(np.max(np.abs(diff)))
        w = float(
            wasserstein_distance(
                x_grid, x_grid, u_weights=pred, v_weights=exact
            )
        )
        raw_mass = asymptotic_mass_from_grid(
            raw, x_grid, cfg.alpha, cfg.domain_bound
        )
        rows.append(
            EvalPoint(
                time=float(t_value),
                l2=l2,
                linf=linf,
                wasserstein=w,
                raw_mass=raw_mass,
                mass_deviation=abs(raw_mass - 1.0),
            )
        )
    return rows


def aggregate_eval(points: Sequence[EvalPoint]) -> Tuple[float, float, float]:
    ws = np.asarray([p.wasserstein for p in points], dtype=np.float64)
    ms = np.asarray([p.mass_deviation for p in points], dtype=np.float64)
    return float(ws.mean()), float(ws.max()), float(ms.mean())


def build_exact_references(
    x_grid: np.ndarray,
    cfg: AblationConfig,
) -> Dict[float, np.ndarray]:
    refs: Dict[float, np.ndarray] = {}
    for t_value in cfg.ablation_eval_times:
        exact = exact_fou_pdf(
            x_grid, float(t_value), cfg.alpha, cfg
        )
        refs[float(t_value)] = normalize_density(exact, x_grid)
    return refs


def mass_loss_for_network(
    ll_net: LLNetwork,
    cfg: AblationConfig,
    device: torch.device,
) -> torch.Tensor:
    def density_func(x_value: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
        return safe_exp(ll_net(x_value, t_value))

    times = torch.linspace(
        cfg.t_final / cfg.mass_time_samples,
        cfg.t_final,
        cfg.mass_time_samples,
        dtype=torch.float32,
        device=device,
    )
    masses = global_mass_torch(
        density_func, times, cfg.alpha, cfg, device
    )
    return torch.log(torch.clamp(masses, min=1.0e-8)).pow(2).mean()


def symmetry_loss(
    ll_net: LLNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    n_sym = min(128, len(x))
    return (ll_net(-x[:n_sym], t[:n_sym]) - q[:n_sym]).pow(2).mean()


def train_variant(
    method: str,
    ll_net: LLNetwork,
    score_net: Optional[ScoreNetwork],
    ll_data: np.ndarray,
    process: FractionalOUProcess,
    cfg: AblationConfig,
    device: torch.device,
    x_grid: np.ndarray,
    exact_refs: Dict[float, np.ndarray],
    stage1_time_s: float,
    rng_seed: int,
) -> VariantResult:
    if method not in VALID_METHODS:
        raise ValueError(method)
    if method in SCORE_METHODS and score_net is None:
        raise ValueError(f"{method} requires a trained Stage-I score network.")

    # Paired random stream for Stage-II optimization across variants.
    set_seed(rng_seed)

    x_train = torch.tensor(ll_data[:, 0:1], dtype=torch.float32, device=device)
    t_train = torch.tensor(ll_data[:, 1:2], dtype=torch.float32, device=device)
    if score_net is not None:
        score_net.eval()

    def density_func(x_value: torch.Tensor, t_value: torch.Tensor) -> torch.Tensor:
        return safe_exp(ll_net(x_value, t_value))

    history: List[Dict[str, float]] = []
    warmup_steps = cfg.ll_warmup_epochs if method in SCORE_METHODS else 0
    if method == "no_score":
        if cfg.no_score_budget == "same_stage2_updates":
            # Preserve the score method's two learning-rate segments while
            # replacing the cheap score warm-up by full PDE updates. This
            # makes the direct log-density baseline differ primarily by the
            # absence of Stage I / score consistency, not by its LR schedule.
            physics_segments = [
                (cfg.ll_warmup_epochs, cfg.ll_warmup_lr, "direct-PDE phase A"),
                (cfg.ll_physics_epochs, cfg.ll_physics_lr, "direct-PDE phase B"),
            ]
        elif cfg.no_score_budget == "same_gl":
            physics_segments = [
                (cfg.ll_physics_epochs, cfg.ll_physics_lr, "equal-GL direct PDE")
            ]
        else:
            raise ValueError(
                "no_score_budget must be 'same_stage2_updates' or 'same_gl'."
            )
    else:
        physics_segments = [
            (cfg.ll_physics_epochs, cfg.ll_physics_lr, "GL physics")
        ]
    physics_budget = int(sum(segment[0] for segment in physics_segments))

    stage2_optim_time = 0.0
    gl_calls = 0
    reached_target = False
    target_physics_step = -1
    target_total_stage2_step = -1
    target_optim_time = -1.0

    # ---------------------------------------------------------------
    # Score-guided cheap warm-up.  no_score deliberately skips this.
    # ---------------------------------------------------------------
    if warmup_steps > 0:
        optimizer = Adam(ll_net.parameters(), lr=cfg.ll_warmup_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(warmup_steps, 1), eta_min=cfg.eta_min
        )
        pbar = tqdm(
            range(warmup_steps),
            desc=f"{method}: score warm-up",
            leave=False,
        )
        for epoch in pbar:
            sync_device(device)
            step_start = time.perf_counter()

            idx = torch.randint(0, len(ll_data), (cfg.batch_size,), device=device)
            x = x_train[idx].clone().detach().requires_grad_(True)
            t = t_train[idx]
            q = ll_net(x, t)
            q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
            with torch.no_grad():
                score_target = score_net(x, t)  # type: ignore[misc]

            loss_score = (q_x - score_target).pow(2).mean()
            loss_mass = mass_loss_for_network(ll_net, cfg, device)
            loss_sym = symmetry_loss(ll_net, x, t, q)
            loss = (
                cfg.score_consistency_weight * loss_score
                + 0.25 * cfg.mass_loss_weight * loss_mass
                + cfg.q_symmetry_weight * loss_sym
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite warm-up loss for {method} at step {epoch + 1}."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sync_device(device)
            stage2_optim_time += time.perf_counter() - step_start

        # A diagnostic at the warm-up/PDE interface. It is *not* eligible for
        # the target criterion because no full nonlocal physics step occurred.
        pts = evaluate_variant(ll_net, x_grid, exact_refs, cfg, device)
        mean_w, max_w, mean_mass = aggregate_eval(pts)
        history.append(
            {
                "phase": 0.0,
                "stage2_step": float(warmup_steps),
                "physics_step": 0.0,
                "gl_calls": 0.0,
                "optim_time_s": float(stage2_optim_time),
                "mean_w": mean_w,
                "max_w": max_w,
                "mean_mass_deviation": mean_mass,
                "pde_weight": 0.0,
                "score_to_q_fraction": 0.0,
            }
        )

    # ---------------------------------------------------------------
    # Full nonlocal physics refinement.
    # ---------------------------------------------------------------
    global_physics_step = 0
    for segment_steps, segment_lr, segment_name in physics_segments:
        if segment_steps <= 0:
            continue
        optimizer = Adam(ll_net.parameters(), lr=segment_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(segment_steps, 1), eta_min=cfg.eta_min
        )

        pbar = tqdm(
            range(segment_steps),
            desc=f"{method}: {segment_name}",
            leave=False,
        )
        for _segment_epoch in pbar:
            global_physics_step += 1
            epoch = global_physics_step - 1
            sync_device(device)
            step_start = time.perf_counter()

            idx = torch.randint(0, len(ll_data), (cfg.batch_size,), device=device)
            x = x_train[idx].clone().detach().requires_grad_(True)
            t = t_train[idx].clone().detach().requires_grad_(True)
            q = ll_net(x, t)
            p = safe_exp(q)
            q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
            q_t = torch.autograd.grad(q.sum(), t, create_graph=True)[0]

            if score_net is not None:
                with torch.no_grad():
                    score_target = score_net(x, t)
            else:
                score_target = None

            pde_weight = min(
                1.0, global_physics_step / max(cfg.pde_ramp_epochs, 1)
            )

            if method == "homotopy":
                q_fraction = min(
                    1.0,
                    global_physics_step / max(cfg.score_to_q_ramp_epochs, 1),
                )
                score_for_pde = (
                    (1.0 - q_fraction) * score_target + q_fraction * q_x  # type: ignore[operator]
                )
                loss_score = (q_x - score_target).pow(2).mean()  # type: ignore[union-attr]
                score_weight = (
                    cfg.score_consistency_weight * (1.0 - 0.75 * q_fraction)
                )
            elif method == "warmstart_only":
                q_fraction = 1.0
                score_for_pde = q_x
                loss_score = torch.zeros((), dtype=q.dtype, device=device)
                score_weight = 0.0
            elif method == "frozen_score":
                q_fraction = 0.0
                score_for_pde = score_target  # type: ignore[assignment]
                loss_score = (q_x - score_target).pow(2).mean()  # type: ignore[union-attr]
                score_weight = cfg.score_consistency_weight
            elif method == "no_score":
                q_fraction = 1.0
                score_for_pde = q_x
                loss_score = torch.zeros((), dtype=q.dtype, device=device)
                score_weight = 0.0
            else:  # pragma: no cover
                raise AssertionError(method)

            # Exact log-density FFP residual for the additive fOU benchmark:
            # p[q_t + f_x + f * score_for_pde] - sigma^alpha D_Riesz^alpha p = 0.
            drift = -process.theta * x
            div_drift = -process.theta
            local = p * (q_t + div_drift + drift * score_for_pde)
            diffusion = (
                process.sigma ** cfg.alpha
                * fractional_derivative_gl(
                    density_func, x, t, cfg.alpha, cfg
                )
            )
            residual = local - diffusion
            loss_residual = residual.pow(2).mean()
            gl_calls += 1

            loss_mass = mass_loss_for_network(ll_net, cfg, device)
            loss_sym = symmetry_loss(ll_net, x, t, q)
            mass_weight = cfg.mass_loss_weight * min(
                1.0, 0.25 + 0.75 * pde_weight
            )
            loss = (
                pde_weight * loss_residual
                + score_weight * loss_score
                + mass_weight * loss_mass
                + cfg.q_symmetry_weight * loss_sym
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite physics loss for {method} at step {global_physics_step}."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            sync_device(device)
            stage2_optim_time += time.perf_counter() - step_start

            should_eval = (
                global_physics_step % cfg.eval_every == 0
                or global_physics_step == physics_budget
            )
            if should_eval:
                pts = evaluate_variant(ll_net, x_grid, exact_refs, cfg, device)
                mean_w, max_w, mean_mass = aggregate_eval(pts)
                history.append(
                    {
                        "phase": 1.0,
                        "stage2_step": float(warmup_steps + global_physics_step),
                        "physics_step": float(global_physics_step),
                        "gl_calls": float(gl_calls),
                        "optim_time_s": float(stage2_optim_time),
                        "mean_w": mean_w,
                        "max_w": max_w,
                        "mean_mass_deviation": mean_mass,
                        "pde_weight": float(pde_weight),
                        "score_to_q_fraction": float(q_fraction),
                    }
                )
                pbar.set_postfix(
                    meanW=f"{mean_w:.3e}",
                    maxW=f"{max_w:.3e}",
                    lam=f"{q_fraction:.2f}",
                )

                if (
                    not reached_target
                    and global_physics_step >= cfg.min_physics_steps_before_target
                    and max_w <= cfg.target_w_max
                ):
                    reached_target = True
                    target_physics_step = global_physics_step
                    target_total_stage2_step = warmup_steps + global_physics_step
                    target_optim_time = stage1_time_s + stage2_optim_time

    final_metrics = evaluate_variant(ll_net, x_grid, exact_refs, cfg, device)
    final_mean_w, final_max_w, final_mean_mass = aggregate_eval(final_metrics)
    gl_stencil_points = int(gl_calls * cfg.batch_size * 2 * cfg.gl_terms)

    return VariantResult(
        method=method,
        seed=-1,  # filled by the caller
        stage1_time_s=float(stage1_time_s),
        stage2_optim_time_s=float(stage2_optim_time),
        total_optim_time_s=float(stage1_time_s + stage2_optim_time),
        warmup_steps=int(warmup_steps),
        physics_steps=int(physics_budget),
        gl_residual_evals=int(gl_calls),
        gl_stencil_points=gl_stencil_points,
        reached_target=bool(reached_target),
        target_physics_step=int(target_physics_step),
        target_total_stage2_step=int(target_total_stage2_step),
        target_optim_time_s=float(target_optim_time),
        final_mean_w=final_mean_w,
        final_max_w=final_max_w,
        final_mean_mass_deviation=final_mean_mass,
        metrics=final_metrics,
        history=history,
        state_dict={
            k: v.detach().cpu().clone() for k, v in ll_net.state_dict().items()
        },
    )


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    x = np.asarray(list(values), dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "Method",
        "Final mean W",
        "Final max W",
        "Mean mass dev.",
        "Total optim. time (s)",
        "GL residual evals",
        "Target success",
        "Time to target (s)",
        "GL evals to target",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        def pm(mean_key: str, std_key: str, fmt: str = ".4g") -> str:
            m = float(row[mean_key])
            s = float(row[std_key])
            return f"{m:{fmt}} ± {s:{fmt}}"

        time_target = (
            pm("target_time_mean", "target_time_std", ".3f")
            if int(row["target_success_n"]) > 0
            else "not reached"
        )
        gl_target = (
            pm("target_gl_mean", "target_gl_std", ".1f")
            if int(row["target_success_n"]) > 0
            else "not reached"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    pm("final_mean_w_mean", "final_mean_w_std"),
                    pm("final_max_w_mean", "final_max_w_std"),
                    pm("mass_dev_mean", "mass_dev_std"),
                    pm("total_time_mean", "total_time_std", ".3f"),
                    pm("gl_calls_mean", "gl_calls_std", ".1f"),
                    f"{int(row['target_success_n'])}/{int(row['n'])}",
                    time_target,
                    gl_target,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_summary_plots(
    output_dir: Path,
    all_results: Sequence[VariantResult],
    target_w_max: float,
) -> None:
    methods = list(dict.fromkeys(r.method for r in all_results))

    # Final max-W paired distribution.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    data = [
        [r.final_max_w for r in all_results if r.method == method]
        for method in methods
    ]
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    ax.set_ylabel("Final max Wasserstein distance")
    ax.set_title("Stage-I score ablation: final distributional error")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_final_maxW.png", dpi=300)
    fig.savefig(output_dir / "ablation_final_maxW.pdf")
    plt.close(fig)

    # Mean convergence path versus GL calls (seed-averaged by common monitor index).
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method in methods:
        histories = [r.history for r in all_results if r.method == method]
        physics_histories = [
            [h for h in hist if h["phase"] > 0.5] for hist in histories
        ]
        min_len = min((len(h) for h in physics_histories), default=0)
        if min_len == 0:
            continue
        x = np.mean(
            [[h[i]["gl_calls"] for i in range(min_len)] for h in physics_histories],
            axis=0,
        )
        y = np.mean(
            [[h[i]["max_w"] for i in range(min_len)] for h in physics_histories],
            axis=0,
        )
        ax.plot(x, y, marker="o", markevery=max(1, len(x) // 8), label=method)
    ax.axhline(
        target_w_max, linestyle="--", linewidth=1.2, label="target max W"
    )
    ax.set_xlabel("Full GL residual evaluations")
    ax.set_ylabel("Max Wasserstein across evaluation times")
    ax.set_title("Accuracy versus expensive nonlocal physics evaluations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_convergence_vs_GL.png", dpi=300)
    fig.savefig(output_dir / "ablation_convergence_vs_GL.pdf")
    plt.close(fig)


def run_ablation(
    cfg: AblationConfig,
    seeds: Sequence[int],
    methods: Sequence[str],
) -> None:
    if not (1.0 < cfg.alpha < 2.0):
        raise ValueError("alpha must be in (1, 2).")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.device)
    warmup_device(device)
    print(f"Device: {device}")
    print(f"Alpha: {cfg.alpha}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Seeds: {list(seeds)}")

    x_grid = np.linspace(-cfg.domain_bound, cfg.domain_bound, cfg.eval_grid_n)
    print("Building exact characteristic-function references...")
    exact_refs = build_exact_references(x_grid, cfg)

    all_results: List[VariantResult] = []
    per_time_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"\n========== Paired seed {seed} ({seed_index}/{len(seeds)}) ==========")

        # Generate one paired dataset for every method.
        set_seed(seed)
        process = FractionalOUProcess(cfg, cfg.alpha)
        score_data, ll_data = generate_stage_datasets(process, cfg)

        # Create one paired Stage-II initialization independent of whether Stage I
        # is present. This avoids RNG-consumption confounding.
        set_seed(seed + 100_000)
        ll_template = LLNetwork(cfg).to(device)
        ll_initial_state = copy.deepcopy(ll_template.state_dict())
        del ll_template

        score_net: Optional[ScoreNetwork] = None
        stage1_time = 0.0
        score_checkpoint: Optional[Dict[str, torch.Tensor]] = None
        if any(method in SCORE_METHODS for method in methods):
            set_seed(seed + 200_000)
            score_net = ScoreNetwork(cfg).to(device)
            print("Training paired Stage-I score network once; reused by score variants...")
            sync_device(device)
            t0 = time.perf_counter()
            _ = train_score_network(score_net, score_data, cfg, device)
            sync_device(device)
            stage1_time = time.perf_counter() - t0
            score_net.eval()
            score_checkpoint = {
                k: v.detach().cpu().clone() for k, v in score_net.state_dict().items()
            }
            print(f"Stage-I time: {stage1_time:.3f} s")

        for method_index, method in enumerate(methods, start=1):
            print(f"\n[{method_index}/{len(methods)}] Variant: {method}")
            ll_net = LLNetwork(cfg).to(device)
            ll_net.load_state_dict(ll_initial_state)

            result = train_variant(
                method=method,
                ll_net=ll_net,
                score_net=score_net if method in SCORE_METHODS else None,
                ll_data=ll_data,
                process=process,
                cfg=cfg,
                device=device,
                x_grid=x_grid,
                exact_refs=exact_refs,
                stage1_time_s=stage1_time if method in SCORE_METHODS else 0.0,
                rng_seed=seed + 300_000,
            )
            result.seed = seed
            all_results.append(result)

            print(
                f"{method}: final mean W={result.final_mean_w:.5f}, "
                f"max W={result.final_max_w:.5f}, "
                f"GL calls={result.gl_residual_evals}, "
                f"optim time={result.total_optim_time_s:.2f}s"
            )
            if result.reached_target:
                print(
                    f"  target reached at physics step {result.target_physics_step}, "
                    f"time={result.target_optim_time_s:.2f}s"
                )
            else:
                print("  target not reached under the configured budget.")

            for point in result.metrics:
                per_time_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "time": point.time,
                        "L2": point.l2,
                        "Linf": point.linf,
                        "Wasserstein": point.wasserstein,
                        "raw_mass": point.raw_mass,
                        "mass_deviation": point.mass_deviation,
                    }
                )
            for h in result.history:
                history_rows.append({"seed": seed, "method": method, **h})

            if cfg.save_checkpoints:
                payload = {
                    "seed": seed,
                    "method": method,
                    "config": asdict(cfg),
                    "ll_initial_state_dict": ll_initial_state,
                    "score_state_dict": score_checkpoint,
                    "ll_final_state_dict": result.state_dict,
                    "summary": {
                        "final_mean_w": result.final_mean_w,
                        "final_max_w": result.final_max_w,
                        "gl_residual_evals": result.gl_residual_evals,
                        "reached_target": result.reached_target,
                        "target_physics_step": result.target_physics_step,
                        "target_optim_time_s": result.target_optim_time_s,
                    },
                }
                torch.save(
                    payload,
                    output_dir / f"checkpoint_seed{seed}_{method}.pt",
                )

    # Per-seed summary.
    per_seed_rows: List[Dict[str, object]] = []
    for r in all_results:
        target_gl = r.target_physics_step if r.reached_target else -1
        per_seed_rows.append(
            {
                "seed": r.seed,
                "method": r.method,
                "final_mean_w": r.final_mean_w,
                "final_max_w": r.final_max_w,
                "final_mean_mass_deviation": r.final_mean_mass_deviation,
                "stage1_time_s": r.stage1_time_s,
                "stage2_optim_time_s": r.stage2_optim_time_s,
                "total_optim_time_s": r.total_optim_time_s,
                "warmup_steps": r.warmup_steps,
                "physics_steps": r.physics_steps,
                "gl_residual_evals": r.gl_residual_evals,
                "gl_stencil_points": r.gl_stencil_points,
                "reached_target": int(r.reached_target),
                "target_physics_step": r.target_physics_step,
                "target_total_stage2_step": r.target_total_stage2_step,
                "target_optim_time_s": r.target_optim_time_s,
                "target_gl_residual_evals": target_gl,
            }
        )

    # Aggregate across paired seeds.
    aggregate_rows: List[Dict[str, object]] = []
    for method in methods:
        subset = [r for r in all_results if r.method == method]
        fwm, fws = mean_std(r.final_mean_w for r in subset)
        fxm, fxs = mean_std(r.final_max_w for r in subset)
        mdm, mds = mean_std(r.final_mean_mass_deviation for r in subset)
        ttm, tts = mean_std(r.total_optim_time_s for r in subset)
        glm, gls = mean_std(float(r.gl_residual_evals) for r in subset)
        reached = [r for r in subset if r.reached_target]
        target_tm, target_ts = mean_std(r.target_optim_time_s for r in reached)
        target_gm, target_gs = mean_std(float(r.target_physics_step) for r in reached)
        aggregate_rows.append(
            {
                "method": method,
                "n": len(subset),
                "final_mean_w_mean": fwm,
                "final_mean_w_std": fws,
                "final_max_w_mean": fxm,
                "final_max_w_std": fxs,
                "mass_dev_mean": mdm,
                "mass_dev_std": mds,
                "total_time_mean": ttm,
                "total_time_std": tts,
                "gl_calls_mean": glm,
                "gl_calls_std": gls,
                "target_success_n": len(reached),
                "target_time_mean": target_tm,
                "target_time_std": target_ts,
                "target_gl_mean": target_gm,
                "target_gl_std": target_gs,
            }
        )

    write_csv(output_dir / "ablation_per_seed.csv", per_seed_rows)
    write_csv(output_dir / "ablation_per_time.csv", per_time_rows)
    write_csv(output_dir / "ablation_history.csv", history_rows)
    write_csv(output_dir / "ablation_aggregate.csv", aggregate_rows)
    save_markdown_summary(output_dir / "ablation_table.md", aggregate_rows)
    (output_dir / "ablation_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if cfg.make_plots:
        make_summary_plots(output_dir, all_results, cfg.target_w_max)

    print("\nAblation finished.")
    print(f"Results: {output_dir.resolve()}")
    print((output_dir / "ablation_table.md").read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="42,123,2024,2025,2026")
    parser.add_argument(
        "--methods",
        default="homotopy,warmstart_only,frozen_score,no_score",
    )
    parser.add_argument("--output_dir", default="results_score_stage_ablation")
    parser.add_argument("--score_epochs", type=int, default=5000)
    parser.add_argument("--ll_warmup_epochs", type=int, default=1000)
    parser.add_argument("--ll_physics_epochs", type=int, default=4000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--target_w_max", type=float, default=0.05)
    parser.add_argument(
        "--no_score_budget",
        choices=["same_stage2_updates", "same_gl"],
        default="same_stage2_updates",
    )
    parser.add_argument("--train_trajectories", type=int, default=800)
    parser.add_argument("--score_train_points", type=int, default=60000)
    parser.add_argument("--ll_train_points", type=int, default=60000)
    parser.add_argument("--tail_ratio", type=float, default=0.40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no_checkpoints", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)

    cfg = AblationConfig(
        alpha=args.alpha,
        device=args.device,
        output_dir=args.output_dir,
        score_epochs=args.score_epochs,
        ll_warmup_epochs=args.ll_warmup_epochs,
        ll_physics_epochs=args.ll_physics_epochs,
        eval_every=args.eval_every,
        target_w_max=args.target_w_max,
        no_score_budget=args.no_score_budget,
        train_trajectories=args.train_trajectories,
        score_train_points=args.score_train_points,
        ll_train_points=args.ll_train_points,
        tail_ratio=args.tail_ratio,
        save_checkpoints=not args.no_checkpoints,
        make_plots=not args.no_plots,
    )

    if args.quick:
        # Syntax/runtime smoke test only; numerical values are not meaningful.
        cfg.train_trajectories = 40
        cfg.score_train_points = 600
        cfg.ll_train_points = 600
        cfg.hidden_dim = 16
        cfg.num_layers = 3
        cfg.batch_size = 8
        cfg.ic_batch_size = 16
        cfg.score_epochs = 2
        cfg.ll_warmup_epochs = 1
        cfg.ll_physics_epochs = 2
        cfg.pde_ramp_epochs = 1
        cfg.score_to_q_ramp_epochs = 1
        cfg.gl_terms = 4
        cfg.mass_quad_points = 11
        cfg.mass_time_samples = 1
        cfg.eval_grid_n = 101
        cfg.exact_k_num = 400
        cfg.eval_every = 1
        cfg.min_physics_steps_before_target = 1
        cfg.save_checkpoints = False
        cfg.output_dir = str(Path(cfg.output_dir).with_name(Path(cfg.output_dir).name + "_quick"))
        seeds = seeds[:1]

    run_ablation(cfg, seeds, methods)


if __name__ == "__main__":
    main()
