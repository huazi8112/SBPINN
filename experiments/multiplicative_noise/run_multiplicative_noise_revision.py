#!/usr/bin/env python3
"""Revision experiment for state-dependent multiplicative alpha-stable Levy noise.

Purpose
-------
This script directly addresses the reviewer concern about replacing the
state-dependent coefficient inside the nonlocal fractional operator by a
constant median.  It implements two paired Stage-II variants:

  exact   : D_Riesz^alpha[ |g_beta(x)|^alpha p(x,t) ]
  median  : median_x(|g_beta(x)|^alpha) D_Riesz^alpha[p(x,t)]

The exact variant keeps the coefficient *inside* the same GL stencil, so no
fractional Leibniz/product rule is used or required.  Both variants reuse the
the same underlying trajectories, Stage-I score checkpoint, and Stage-II initialization.
Stage I uses unbiased empirical samples, while only Stage II is deliberately
tail-enriched for nonlocal PDE collocation.

A cheap operator-only diagnostic is also included.  It measures the discrete
operator discrepancy between the exact and median forms as beta increases in
    g_beta(x) = 1 + beta sin^2(x).

The script is intentionally separate from the legacy multiplicative-noise
experiment so that the original files remain untouched during revision.

Evaluation note
---------------
For consistency with the legacy multiplicative-noise experiment, Wasserstein
distance is evaluated on MC samples lying inside the represented interval
[-L,L].  The fraction of independent MC samples outside that interval is
reported separately in multiplicative_per_time.csv.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.integrate import simpson
from scipy.stats import gaussian_kde, levy_stable, wasserstein_distance
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Reproducibility and utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_exp(q: torch.Tensor) -> torch.Tensor:
    # Avoid numerical overflow without altering the well-resolved density core.
    return torch.exp(torch.clamp(q, min=-45.0, max=20.0))


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


@dataclass
class Config:
    alpha: float = 1.5
    beta: float = 0.5
    sigma_const: float = 1.0

    t_end: float = 1.0
    dt: float = 0.001
    eval_times: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    initial_std: float = 0.5

    physical_limit: float = 8.0
    gl_dx: float = 0.005
    gl_terms: int = 1000
    gl_chunk_size: int = 65536

    train_trajectories: int = 10000
    train_snapshot_times: int = 9
    # Stage I must sample the empirical p_t itself.  Tail reweighting is reserved
    # for Stage-II PDE collocation, where it does not bias score matching.
    score_train_points: int = 60000
    ll_train_points: int = 60000
    tail_ratio: float = 0.40
    eval_trajectories: int = 30000

    # A deterministic, higher-capacity teacher is used in Stage I.  Stage II
    # keeps the lighter network to remain safe on a 4-GB GPU with 1000 GL terms.
    score_hidden_dim: int = 128
    ll_hidden_dim: int = 64
    num_hidden: int = 3
    score_dropout: float = 0.0

    batch_size: int = 256
    ic_batch_size: int = 512
    score_epochs: int = 5000
    score_lr: float = 5.0e-4
    score_ic_weight: float = 10.0
    score_symmetry_weight: float = 0.10
    q_symmetry_weight: float = 0.10

    # Stage II is intentionally aligned with the validated linear-fOU protocol.
    # The initial condition is enforced *hard* by LLNetwork, so no separate IC
    # warm-up or soft IC penalty is required.
    ic_mode: str = "hard"
    ic_warmup_epochs: int = 0
    score_warmup_epochs: int = 1000
    physics_epochs: int = 4000
    ll_warmup_lr: float = 8.0e-4
    ll_physics_lr: float = 2.0e-4
    eta_min: float = 1.0e-5
    score_consistency_weight: float = 1.0
    score_to_q_ramp_epochs: int = 1200
    pde_ramp_epochs: int = 1200

    # The symmetric alpha-stable law has algebraic tails.  Therefore artificial
    # vacuum / zero-boundary penalties are disabled in the formal protocol; they
    # conflict with the power-law exterior padding used by the GL operator.
    ic_weight: float = 0.0
    boundary_weight: float = 0.0
    decay_weight: float = 0.0
    # The stabilized PDE residual is homogeneous in p and score consistency
    # constrains only spatial derivatives of q.  A multi-time mass constraint
    # removes the otherwise admissible near-zero-density collapse.
    mass_loss_weight: float = 1.0
    mass_quad_points: int = 161
    mass_time_samples: int = 4

    eval_grid_n: int = 1201
    operator_grid_n: int = 401

    output_dir: str = "results_multiplicative_noise_revision"
    device: str = "auto"
    make_plots: bool = True
    save_checkpoints: bool = True


# -----------------------------------------------------------------------------
# Model and SDE
# -----------------------------------------------------------------------------


class ScoreNetwork(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(cfg.num_hidden):
            layers += [nn.Linear(in_dim, cfg.score_hidden_dim), nn.Tanh()]
            if cfg.score_dropout > 0:
                layers.append(nn.Dropout(cfg.score_dropout))
            in_dim = cfg.score_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


class LLNetwork(nn.Module):
    """Log-density network supporting a controlled hard-vs-soft IC diagnostic.

    hard: q(x,t)=q0(x)+t*N(x,t), exactly enforcing the Gaussian IC.
    soft: q(x,t)=N(x,t), with the original log-density IC warm-up/penalty
          enforced only on the central interval |x|<=3.

    The soft option is intended as a diagnostic for the instantaneous algebraic
    tails generated by alpha-stable noise.  All other Stage-II settings remain
    unchanged.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(cfg.num_hidden):
            layers += [nn.Linear(in_dim, cfg.ll_hidden_dim), nn.Softplus()]
            in_dim = cfg.ll_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.core = nn.Sequential(*layers)
        self.initial_std = float(cfg.initial_std)
        self.ic_mode = str(cfg.ic_mode)
        if self.ic_mode not in {"hard", "soft"}:
            raise ValueError(f"Unknown ic_mode={self.ic_mode}")
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raw = self.core(xt)
        if self.ic_mode == "soft":
            return raw
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        s = self.initial_std
        q0 = -0.5 * (x / s).pow(2) - math.log(math.sqrt(2.0 * math.pi) * s)
        return q0 + t * raw


def drift_np(x: np.ndarray) -> np.ndarray:
    return -x


def g_np(x: np.ndarray, beta: float) -> np.ndarray:
    return 1.0 + beta * np.sin(x) ** 2


def g_torch(x: torch.Tensor, beta: float) -> torch.Tensor:
    return 1.0 + beta * torch.sin(x) ** 2


def a_np(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return np.abs(g_np(x, beta)) ** alpha


def a_torch(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return torch.abs(g_torch(x, beta)) ** alpha


def spatial_median_a(cfg: Config, n: int = 20001) -> float:
    """Median of a(x)=|g(x)|^alpha over the *spatial domain*.

    This is deliberately defined on a uniform spatial grid.  It matches the
    manuscript's stated spatial-median idea and avoids the legacy ambiguity of
    evaluating g at the median sample coordinate.
    """
    x = np.linspace(-cfg.physical_limit, cfg.physical_limit, n)
    return float(np.median(a_np(x, cfg.alpha, cfg.beta)))


# -----------------------------------------------------------------------------
# Monte-Carlo trajectories and paired data
# -----------------------------------------------------------------------------


def _step_sde(x: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    scale = cfg.dt ** (1.0 / cfg.alpha)
    # scipy's levy_stable receives an explicit RandomState/Generator.
    z = levy_stable.rvs(
        cfg.alpha,
        0,
        scale=scale,
        size=len(x),
        random_state=rng,
    )
    return x + drift_np(x) * cfg.dt + g_np(x, cfg.beta) * z


def simulate_snapshots(
    cfg: Config,
    n_trajectories: int,
    times: Sequence[float],
    seed: int,
) -> Dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, cfg.initial_std, n_trajectories)
    requested = sorted({float(t) for t in times if t >= 0.0})
    snapshots: Dict[float, np.ndarray] = {}
    if any(abs(t) < 1e-12 for t in requested):
        snapshots[0.0] = x.copy()

    step_map: Dict[int, List[float]] = {}
    for t in requested:
        if t <= 0.0:
            continue
        step = int(round(t / cfg.dt))
        step_map.setdefault(step, []).append(t)

    n_steps = int(round(cfg.t_end / cfg.dt))
    for step in range(1, n_steps + 1):
        x = _step_sde(x, cfg, rng)
        if step in step_map:
            for t in step_map[step]:
                snapshots[t] = x.copy()
    return snapshots


def build_stage_datasets(
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """Construct unbiased Stage-I samples and tail-enhanced Stage-II points.

    Score matching is an expectation under the target density p_t.  Therefore
    Stage I must *not* use the legacy core/tail oversampling without importance
    weights: doing so changes the density whose score is learned.  Stage II is a
    collocation problem rather than a score-matching expectation, so deliberate
    tail enrichment is appropriate there for the nonlocal GL residual.
    """
    train_times = np.linspace(
        cfg.t_end / cfg.train_snapshot_times,
        cfg.t_end,
        cfg.train_snapshot_times,
    )
    snaps = simulate_snapshots(
        cfg,
        n_trajectories=cfg.train_trajectories,
        times=train_times,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 17)

    blocks: List[np.ndarray] = []
    for t in train_times:
        xvals = snaps[float(t)]
        xvals = xvals[np.abs(xvals) <= cfg.physical_limit]
        if len(xvals) == 0:
            continue
        tt = np.full_like(xvals, float(t), dtype=np.float64)
        blocks.append(np.column_stack([xvals, tt]))
    if not blocks:
        raise RuntimeError("No trajectory samples fall inside the physical domain.")
    physical_points = np.concatenate(blocks, axis=0)

    # Stage I: unbiased empirical p_t samples.  Antithetic reflection is valid
    # because the drift, g_beta(x), symmetric Levy noise and initial law preserve
    # p(-x,t)=p(x,t); it reduces finite-sample asymmetry without tail reweighting.
    mirrored = physical_points.copy()
    mirrored[:, 0] *= -1.0
    score_pool = np.vstack([physical_points, mirrored])
    score_idx = rng.choice(
        len(score_pool),
        cfg.score_train_points,
        replace=len(score_pool) < cfg.score_train_points,
    )
    score_data = score_pool[score_idx].astype(np.float32)

    # Stage II: fOU-style core + empirical-tail + uniform-tail collocation.
    abs_x = np.abs(physical_points[:, 0])
    threshold = float(np.percentile(abs_x, 70.0))
    core = physical_points[abs_x <= threshold]
    empirical_tail = physical_points[abs_x > threshold]

    n_tail = int(round(cfg.ll_train_points * cfg.tail_ratio))
    n_core = cfg.ll_train_points - n_tail
    core_idx = rng.choice(len(core), n_core, replace=len(core) < n_core)
    core_sample = core[core_idx]

    n_emp_tail = min(len(empirical_tail), n_tail // 2)
    if n_emp_tail > 0:
        tail_idx = rng.choice(
            len(empirical_tail), n_emp_tail,
            replace=len(empirical_tail) < n_emp_tail,
        )
        tail_sample = empirical_tail[tail_idx]
    else:
        tail_sample = np.empty((0, 2), dtype=np.float64)

    n_uniform = n_tail - len(tail_sample)
    uniform_blocks: List[np.ndarray] = []
    total = 0
    while total < n_uniform:
        n_draw = max(2 * (n_uniform - total), 2048)
        x_can = rng.uniform(-cfg.physical_limit, cfg.physical_limit, size=n_draw)
        x_can = x_can[np.abs(x_can) > threshold]
        if len(x_can) == 0:
            continue
        t_can = rng.uniform(0.0, cfg.t_end, size=len(x_can))
        block = np.column_stack([x_can, t_can])
        uniform_blocks.append(block)
        total += len(block)
    uniform_tail = (
        np.concatenate(uniform_blocks, axis=0)[:n_uniform]
        if n_uniform > 0 else np.empty((0, 2), dtype=np.float64)
    )

    ll_data = np.vstack([core_sample, tail_sample, uniform_tail]).astype(np.float32)
    rng.shuffle(score_data)
    rng.shuffle(ll_data)
    return score_data, ll_data, snaps


# -----------------------------------------------------------------------------
# GL operator: exact variable coefficient versus median approximation
# -----------------------------------------------------------------------------


def gl_weights(alpha: float, M: int, device: torch.device) -> torch.Tensor:
    w = torch.empty(M, dtype=torch.float32, device=device)
    wk = 1.0
    for k in range(M):
        w[k] = wk
        wk = wk * (1.0 - (alpha + 1.0) / (k + 1.0))
    return w.view(1, M)


def padded_density(
    ll_net: LLNetwork,
    x_in: torch.Tensor,
    t_in: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Network density inside [-L,L], power-law density padding outside."""
    L = cfg.physical_limit
    mask_in = torch.abs(x_in) <= L
    x_safe = torch.clamp(x_in, -L, L)
    p_inside = safe_exp(ll_net(torch.cat([x_safe, t_in], dim=1)))

    sign = torch.where(x_in >= 0.0, torch.ones_like(x_in), -torch.ones_like(x_in))
    x_b = sign * L
    p_b = safe_exp(ll_net(torch.cat([x_b, t_in], dim=1)))
    r = torch.clamp(torch.abs(x_in), min=L)
    p_tail = p_b * (L / r) ** (cfg.alpha + 1.0)
    return torch.where(mask_in, p_inside, p_tail)


def gl_fractional_operator_network(
    ll_net: LLNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: Config,
    weights: torch.Tensor,
    mode: str,
    median_a: float,
) -> torch.Tensor:
    """Evaluate the same GL discretization under two coefficient treatments.

    exact:
        D^alpha[ a(x) p(x,t) ]
    median:
        a_bar D^alpha[ p(x,t) ]

    The exact mode simply multiplies each stencil value of p by a at that same
    auxiliary coordinate.  No fractional product rule is invoked.
    """
    if mode not in {"exact", "median"}:
        raise ValueError(f"Unknown operator mode: {mode}")

    B = x.shape[0]
    M = cfg.gl_terms
    k_vec = torch.arange(M, dtype=x.dtype, device=x.device).view(1, M)
    x_left = x - k_vec * cfg.gl_dx
    x_right = x + k_vec * cfg.gl_dx
    t_mat = t.expand(B, M)

    # Flatten and chunk network calls to be safe on 4-GB GPUs.
    def eval_side(xmat: torch.Tensor) -> torch.Tensor:
        xf = xmat.reshape(-1, 1)
        tf = t_mat.reshape(-1, 1)
        vals: List[torch.Tensor] = []
        chunk = max(1, cfg.gl_chunk_size)
        for start in range(0, len(xf), chunk):
            xs = xf[start : start + chunk]
            ts = tf[start : start + chunk]
            p = padded_density(ll_net, xs, ts, cfg)
            if mode == "exact":
                p = a_torch(xs, cfg.alpha, cfg.beta) * p
            vals.append(p)
        out = torch.cat(vals, dim=0).view(B, M)
        if mode == "median":
            out = float(median_a) * out
        return out

    val_left = eval_side(x_left)
    val_right = eval_side(x_right)
    sum_lr = torch.sum((val_left + val_right) * weights, dim=1, keepdim=True)
    riesz_coeff = -1.0 / (2.0 * math.cos(cfg.alpha * math.pi / 2.0))
    return riesz_coeff * sum_lr / (cfg.gl_dx ** cfg.alpha)


# -----------------------------------------------------------------------------
# Operator-only diagnostic (no neural training)
# -----------------------------------------------------------------------------


def test_density_np(x: np.ndarray, cfg: Config, family: str) -> np.ndarray:
    if family == "gaussian":
        s = 0.9
        p = np.exp(-0.5 * (x / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
        return p
    if family == "heavy_tail":
        # Smooth algebraic density with the same |x|^{-(1+alpha)} tail exponent.
        p = (1.0 + x * x) ** (-(1.0 + cfg.alpha) / 2.0)
        return p
    raise ValueError(family)


def padded_test_density_np(x: np.ndarray, cfg: Config, family: str) -> np.ndarray:
    L = cfg.physical_limit
    inside = np.abs(x) <= L
    x_safe = np.clip(x, -L, L)
    p_in = test_density_np(x_safe, cfg, family)
    p_b = test_density_np(np.where(x >= 0.0, L, -L), cfg, family)
    r = np.maximum(np.abs(x), L)
    p_tail = p_b * (L / r) ** (cfg.alpha + 1.0)
    return np.where(inside, p_in, p_tail)


def gl_operator_numpy(
    x: np.ndarray,
    cfg: Config,
    family: str,
    mode: str,
    median_a: float,
) -> np.ndarray:
    M = cfg.gl_terms
    weights = np.empty(M, dtype=np.float64)
    wk = 1.0
    for k in range(M):
        weights[k] = wk
        wk = wk * (1.0 - (cfg.alpha + 1.0) / (k + 1.0))
    k = np.arange(M, dtype=np.float64)
    left = x[:, None] - k[None, :] * cfg.gl_dx
    right = x[:, None] + k[None, :] * cfg.gl_dx
    p_l = padded_test_density_np(left, cfg, family)
    p_r = padded_test_density_np(right, cfg, family)
    if mode == "exact":
        p_l = a_np(left, cfg.alpha, cfg.beta) * p_l
        p_r = a_np(right, cfg.alpha, cfg.beta) * p_r
    elif mode == "median":
        p_l = median_a * p_l
        p_r = median_a * p_r
    else:
        raise ValueError(mode)
    coeff = -1.0 / (2.0 * math.cos(cfg.alpha * math.pi / 2.0))
    return coeff * ((p_l + p_r) @ weights) / (cfg.gl_dx ** cfg.alpha)


def operator_diagnostic(cfg: Config, betas: Sequence[float], outdir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    x = np.linspace(-0.5 * cfg.physical_limit, 0.5 * cfg.physical_limit, cfg.operator_grid_n)
    for beta in betas:
        c = replace(cfg, beta=float(beta))
        med = spatial_median_a(c)
        a_grid = a_np(np.linspace(-c.physical_limit, c.physical_limit, 20001), c.alpha, c.beta)
        variation = (float(a_grid.max()) - float(a_grid.min())) / max(abs(med), 1e-12)
        for family in ("gaussian", "heavy_tail"):
            exact = gl_operator_numpy(x, c, family, "exact", med)
            approx = gl_operator_numpy(x, c, family, "median", med)
            diff = approx - exact
            rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(exact), 1e-14)
            rel_linf = np.max(np.abs(diff)) / max(np.max(np.abs(exact)), 1e-14)
            rows.append(
                {
                    "alpha": c.alpha,
                    "beta": c.beta,
                    "density": family,
                    "a_min": float(a_grid.min()),
                    "a_median": med,
                    "a_max": float(a_grid.max()),
                    "relative_coefficient_range": variation,
                    "operator_rel_l2": float(rel_l2),
                    "operator_rel_linf": float(rel_linf),
                }
            )
    write_csv(outdir / "operator_consistency.csv", rows)
    if cfg.make_plots:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for family in ("gaussian", "heavy_tail"):
                rr = [r for r in rows if r["density"] == family]
                ax.plot([r["beta"] for r in rr], [r["operator_rel_l2"] for r in rr], marker="o", label=family)
            ax.set_xlabel(r"Spatial-variation amplitude $\beta$")
            ax.set_ylabel("Relative GL-operator error of median approximation")
            ax.set_title("Variable-coefficient operator sensitivity")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / "operator_error_vs_beta.png", dpi=300)
            fig.savefig(outdir / "operator_error_vs_beta.pdf")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: operator plot failed: {exc}")
    return rows



def beta0_exact_score_diagnostic(
    score_net: ScoreNetwork,
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Evaluate Stage-I against the analytic beta=0 transient score.

    This diagnostic is evaluation-only.  For dX=-X dt + sigma dL_alpha and a
    Gaussian initial law, the characteristic function is known exactly.
    """
    if abs(cfg.beta) > 1.0e-14:
        return []
    x = np.linspace(-4.0, 4.0, 401, dtype=np.float64)
    k = np.linspace(0.0, 100.0, 10000, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for tval in cfg.eval_times:
        gaussian = 0.5 * (cfg.initial_std ** 2) * math.exp(-2.0 * tval) * k * k
        stable = (
            (cfg.sigma_const ** cfg.alpha)
            * (1.0 - math.exp(-cfg.alpha * tval))
            / cfg.alpha
            * np.abs(k) ** cfg.alpha
        )
        phi = np.exp(-(gaussian + stable))
        phase = np.outer(x, k)
        density = simpson(np.cos(phase) * phi[None, :], x=k, axis=1) / math.pi
        px = -simpson(np.sin(phase) * (k * phi)[None, :], x=k, axis=1) / math.pi
        exact = px / np.maximum(density, 1.0e-12)
        xt = torch.tensor(
            np.column_stack([x, np.full_like(x, float(tval))]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            pred = score_net(xt).cpu().numpy().reshape(-1)
        mask = density > 1.0e-7
        dx = x[1] - x[0]
        norm = max(float(np.sum(density[mask]) * dx), 1.0e-12)
        wrmse = math.sqrt(
            float(np.sum(((pred[mask] - exact[mask]) ** 2) * density[mask]) * dx / norm)
        )
        exact_norm = math.sqrt(
            float(np.sum((exact[mask] ** 2) * density[mask]) * dx / norm)
        )
        rows.append({
            "time": float(tval),
            "weighted_score_rmse": float(wrmse),
            "weighted_score_relative_rmse": float(wrmse / max(exact_norm, 1.0e-12)),
            "max_abs_score_error_core": float(np.max(np.abs(pred[mask] - exact[mask]))),
        })
    return rows

# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def train_score(
    score_net: ScoreNetwork,
    data: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> Tuple[float, List[float]]:
    """Stage-I sliced score matching with exact-IC and odd-symmetry anchors."""
    score_net.train()
    xt_all = torch.tensor(data, dtype=torch.float32, device=device)
    opt = Adam(score_net.parameters(), lr=cfg.score_lr)
    scheduler = CosineAnnealingLR(
        opt, T_max=max(cfg.score_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    sync_device(device)
    t0 = time.perf_counter()
    pbar = tqdm(range(cfg.score_epochs), desc="Stage I score", leave=False)
    for epoch in pbar:
        idx = torch.randint(0, len(xt_all), (cfg.batch_size,), device=device)
        batch = xt_all[idx].clone().detach().requires_grad_(True)
        s = score_net(batch)
        grad = autograd.grad(s.sum(), batch, create_graph=True)[0]
        div_x = grad[:, 0:1]
        loss_ssm = (0.5 * s.pow(2) + div_x).mean()

        # Exact score of N(0, initial_std^2): d_x log p0 = -x / initial_std^2.
        n_gaussian = int(round(0.8 * cfg.ic_batch_size))
        x0 = torch.cat([
            cfg.initial_std * torch.randn(n_gaussian, 1, device=device),
            (2.0 * torch.rand(cfg.ic_batch_size - n_gaussian, 1, device=device) - 1.0)
            * cfg.physical_limit,
        ])
        t0_batch = torch.zeros_like(x0)
        xt0 = torch.cat([x0, t0_batch], dim=1)
        target0 = -x0 / (cfg.initial_std ** 2)
        loss_ic = (score_net(xt0) - target0).pow(2).mean()

        # Symmetry of the entire multiplicative family implies an odd score.
        n_sym = min(128, len(batch))
        reflected = batch[:n_sym].clone()
        reflected[:, 0] *= -1.0
        loss_sym = (score_net(reflected) + s[:n_sym]).pow(2).mean()

        loss = (
            loss_ssm
            + cfg.score_ic_weight * loss_ic
            + cfg.score_symmetry_weight * loss_sym
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite score loss at step {epoch + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if epoch % max(1, cfg.score_epochs // 100) == 0 or epoch + 1 == cfg.score_epochs:
            hist.append(float(loss.detach().cpu()))
    sync_device(device)
    elapsed = time.perf_counter() - t0
    score_net.eval()
    for param in score_net.parameters():
        param.requires_grad_(False)
    return elapsed, hist


def initial_q_true(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    s = cfg.initial_std
    return -0.5 * (x / s) ** 2 - math.log(math.sqrt(2.0 * math.pi) * s)


def warmup_initial_condition(ll_net: LLNetwork, cfg: Config, device: torch.device) -> float:
    if cfg.ic_mode == "hard":
        return 0.0
    if cfg.ic_warmup_epochs <= 0:
        return 0.0
    opt = Adam(ll_net.parameters(), lr=cfg.ll_warmup_lr)
    sync_device(device)
    t0 = time.perf_counter()
    for _ in tqdm(range(cfg.ic_warmup_epochs), desc="Stage II soft-IC warm-up", leave=False):
        # Reuse the original manuscript protocol: enforce log-density IC in the
        # well-resolved core only, avoiding an enormous Gaussian log-tail penalty.
        x = torch.rand(cfg.batch_size, 1, device=device) * 6.0 - 3.0
        t = torch.zeros_like(x)
        q = ll_net(torch.cat([x, t], dim=1))
        loss = (q - initial_q_true(x, cfg)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        opt.step()
    sync_device(device)
    return time.perf_counter() - t0


def warmup_score_consistency(
    ll_net: LLNetwork,
    score_net: ScoreNetwork,
    data: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> float:
    xt_all = torch.tensor(data, dtype=torch.float32, device=device)
    opt = Adam(ll_net.parameters(), lr=cfg.ll_warmup_lr)
    sync_device(device)
    t0 = time.perf_counter()
    for _ in tqdm(range(cfg.score_warmup_epochs), desc="Stage II score warm-up", leave=False):
        idx = torch.randint(0, len(xt_all), (cfg.batch_size,), device=device)
        xt = xt_all[idx].clone().detach().requires_grad_(True)
        q = ll_net(xt)
        qx = autograd.grad(q.sum(), xt, create_graph=True)[0][:, 0:1]
        with torch.no_grad():
            s = score_net(xt)
        loss_score = (qx - s).pow(2).mean()
        reflected = xt.clone()
        reflected[:, 0] *= -1.0
        loss_sym = (ll_net(reflected) - q).pow(2).mean()
        loss_mass = mass_loss_for_network(ll_net, cfg, device)
        loss = (
            cfg.score_consistency_weight * loss_score
            + cfg.q_symmetry_weight * loss_sym
            + 0.25 * cfg.mass_loss_weight * loss_mass
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        opt.step()
    sync_device(device)
    return time.perf_counter() - t0


def mass_on_grid(ll_net: LLNetwork, cfg: Config, device: torch.device, tval: float) -> float:
    x = np.linspace(-cfg.physical_limit, cfg.physical_limit, cfg.eval_grid_n)
    xt = torch.tensor(np.column_stack([x, np.full_like(x, tval)]), dtype=torch.float32, device=device)
    with torch.no_grad():
        p = safe_exp(ll_net(xt)).cpu().numpy().reshape(-1)
    return float(simpson(p, x=x))


def trapezoid_weights(n: int, left: float, right: float, device: torch.device) -> torch.Tensor:
    h = (right - left) / max(n - 1, 1)
    w = torch.full((n,), h, dtype=torch.float32, device=device)
    if n >= 2:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w


def global_mass_torch(ll_net: LLNetwork, cfg: Config, device: torch.device, times: torch.Tensor) -> torch.Tensor:
    """Differentiable whole-space mass consistent with the adopted tail padding.

    The interior integral is completed analytically under the same
    p(x) ~ C_pm |x|^{-(1+alpha)} exterior model used by the GL padding.
    Thus each tail contributes L/alpha times the boundary density.
    """
    x = torch.linspace(
        -cfg.physical_limit, cfg.physical_limit, cfg.mass_quad_points,
        dtype=torch.float32, device=device
    )
    w = trapezoid_weights(
        cfg.mass_quad_points, -cfg.physical_limit, cfg.physical_limit, device
    )
    masses: List[torch.Tensor] = []
    for tval in times.reshape(-1):
        tt = torch.full(
            (cfg.mass_quad_points, 1), tval, dtype=torch.float32, device=device
        )
        p = safe_exp(ll_net(torch.cat([x.view(-1, 1), tt], dim=1))).reshape(-1)
        interior = torch.sum(p * w)
        tail = (cfg.physical_limit / cfg.alpha) * (p[0] + p[-1])
        masses.append(interior + tail)
    return torch.stack(masses)


def mass_loss_for_network(ll_net: LLNetwork, cfg: Config, device: torch.device) -> torch.Tensor:
    times = torch.linspace(
        cfg.t_end / cfg.mass_time_samples, cfg.t_end, cfg.mass_time_samples,
        dtype=torch.float32, device=device
    )
    masses = global_mass_torch(ll_net, cfg, device, times)
    # log-mass penalization is scale symmetric and strongly resists p -> 0.
    return torch.log(torch.clamp(masses, min=1.0e-8)).pow(2).mean()


def train_stage2(
    mode: str,
    ll_net: LLNetwork,
    score_net: ScoreNetwork,
    data: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[float]]:
    if mode not in {"exact", "median"}:
        raise ValueError(mode)
    xt_all = torch.tensor(data, dtype=torch.float32, device=device)
    med_a = spatial_median_a(cfg)
    weights = gl_weights(cfg.alpha, cfg.gl_terms, device)

    time_ic = warmup_initial_condition(ll_net, cfg, device)
    time_score = warmup_score_consistency(ll_net, score_net, data, cfg, device)

    opt = Adam(ll_net.parameters(), lr=cfg.ll_physics_lr)
    scheduler = CosineAnnealingLR(
        opt, T_max=max(cfg.physics_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    sync_device(device)
    t0 = time.perf_counter()

    pbar = tqdm(range(cfg.physics_epochs), desc=f"Stage II physics [{mode}]", leave=False)
    for epoch in pbar:
        idx = torch.randint(0, len(xt_all), (cfg.batch_size // 2,), device=device)
        batch = xt_all[idx].clone().detach().requires_grad_(True)
        x = batch[:, 0:1]
        t = batch[:, 1:2]
        q = ll_net(batch)
        p = safe_exp(q)
        grad_q = autograd.grad(q.sum(), batch, create_graph=True)[0]
        qx = grad_q[:, 0:1]
        qt = grad_q[:, 1:2]
        with torch.no_grad():
            score_target = score_net(batch)

        lam = min(1.0, float(epoch + 1) / max(cfg.score_to_q_ramp_epochs, 1))
        score_for_pde = (1.0 - lam) * score_target + lam * qx
        pde_weight = min(1.0, float(epoch + 1) / max(cfg.pde_ramp_epochs, 1))

        term_time = p * qt
        # For f(x)=-x, -d_x(fp)=p + x p_x = p(1+x*q_x).
        term_drift = p * (1.0 + x * score_for_pde)
        frac = gl_fractional_operator_network(
            ll_net, x, t, cfg, weights, mode=mode, median_a=med_a
        )
        term_diff = (cfg.sigma_const ** cfg.alpha) * frac
        residual = term_time - (term_drift + term_diff)
        loss_pde = residual.pow(2).mean()

        score_weight = cfg.score_consistency_weight * (1.0 - 0.75 * lam)
        loss_score = (qx - score_target).pow(2).mean()
        reflected = batch.clone()
        reflected[:, 0] *= -1.0
        loss_sym = (ll_net(reflected) - q).pow(2).mean()

        # Do not impose artificial vacuum or zero-boundary conditions on an
        # alpha-stable density.  Optional terms remain available only for
        # diagnostic experiments through explicit nonzero weights.
        if cfg.decay_weight > 0.0:
            n_u = cfg.batch_size // 2
            xu = (torch.rand(n_u, 1, device=device) * 2.0 - 1.0) * cfg.physical_limit
            tu = torch.rand(n_u, 1, device=device) * cfg.t_end
            pu = safe_exp(ll_net(torch.cat([xu, tu], dim=1)))
            vac = torch.abs(xu) > 3.5
            loss_decay = pu[vac].pow(2).mean() if torch.any(vac) else torch.zeros((), device=device)
        else:
            loss_decay = torch.zeros((), device=device)

        if cfg.boundary_weight > 0.0:
            nb = max(8, cfg.batch_size // 8)
            xb_abs = torch.empty(nb, 1, device=device).uniform_(
                0.94 * cfg.physical_limit, 1.03 * cfg.physical_limit
            )
            sign = torch.where(torch.rand(nb, 1, device=device) > 0.5, 1.0, -1.0)
            xb = xb_abs * sign
            tb = torch.rand(nb, 1, device=device) * cfg.t_end
            pb = safe_exp(ll_net(torch.cat([xb, tb], dim=1)))
            loss_bc = pb.mean()
        else:
            loss_bc = torch.zeros((), device=device)

        if cfg.ic_mode == "soft" and cfg.ic_weight > 0.0:
            x0 = torch.rand(cfg.batch_size, 1, device=device) * 6.0 - 3.0
            t0_batch = torch.zeros_like(x0)
            q0_pred = ll_net(torch.cat([x0, t0_batch], dim=1))
            loss_ic = (q0_pred - initial_q_true(x0, cfg)).pow(2).mean()
        else:
            loss_ic = torch.zeros((), device=device)
        loss_mass = mass_loss_for_network(ll_net, cfg, device)
        mass_weight = cfg.mass_loss_weight * min(1.0, 0.25 + 0.75 * pde_weight)

        loss = (
            pde_weight * loss_pde
            + score_weight * loss_score
            + cfg.q_symmetry_weight * loss_sym
            + cfg.ic_weight * loss_ic
            + cfg.decay_weight * loss_decay
            + cfg.boundary_weight * loss_bc
            + mass_weight * loss_mass
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage-II loss for {mode} at step {epoch + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if epoch % max(1, cfg.physics_epochs // 100) == 0 or epoch + 1 == cfg.physics_epochs:
            hist.append(float(loss.detach().cpu()))

    sync_device(device)
    physics_time = time.perf_counter() - t0
    timing = {
        "ic_warmup_s": time_ic,
        "score_warmup_s": time_score,
        "physics_s": physics_time,
        "stage2_total_s": time_ic + time_score + physics_time,
    }
    return med_a, timing, hist


def evaluate_model(
    ll_net: LLNetwork,
    refs: Dict[float, np.ndarray],
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, float]]:
    x_grid = np.linspace(-cfg.physical_limit, cfg.physical_limit, cfg.eval_grid_n)
    rows: List[Dict[str, float]] = []
    for tval in cfg.eval_times:
        xt = torch.tensor(
            np.column_stack([x_grid, np.full_like(x_grid, float(tval))]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            p_raw = safe_exp(ll_net(xt)).cpu().numpy().reshape(-1)
        mass_interior = float(simpson(p_raw, x=x_grid))
        tail_mass = (cfg.physical_limit / cfg.alpha) * float(p_raw[0] + p_raw[-1])
        mass_total = mass_interior + tail_mass
        # Shape metrics are conditional on the represented interval, so normalize
        # by the interior mass.  Mass conservation is assessed with the same
        # asymptotic tail completion used by training and GL padding.
        p_norm = p_raw / max(mass_interior, 1e-12)

        samples = refs[float(tval)]
        # Match the legacy multiplicative-noise evaluation protocol: compare
        # density shapes on the represented physical interval [-L,L].  Using
        # untruncated alpha-stable MC samples against a grid restricted to
        # [-L,L] would make W1 depend strongly on rare samples that the grid
        # does not represent.  We therefore report the conditional/truncated
        # W1 on [-L,L] and record the outside-domain MC fraction separately.
        clipped = samples[(samples >= -cfg.physical_limit) & (samples <= cfg.physical_limit)]
        inside_fraction = float(len(clipped) / max(len(samples), 1))
        outside_fraction = 1.0 - inside_fraction
        if len(clipped) > 0:
            w1 = float(
                wasserstein_distance(
                    clipped,
                    x_grid,
                    u_weights=None,
                    v_weights=np.maximum(p_norm, 0.0),
                )
            )
        else:
            w1 = float("nan")

        # KDE is used only as an independent smooth MC reference for shape errors.
        if len(clipped) >= 20:
            kde = gaussian_kde(clipped)
            p_ref = kde(x_grid)
            p_ref /= max(simpson(p_ref, x=x_grid), 1e-12)
            l2_rel = float(np.linalg.norm(p_norm - p_ref) / max(np.linalg.norm(p_ref), 1e-12))
            linf = float(np.max(np.abs(p_norm - p_ref)))
        else:
            l2_rel = float("nan")
            linf = float("nan")

        rows.append(
            {
                "time": float(tval),
                "wasserstein": w1,
                "rel_l2": l2_rel,
                "linf": linf,
                "raw_mass_interior": mass_interior,
                "raw_mass_asymptotic": mass_total,
                "mass_deviation": abs(mass_total - 1.0),
                "mc_inside_fraction": inside_fraction,
                "mc_outside_fraction": outside_fraction,
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_table(path: Path, aggregate: Sequence[Dict[str, object]]) -> None:
    lines = [
        "| beta | mode | mean W1 (|x|<=L) | max W1 (|x|<=L) | mean rel L2 | mean mass dev | n |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in aggregate:
        lines.append(
            "| {beta:.3g} | {mode} | {mean_w:.5f} ± {mean_w_sd:.5f} | "
            "{max_w:.5f} ± {max_w_sd:.5f} | {mean_l2:.5f} ± {mean_l2_sd:.5f} | "
            "{mass:.5f} ± {mass_sd:.5f} | {n} |".format(
                beta=float(r["beta"]),
                mode=str(r["mode"]),
                mean_w=float(r["mean_w_mean"]),
                mean_w_sd=float(r["mean_w_std"]),
                max_w=float(r["max_w_mean"]),
                max_w_sd=float(r["max_w_std"]),
                mean_l2=float(r["mean_l2_mean"]),
                mean_l2_sd=float(r["mean_l2_std"]),
                mass=float(r["mass_dev_mean"]),
                mass_sd=float(r["mass_dev_std"]),
                n=int(r["n"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_training_plots(outdir: Path, aggregate: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt

        modes = sorted({str(r["mode"]) for r in aggregate})
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for mode in modes:
            rr = sorted([r for r in aggregate if r["mode"] == mode], key=lambda r: float(r["beta"]))
            ax.errorbar(
                [float(r["beta"]) for r in rr],
                [float(r["mean_w_mean"]) for r in rr],
                yerr=[float(r["mean_w_std"]) for r in rr],
                marker="o",
                capsize=3,
                label=mode,
            )
        ax.set_xlabel(r"Spatial-variation amplitude $\beta$")
        ax.set_ylabel("Mean Wasserstein distance")
        ax.set_title("Multiplicative-noise sensitivity")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "wasserstein_vs_beta.png", dpi=300)
        fig.savefig(outdir / "wasserstein_vs_beta.pdf")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: summary plot failed: {exc}")


# -----------------------------------------------------------------------------
# Paired experiment driver
# -----------------------------------------------------------------------------


def run_training_experiment(
    base_cfg: Config,
    betas: Sequence[float],
    modes: Sequence[str],
    seeds: Sequence[int],
    outdir: Path,
) -> None:
    if any(m not in {"exact", "median"} for m in modes):
        raise ValueError("modes must be chosen from exact,median")

    device = torch.device(
        "cuda" if base_cfg.device == "auto" and torch.cuda.is_available() else
        "cpu" if base_cfg.device == "auto" else base_cfg.device
    )
    print(f"Device: {device}")
    print(f"alpha={base_cfg.alpha}; betas={list(betas)}; modes={list(modes)}; seeds={list(seeds)}")
    print(
        "Protocol v3.1 diagnostic: unbiased Stage-I score data + IC/symmetry anchors; "
        f"tail-enhanced Stage-II collocation; IC mode={base_cfg.ic_mode}; no artificial tail/boundary vacuum penalty; "
        f"score warm-up={base_cfg.score_warmup_epochs}; physics={base_cfg.physics_epochs}; "
        f"PDE ramp={base_cfg.pde_ramp_epochs}; score->qx ramp={base_cfg.score_to_q_ramp_epochs}; "
        f"mass weight={base_cfg.mass_loss_weight}"
    )

    per_seed_rows: List[Dict[str, object]] = []
    per_time_rows: List[Dict[str, object]] = []

    for beta in betas:
        cfg = replace(base_cfg, beta=float(beta))
        med = spatial_median_a(cfg)
        print("\n" + "=" * 78)
        print(f"beta={cfg.beta:g}: a_bar(spatial median)={med:.6f}")
        print("=" * 78)

        for seed in seeds:
            print(f"\n--- Paired seed {seed} ---")
            set_seed(seed)
            score_data, ll_data, _train_snaps = build_stage_datasets(cfg, seed=seed + 1000)
            print(
                f"Data: Stage-I unbiased points={len(score_data)}, "
                f"Stage-II collocation={len(ll_data)}, tail ratio={cfg.tail_ratio:.2f}"
            )
            refs = simulate_snapshots(
                cfg,
                n_trajectories=cfg.eval_trajectories,
                times=cfg.eval_times,
                seed=seed + 900_000,
            )

            # One shared Stage-I network for both coefficient treatments.
            set_seed(seed + 200_000)
            score_net = ScoreNetwork(cfg).to(device)
            stage1_time, _score_hist = train_score(score_net, score_data, cfg, device)
            print(f"Stage-I time: {stage1_time:.2f}s")
            if abs(cfg.beta) < 1.0e-14:
                score_diag = beta0_exact_score_diagnostic(score_net, cfg, device)
                write_csv(outdir / f"beta0_score_diagnostic_seed{seed}.csv", score_diag)
                if score_diag:
                    mean_rel = float(np.mean([r["weighted_score_relative_rmse"] for r in score_diag]))
                    print(f"Stage-I beta=0 exact-score diagnostic: mean weighted relative RMSE={mean_rel:.4f}")

            # Paired Stage-II initialization.
            set_seed(seed + 300_000)
            template = LLNetwork(cfg).to(device)
            init_state = copy.deepcopy(template.state_dict())
            del template

            for mode in modes:
                set_seed(seed + 400_000)
                ll_net = LLNetwork(cfg).to(device)
                ll_net.load_state_dict(copy.deepcopy(init_state))
                print(f"Training mode={mode} ...")
                median_used, timing, _hist = train_stage2(
                    mode, ll_net, score_net, ll_data, cfg, device
                )
                points = evaluate_model(ll_net, refs, cfg, device)
                ws = [r["wasserstein"] for r in points]
                l2s = [r["rel_l2"] for r in points]
                mdev = [r["mass_deviation"] for r in points]
                row = {
                    "alpha": cfg.alpha,
                    "beta": cfg.beta,
                    "seed": seed,
                    "mode": mode,
                    "median_a": median_used,
                    "mean_w": float(np.mean(ws)),
                    "max_w": float(np.max(ws)),
                    "mean_rel_l2": float(np.nanmean(l2s)),
                    "max_rel_l2": float(np.nanmax(l2s)),
                    "mean_mass_deviation": float(np.mean(mdev)),
                    "stage1_time_s": stage1_time,
                    **timing,
                    "end_to_end_time_s": stage1_time + timing["stage2_total_s"],
                }
                per_seed_rows.append(row)
                for p in points:
                    per_time_rows.append(
                        {
                            "alpha": cfg.alpha,
                            "beta": cfg.beta,
                            "seed": seed,
                            "mode": mode,
                            **p,
                        }
                    )
                print(
                    f"  {mode}: mean W={row['mean_w']:.5f}, max W={row['max_w']:.5f}, "
                    f"mean relL2={row['mean_rel_l2']:.5f}, mass dev={row['mean_mass_deviation']:.5f}"
                )

                if cfg.save_checkpoints:
                    ckdir = outdir / "checkpoints"
                    ckdir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "ll_state_dict": ll_net.state_dict(),
                            "score_state_dict": score_net.state_dict(),
                            "config": cfg.__dict__,
                            "mode": mode,
                            "seed": seed,
                        },
                        ckdir / f"alpha{cfg.alpha:g}_beta{cfg.beta:g}_seed{seed}_{mode}.pt",
                    )

                del ll_net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del score_net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(outdir / "multiplicative_per_seed.csv", per_seed_rows)
    write_csv(outdir / "multiplicative_per_time.csv", per_time_rows)

    aggregate: List[Dict[str, object]] = []
    for beta in betas:
        for mode in modes:
            rr = [r for r in per_seed_rows if abs(float(r["beta"]) - float(beta)) < 1e-12 and r["mode"] == mode]
            if not rr:
                continue
            mw, mw_sd = mean_std([float(r["mean_w"]) for r in rr])
            xw, xw_sd = mean_std([float(r["max_w"]) for r in rr])
            ml, ml_sd = mean_std([float(r["mean_rel_l2"]) for r in rr])
            md, md_sd = mean_std([float(r["mean_mass_deviation"]) for r in rr])
            e2e, e2e_sd = mean_std([float(r["end_to_end_time_s"]) for r in rr])
            aggregate.append(
                {
                    "alpha": base_cfg.alpha,
                    "beta": beta,
                    "mode": mode,
                    "n": len(rr),
                    "mean_w_mean": mw,
                    "mean_w_std": mw_sd,
                    "max_w_mean": xw,
                    "max_w_std": xw_sd,
                    "mean_l2_mean": ml,
                    "mean_l2_std": ml_sd,
                    "mass_dev_mean": md,
                    "mass_dev_std": md_sd,
                    "end_to_end_time_mean": e2e,
                    "end_to_end_time_std": e2e_sd,
                }
            )
    write_csv(outdir / "multiplicative_aggregate.csv", aggregate)
    make_table(outdir / "multiplicative_table.md", aggregate)
    if base_cfg.make_plots:
        make_training_plots(outdir, aggregate)

    print("\nTraining experiment finished.")
    print(f"Results: {outdir.resolve()}")
    print((outdir / "multiplicative_table.md").read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--betas", default="0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--modes", default="exact,median")
    p.add_argument("--seeds", default="42")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output_dir", default="results_multiplicative_noise_revision")
    p.add_argument("--operator_only", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_checkpoints", action="store_true")
    p.add_argument("--score_epochs", type=int, default=5000)
    p.add_argument("--ic_mode", choices=["hard", "soft"], default="hard")
    p.add_argument("--ic_warmup_epochs", type=int, default=0)
    p.add_argument("--score_warmup_epochs", type=int, default=1000)
    p.add_argument("--physics_epochs", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gl_terms", type=int, default=1000)
    p.add_argument("--gl_dx", type=float, default=0.005)
    p.add_argument("--train_trajectories", type=int, default=10000)
    p.add_argument("--train_points", type=int, default=None, help="Legacy alias: if set, use the same count for Stage-I and Stage-II data")
    p.add_argument("--score_train_points", type=int, default=60000)
    p.add_argument("--ll_train_points", type=int, default=60000)
    p.add_argument("--tail_ratio", type=float, default=0.40)
    p.add_argument("--eval_trajectories", type=int, default=30000)
    p.add_argument("--pde_ramp_epochs", type=int, default=1200)
    p.add_argument("--score_to_q_ramp_epochs", type=int, default=1200)
    p.add_argument("--score_consistency_weight", type=float, default=1.0)
    p.add_argument("--score_lr", type=float, default=5.0e-4)
    p.add_argument("--ll_warmup_lr", type=float, default=8.0e-4)
    p.add_argument("--ll_physics_lr", type=float, default=2.0e-4)
    p.add_argument("--mass_loss_weight", type=float, default=1.0)
    p.add_argument("--ic_weight", type=float, default=0.0)
    p.add_argument("--boundary_weight", type=float, default=0.0)
    p.add_argument("--decay_weight", type=float, default=0.0)
    p.add_argument("--mass_quad_points", type=int, default=161)
    p.add_argument("--mass_time_samples", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    betas = parse_float_list(args.betas)
    modes = parse_str_list(args.modes)
    seeds = parse_int_list(args.seeds)
    cfg = Config(
        alpha=args.alpha,
        device=args.device,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        save_checkpoints=not args.no_checkpoints,
        score_epochs=args.score_epochs,
        score_lr=args.score_lr,
        ic_mode=args.ic_mode,
        ic_warmup_epochs=args.ic_warmup_epochs,
        score_warmup_epochs=args.score_warmup_epochs,
        physics_epochs=args.physics_epochs,
        ll_warmup_lr=args.ll_warmup_lr,
        ll_physics_lr=args.ll_physics_lr,
        batch_size=args.batch_size,
        gl_terms=args.gl_terms,
        gl_dx=args.gl_dx,
        train_trajectories=args.train_trajectories,
        score_train_points=(args.train_points if args.train_points is not None else args.score_train_points),
        ll_train_points=(args.train_points if args.train_points is not None else args.ll_train_points),
        tail_ratio=args.tail_ratio,
        eval_trajectories=args.eval_trajectories,
        pde_ramp_epochs=args.pde_ramp_epochs,
        score_to_q_ramp_epochs=args.score_to_q_ramp_epochs,
        score_consistency_weight=args.score_consistency_weight,
        mass_loss_weight=args.mass_loss_weight,
        ic_weight=args.ic_weight,
        boundary_weight=args.boundary_weight,
        decay_weight=args.decay_weight,
        mass_quad_points=args.mass_quad_points,
        mass_time_samples=args.mass_time_samples,
    )

    if not (1.0 < cfg.alpha < 2.0):
        raise ValueError("alpha must lie in (1,2)")
    if any(b < 0 for b in betas):
        raise ValueError("beta must be non-negative")

    if args.quick:
        cfg.t_end = 0.20
        cfg.dt = 0.05
        cfg.eval_times = (0.10, 0.20)
        cfg.train_trajectories = 80
        cfg.train_snapshot_times = 2
        cfg.score_train_points = 120
        cfg.ll_train_points = 120
        cfg.eval_trajectories = 150
        cfg.score_hidden_dim = 16
        cfg.ll_hidden_dim = 16
        cfg.num_hidden = 2
        cfg.batch_size = 8
        cfg.score_epochs = 2
        cfg.ic_warmup_epochs = 1
        cfg.score_warmup_epochs = 1
        cfg.physics_epochs = 2
        cfg.score_to_q_ramp_epochs = 1
        cfg.gl_terms = 4
        cfg.gl_dx = 0.25
        cfg.gl_chunk_size = 2048
        cfg.pde_ramp_epochs = 1
        cfg.mass_quad_points = 21
        cfg.mass_time_samples = 1
        cfg.eval_grid_n = 101
        cfg.operator_grid_n = 41
        cfg.save_checkpoints = False
        betas = betas[: min(2, len(betas))]
        seeds = seeds[:1]
        cfg.output_dir = str(Path(cfg.output_dir).with_name(Path(cfg.output_dir).name + "_quick"))

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running operator consistency diagnostic...")
    rows = operator_diagnostic(cfg, betas, outdir)
    for r in rows:
        print(
            f"beta={r['beta']:.3g}, {r['density']}: coeff-range={r['relative_coefficient_range']:.3g}, "
            f"operator relL2={r['operator_rel_l2']:.4g}"
        )

    if not args.operator_only:
        run_training_experiment(cfg, betas, modes, seeds, outdir)


if __name__ == "__main__":
    main()