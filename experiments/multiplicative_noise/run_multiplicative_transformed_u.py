#!/usr/bin/env python3
"""Zhang-et-al.-inspired transformed-variable pilot for multiplicative alpha-stable noise.

This script is a *separate pilot* built on the revision v3.1 protocol in
``run_multiplicative_noise_revision.py``.  It does not replace or modify the
existing exact/median implementation.

Reference transformation
------------------------
For the multiplicative symmetric alpha-stable FPE, Zhang et al. introduce

    u(x,t) = a(x) p(x,t),       a(x) = |sigma(x)|^alpha,

and, when the Brownian term is absent,

    u_t = -a(x) d_x[ f(x) u/a(x) ] + a(x) D_Riesz^alpha[u].

For the present experiment

    f(x) = -x,
    sigma(x) = 1 + beta sin^2(x),
    a(x) = |1 + beta sin^2(x)|^alpha,

so the transformed equation is

    u_t = u * {1 + x [s_u - d_x log a]} + a(x) D_Riesz^alpha[u],

where s_u = d_x log u.  Stage I still learns the score of the *probability
density* p, denoted s_p.  Because u=a p,

    s_u = s_p + d_x log a.

Therefore the existing Stage-I teacher can be reused exactly, with the known
analytic correction d_x log a added before Stage-II score consistency and
homotopy.

Important implementation choices
--------------------------------
* Standard GL stencil x +/- k h is retained.
* No spatial-median replacement is used in the transformed-u PDE.
* Stage-I data/teacher are exactly the v3.1 unbiased protocol.
* Stage-II uses the same soft-IC, tail-enriched collocation, homotopy,
  optimizer schedule and mass constraint as the v3.1 pilot.
* The NN state is r=log u; reported quantities are always converted back to
  p=u/a before Wasserstein/L2/mass evaluation.
* Exterior padding is imposed on p with the same algebraic tail model used by
  the current manuscript code, then converted to u=a p at each auxiliary GL
  coordinate.  This preserves the variable coefficient outside the represented
  interval instead of pretending that u itself has a constant tail amplitude.

This is an SBPINN-specific adaptation of Zhang et al.'s transformed variable;
the cited paper itself develops finite-difference schemes rather than PINNs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.integrate import simpson
from scipy.stats import gaussian_kde, wasserstein_distance
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Allow direct execution from experiments/multiplicative_noise_revision.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_multiplicative_noise_revision as base  # noqa: E402


# -----------------------------------------------------------------------------
# Transformed state u=a p and its score correction
# -----------------------------------------------------------------------------


def total_a_torch(x: torch.Tensor, cfg: base.Config) -> torch.Tensor:
    """Full Levy coefficient a(x)=|sigma_const*g_beta(x)|^alpha."""
    return (float(cfg.sigma_const) ** cfg.alpha) * base.a_torch(
        x, cfg.alpha, cfg.beta
    )


def total_a_np(x: np.ndarray, cfg: base.Config) -> np.ndarray:
    return (float(cfg.sigma_const) ** cfg.alpha) * base.a_np(
        x, cfg.alpha, cfg.beta
    )


def log_a_torch(x: torch.Tensor, cfg: base.Config) -> torch.Tensor:
    a = total_a_torch(x, cfg)
    return torch.log(torch.clamp(a, min=1.0e-12))


def dlog_a_dx_torch(x: torch.Tensor, cfg: base.Config) -> torch.Tensor:
    """Analytic derivative of log a for g_beta=1+beta sin^2(x).

    The constant sigma_const drops out of d_x log a.
    """
    g = 1.0 + cfg.beta * torch.sin(x).pow(2)
    return cfg.alpha * cfg.beta * torch.sin(2.0 * x) / g


class TransformedUNetwork(nn.Module):
    """Network for r(x,t)=log u(x,t), u=a(x)p(x,t).

    The decisive pilot uses a soft initial condition because the preceding
    v3.1 diagnostic showed that hard q0+tN parameterization impedes rapid
    formation of alpha-stable algebraic tails.
    """

    def __init__(self, cfg: base.Config):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(cfg.num_hidden):
            layers += [nn.Linear(in_dim, cfg.ll_hidden_dim), nn.Softplus()]
            in_dim = cfg.ll_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.core = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.core(xt)


def initial_r_true(x: torch.Tensor, cfg: base.Config) -> torch.Tensor:
    """r0=log u0=log p0+log a."""
    return base.initial_q_true(x, cfg) + log_a_torch(x, cfg)


def u_inside(
    u_net: TransformedUNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return base.safe_exp(u_net(torch.cat([x, t], dim=1)))


def p_inside_from_u(
    u_net: TransformedUNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: base.Config,
) -> torch.Tensor:
    return u_inside(u_net, x, t) / torch.clamp(total_a_torch(x, cfg), min=1.0e-12)


# -----------------------------------------------------------------------------
# Exterior padding and standard-GL operator for D^alpha[u]
# -----------------------------------------------------------------------------


def padded_u(
    u_net: TransformedUNetwork,
    x_in: torch.Tensor,
    t_in: torch.Tensor,
    cfg: base.Config,
) -> torch.Tensor:
    """Evaluate u inside and use the existing algebraic *p-tail* outside.

    For |x|>L, first estimate the p-boundary value via u(L)/a(L), extend
    p ~ C |x|^{-(1+alpha)}, then return u(x)=a(x)p(x).  This is the transformed
    counterpart of the current manuscript's exterior-density padding.
    """
    L = cfg.physical_limit
    mask_in = torch.abs(x_in) <= L
    x_safe = torch.clamp(x_in, -L, L)
    u_in = u_inside(u_net, x_safe, t_in)

    sign = torch.where(x_in >= 0.0, torch.ones_like(x_in), -torch.ones_like(x_in))
    x_b = sign * L
    u_b = u_inside(u_net, x_b, t_in)
    a_b = torch.clamp(total_a_torch(x_b, cfg), min=1.0e-12)
    p_b = u_b / a_b

    r = torch.clamp(torch.abs(x_in), min=L)
    p_tail = p_b * (L / r).pow(cfg.alpha + 1.0)
    u_tail = total_a_torch(x_in, cfg) * p_tail
    return torch.where(mask_in, u_in, u_tail)


def gl_fractional_u(
    u_net: TransformedUNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: base.Config,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Standard GL approximation of D_Riesz^alpha[u] on x +/- k h."""
    B = x.shape[0]
    M = cfg.gl_terms
    k_vec = torch.arange(M, dtype=x.dtype, device=x.device).view(1, M)
    x_left = x - k_vec * cfg.gl_dx
    x_right = x + k_vec * cfg.gl_dx
    t_mat = t.expand(B, M)

    def eval_side(xmat: torch.Tensor) -> torch.Tensor:
        xf = xmat.reshape(-1, 1)
        tf = t_mat.reshape(-1, 1)
        vals: List[torch.Tensor] = []
        chunk = max(1, cfg.gl_chunk_size)
        for start in range(0, len(xf), chunk):
            xs = xf[start : start + chunk]
            ts = tf[start : start + chunk]
            vals.append(padded_u(u_net, xs, ts, cfg))
        return torch.cat(vals, dim=0).view(B, M)

    val_left = eval_side(x_left)
    val_right = eval_side(x_right)
    sum_lr = torch.sum((val_left + val_right) * weights, dim=1, keepdim=True)
    riesz_coeff = -1.0 / (2.0 * math.cos(cfg.alpha * math.pi / 2.0))
    return riesz_coeff * sum_lr / (cfg.gl_dx ** cfg.alpha)


# -----------------------------------------------------------------------------
# p-mass constraint and Stage-II warm-ups in transformed coordinates
# -----------------------------------------------------------------------------


def global_p_mass_torch(
    u_net: TransformedUNetwork,
    cfg: base.Config,
    device: torch.device,
    times: torch.Tensor,
) -> torch.Tensor:
    x = torch.linspace(
        -cfg.physical_limit,
        cfg.physical_limit,
        cfg.mass_quad_points,
        dtype=torch.float32,
        device=device,
    )
    w = base.trapezoid_weights(
        cfg.mass_quad_points,
        -cfg.physical_limit,
        cfg.physical_limit,
        device,
    )
    masses: List[torch.Tensor] = []
    for tval in times.reshape(-1):
        tt = torch.full(
            (cfg.mass_quad_points, 1),
            tval,
            dtype=torch.float32,
            device=device,
        )
        xx = x.view(-1, 1)
        p = p_inside_from_u(u_net, xx, tt, cfg).reshape(-1)
        interior = torch.sum(p * w)
        # Tail completion is for p, not u.
        tail = (cfg.physical_limit / cfg.alpha) * (p[0] + p[-1])
        masses.append(interior + tail)
    return torch.stack(masses)


def mass_loss_u(
    u_net: TransformedUNetwork,
    cfg: base.Config,
    device: torch.device,
) -> torch.Tensor:
    times = torch.linspace(
        cfg.t_end / cfg.mass_time_samples,
        cfg.t_end,
        cfg.mass_time_samples,
        dtype=torch.float32,
        device=device,
    )
    masses = global_p_mass_torch(u_net, cfg, device, times)
    return torch.log(torch.clamp(masses, min=1.0e-8)).pow(2).mean()


def warmup_initial_u(
    u_net: TransformedUNetwork,
    cfg: base.Config,
    device: torch.device,
) -> float:
    """Soft IC warm-up matching the successful v3.1 central-core protocol."""
    if cfg.ic_warmup_epochs <= 0:
        return 0.0
    opt = Adam(u_net.parameters(), lr=cfg.ll_warmup_lr)
    base.sync_device(device)
    t0 = time.perf_counter()
    for _ in tqdm(
        range(cfg.ic_warmup_epochs),
        desc="Stage II transformed-u soft-IC warm-up",
        leave=False,
    ):
        x = torch.rand(cfg.batch_size, 1, device=device) * 6.0 - 3.0
        t = torch.zeros_like(x)
        r = u_net(torch.cat([x, t], dim=1))
        loss = (r - initial_r_true(x, cfg)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(u_net.parameters(), 1.0)
        opt.step()
    base.sync_device(device)
    return time.perf_counter() - t0


def warmup_score_u(
    u_net: TransformedUNetwork,
    score_net: base.ScoreNetwork,
    data: np.ndarray,
    cfg: base.Config,
    device: torch.device,
) -> float:
    xt_all = torch.tensor(data, dtype=torch.float32, device=device)
    opt = Adam(u_net.parameters(), lr=cfg.ll_warmup_lr)
    base.sync_device(device)
    t0 = time.perf_counter()
    for _ in tqdm(
        range(cfg.score_warmup_epochs),
        desc="Stage II transformed-u score warm-up",
        leave=False,
    ):
        idx = torch.randint(0, len(xt_all), (cfg.batch_size,), device=device)
        xt = xt_all[idx].clone().detach().requires_grad_(True)
        r = u_net(xt)
        rx = autograd.grad(r.sum(), xt, create_graph=True)[0][:, 0:1]
        x = xt[:, 0:1]
        with torch.no_grad():
            s_p = score_net(xt)
        s_u_target = s_p + dlog_a_dx_torch(x, cfg)
        loss_score = (rx - s_u_target).pow(2).mean()

        reflected = xt.clone()
        reflected[:, 0] *= -1.0
        loss_sym = (u_net(reflected) - r).pow(2).mean()
        loss_mass = mass_loss_u(u_net, cfg, device)
        loss = (
            cfg.score_consistency_weight * loss_score
            + cfg.q_symmetry_weight * loss_sym
            + 0.25 * cfg.mass_loss_weight * loss_mass
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite transformed-u score warm-up loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(u_net.parameters(), 1.0)
        opt.step()
    base.sync_device(device)
    return time.perf_counter() - t0


# -----------------------------------------------------------------------------
# Transformed-u Stage II
# -----------------------------------------------------------------------------


def train_stage2_transformed_u(
    u_net: TransformedUNetwork,
    score_net: base.ScoreNetwork,
    data: np.ndarray,
    cfg: base.Config,
    device: torch.device,
) -> Tuple[Dict[str, float], List[float]]:
    xt_all = torch.tensor(data, dtype=torch.float32, device=device)
    weights = base.gl_weights(cfg.alpha, cfg.gl_terms, device)

    time_ic = warmup_initial_u(u_net, cfg, device)
    time_score = warmup_score_u(u_net, score_net, data, cfg, device)

    opt = Adam(u_net.parameters(), lr=cfg.ll_physics_lr)
    scheduler = CosineAnnealingLR(
        opt, T_max=max(cfg.physics_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    base.sync_device(device)
    t0 = time.perf_counter()

    pbar = tqdm(
        range(cfg.physics_epochs),
        desc="Stage II physics [transformed_u]",
        leave=False,
    )
    for epoch in pbar:
        idx = torch.randint(0, len(xt_all), (cfg.batch_size // 2,), device=device)
        batch = xt_all[idx].clone().detach().requires_grad_(True)
        x = batch[:, 0:1]
        t = batch[:, 1:2]

        r = u_net(batch)
        u = base.safe_exp(r)
        grad_r = autograd.grad(r.sum(), batch, create_graph=True)[0]
        rx = grad_r[:, 0:1]
        rt = grad_r[:, 1:2]

        with torch.no_grad():
            s_p_teacher = score_net(batch)
        dlog_a = dlog_a_dx_torch(x, cfg)
        s_u_teacher = s_p_teacher + dlog_a

        lam = min(1.0, float(epoch + 1) / max(cfg.score_to_q_ramp_epochs, 1))
        s_u_for_pde = (1.0 - lam) * s_u_teacher + lam * rx
        # Drift in the transformed equation needs s_p=s_u-d_x log a.
        s_p_for_drift = s_u_for_pde - dlog_a
        pde_weight = min(1.0, float(epoch + 1) / max(cfg.pde_ramp_epochs, 1))

        term_time = u * rt
        # a[-d_x(f u/a)] with f=-x equals u[1+x(s_u-d_x log a)].
        term_drift = u * (1.0 + x * s_p_for_drift)
        frac_u = gl_fractional_u(u_net, x, t, cfg, weights)
        term_diff = total_a_torch(x, cfg) * frac_u
        residual = term_time - (term_drift + term_diff)
        loss_pde = residual.pow(2).mean()

        score_weight = cfg.score_consistency_weight * (1.0 - 0.75 * lam)
        loss_score = (rx - s_u_teacher).pow(2).mean()

        reflected = batch.clone()
        reflected[:, 0] *= -1.0
        loss_sym = (u_net(reflected) - r).pow(2).mean()

        # Same soft IC protocol as the successful v3.1 p-representation pilot.
        x0 = torch.rand(cfg.batch_size, 1, device=device) * 6.0 - 3.0
        t0_batch = torch.zeros_like(x0)
        r0_pred = u_net(torch.cat([x0, t0_batch], dim=1))
        loss_ic = (r0_pred - initial_r_true(x0, cfg)).pow(2).mean()

        loss_mass = mass_loss_u(u_net, cfg, device)
        mass_weight = cfg.mass_loss_weight * min(1.0, 0.25 + 0.75 * pde_weight)

        loss = (
            pde_weight * loss_pde
            + score_weight * loss_score
            + cfg.q_symmetry_weight * loss_sym
            + cfg.ic_weight * loss_ic
            + mass_weight * loss_mass
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite transformed-u Stage-II loss at step {epoch + 1}"
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(u_net.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % max(1, cfg.physics_epochs // 100) == 0 or epoch + 1 == cfg.physics_epochs:
            hist.append(float(loss.detach().cpu()))

    base.sync_device(device)
    physics_time = time.perf_counter() - t0
    timing = {
        "ic_warmup_s": time_ic,
        "score_warmup_s": time_score,
        "physics_s": physics_time,
        "stage2_total_s": time_ic + time_score + physics_time,
    }
    return timing, hist


# -----------------------------------------------------------------------------
# Evaluation: always convert u back to the probability density p=u/a
# -----------------------------------------------------------------------------


def evaluate_transformed_u(
    u_net: TransformedUNetwork,
    refs: Dict[float, np.ndarray],
    cfg: base.Config,
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
        x_t = xt[:, 0:1]
        t_t = xt[:, 1:2]
        with torch.no_grad():
            p_raw = p_inside_from_u(u_net, x_t, t_t, cfg).cpu().numpy().reshape(-1)

        mass_interior = float(simpson(p_raw, x=x_grid))
        tail_mass = (cfg.physical_limit / cfg.alpha) * float(p_raw[0] + p_raw[-1])
        mass_total = mass_interior + tail_mass
        p_norm = p_raw / max(mass_interior, 1.0e-12)

        samples = refs[float(tval)]
        clipped = samples[
            (samples >= -cfg.physical_limit) & (samples <= cfg.physical_limit)
        ]
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

        if len(clipped) >= 20:
            kde = gaussian_kde(clipped)
            p_ref = kde(x_grid)
            p_ref /= max(simpson(p_ref, x=x_grid), 1.0e-12)
            l2_rel = float(
                np.linalg.norm(p_norm - p_ref) /
                max(np.linalg.norm(p_ref), 1.0e-12)
            )
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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transformation_identity_diagnostic(cfg: base.Config, outdir: Path) -> None:
    """Save analytic checks for the transformed score/drift identities."""
    x = np.linspace(-cfg.physical_limit, cfg.physical_limit, 2001)
    g = 1.0 + cfg.beta * np.sin(x) ** 2
    a = (cfg.sigma_const ** cfg.alpha) * np.abs(g) ** cfg.alpha
    dlog = cfg.alpha * cfg.beta * np.sin(2.0 * x) / g
    # Analytic consistency: s_u=s_p+dlog(a), hence transformed drift equals
    # a times the original p-drift pointwise for any positive p.  Use a smooth
    # test p only to make this an executable floating-point check.
    s = 0.9
    p = np.exp(-0.5 * (x / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
    s_p = -x / (s * s)
    u = a * p
    s_u = s_p + dlog
    lhs = a * p * (1.0 + x * s_p)
    rhs = u * (1.0 + x * (s_u - dlog))
    rel = np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), 1.0e-14)
    rows = [{
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "max_a": float(np.max(a)),
        "min_a": float(np.min(a)),
        "drift_identity_relative_l2": float(rel),
    }]
    write_csv(outdir / "transformation_identity.csv", rows)
    print(f"Transformation drift identity relative L2={rel:.3e}")


# -----------------------------------------------------------------------------
# Pilot driver
# -----------------------------------------------------------------------------


def run_pilot(cfg: base.Config, seed: int, outdir: Path) -> None:
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available() else
        "cpu" if cfg.device == "auto" else cfg.device
    )
    print(f"Device: {device}")
    print(
        f"alpha={cfg.alpha}; beta={cfg.beta}; seed={seed}; "
        "representation=transformed_u"
    )
    print(
        "Zhang-et-al.-inspired protocol: u=a(x)p, r=log u; "
        "standard GL on D^alpha[u]; central coefficient a(x) retained; "
        "Stage-I p-score + analytic dlog(a)/dx correction; soft IC; "
        f"score warm-up={cfg.score_warmup_epochs}; physics={cfg.physics_epochs}; "
        f"PDE ramp={cfg.pde_ramp_epochs}; score->r_x ramp={cfg.score_to_q_ramp_epochs}; "
        f"mass weight={cfg.mass_loss_weight}"
    )

    transformation_identity_diagnostic(cfg, outdir)

    # Keep the same operator discrepancy diagnostic used in the revision.
    op_rows = base.operator_diagnostic(cfg, [cfg.beta], outdir)
    for row in op_rows:
        print(
            f"operator check beta={row['beta']:.3g}, {row['density']}: "
            f"median-vs-exact relL2={row['operator_rel_l2']:.4g}"
        )

    base.set_seed(seed)
    score_data, ll_data, _ = base.build_stage_datasets(cfg, seed=seed + 1000)
    print(
        f"Data: Stage-I unbiased points={len(score_data)}, "
        f"Stage-II collocation={len(ll_data)}, tail ratio={cfg.tail_ratio:.2f}"
    )
    refs = base.simulate_snapshots(
        cfg,
        n_trajectories=cfg.eval_trajectories,
        times=cfg.eval_times,
        seed=seed + 900_000,
    )

    # Exactly the same v3.1 Stage-I p-score teacher.
    base.set_seed(seed + 200_000)
    score_net = base.ScoreNetwork(cfg).to(device)
    stage1_time, _ = base.train_score(score_net, score_data, cfg, device)
    print(f"Stage-I time: {stage1_time:.2f}s")
    if abs(cfg.beta) < 1.0e-14:
        diag = base.beta0_exact_score_diagnostic(score_net, cfg, device)
        write_csv(outdir / f"beta0_score_diagnostic_seed{seed}.csv", diag)
        if diag:
            mean_rel = float(
                np.mean([r["weighted_score_relative_rmse"] for r in diag])
            )
            print(
                "Stage-I beta=0 exact-score diagnostic: "
                f"mean weighted relative RMSE={mean_rel:.4f}"
            )

    # Deterministic Stage-II initialization for this representation.
    base.set_seed(seed + 300_000)
    u_net = TransformedUNetwork(cfg).to(device)
    print("Training representation=transformed_u ...")
    timing, hist = train_stage2_transformed_u(
        u_net, score_net, ll_data, cfg, device
    )
    points = evaluate_transformed_u(u_net, refs, cfg, device)

    ws = [r["wasserstein"] for r in points]
    l2s = [r["rel_l2"] for r in points]
    mdev = [r["mass_deviation"] for r in points]
    row = {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "seed": seed,
        "mode": "transformed_u",
        "mean_w": float(np.mean(ws)),
        "max_w": float(np.max(ws)),
        "mean_rel_l2": float(np.nanmean(l2s)),
        "max_rel_l2": float(np.nanmax(l2s)),
        "mean_mass_deviation": float(np.mean(mdev)),
        "stage1_time_s": stage1_time,
        **timing,
        "end_to_end_time_s": stage1_time + timing["stage2_total_s"],
    }

    per_time_rows = [
        {
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "seed": seed,
            "mode": "transformed_u",
            **p,
        }
        for p in points
    ]
    write_csv(outdir / "transformed_u_per_time.csv", per_time_rows)
    write_csv(outdir / "transformed_u_summary.csv", [row])

    protocol = {
        "reference": "Zhang et al., arXiv:1811.05610, Eq. (3.4)",
        "transformation": "u(x,t)=a(x)p(x,t), a(x)=|g_beta(x)|^alpha",
        "network_state": "r=log u",
        "score_relation": "d_x log u = d_x log p + d_x log a",
        "drift": "u_t drift = u[1+x(s_u-d_x log a)] for f=-x",
        "fractional": "a(x) * D_Riesz^alpha[u]",
        "gl_stencil": "standard x +/- k h",
        "ic": "soft central-core r0=log p0+log a",
        "stage1": "same unbiased p-score teacher as v3.1",
        "tail_padding": "power-law padding on p, then u=a p at auxiliary coordinates",
        "config": cfg.__dict__,
    }
    with (outdir / "protocol_transformed_u.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2, default=list)

    if cfg.save_checkpoints:
        ckdir = outdir / "checkpoints"
        ckdir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "u_state_dict": u_net.state_dict(),
                "score_state_dict": score_net.state_dict(),
                "config": cfg.__dict__,
                "mode": "transformed_u",
                "seed": seed,
            },
            ckdir / f"alpha{cfg.alpha:g}_beta{cfg.beta:g}_seed{seed}_transformed_u.pt",
        )

    print(
        "  transformed_u: "
        f"mean W={row['mean_w']:.5f}, max W={row['max_w']:.5f}, "
        f"mean relL2={row['mean_rel_l2']:.5f}, "
        f"mass dev={row['mean_mass_deviation']:.5f}"
    )
    print(f"Results: {outdir.resolve()}")
    print("Per-time file: transformed_u_per_time.csv")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--output_dir",
        default="revision_results/05_multiplicative/transformed_u_beta05_seed42",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_checkpoints", action="store_true")

    # Defaults intentionally match the successful v3.1 soft-IC pilot.
    p.add_argument("--score_epochs", type=int, default=5000)
    p.add_argument("--ic_warmup_epochs", type=int, default=800)
    p.add_argument("--score_warmup_epochs", type=int, default=1000)
    p.add_argument("--physics_epochs", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gl_terms", type=int, default=1000)
    p.add_argument("--gl_dx", type=float, default=0.005)
    p.add_argument("--train_trajectories", type=int, default=10000)
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
    p.add_argument("--ic_weight", type=float, default=2.0)
    p.add_argument("--mass_quad_points", type=int, default=161)
    p.add_argument("--mass_time_samples", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = base.Config(
        alpha=args.alpha,
        beta=args.beta,
        device=args.device,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        save_checkpoints=not args.no_checkpoints,
        score_epochs=args.score_epochs,
        score_lr=args.score_lr,
        ic_mode="soft",
        ic_warmup_epochs=args.ic_warmup_epochs,
        score_warmup_epochs=args.score_warmup_epochs,
        physics_epochs=args.physics_epochs,
        ll_warmup_lr=args.ll_warmup_lr,
        ll_physics_lr=args.ll_physics_lr,
        batch_size=args.batch_size,
        gl_terms=args.gl_terms,
        gl_dx=args.gl_dx,
        train_trajectories=args.train_trajectories,
        score_train_points=args.score_train_points,
        ll_train_points=args.ll_train_points,
        tail_ratio=args.tail_ratio,
        eval_trajectories=args.eval_trajectories,
        pde_ramp_epochs=args.pde_ramp_epochs,
        score_to_q_ramp_epochs=args.score_to_q_ramp_epochs,
        score_consistency_weight=args.score_consistency_weight,
        mass_loss_weight=args.mass_loss_weight,
        ic_weight=args.ic_weight,
        boundary_weight=0.0,
        decay_weight=0.0,
        mass_quad_points=args.mass_quad_points,
        mass_time_samples=args.mass_time_samples,
    )

    if not (1.0 < cfg.alpha < 2.0):
        raise ValueError("alpha must lie in (1,2)")
    if cfg.beta < 0.0:
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
        cfg.output_dir = str(
            Path(cfg.output_dir).with_name(Path(cfg.output_dir).name + "_quick")
        )

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_pilot(cfg, args.seed, outdir)


if __name__ == "__main__":
    main()
