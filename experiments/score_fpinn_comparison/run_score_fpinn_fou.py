"""
Reviewer-1 Comment-2 direct comparison baseline: Score-fPINN on the same 1D fOU problem.

This is a PyTorch reimplementation of the published Score-fPINN route of
Hu, Zhang, Karniadakis & Kawaguchi (arXiv:2406.11676; CiCP 2026), adapted to
exactly the one-dimensional fOU SDE used by the revised SBPINN manuscript:

    dX_t = -theta X_t dt + sigma dL_t^alpha,
    X_0 ~ N(0,1), theta=0.5, sigma=1.

Published Score-fPINN structure used here:
  1) learn the ordinary score s = d_x log p by the Hyvarinen score-matching
     objective (Eq. 13; in 1D the divergence is evaluated exactly by autograd);
  2) learn the fractional score r = S^(alpha) through the score-fPDE (Eq. 12/14);
  3) reconstruct q=log p from the local LL-PDE (Eq. 8/15), which contains no
     fractional GL/FFT operator after r is learned.

For G=0 in 1D, define A = f - sigma^alpha r. Then
    q_t = -A q_x - A_x,
and differentiating in x gives the score-fPDE
    s_t = -(A_x s + A s_x + A_xx).

The ordinary-score and log-density networks use hard Gaussian initial
constraints as in the published Score-fPINN experiments:
    s(x,t) = -x + t N_s(x,t),
    q(x,t) = log p0(x) + t N_q(x,t).

The fractional-score network is unconstrained at t=0, matching the published
Score-fPINN implementation. All three networks are 4-layer tanh MLPs with
hidden width 128 by default.

IMPORTANT FAIRNESS CHOICES
--------------------------
* Same fOU SDE, initial law, domain [-6,6], trajectory generator, exact
  characteristic-function reference, evaluation grid/times, and default
  Stage-I data size as revised SBPINN.
* Physics training points are sampled from the empirical p_t distribution,
  as required by Score-fPINN Eq. (14)-(15), rather than SBPINN's tail-enriched
  GL collocation distribution.
* No exact solution is used for training or checkpoint selection.
* No fractional GL/FFT operator is used anywhere in Score-fPINN training.
* The script reports wall-clock time and CUDA peak memory, plus full-domain and
  tail-region density errors against the same exact reference.

This script is intended as a controlled reviewer-response baseline, not as a
claim that either method dominates the other outside their shared setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import simpson
from scipy.stats import levy_stable, wasserstein_distance
from torch.optim import Adam
from tqdm import tqdm

try:
    from numpy import trapezoid as np_trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as np_trapz


@dataclass
class Config:
    # Same 1D fOU as revised SBPINN Sec. 3.1.
    theta: float = 0.5
    sigma: float = 1.0
    alpha: float = 1.75
    t_final: float = 1.0
    n_sde_steps: int = 100
    initial_std: float = 1.0
    domain_bound: float = 6.0

    # Same empirical-data scale as revised SBPINN.
    train_trajectories: int = 800
    train_points: int = 60000
    batch_size: int = 384

    # Published Score-fPINN model scale: 4-layer FCN, hidden width 128.
    hidden_dim: int = 128
    num_layers: int = 4

    # Pilot budgets. Score-fPINN has three optimization stages.
    ordinary_score_epochs: int = 5000
    fractional_score_epochs: int = 5000
    ll_epochs: int = 5000
    score_lr: float = 5.0e-4
    fractional_score_lr: float = 5.0e-4
    ll_lr: float = 5.0e-4
    eta_min: float = 1.0e-5
    smooth_l1_beta: float = 1.0
    grad_clip: float = 1.0

    # Same exact-reference/evaluation conventions as revised fOU.
    eval_grid_n: int = 1201
    eval_times: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    exact_k_max: float = 100.0
    exact_k_num: int = 16000
    tail_start: float = 4.0

    seed: int = 42
    device: str = "auto"
    output_dir: str = "results_score_fpinn_fou"
    save_checkpoints: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def safe_exp(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(q, min=-60.0, max=20.0))


def smooth_l1_paper(residual: torch.Tensor, beta: float) -> torch.Tensor:
    """Piecewise smooth-L1 convention used in Hu et al. Eq. (16)."""
    a = residual.abs()
    return torch.where(
        a < beta,
        a.square(),
        2.0 * beta * a - beta * beta,
    ).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CoreMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must include at least input/output layers")
        layers: List[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


class OrdinaryScoreNet(nn.Module):
    """Published hard initial score: s(x,t) = -x + t*N(x,t)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.core = CoreMLP(cfg.hidden_dim, cfg.num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x + t * self.core(x, t)


class FractionalScoreNet(nn.Module):
    """Fractional score S^(alpha); no artificial initial target is imposed."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.core = CoreMLP(cfg.hidden_dim, cfg.num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.core(x, t)


class LogDensityNet(nn.Module):
    """Published hard initial LL: q(x,t)=log p0(x)+t*N(x,t)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.core = CoreMLP(cfg.hidden_dim, cfg.num_layers)
        self.initial_std = cfg.initial_std

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = self.initial_std
        q0 = -0.5 * (x / s).square() - math.log(math.sqrt(2.0 * math.pi) * s)
        return q0 + t * self.core(x, t)


class FractionalOUProcess:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def simulate(self, n_particles: int) -> np.ndarray:
        cfg = self.cfg
        dt = cfg.t_final / cfg.n_sde_steps
        traj = np.zeros((n_particles, cfg.n_sde_steps + 1), dtype=np.float64)
        traj[:, 0] = np.random.normal(0.0, cfg.initial_std, n_particles)
        increments = levy_stable.rvs(
            alpha=cfg.alpha,
            beta=0.0,
            loc=0.0,
            scale=dt ** (1.0 / cfg.alpha),
            size=(n_particles, cfg.n_sde_steps),
        )
        for n in range(cfg.n_sde_steps):
            x = traj[:, n]
            traj[:, n + 1] = (
                x - cfg.theta * x * dt + cfg.sigma * increments[:, n]
            )
        return traj


def build_empirical_training_data(cfg: Config) -> np.ndarray:
    """Unbiased samples from empirical p_t, with valid antithetic augmentation."""
    traj = FractionalOUProcess(cfg).simulate(cfg.train_trajectories)
    times = np.linspace(0.0, cfg.t_final, cfg.n_sde_steps + 1, dtype=np.float64)
    tt = np.broadcast_to(times[None, :], traj.shape)
    points = np.stack([traj, tt], axis=-1).reshape(-1, 2)
    points = points[np.abs(points[:, 0]) <= cfg.domain_bound]

    # Symmetry augmentation preserves p_t exactly for this symmetric fOU.
    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    pool = np.vstack([points, mirrored])
    idx = np.random.choice(len(pool), cfg.train_points, replace=len(pool) < cfg.train_points)
    data = pool[idx].astype(np.float32)
    np.random.shuffle(data)
    return data


def derivative(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        y.sum(), x, create_graph=create_graph, retain_graph=True
    )[0]


def make_training_tensors(data: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(data[:, 0:1], dtype=torch.float32, device=device),
        torch.as_tensor(data[:, 1:2], dtype=torch.float32, device=device),
    )


def sample_batch(
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    batch_size: int,
    positive_time: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positive_time:
        valid = torch.nonzero(t_all[:, 0] > 1.0e-8, as_tuple=False).view(-1)
        j = torch.randint(0, len(valid), (batch_size,), device=x_all.device)
        idx = valid[j]
    else:
        idx = torch.randint(0, len(x_all), (batch_size,), device=x_all.device)
    x = x_all[idx].clone().detach().requires_grad_(True)
    t = t_all[idx].clone().detach().requires_grad_(True)
    return x, t


def train_ordinary_score(
    model: OrdinaryScoreNet,
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    cfg: Config,
) -> Tuple[List[float], float]:
    opt = Adam(model.parameters(), lr=cfg.score_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.ordinary_score_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    start = time.perf_counter()
    pbar = tqdm(range(cfg.ordinary_score_epochs), desc="Score-fPINN 1/3 ordinary score", leave=False)
    for k in pbar:
        x, t = sample_batch(x_all, t_all, cfg.batch_size, positive_time=False)
        s = model(x, t)
        s_x = derivative(s, x, create_graph=True)
        loss = (0.5 * s.square() + s_x).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"ordinary-score loss nonfinite at {k+1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        hist.append(float(loss.detach().cpu()))
        if (k + 1) % 500 == 0:
            pbar.set_postfix(loss=f"{hist[-1]:.3e}")
    elapsed = time.perf_counter() - start
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return hist, elapsed


def train_fractional_score(
    ordinary: OrdinaryScoreNet,
    frac: FractionalScoreNet,
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    cfg: Config,
) -> Tuple[List[float], float]:
    """Train S^(alpha) from the 1D specialization of Hu et al. Eq. (12)/(14)."""
    opt = Adam(frac.parameters(), lr=cfg.fractional_score_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.fractional_score_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    sigma_a = cfg.sigma ** cfg.alpha
    start = time.perf_counter()
    pbar = tqdm(range(cfg.fractional_score_epochs), desc="Score-fPINN 2/3 fractional score", leave=False)
    for k in pbar:
        x, t = sample_batch(x_all, t_all, cfg.batch_size, positive_time=True)

        # Fixed learned ordinary score and its input derivatives.
        s = ordinary(x, t)
        s_x = derivative(s, x, create_graph=True)
        s_t = derivative(s, t, create_graph=True)
        s_det = s.detach(); sx_det = s_x.detach(); st_det = s_t.detach()

        r = frac(x, t)
        r_x = derivative(r, x, create_graph=True)
        r_xx = derivative(r_x, x, create_graph=True)

        # A = f - sigma^alpha r; f=-theta*x.
        A = -cfg.theta * x - sigma_a * r
        A_x = -cfg.theta - sigma_a * r_x
        A_xx = -sigma_a * r_xx

        # d_t s + d_x(A*s + A_x) = 0.
        residual = st_det + A_x * s_det + A * sx_det + A_xx
        loss = smooth_l1_paper(residual, cfg.smooth_l1_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"fractional-score loss nonfinite at {k+1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(frac.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        hist.append(float(loss.detach().cpu()))
        if (k + 1) % 500 == 0:
            pbar.set_postfix(loss=f"{hist[-1]:.3e}")
    elapsed = time.perf_counter() - start
    frac.eval()
    for p in frac.parameters():
        p.requires_grad_(False)
    return hist, elapsed


def train_log_density(
    frac: FractionalScoreNet,
    qnet: LogDensityNet,
    x_all: torch.Tensor,
    t_all: torch.Tensor,
    cfg: Config,
) -> Tuple[List[float], float]:
    """Train q from the local LL-PDE after the fractional operator is removed."""
    opt = Adam(qnet.parameters(), lr=cfg.ll_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.ll_epochs, 1), eta_min=cfg.eta_min
    )
    hist: List[float] = []
    sigma_a = cfg.sigma ** cfg.alpha
    start = time.perf_counter()
    pbar = tqdm(range(cfg.ll_epochs), desc="Score-fPINN 3/3 local LL-PDE", leave=False)
    for k in pbar:
        x, t = sample_batch(x_all, t_all, cfg.batch_size, positive_time=True)
        q = qnet(x, t)
        q_x = derivative(q, x, create_graph=True)
        q_t = derivative(q, t, create_graph=True)

        r = frac(x, t)
        r_x = derivative(r, x, create_graph=True)
        A = (-cfg.theta * x - sigma_a * r).detach()
        A_x = (-cfg.theta - sigma_a * r_x).detach()

        # q_t = -A*q_x - A_x.
        residual = q_t + A * q_x + A_x
        loss = smooth_l1_paper(residual, cfg.smooth_l1_beta)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"LL-PDE loss nonfinite at {k+1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        hist.append(float(loss.detach().cpu()))
        if (k + 1) % 500 == 0:
            pbar.set_postfix(loss=f"{hist[-1]:.3e}")
    elapsed = time.perf_counter() - start
    qnet.eval()
    return hist, elapsed


def exact_fou_pdf(x_grid: np.ndarray, t: float, cfg: Config, chunk_size: int = 256) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if t <= 1.0e-14:
        s = cfg.initial_std
        return np.exp(-0.5 * (x_grid / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)
    a_t = (
        t if abs(cfg.theta) < 1.0e-14
        else (1.0 - math.exp(-cfg.alpha * cfg.theta * t)) / (cfg.alpha * cfg.theta)
    )
    gaussian_variance = cfg.initial_std ** 2 * math.exp(-2.0 * cfg.theta * t)
    levy_coeff = cfg.sigma ** cfg.alpha * a_t
    k = np.linspace(0.0, cfg.exact_k_max, cfg.exact_k_num, dtype=np.float64)
    decay = np.exp(-0.5 * gaussian_variance * k * k - levy_coeff * k ** cfg.alpha)
    pdf = np.empty_like(x_grid)
    for start in range(0, len(x_grid), chunk_size):
        block = x_grid[start:start + chunk_size, None]
        integrand = np.cos(block * k[None, :]) * decay[None, :]
        pdf[start:start + chunk_size] = np_trapz(integrand, k, axis=1) / math.pi
    return np.maximum(pdf, 0.0)


def normalize_density(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    m = float(simpson(p, x=x))
    if not np.isfinite(m) or m <= 1e-14:
        raise RuntimeError("invalid density mass")
    return p / m


def predict_density(qnet: LogDensityNet, x: np.ndarray, t: float, device: torch.device) -> np.ndarray:
    xt = torch.as_tensor(x[:, None], dtype=torch.float32, device=device)
    tt = torch.full_like(xt, float(t))
    with torch.no_grad():
        return safe_exp(qnet(xt, tt)).cpu().numpy().reshape(-1).astype(np.float64)


def metrics_for_time(qnet: LogDensityNet, x: np.ndarray, t: float, cfg: Config, device: torch.device) -> Dict[str, float]:
    exact_raw = exact_fou_pdf(x, t, cfg)
    pred_raw = predict_density(qnet, x, t, device)
    exact_mass = float(simpson(exact_raw, x=x))
    pred_mass = float(simpson(pred_raw, x=x))

    exact = normalize_density(exact_raw, x)
    pred = normalize_density(pred_raw, x)
    diff = pred - exact
    abs_l2 = float(math.sqrt(simpson(diff * diff, x=x)))
    rel_l2 = float(abs_l2 / max(math.sqrt(simpson(exact * exact, x=x)), 1e-14))
    linf = float(np.max(np.abs(diff)))
    w1 = float(wasserstein_distance(x, x, u_weights=pred, v_weights=exact))

    # Tail region is the disjoint union [-L,-tail_start] U [tail_start,L].
    # Integrate the two pieces separately; never bridge across the omitted core.
    left = x <= -cfg.tail_start
    right = x >= cfg.tail_start

    def two_tail_integral(values: np.ndarray) -> float:
        total = 0.0
        if np.count_nonzero(left) >= 2:
            total += float(simpson(values[left], x=x[left]))
        if np.count_nonzero(right) >= 2:
            total += float(simpson(values[right], x=x[right]))
        return total

    tail_diff_sq = (pred - exact) ** 2
    tail_ref_sq = exact ** 2
    tail_ref_norm = math.sqrt(max(two_tail_integral(tail_ref_sq), 0.0))
    tail_rel_l2 = float(math.sqrt(max(two_tail_integral(tail_diff_sq), 0.0)) / max(tail_ref_norm, 1e-14))
    tail_mass_pred = two_tail_integral(pred)
    tail_mass_exact = two_tail_integral(exact)

    return {
        "alpha": cfg.alpha,
        "seed": cfg.seed,
        "time": float(t),
        "L2_error_normalized": abs_l2,
        "rel_L2_normalized": rel_l2,
        "Linf_error_normalized": linf,
        "Wasserstein_normalized": w1,
        "raw_mass_in_domain": pred_mass,
        "exact_mass_in_domain": exact_mass,
        "abs_in_domain_mass_error": abs(pred_mass - exact_mass),
        "tail_start": cfg.tail_start,
        "tail_rel_L2_normalized": tail_rel_l2,
        "tail_mass_normalized_pred": tail_mass_pred,
        "tail_mass_normalized_exact": tail_mass_exact,
        "tail_mass_abs_error_normalized": abs(tail_mass_pred - tail_mass_exact),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run(cfg: Config) -> None:
    if not (1.0 < cfg.alpha < 2.0):
        raise ValueError("alpha must lie in (1,2)")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    print(f"Device: {device}")
    print(f"Score-fPINN direct fOU baseline: alpha={cfg.alpha}, seed={cfg.seed}")
    print("SDE: dX=-0.5 X dt + dL_alpha; X0~N(0,1)")
    print("Published route: ordinary score -> fractional score via score-fPDE -> local LL-PDE")
    print("No GL/FFT fractional operator is evaluated in this baseline.")
    print(f"Data: trajectories={cfg.train_trajectories}, empirical points={cfg.train_points}, batch={cfg.batch_size}")
    print(f"Budget: ordinary={cfg.ordinary_score_epochs}, fractional={cfg.fractional_score_epochs}, LL={cfg.ll_epochs}")

    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)

    data_start = time.perf_counter()
    data = build_empirical_training_data(cfg)
    data_time = time.perf_counter() - data_start
    x_all, t_all = make_training_tensors(data, device)

    ordinary = OrdinaryScoreNet(cfg).to(device)
    frac = FractionalScoreNet(cfg).to(device)
    qnet = LogDensityNet(cfg).to(device)

    n_ord = count_parameters(ordinary)
    n_frac = count_parameters(frac)
    n_q = count_parameters(qnet)
    print(f"Parameters: ordinary={n_ord}, fractional={n_frac}, q={n_q}, total={n_ord+n_frac+n_q}")

    h1, t1 = train_ordinary_score(ordinary, x_all, t_all, cfg)
    h2, t2 = train_fractional_score(ordinary, frac, x_all, t_all, cfg)
    h3, t3 = train_log_density(frac, qnet, x_all, t_all, cfg)

    peak_mb = float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0**2

    grid = np.linspace(-cfg.domain_bound, cfg.domain_bound, cfg.eval_grid_n)
    rows = [metrics_for_time(qnet, grid, t, cfg, device) for t in cfg.eval_times]
    write_csv(out / "score_fpinn_fou_per_time.csv", rows)

    positive = [r for r in rows if r["time"] > 1e-12]
    summary = {
        "method": "Score-fPINN",
        "alpha": cfg.alpha,
        "seed": cfg.seed,
        "mean_W1_positive_times": float(np.mean([r["Wasserstein_normalized"] for r in positive])),
        "max_W1_positive_times": float(np.max([r["Wasserstein_normalized"] for r in positive])),
        "mean_rel_L2_positive_times": float(np.mean([r["rel_L2_normalized"] for r in positive])),
        "mean_abs_in_domain_mass_error_positive_times": float(np.mean([r["abs_in_domain_mass_error"] for r in positive])),
        "mean_tail_rel_L2_positive_times": float(np.mean([r["tail_rel_L2_normalized"] for r in positive])),
        "data_time_s": data_time,
        "ordinary_score_time_s": t1,
        "fractional_score_time_s": t2,
        "ll_time_s": t3,
        "optimization_time_s": t1 + t2 + t3,
        "total_time_s": data_time + t1 + t2 + t3,
        "peak_cuda_memory_mb": peak_mb,
        "ordinary_score_params": n_ord,
        "fractional_score_params": n_frac,
        "ll_params": n_q,
        "total_params": n_ord + n_frac + n_q,
    }
    write_csv(out / "score_fpinn_fou_summary.csv", [summary])

    protocol = asdict(cfg)
    protocol.update({
        "method": "Score-fPINN",
        "paper_equations": "ordinary score Eq13; score-fPDE Eq12/14; LL-PDE Eq8/15",
        "fractional_operator_in_training": "none",
        "fairness_note": "same 1D fOU SDE/data scale/domain/exact reference as revised SBPINN; empirical p_t collocation retained for Score-fPINN",
    })
    with (out / "protocol_score_fpinn_fou.json").open("w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    np.savez_compressed(
        out / "training_histories.npz",
        ordinary_score=np.asarray(h1),
        fractional_score=np.asarray(h2),
        ll=np.asarray(h3),
    )
    if cfg.save_checkpoints:
        torch.save({
            "config": protocol,
            "ordinary_score": ordinary.state_dict(),
            "fractional_score": frac.state_dict(),
            "log_density": qnet.state_dict(),
            "summary": summary,
            "per_time": rows,
        }, out / "score_fpinn_fou_checkpoint.pt")

    print("\nScore-fPINN evaluation")
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
        f"time={summary['optimization_time_s']:.2f}s, peakCUDA={peak_mb:.1f} MB"
    )
    print(f"Results: {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=1.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output_dir", default="results_score_fpinn_fou")
    p.add_argument("--ordinary_score_epochs", type=int, default=5000)
    p.add_argument("--fractional_score_epochs", type=int, default=5000)
    p.add_argument("--ll_epochs", type=int, default=5000)
    p.add_argument("--train_trajectories", type=int, default=800)
    p.add_argument("--train_points", type=int, default=60000)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_checkpoint", action="store_true")
    return p


def main() -> None:
    a = build_parser().parse_args()
    cfg = Config(
        alpha=a.alpha,
        seed=a.seed,
        device=a.device,
        output_dir=a.output_dir,
        ordinary_score_epochs=a.ordinary_score_epochs,
        fractional_score_epochs=a.fractional_score_epochs,
        ll_epochs=a.ll_epochs,
        train_trajectories=a.train_trajectories,
        train_points=a.train_points,
        batch_size=a.batch_size,
        save_checkpoints=not a.no_checkpoint,
    )
    if a.quick:
        cfg.train_trajectories = min(cfg.train_trajectories, 80)
        cfg.train_points = min(cfg.train_points, 4000)
        cfg.batch_size = min(cfg.batch_size, 64)
        cfg.ordinary_score_epochs = min(cfg.ordinary_score_epochs, 5)
        cfg.fractional_score_epochs = min(cfg.fractional_score_epochs, 5)
        cfg.ll_epochs = min(cfg.ll_epochs, 5)
        cfg.eval_grid_n = 301
        cfg.exact_k_num = 4000
        cfg.save_checkpoints = False
        cfg.output_dir = str(Path(cfg.output_dir).with_name(Path(cfg.output_dir).name + "_quick"))
    run(cfg)


if __name__ == "__main__":
    main()
