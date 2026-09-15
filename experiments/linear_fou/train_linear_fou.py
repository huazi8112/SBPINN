"""
Stable, publication-ready one-dimensional fractional OU experiment.

Main revisions
--------------
1. Each alpha run starts from the same reproducible random seed.
2. Stage-I score matching uses unbiased trajectory samples rather than the
   tail-reweighted PDE collocation distribution.
3. The score network is deterministic (no dropout), uses a smaller learning
   rate, exact initial-score anchoring, and a light odd-symmetry constraint.
4. Stage II keeps the two-stage framework and does not use score-integration
   initialization.  It uses a score-consistency warm-up followed by gradual
   activation of the standard-GL FFP residual, score-to-grad(q) continuation,
   multi-time mass regularization, and a hard initial condition.
5. The collocation set contains stronger tail coverage, which is especially
   useful for alpha=1.5.
6. Figures omit the overall title and panel letters, use distinguishable line
   styles/markers, and employ publication-scale typography.

The exact transient density is computed by characteristic-function inversion.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import simpson
from scipy.stats import levy_stable, wasserstein_distance
from torch.optim import Adam
from tqdm import tqdm

try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz


# =============================================================================
# Publication plotting
# =============================================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "axes.linewidth": 1.15,
    "lines.linewidth": 2.5,
    "savefig.dpi": 600,
})


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    theta: float = 0.5
    sigma: float = 1.0
    t_final: float = 1.0
    n_sde_steps: int = 100
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results_linear_fou_exact_stable"

    # Data. Stage I and Stage II use separate sampling distributions.
    train_trajectories: int = 800
    score_train_points: int = 60000
    ll_train_points: int = 60000
    tail_ratio: float = 0.40
    domain_bound: float = 6.0

    # Network and optimization.
    hidden_dim: int = 128
    num_layers: int = 4
    batch_size: int = 384
    ic_batch_size: int = 512
    score_epochs: int = 5000
    ll_warmup_epochs: int = 1000
    ll_physics_epochs: int = 4000
    score_lr: float = 5.0e-4
    ll_warmup_lr: float = 8.0e-4
    ll_physics_lr: float = 2.0e-4
    eta_min: float = 1.0e-5

    # Stage-I and Stage-II constraints.
    score_ic_weight: float = 10.0
    score_symmetry_weight: float = 0.10
    score_consistency_weight: float = 1.0
    q_symmetry_weight: float = 0.10
    mass_loss_weight: float = 1.0
    pde_ramp_epochs: int = 1200
    score_to_q_ramp_epochs: int = 1200

    # Standard GL and mass quadrature.
    gl_dx: float = 0.08
    gl_terms: int = 80
    mass_quad_points: int = 161
    mass_time_samples: int = 4

    # Exact reference and figure.
    eval_grid_n: int = 1201
    exact_k_max: float = 100.0
    exact_k_num: int = 16000
    eval_times: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
    loss_smooth_window: int = 200
    save_model: bool = True


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def safe_exp(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(q, min=-60.0, max=20.0))


def trapezoid_weights(
    n: int, a: float, b: float, device: torch.device
) -> torch.Tensor:
    h = (b - a) / (n - 1)
    weights = torch.full((n,), h, dtype=torch.float32, device=device)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < window:
        return array
    return np.convolve(array, np.ones(window) / window, mode="valid")


# =============================================================================
# Networks
# =============================================================================
class MLP(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class ScoreNetwork(MLP):
    def __init__(self, cfg: Config) -> None:
        # No dropout: the frozen score target must be deterministic in Stage II.
        super().__init__(1, cfg.hidden_dim, cfg.num_layers)


class LLNetwork(nn.Module):
    """q(x,t)=q0(x)+t*N(x,t), enforcing the Gaussian initial density exactly."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.correction = MLP(1, cfg.hidden_dim, cfg.num_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q0 = -0.5 * x.pow(2) - 0.5 * math.log(2.0 * math.pi)
        return q0 + t * self.correction(x, t)


# =============================================================================
# Fractional OU trajectories and stage-specific datasets
# =============================================================================
class FractionalOUProcess:
    def __init__(self, cfg: Config, alpha: float) -> None:
        self.theta = cfg.theta
        self.sigma = cfg.sigma
        self.alpha = alpha

    def simulate(self, n_particles: int, cfg: Config) -> np.ndarray:
        dt = cfg.t_final / cfg.n_sde_steps
        trajectories = np.zeros(
            (n_particles, cfg.n_sde_steps + 1), dtype=np.float64
        )
        trajectories[:, 0] = np.random.normal(0.0, 1.0, n_particles)
        scale = dt ** (1.0 / self.alpha)
        increments = levy_stable.rvs(
            alpha=self.alpha,
            beta=0.0,
            loc=0.0,
            scale=scale,
            size=(n_particles, cfg.n_sde_steps),
        )
        for n in range(cfg.n_sde_steps):
            x = trajectories[:, n]
            trajectories[:, n + 1] = (
                x - self.theta * x * dt + self.sigma * increments[:, n]
            )
        return trajectories


def generate_stage_datasets(
    process: FractionalOUProcess,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate unbiased score samples and tail-enhanced PDE collocation points."""
    trajectories = process.simulate(cfg.train_trajectories, cfg)
    times = np.arange(cfg.n_sde_steps + 1, dtype=np.float64)
    times *= cfg.t_final / cfg.n_sde_steps
    tt = np.broadcast_to(times[None, :], trajectories.shape)
    all_points = np.stack([trajectories, tt], axis=-1).reshape(-1, 2)

    inside = np.abs(all_points[:, 0]) <= cfg.domain_bound
    physical_points = all_points[inside]
    if len(physical_points) < cfg.batch_size:
        raise RuntimeError("Too few trajectory points inside the neural domain.")

    # Stage I: true empirical distribution, augmented by antithetic reflection.
    mirrored = physical_points.copy()
    mirrored[:, 0] *= -1.0
    score_pool = np.vstack([physical_points, mirrored])
    score_idx = np.random.choice(
        len(score_pool),
        cfg.score_train_points,
        replace=len(score_pool) < cfg.score_train_points,
    )
    score_data = score_pool[score_idx].astype(np.float32)

    # Stage II: stronger tail coverage for the nonlocal residual.
    abs_x = np.abs(physical_points[:, 0])
    threshold = float(np.percentile(abs_x, 70.0))
    core = physical_points[abs_x <= threshold]
    trajectory_tail = physical_points[abs_x > threshold]

    n_tail = int(cfg.ll_train_points * cfg.tail_ratio)
    n_core = cfg.ll_train_points - n_tail
    core_idx = np.random.choice(
        len(core), n_core, replace=len(core) < n_core
    )
    core_sample = core[core_idx]

    n_traj_tail = min(len(trajectory_tail), n_tail // 2)
    if n_traj_tail > 0:
        tail_idx = np.random.choice(
            len(trajectory_tail),
            n_traj_tail,
            replace=len(trajectory_tail) < n_traj_tail,
        )
        tail_sample = trajectory_tail[tail_idx]
    else:
        tail_sample = np.empty((0, 2), dtype=np.float64)

    n_uniform = n_tail - len(tail_sample)
    uniform_blocks: List[np.ndarray] = []
    while sum(len(block) for block in uniform_blocks) < n_uniform:
        candidates = np.random.uniform(
            -cfg.domain_bound,
            cfg.domain_bound,
            size=max(2 * n_uniform, 2048),
        )
        candidates = candidates[np.abs(candidates) > threshold]
        t_values = np.random.uniform(
            0.0, cfg.t_final, size=len(candidates)
        )
        uniform_blocks.append(np.column_stack([candidates, t_values]))
    uniform_tail = np.vstack(uniform_blocks)[:n_uniform]

    ll_data = np.vstack([core_sample, tail_sample, uniform_tail]).astype(np.float32)
    np.random.shuffle(score_data)
    np.random.shuffle(ll_data)
    return score_data, ll_data


# =============================================================================
# Standard GL derivative and global mass
# =============================================================================
def gl_coefficients(
    alpha: float, n_terms: int, device: torch.device
) -> torch.Tensor:
    coefficients = [1.0]
    for k in range(1, n_terms):
        coefficients.append(coefficients[-1] * (k - 1.0 - alpha) / k)
    return torch.tensor(coefficients, dtype=torch.float32, device=device)


def fractional_derivative_gl(
    density_func,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    cfg: Config,
) -> torch.Tensor:
    batch = x.shape[0]
    device = x.device
    coefficients = gl_coefficients(alpha, cfg.gl_terms, device).view(
        1, cfg.gl_terms
    )
    shifts = (
        torch.arange(cfg.gl_terms, dtype=x.dtype, device=device)
        * cfg.gl_dx
    ).view(1, cfg.gl_terms)

    left_x = x - shifts
    right_x = x + shifts
    t_rep = t.repeat(1, cfg.gl_terms)

    def evaluate(points: torch.Tensor) -> torch.Tensor:
        flat_x = points.reshape(-1, 1)
        flat_t = t_rep.reshape(-1, 1)
        inside = (flat_x.abs() <= cfg.domain_bound).view(-1)
        values = torch.zeros_like(flat_x)
        if inside.any():
            values[inside] = density_func(flat_x[inside], flat_t[inside])
        outside = ~inside
        if outside.any():
            x_out = flat_x[outside]
            x_anchor = cfg.domain_bound * torch.sign(x_out)
            p_anchor = density_func(x_anchor, flat_t[outside])
            decay = (x_out.abs() / cfg.domain_bound).pow(-(1.0 + alpha))
            values[outside] = p_anchor * decay
        return values.view(batch, cfg.gl_terms)

    weighted = torch.sum(
        coefficients * (evaluate(left_x) + evaluate(right_x)),
        dim=1,
        keepdim=True,
    )
    factor = -1.0 / (
        2.0
        * math.cos(alpha * math.pi / 2.0)
        * cfg.gl_dx ** alpha
    )
    return factor * weighted


def global_mass_torch(
    density_func,
    times: torch.Tensor,
    alpha: float,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    x = torch.linspace(
        -cfg.domain_bound,
        cfg.domain_bound,
        cfg.mass_quad_points,
        dtype=torch.float32,
        device=device,
    )
    weights = trapezoid_weights(
        cfg.mass_quad_points,
        -cfg.domain_bound,
        cfg.domain_bound,
        device,
    )
    masses: List[torch.Tensor] = []
    for t_value in times.reshape(-1):
        t_line = torch.full(
            (cfg.mass_quad_points, 1),
            t_value,
            dtype=torch.float32,
            device=device,
        )
        p = density_func(x.view(-1, 1), t_line).reshape(-1)
        interior = torch.sum(p * weights)
        tail = (cfg.domain_bound / alpha) * (p[0] + p[-1])
        masses.append(interior + tail)
    return torch.stack(masses)


# =============================================================================
# Training
# =============================================================================
def train_score_network(
    score_net: ScoreNetwork,
    score_data: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> List[float]:
    x_train = torch.tensor(
        score_data[:, 0:1], dtype=torch.float32, device=device
    )
    t_train = torch.tensor(
        score_data[:, 1:2], dtype=torch.float32, device=device
    )
    optimizer = Adam(score_net.parameters(), lr=cfg.score_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.score_epochs, eta_min=cfg.eta_min
    )

    losses: List[float] = []
    pbar = tqdm(range(cfg.score_epochs), desc="Stage I score", leave=False)
    for epoch in pbar:
        idx = torch.randint(
            0, len(score_data), (cfg.batch_size,), device=device
        )
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx]
        score = score_net(x, t)
        score_x = torch.autograd.grad(
            score.sum(), x, create_graph=True
        )[0]
        loss_ssm = 0.5 * score.pow(2).mean() + score_x.mean()

        # Exact initial score in every iteration.
        n_gaussian = int(0.8 * cfg.ic_batch_size)
        x0 = torch.cat([
            torch.randn(n_gaussian, 1, device=device),
            (
                2.0
                * torch.rand(
                    cfg.ic_batch_size - n_gaussian, 1, device=device
                )
                - 1.0
            )
            * cfg.domain_bound,
        ])
        t0 = torch.zeros_like(x0)
        loss_ic = (score_net(x0, t0) + x0).pow(2).mean()

        # The symmetric fOU score is odd: s(-x,t)=-s(x,t).
        n_sym = min(128, len(x))
        score_reflected = score_net(-x[:n_sym], t[:n_sym])
        loss_symmetry = (
            score_reflected + score[:n_sym]
        ).pow(2).mean()

        loss = (
            loss_ssm
            + cfg.score_ic_weight * loss_ic
            + cfg.score_symmetry_weight * loss_symmetry
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage-I objective at iteration {epoch + 1}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if (epoch + 1) % 250 == 0:
            pbar.set_postfix(objective=f"{losses[-1]:.3e}")
    return losses


def train_ll_network(
    ll_net: LLNetwork,
    score_net: ScoreNetwork,
    ll_data: np.ndarray,
    process: FractionalOUProcess,
    alpha: float,
    cfg: Config,
    device: torch.device,
) -> List[float]:
    x_train = torch.tensor(
        ll_data[:, 0:1], dtype=torch.float32, device=device
    )
    t_train = torch.tensor(
        ll_data[:, 1:2], dtype=torch.float32, device=device
    )
    score_net.eval()

    def density_func(
        x_value: torch.Tensor, t_value: torch.Tensor
    ) -> torch.Tensor:
        return safe_exp(ll_net(x_value, t_value))

    def mass_loss() -> torch.Tensor:
        times = torch.linspace(
            cfg.t_final / cfg.mass_time_samples,
            cfg.t_final,
            cfg.mass_time_samples,
            dtype=torch.float32,
            device=device,
        )
        masses = global_mass_torch(
            density_func, times, alpha, cfg, device
        )
        return torch.log(torch.clamp(masses, min=1.0e-8)).pow(2).mean()

    def symmetry_loss(
        x: torch.Tensor, t: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        n_sym = min(128, len(x))
        return (
            ll_net(-x[:n_sym], t[:n_sym]) - q[:n_sym]
        ).pow(2).mean()

    losses: List[float] = []

    # Stage II-A: score-consistency and mass-scale warm-up.
    optimizer = Adam(ll_net.parameters(), lr=cfg.ll_warmup_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.ll_warmup_epochs, 1), eta_min=cfg.eta_min
    )
    pbar = tqdm(
        range(cfg.ll_warmup_epochs),
        desc="Stage II consistency warm-up",
        leave=False,
    )
    for epoch in pbar:
        idx = torch.randint(
            0, len(ll_data), (cfg.batch_size,), device=device
        )
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx]
        q = ll_net(x, t)
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        with torch.no_grad():
            target = score_net(x, t)
        loss_score = (q_x - target).pow(2).mean()
        loss_mass = mass_loss()
        loss_sym = symmetry_loss(x, t, q)
        loss = (
            cfg.score_consistency_weight * loss_score
            + 0.25 * cfg.mass_loss_weight * loss_mass
            + cfg.q_symmetry_weight * loss_sym
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage-II warm-up loss at iteration {epoch + 1}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    # Stage II-B: gradual nonlocal physics refinement.
    optimizer = Adam(ll_net.parameters(), lr=cfg.ll_physics_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.ll_physics_epochs, 1), eta_min=cfg.eta_min
    )
    pbar = tqdm(
        range(cfg.ll_physics_epochs),
        desc="Stage II physics refinement",
        leave=False,
    )
    for epoch in pbar:
        idx = torch.randint(
            0, len(ll_data), (cfg.batch_size,), device=device
        )
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx].clone().detach().requires_grad_(True)

        q = ll_net(x, t)
        p = safe_exp(q)
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        q_t = torch.autograd.grad(q.sum(), t, create_graph=True)[0]
        with torch.no_grad():
            score_target = score_net(x, t)

        pde_weight = min(
            1.0, (epoch + 1) / max(cfg.pde_ramp_epochs, 1)
        )
        q_fraction = min(
            1.0, (epoch + 1) / max(cfg.score_to_q_ramp_epochs, 1)
        )
        score_for_pde = (
            (1.0 - q_fraction) * score_target + q_fraction * q_x
        )

        drift = -process.theta * x
        div_drift = -process.theta
        local = p * (
            q_t + div_drift + drift * score_for_pde
        )
        diffusion = (
            process.sigma ** alpha
            * fractional_derivative_gl(
                density_func, x, t, alpha, cfg
            )
        )
        residual = local - diffusion
        loss_residual = residual.pow(2).mean()
        loss_score = (q_x - score_target).pow(2).mean()
        loss_mass = mass_loss()
        loss_sym = symmetry_loss(x, t, q)

        score_weight = (
            cfg.score_consistency_weight * (1.0 - 0.75 * q_fraction)
        )
        mass_weight = (
            cfg.mass_loss_weight * min(1.0, 0.25 + 0.75 * pde_weight)
        )
        loss = (
            pde_weight * loss_residual
            + score_weight * loss_score
            + mass_weight * loss_mass
            + cfg.q_symmetry_weight * loss_sym
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage-II physics loss at iteration {epoch + 1}."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if (epoch + 1) % 250 == 0:
            pbar.set_postfix(
                loss=f"{losses[-1]:.3e}",
                pde=f"{pde_weight:.2f}",
            )
    return losses


# =============================================================================
# Exact density and metrics
# =============================================================================
def exact_fou_pdf(
    x_grid: np.ndarray,
    t: float,
    alpha: float,
    cfg: Config,
    chunk_size: int = 256,
) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if t <= 1.0e-14:
        return np.exp(-0.5 * x_grid ** 2) / math.sqrt(2.0 * math.pi)

    a_t = (
        t
        if abs(cfg.theta) < 1.0e-14
        else (
            1.0 - math.exp(-alpha * cfg.theta * t)
        ) / (alpha * cfg.theta)
    )
    gaussian_variance = math.exp(-2.0 * cfg.theta * t)
    levy_coefficient = cfg.sigma ** alpha * a_t

    k = np.linspace(
        0.0, cfg.exact_k_max, cfg.exact_k_num, dtype=np.float64
    )
    decay = np.exp(
        -0.5 * gaussian_variance * k ** 2
        - levy_coefficient * k ** alpha
    )
    pdf = np.empty_like(x_grid)
    for start in range(0, len(x_grid), chunk_size):
        block = x_grid[start:start + chunk_size, None]
        integrand = np.cos(block * k[None, :]) * decay[None, :]
        pdf[start:start + chunk_size] = (
            np_trapz(integrand, k, axis=1) / math.pi
        )
    return np.maximum(pdf, 0.0)


def normalize_density(
    density: np.ndarray, x_grid: np.ndarray
) -> np.ndarray:
    positive = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    mass = float(simpson(positive, x=x_grid))
    if not np.isfinite(mass) or mass <= 1.0e-14:
        raise RuntimeError("Invalid density mass.")
    return positive / mass


def evaluate_density(
    ll_net: LLNetwork,
    x_grid: np.ndarray,
    t: float,
    device: torch.device,
) -> np.ndarray:
    x_tensor = torch.tensor(
        x_grid[:, None], dtype=torch.float32, device=device
    )
    t_tensor = torch.full_like(x_tensor, float(t))
    ll_net.eval()
    with torch.no_grad():
        raw = safe_exp(ll_net(x_tensor, t_tensor)).cpu().numpy().reshape(-1)
    return normalize_density(raw, x_grid)


def error_metrics(
    prediction: np.ndarray,
    exact: np.ndarray,
    x_grid: np.ndarray,
) -> Tuple[float, float, float]:
    pred = normalize_density(prediction, x_grid)
    ref = normalize_density(exact, x_grid)
    difference = pred - ref
    l2 = float(math.sqrt(simpson(difference ** 2, x=x_grid)))
    linf = float(np.max(np.abs(difference)))
    w = float(
        wasserstein_distance(
            x_grid, x_grid, u_weights=pred, v_weights=ref
        )
    )
    return l2, linf, w


# =============================================================================
# Figure
# =============================================================================
def plot_results(
    alpha: float,
    cfg: Config,
    x_grid: np.ndarray,
    rows: Sequence[Dict[str, float]],
    densities: Dict[float, Tuple[np.ndarray, np.ndarray]],
    score_losses: Sequence[float],
    ll_losses: Sequence[float],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(21.0, 12.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 3, left=0.065, right=0.985, bottom=0.085, top=0.965,
        wspace=0.25, hspace=0.28
    )

    for index, row in enumerate(rows):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        t = float(row["time"])
        exact, prediction = densities[t]

        # Draw the red prediction first and place a dashed exact curve on top.
        prediction_line, = ax.plot(
            x_grid,
            prediction,
            color="red",
            linestyle="-",
            linewidth=2.7,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markevery=70,
            label="Score-based PINN",
            zorder=2,
        )
        exact_line, = ax.plot(
            x_grid,
            exact,
            color="black",
            linestyle=(0, (7, 4)),
            linewidth=3.0,
            label="Exact",
            zorder=3,
        )

        ax.set_xlim(-cfg.domain_bound, cfg.domain_bound)
        ax.set_ylim(
            0.0,
            1.12 * max(float(exact.max()), float(prediction.max())),
        )
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel("Probability density")
        ax.set_title(rf"$t={t:.2f}$", pad=8)
        ax.grid(True, linestyle="--", alpha=0.24)
        ax.tick_params(direction="out", length=5.5, width=1.0)
        ax.text(
            0.035,
            0.95,
            rf"$W={float(row['Wasserstein']):.4f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=18,
            bbox=dict(
                facecolor="white",
                edgecolor="0.72",
                alpha=0.92,
                boxstyle="round,pad=0.28",
            ),
        )
        if index == 0:
            # Exact first in the legend even though it is drawn second.
            ax.legend(
                handles=[exact_line, prediction_line],
                loc="upper right",
                frameon=True,
            )

    skip = min(100, max(0, len(score_losses) // 50))
    ax_score = fig.add_subplot(grid[0, 2])
    score_array = np.asarray(score_losses[skip:], dtype=np.float64)
    score_smooth = moving_average(
        score_array, min(cfg.loss_smooth_window, max(1, len(score_array)))
    )
    score_x = np.arange(skip, len(score_losses))[-len(score_smooth):]
    ax_score.plot(score_x, score_smooth, linewidth=2.5)
    ax_score.set_title("Stage I score-matching objective", pad=8)
    ax_score.set_xlabel("Iteration")
    ax_score.set_ylabel("Objective")
    ax_score.grid(True, linestyle="--", alpha=0.28)
    ax_score.tick_params(direction="out", length=5.5, width=1.0)

    ax_ll = fig.add_subplot(grid[1, 2])
    ll_array = np.asarray(ll_losses[skip:], dtype=np.float64)
    ll_smooth = moving_average(
        ll_array, min(cfg.loss_smooth_window, max(1, len(ll_array)))
    )
    ll_x = np.arange(skip, len(ll_losses))[-len(ll_smooth):]
    ax_ll.plot(ll_x, np.maximum(ll_smooth, 1.0e-12), linewidth=2.5)
    ax_ll.axvline(
        cfg.ll_warmup_epochs,
        color="0.25",
        linestyle=":",
        linewidth=2.0,
        label="Physics refinement begins",
    )
    ax_ll.set_yscale("log")
    ax_ll.set_title("Stage II log-density loss", pad=8)
    ax_ll.set_xlabel("Iteration")
    ax_ll.set_ylabel("Loss")
    ax_ll.grid(True, linestyle="--", alpha=0.28)
    ax_ll.tick_params(direction="out", length=5.5, width=1.0)
    ax_ll.legend(loc="best", frameon=True)

    tag = alpha_tag(alpha)
    fig.savefig(
        output_dir / f"linear_fou_exact_alpha_{tag}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        output_dir / f"linear_fou_exact_alpha_{tag}.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


# =============================================================================
# Experiment
# =============================================================================
def run_experiment(
    alpha: float,
    cfg: Config,
    device: torch.device,
    make_figure: bool = True,
) -> List[Dict[str, float]]:
    # Reset before every alpha. This removes the isolated alpha=1.7 RNG-state run.
    set_seed(cfg.seed)
    print(f"\n========== Stable linear fOU experiment: alpha={alpha} ==========")

    process = FractionalOUProcess(cfg, alpha)
    print("Generating separate Stage-I and Stage-II datasets...")
    score_data, ll_data = generate_stage_datasets(process, cfg)

    score_net = ScoreNetwork(cfg).to(device)
    ll_net = LLNetwork(cfg).to(device)

    print("Training Stage I score network...")
    score_losses = train_score_network(score_net, score_data, cfg, device)
    score_net.eval()

    print("Training Stage II log-density network...")
    ll_losses = train_ll_network(
        ll_net, score_net, ll_data, process, alpha, cfg, device
    )
    ll_net.eval()

    x_grid = np.linspace(
        -cfg.domain_bound, cfg.domain_bound, cfg.eval_grid_n
    )
    rows: List[Dict[str, float]] = []
    densities: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    print("Evaluating against the exact transient density...")
    for t in cfg.eval_times:
        exact = normalize_density(
            exact_fou_pdf(x_grid, float(t), alpha, cfg), x_grid
        )
        prediction = evaluate_density(
            ll_net, x_grid, float(t), device
        )
        l2, linf, w = error_metrics(prediction, exact, x_grid)
        row = {
            "alpha": float(alpha),
            "time": float(t),
            "L2_error": l2,
            "Linf_error": linf,
            "Wasserstein": w,
        }
        rows.append(row)
        densities[float(t)] = (exact, prediction)
        print(
            f"t={float(t):.2f} | L2={l2:.4e}, "
            f"Linf={linf:.4e}, W={w:.4e}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = alpha_tag(alpha)

    with (output_dir / f"linear_fou_exact_metrics_alpha_{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if make_figure:
        plot_results(
            alpha,
            cfg,
            x_grid,
            rows,
            densities,
            score_losses,
            ll_losses,
            output_dir,
        )

    if cfg.save_model:
        torch.save(
            {
                "alpha": float(alpha),
                "config": asdict(cfg),
                "score_state_dict": score_net.state_dict(),
                "ll_state_dict": ll_net.state_dict(),
                "score_losses": list(score_losses),
                "ll_losses": list(ll_losses),
                "metrics": rows,
            },
            output_dir / f"linear_fou_exact_model_alpha_{tag}.pt",
        )
    return rows


def parse_alphas(value: str) -> List[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas or any(not (1.0 < alpha < 2.0) for alpha in alphas):
        raise argparse.ArgumentTypeError("All alpha values must lie in (1,2).")
    return alphas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="1.5,1.6,1.7,1.8")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output_dir", default="results_linear_fou_exact_stable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_epochs", type=int, default=5000)
    parser.add_argument("--ll_warmup_epochs", type=int, default=1000)
    parser.add_argument("--ll_physics_epochs", type=int, default=4000)
    parser.add_argument("--train_trajectories", type=int, default=800)
    parser.add_argument("--score_train_points", type=int, default=60000)
    parser.add_argument("--ll_train_points", type=int, default=60000)
    parser.add_argument("--tail_ratio", type=float, default=0.40)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        score_epochs=args.score_epochs,
        ll_warmup_epochs=args.ll_warmup_epochs,
        ll_physics_epochs=args.ll_physics_epochs,
        train_trajectories=args.train_trajectories,
        score_train_points=args.score_train_points,
        ll_train_points=args.ll_train_points,
        tail_ratio=args.tail_ratio,
        save_model=not args.no_save_model,
    )

    if args.quick:
        cfg.train_trajectories = 60
        cfg.score_train_points = 1000
        cfg.ll_train_points = 1000
        cfg.hidden_dim = 24
        cfg.num_layers = 3
        cfg.batch_size = 16
        cfg.ic_batch_size = 32
        cfg.score_epochs = 2
        cfg.ll_warmup_epochs = 1
        cfg.ll_physics_epochs = 1
        cfg.pde_ramp_epochs = 1
        cfg.score_to_q_ramp_epochs = 1
        cfg.gl_terms = 5
        cfg.mass_quad_points = 21
        cfg.mass_time_samples = 1
        cfg.eval_grid_n = 201
        cfg.exact_k_num = 1000
        cfg.loss_smooth_window = 2
        cfg.output_dir += "_quick"

    device = get_device(cfg.device)
    print(f"Using device: {device}")

    all_rows: List[Dict[str, float]] = []
    for alpha in parse_alphas(args.alphas):
        all_rows.extend(run_experiment(alpha, cfg, device, make_figure=True))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "linear_fou_exact_metrics_all_alpha.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nAll stable one-dimensional fOU experiments finished.")


if __name__ == "__main__":
    main()
