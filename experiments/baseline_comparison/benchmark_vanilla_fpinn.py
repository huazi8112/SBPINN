"""
Corrected Fig. 2 generator for the one-dimensional fractional OU benchmark.

This script uses exactly the same corrected two-stage training strategy as
``generate_111_revised_no_score_integration.py``:

    Stage I  : score learning from unbiased trajectory samples;
    Stage II : score-consistency warm-up followed by gradual FFP-residual refinement.

No score-integration initialization is used.  The Vanilla fPINN and the proposed
method solve the same fractional Fokker--Planck equation with the corrected OU
drift sign and the same shifted GL approximation.  The linear fOU exact density
is computed by characteristic-function inversion and is used as the common
reference for every panel.

The generated Fig. 2 contains:
    (a)--(c) exact/Ours/Vanilla density comparisons at t=0.25, 0.50, 0.75;
    (d) Wasserstein distance over physical time;
    (e) negative probability mass over physical time;
    (f) raw global probability-mass deviation over physical time.

Both Wasserstein distances use positive normalized shapes on the common interval.
Mass and negativity metrics are computed from the unnormalized raw outputs.  The
same alpha-stable asymptotic tail completion is used for both neural solvers.

Examples
--------
Train from scratch and generate the figure:
    python benchmark_vanilla_fpinn.py --device cpu

Reuse the exact checkpoints produced by the revised Table-2 script:
    python benchmark_vanilla_fpinn.py \
        --checkpoint results_table2_no_score_integration/trained_models.pt

A quick smoke test (not for publication):
    python benchmark_vanilla_fpinn.py --quick_test
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.integrate import simpson
from scipy.stats import levy_stable, wasserstein_distance
from torch.optim import Adam
from tqdm import tqdm

try:
    from numpy import trapezoid as np_trapz
except ImportError:  # NumPy < 2.0
    from numpy import trapz as np_trapz


# =============================================================================
# 配置
# =============================================================================
@dataclass
class Config:
    alpha: float = 1.5
    theta: float = 0.5
    sigma: float = 1.0
    t_final: float = 1.0
    n_sde_steps: int = 100

    seed: int = 42
    device: str = "auto"
    output_dir: str = "results_table2_fem_revised"

    # 训练数据
    train_trajectories: int = 800
    train_points: int = 50000
    tail_ratio: float = 0.30
    trajectory_truncate_bound: float = 6.0

    # 网络/训练
    hidden_dim: int = 128
    num_layers: int = 4
    batch_size: int = 256
    lr: float = 1e-3
    score_lr: float = 5e-4
    ll_warmup_lr: float = 1e-3
    ll_lr: float = 2e-4
    eta_min: float = 1e-5
    vanilla_epochs: int = 3000
    score_epochs: int = 3000
    ll_epochs: int = 2000
    eval_every: int = 50
    target_w: float = 0.1

    # 神经网络 GL 设置
    gl_dx: float = 0.08
    gl_terms: int = 80
    network_bound: float = 6.0

    # Stage II 损失权重
    score_loss_weight: float = 1.0
    ll_ic_weight: float = 0.0  # LLNetwork 使用硬初值，该项仅保留兼容性
    score_ic_weight: float = 10.0
    mass_loss_weight: float = 1.0
    ic_batch_size: int = 512
    # Stage II-A 只使用论文原有的 score consistency + mass losses 进行预热，
    # 不构造任何 score 积分密度标签。
    ll_warmup_epochs: int = 800
    ll_pde_ramp_epochs: int = 1500
    min_physics_epochs_before_stop: int = 50
    mass_quad_points: int = 161
    mass_time_samples: int = 3

    # 评价网格；W 只比较公共核心区间上的形状
    eval_bound: float = 6.0
    eval_grid_n: int = 1201
    exact_k_max: float = 100.0
    exact_k_num: int = 16000

    # P1 FEM：扩大区间和 h-dt 细化序列
    fem_bound: float = 12.0
    fem_refinements: Tuple[Tuple[int, float], ...] = (
        (50, 0.0200),
        (100, 0.0100),
        (200, 0.0050),
        (300, 0.0025),
    )

    # 是否把生成 Stage I/II 所必需的训练数据时间计入端到端总时间
    include_data_generation_time: bool = True


@dataclass
class EvalMetrics:
    wasserstein: float
    raw_mass: float
    mass_deviation: float
    negative_mass: float
    l2_rel: float


@dataclass
class MethodResult:
    method: str
    converged: bool
    work: str
    average_step_ms: float
    total_time_s: float
    metrics: EvalMetrics
    stage1_iterations: int = 0
    stage2_iterations: int = 0
    data_time_s: float = 0.0
    stage1_time_s: float = 0.0
    stage2_time_s: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 通用工具
# =============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 未检测到可用 GPU。")
    return requested


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_exp(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(q, min=-60.0, max=20.0))


def trapezoid_weights(n: int, a: float, b: float, device: torch.device) -> torch.Tensor:
    if n < 2:
        raise ValueError("积分点数至少为 2。")
    h = (b - a) / (n - 1)
    w = torch.ones(n, dtype=torch.float32, device=device) * h
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


# =============================================================================
# 网络
# =============================================================================
class MLP(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class ScoreNetwork(MLP):
    """通用一维 score 网络；不在网络结构中使用 fOU 对称性。"""
    def __init__(self, cfg: Config):
        super().__init__(1, cfg.hidden_dim, cfg.num_layers, dropout=0.0)


class LLNetwork(nn.Module):
    """对数密度网络，使用 q(x,t)=q0(x)+t*N(x,t) 精确满足初值。"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.correction = MLP(1, cfg.hidden_dim, cfg.num_layers, dropout=0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q0 = -0.5 * x.pow(2) - 0.5 * math.log(2.0 * math.pi)
        return q0 + t * self.correction(x, t)


class VanillaNetwork(MLP):
    def __init__(self, cfg: Config):
        super().__init__(1, cfg.hidden_dim, cfg.num_layers, dropout=0.0)


# =============================================================================
# fractional OU 轨迹和训练数据
# =============================================================================
class FractionalOUProcess:
    def __init__(self, cfg: Config):
        self.theta = cfg.theta
        self.sigma = cfg.sigma
        self.alpha = cfg.alpha

    def drift(self, x):
        return -self.theta * x

    def simulate_trajectory(self, x0: np.ndarray, t_final: float, n_steps: int) -> np.ndarray:
        n_particles = len(x0)
        dt = t_final / n_steps
        trajectories = np.zeros((n_particles, n_steps + 1), dtype=np.float64)
        trajectories[:, 0] = x0
        scale = dt ** (1.0 / self.alpha)
        increments = levy_stable.rvs(
            alpha=self.alpha,
            beta=0,
            loc=0,
            scale=scale,
            size=(n_particles, n_steps),
        )
        for n in range(n_steps):
            x = trajectories[:, n]
            trajectories[:, n + 1] = (
                x - self.theta * x * dt + self.sigma * increments[:, n]
            )
        return trajectories


def generate_training_data(
    process: FractionalOUProcess, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """分别生成 Stage-I 真实分布样本和 Stage-II 混合配点。

    score_data 保留轨迹经验分布，不进行尾部重加权，否则 SSM 会学习到
    重加权分布的 score。collocation_data 才使用 core/tail 混合采样。
    """
    x0 = np.random.normal(0.0, 1.0, cfg.train_trajectories)
    trajectories = process.simulate_trajectory(x0, cfg.t_final, cfg.n_sde_steps)
    dt = cfg.t_final / cfg.n_sde_steps

    blocks: List[List[float]] = []
    for m in range(cfg.train_trajectories):
        for n in range(cfg.n_sde_steps + 1):
            x = trajectories[m, n]
            if abs(x) <= cfg.trajectory_truncate_bound:
                blocks.append([x, n * dt])

    all_points = np.asarray(blocks, dtype=np.float64)
    if all_points.ndim != 2 or len(all_points) < cfg.batch_size:
        raise RuntimeError("生成的训练点过少，请增加 train_trajectories 或扩大截断区间。")

    # Stage I：从真实经验分布均匀抽样，不改变 core/tail 比例。
    # 该模型及初值关于原点对称，加入镜像样本消除有限 Monte Carlo 偏斜。
    mirrored = all_points.copy()
    mirrored[:, 0] *= -1.0
    score_pool = np.vstack([all_points, mirrored])
    score_idx = np.random.choice(
        len(score_pool), cfg.train_points, replace=len(score_pool) < cfg.train_points
    )
    score_data = score_pool[score_idx].copy()
    np.random.shuffle(score_data)

    # Stage II/Vanilla：采用重要性混合配点，加强尾部 PDE 约束。
    abs_x = np.abs(all_points[:, 0])
    threshold = np.percentile(abs_x, 70.0)
    core = all_points[abs_x <= threshold]
    tail = all_points[abs_x > threshold]

    n_tail = int(cfg.train_points * cfg.tail_ratio)
    n_core = cfg.train_points - n_tail
    core_idx = np.random.choice(len(core), n_core, replace=len(core) < n_core)
    if len(tail) == 0:
        tail_points = np.empty((0, 2), dtype=np.float64)
    else:
        tail_idx = np.random.choice(len(tail), n_tail, replace=len(tail) < n_tail)
        tail_points = tail[tail_idx]

    collocation_data = np.vstack([core[core_idx], tail_points])
    np.random.shuffle(collocation_data)
    return score_data.astype(np.float32), collocation_data.astype(np.float32)


# =============================================================================
# 神经网络中的 GL Riesz 导数和质量公式
# =============================================================================
def gl_coefficients(alpha: float, n_terms: int, device: torch.device) -> torch.Tensor:
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append(coeffs[-1] * (k - 1.0 - alpha) / k)
    return torch.tensor(coeffs, dtype=torch.float32, device=device)


def fractional_derivative_gl_with_boundary(
    density_func,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    dx: float,
    domain_bound: float,
    n_terms: int,
) -> torch.Tensor:
    """对称 Riesz 导数的 GL 近似，域外使用 alpha-stable 幂律尾部。"""
    device = x.device
    batch = x.shape[0]
    coeffs = gl_coefficients(alpha, n_terms, device).view(1, n_terms)
    # 一阶 shifted GL：第 k 项取 x-(k-1)h 与 x+(k-1)h。
    # 相比未移位格式，它对 alpha in (1,2) 的 Riesz 算子更稳定且误差更小。
    shifts = ((torch.arange(n_terms, dtype=x.dtype, device=device) - 1.0) * dx).view(1, n_terms)

    x_left = x - shifts
    x_right = x + shifts
    t_rep = t.repeat(1, n_terms)

    def evaluate(points: torch.Tensor) -> torch.Tensor:
        flat_x = points.reshape(-1, 1)
        flat_t = t_rep.reshape(-1, 1)
        inside = (flat_x.abs() <= domain_bound).view(-1)
        values = torch.zeros_like(flat_x)
        if inside.any():
            values[inside] = density_func(flat_x[inside], flat_t[inside])
        outside = ~inside
        if outside.any():
            x_out = flat_x[outside]
            x_anchor = domain_bound * torch.sign(x_out)
            p_anchor = density_func(x_anchor, flat_t[outside])
            decay = (x_out.abs() / domain_bound).pow(-(1.0 + alpha))
            values[outside] = p_anchor * decay
        return values.view(batch, n_terms)

    left = evaluate(x_left)
    right = evaluate(x_right)
    weighted = torch.sum(coeffs * (left + right), dim=1, keepdim=True)
    cos_value = math.cos(alpha * math.pi / 2.0)
    factor = -1.0 / (2.0 * cos_value * dx ** alpha)
    return factor * weighted


def global_mass_from_network_torch(
    density_func,
    times: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    """按论文渐近尾部公式，估计每个时刻在 R 上的总质量；保持可微。"""
    n_t = times.shape[0]
    x_line = torch.linspace(
        -cfg.network_bound,
        cfg.network_bound,
        cfg.mass_quad_points,
        dtype=torch.float32,
        device=device,
    )
    weights = trapezoid_weights(
        cfg.mass_quad_points, -cfg.network_bound, cfg.network_bound, device
    )
    x_batch = x_line.repeat(n_t).view(-1, 1)
    t_batch = times.repeat_interleave(cfg.mass_quad_points, dim=0)
    p = density_func(x_batch, t_batch).view(n_t, cfg.mass_quad_points)
    interior_mass = torch.sum(p * weights.view(1, -1), dim=1)
    tail_mass = (cfg.network_bound / cfg.alpha) * (p[:, 0] + p[:, -1])
    return interior_mass + tail_mass


def evaluate_network_raw(
    net: nn.Module,
    x_grid: np.ndarray,
    t_value: float,
    is_log_density: bool,
    device: torch.device,
) -> np.ndarray:
    net.eval()
    x_t = torch.tensor(x_grid[:, None], dtype=torch.float32, device=device)
    t_t = torch.full_like(x_t, float(t_value))
    with torch.no_grad():
        out = net(x_t, t_t)
        p = safe_exp(out) if is_log_density else out
    return p.cpu().numpy().reshape(-1).astype(np.float64)


def asymptotic_global_mass_from_grid(
    p_inside: np.ndarray, x_grid: np.ndarray, alpha: float
) -> float:
    """在 [-B,B] 内积分并用两侧幂律尾部补全到无穷远。"""
    bound = float(max(abs(x_grid[0]), abs(x_grid[-1])))
    interior = float(simpson(p_inside, x=x_grid))
    tail = (bound / alpha) * float(p_inside[0] + p_inside[-1])
    return interior + tail


# =============================================================================
# 精确 fractional OU 参考解与统一评价
# =============================================================================
def exact_fou_pdf(
    x_grid: np.ndarray,
    t: float,
    cfg: Config,
    chunk_size: int = 256,
) -> np.ndarray:
    """通过特征函数 Fourier 反演得到线性 fOU 的精确密度。"""
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if t <= 1e-14:
        return np.exp(-0.5 * x_grid ** 2) / math.sqrt(2.0 * math.pi)

    if abs(cfg.theta) < 1e-14:
        a_t = t
    else:
        a_t = (1.0 - math.exp(-cfg.alpha * cfg.theta * t)) / (
            cfg.alpha * cfg.theta
        )
    gaussian_variance = math.exp(-2.0 * cfg.theta * t)
    levy_coefficient = (cfg.sigma ** cfg.alpha) * a_t

    k = np.linspace(0.0, cfg.exact_k_max, cfg.exact_k_num, dtype=np.float64)
    decay = np.exp(
        -0.5 * gaussian_variance * k ** 2
        - levy_coefficient * k ** cfg.alpha
    )
    pdf = np.empty_like(x_grid)
    for start in range(0, len(x_grid), chunk_size):
        block = x_grid[start : start + chunk_size, None]
        integrand = np.cos(block * k[None, :]) * decay[None, :]
        pdf[start : start + chunk_size] = np_trapz(integrand, k, axis=1) / math.pi
    return np.maximum(pdf, 0.0)


def normalize_positive_density(p: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    p_pos = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    mass = float(simpson(p_pos, x=x_grid))
    if not np.isfinite(mass) or mass <= 1e-14:
        return np.ones_like(p_pos) / len(p_pos)
    return p_pos / mass


def compute_metrics(
    p_raw: np.ndarray,
    p_exact: np.ndarray,
    x_grid: np.ndarray,
    raw_global_mass: float,
) -> EvalMetrics:
    p_shape = normalize_positive_density(p_raw, x_grid)
    exact_shape = normalize_positive_density(p_exact, x_grid)
    w = wasserstein_distance(
        x_grid,
        x_grid,
        u_weights=p_shape,
        v_weights=exact_shape,
    )
    l2_num = float(simpson((p_shape - exact_shape) ** 2, x=x_grid))
    l2_den = float(simpson(exact_shape ** 2, x=x_grid))
    negative_mass = float(simpson(np.maximum(-p_raw, 0.0), x=x_grid))
    return EvalMetrics(
        wasserstein=float(w),
        raw_mass=float(raw_global_mass),
        mass_deviation=float(abs(raw_global_mass - 1.0)),
        negative_mass=negative_mass,
        l2_rel=math.sqrt(max(l2_num, 0.0) / max(l2_den, 1e-30)),
    )


def evaluate_network(
    net: nn.Module,
    is_log_density: bool,
    cfg: Config,
    device: torch.device,
    x_eval: np.ndarray,
    p_exact_eval: np.ndarray,
) -> EvalMetrics:
    p_raw = evaluate_network_raw(net, x_eval, cfg.t_final, is_log_density, device)
    global_mass = asymptotic_global_mass_from_grid(p_raw, x_eval, cfg.alpha)
    return compute_metrics(p_raw, p_exact_eval, x_eval, global_mass)


# =============================================================================
# 网络训练；首次达标状态立即停止并用于表格
# =============================================================================
def train_vanilla_network(
    net: VanillaNetwork,
    process: FractionalOUProcess,
    training_data: np.ndarray,
    cfg: Config,
    device: torch.device,
    x_eval: np.ndarray,
    p_exact_eval: np.ndarray,
    data_time_s: float,
) -> MethodResult:
    optimizer = Adam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.vanilla_epochs, eta_min=cfg.eta_min
    )
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32, device=device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32, device=device)

    epoch_times: List[float] = []
    best_metrics: Optional[EvalMetrics] = None
    converged_epoch = -1
    sync_device(device)
    train_start = time.perf_counter()

    pbar = tqdm(range(cfg.vanilla_epochs), desc="Vanilla fPINN", leave=False)
    for epoch in pbar:
        sync_device(device)
        epoch_start = time.perf_counter()
        idx = torch.randperm(len(training_data), device=device)[: cfg.batch_size]
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx].clone().detach().requires_grad_(True)

        p = net(x, t)
        p_t = torch.autograd.grad(p.sum(), t, create_graph=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]

        # 正确符号：-d(f p)/dx，f=-theta*x，因此为 theta*p + theta*x*p_x。
        drift = process.theta * p + process.theta * x * p_x
        diffusion = (process.sigma ** cfg.alpha) * fractional_derivative_gl_with_boundary(
            net,
            x,
            t,
            cfg.alpha,
            cfg.gl_dx,
            cfg.network_bound,
            cfg.gl_terms,
        )
        loss_res = ((p_t - drift - diffusion) ** 2).mean()

        t0_mask = t[:, 0] < 0.05
        if t0_mask.any():
            p0 = torch.exp(-0.5 * x[t0_mask] ** 2) / math.sqrt(2.0 * math.pi)
            loss_ic = ((p[t0_mask] - p0) ** 2).mean()
        else:
            loss_ic = torch.zeros((), device=device)

        loss_pos = 10.0 * torch.relu(-p).pow(2).mean()
        x_bc = torch.tensor(
            [[-cfg.network_bound], [cfg.network_bound]],
            dtype=torch.float32,
            device=device,
        )
        t_bc = torch.rand((2, 1), dtype=torch.float32, device=device) * cfg.t_final
        loss_bc = net(x_bc, t_bc).pow(2).mean()
        loss = loss_res + 20.0 * loss_ic + loss_bc + loss_pos

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        sync_device(device)
        epoch_times.append(time.perf_counter() - epoch_start)

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate_network(net, False, cfg, device, x_eval, p_exact_eval)
            pbar.set_postfix(W=f"{metrics.wasserstein:.3e}", M=f"{metrics.raw_mass:.4f}")
            best_metrics = metrics
            if metrics.wasserstein < cfg.target_w:
                converged_epoch = epoch + 1
                break

    sync_device(device)
    train_time = time.perf_counter() - train_start
    if best_metrics is None or converged_epoch < 0:
        best_metrics = evaluate_network(net, False, cfg, device, x_eval, p_exact_eval)

    total_time = train_time + (data_time_s if cfg.include_data_generation_time else 0.0)
    used_epochs = converged_epoch if converged_epoch > 0 else len(epoch_times)
    return MethodResult(
        method="Vanilla fPINN",
        converged=converged_epoch > 0,
        work=str(converged_epoch) if converged_epoch > 0 else "未收敛",
        average_step_ms=1000.0 * float(np.mean(epoch_times)),
        total_time_s=total_time if converged_epoch > 0 else -1.0,
        metrics=best_metrics,
        stage2_iterations=used_epochs,
        data_time_s=data_time_s,
        stage2_time_s=train_time,
    )


def train_score_network(
    net: ScoreNetwork,
    training_data: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> Tuple[float, List[float]]:
    optimizer = Adam(net.parameters(), lr=cfg.score_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.score_epochs, eta_min=cfg.eta_min
    )
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32, device=device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32, device=device)
    epoch_times: List[float] = []

    sync_device(device)
    start = time.perf_counter()
    pbar = tqdm(range(cfg.score_epochs), desc="Stage I score", leave=False)
    for _ in pbar:
        sync_device(device)
        epoch_start = time.perf_counter()
        idx = torch.randperm(len(training_data), device=device)[: cfg.batch_size]
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx]
        score = net(x, t)
        score_x = torch.autograd.grad(score.sum(), x, create_graph=True)[0]
        loss_ssm = 0.5 * score.pow(2).mean() + score_x.mean()

        # 每个 epoch 独立采样精确 t=0 点，避免随机批次缺少初值样本。
        n_ic = cfg.ic_batch_size
        n_gauss = int(0.8 * n_ic)
        x0_gauss = torch.randn((n_gauss, 1), dtype=torch.float32, device=device)
        x0_uniform = (2.0 * torch.rand((n_ic - n_gauss, 1), dtype=torch.float32, device=device) - 1.0) * cfg.network_bound
        x0 = torch.cat([x0_gauss, x0_uniform], dim=0)
        t0 = torch.zeros_like(x0)
        score0 = net(x0, t0)
        loss_ic = (score0 + x0).pow(2).mean()
        loss = loss_ssm + cfg.score_ic_weight * loss_ic

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        sync_device(device)
        epoch_times.append(time.perf_counter() - epoch_start)

    sync_device(device)
    return time.perf_counter() - start, epoch_times


def train_ll_network(
    ll_net: LLNetwork,
    score_net: ScoreNetwork,
    process: FractionalOUProcess,
    training_data: np.ndarray,
    cfg: Config,
    device: torch.device,
    x_eval: np.ndarray,
    p_exact_eval: np.ndarray,
    data_time_s: float,
    stage1_time_s: float,
    stage1_epoch_times: List[float],
) -> MethodResult:
    """原始 Stage II：score-consistency 预热后逐步引入 FFP 物理残差。

    本函数不对 score 做空间积分，也不生成任何伪 log-density 监督标签。
    Stage II-A 仅最小化论文已有的 score consistency 和质量约束；
    Stage II-B 在同一网络上平滑加入非局部 FFP 残差。
    """
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32, device=device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32, device=device)
    score_net.eval()

    def density_func(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return safe_exp(ll_net(x, t))

    all_stage2_times: List[float] = []
    best_w = float("inf")
    best_epoch = -1
    best_metrics: Optional[EvalMetrics] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    converged_epoch = -1

    # ------------------------------------------------------------------
    # Stage II-A: score-consistency warm-up (still part of original Stage II)
    # ------------------------------------------------------------------
    warmup_optimizer = Adam(ll_net.parameters(), lr=cfg.ll_warmup_lr)
    warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        warmup_optimizer, T_max=max(1, cfg.ll_warmup_epochs), eta_min=cfg.eta_min
    )
    sync_device(device)
    stage2_start = time.perf_counter()
    pbar = tqdm(range(cfg.ll_warmup_epochs), desc="Stage II-A consistency warm-up", leave=False)
    for epoch in pbar:
        sync_device(device)
        epoch_start = time.perf_counter()
        idx = torch.randperm(len(training_data), device=device)[: cfg.batch_size]
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx].clone().detach().requires_grad_(True)

        q = ll_net(x, t)
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        with torch.no_grad():
            score_target = score_net(x, t)
        loss_score = (q_x - score_target).pow(2).mean()

        if cfg.mass_loss_weight > 0.0:
            # 固定包含中间与终点时刻，减少纯随机质量估计的波动。
            if cfg.mass_time_samples <= 1:
                mass_times = torch.ones((1, 1), dtype=torch.float32, device=device) * cfg.t_final
            else:
                base_times = torch.linspace(
                    cfg.t_final / cfg.mass_time_samples,
                    cfg.t_final,
                    cfg.mass_time_samples,
                    dtype=torch.float32,
                    device=device,
                ).view(-1, 1)
                jitter = (torch.rand_like(base_times) - 0.5) * (
                    cfg.t_final / cfg.mass_time_samples * 0.2
                )
                mass_times = torch.clamp(base_times + jitter, 0.0, cfg.t_final)
            masses = global_mass_from_network_torch(density_func, mass_times, cfg, device)
            loss_mass = torch.log(torch.clamp(masses, min=1e-8)).pow(2).mean()
        else:
            loss_mass = torch.zeros((), device=device)

        # 预热阶段先学习空间形状，再用较小质量权重确定每个时刻的积分常数。
        warm_mass_weight = 0.25 * cfg.mass_loss_weight
        loss = cfg.score_loss_weight * loss_score + warm_mass_weight * loss_mass

        warmup_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        warmup_optimizer.step()
        warmup_scheduler.step()
        sync_device(device)
        all_stage2_times.append(time.perf_counter() - epoch_start)

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = evaluate_network(ll_net, True, cfg, device, x_eval, p_exact_eval)
            if metrics.wasserstein < best_w:
                best_w = metrics.wasserstein
                best_epoch = epoch + 1
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in ll_net.state_dict().items()}
            pbar.set_postfix(
                W=f"{metrics.wasserstein:.3e}",
                M=f"{metrics.raw_mass:.4f}",
                Ls=f"{float(loss_score.detach().cpu()):.2e}",
            )

    # ------------------------------------------------------------------
    # Stage II-B: physics refinement with smooth PDE ramp
    # ------------------------------------------------------------------
    optimizer = Adam(ll_net.parameters(), lr=cfg.ll_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.ll_epochs), eta_min=cfg.eta_min
    )
    pbar = tqdm(range(cfg.ll_epochs), desc="Stage II-B physics refinement", leave=False)
    for epoch in pbar:
        sync_device(device)
        epoch_start = time.perf_counter()
        idx = torch.randperm(len(training_data), device=device)[: cfg.batch_size]
        x = x_train[idx].clone().detach().requires_grad_(True)
        t = t_train[idx].clone().detach().requires_grad_(True)

        q = ll_net(x, t)
        p = safe_exp(q)
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        q_t = torch.autograd.grad(q.sum(), t, create_graph=True)[0]
        with torch.no_grad():
            score_target = score_net(x, t)

        # 原论文两阶段方程：Stage-I score 直接进入 drift，q_x 通过一致性损失约束。
        drift = p * (process.theta + process.theta * x * score_target)
        diffusion = (process.sigma ** cfg.alpha) * fractional_derivative_gl_with_boundary(
            density_func,
            x,
            t,
            cfg.alpha,
            cfg.gl_dx,
            cfg.network_bound,
            cfg.gl_terms,
        )
        residual = p * q_t - drift - diffusion
        loss_res = residual.pow(2).mean()
        loss_score = (q_x - score_target).pow(2).mean()

        if cfg.mass_loss_weight > 0.0:
            mass_times = torch.linspace(
                cfg.t_final / cfg.mass_time_samples,
                cfg.t_final,
                cfg.mass_time_samples,
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)
            masses = global_mass_from_network_torch(density_func, mass_times, cfg, device)
            loss_mass = torch.log(torch.clamp(masses, min=1e-8)).pow(2).mean()
        else:
            loss_mass = torch.zeros((), device=device)

        pde_weight = min(1.0, (epoch + 1) / max(1, cfg.ll_pde_ramp_epochs))
        # PDE 权重增加时保留 score 一致性，防止非局部残差破坏已学到的密度形状。
        score_weight = cfg.score_loss_weight * (1.0 - 0.5 * pde_weight)
        mass_weight = cfg.mass_loss_weight * min(1.0, 0.25 + 0.75 * pde_weight)
        loss = pde_weight * loss_res + score_weight * loss_score + mass_weight * loss_mass

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        sync_device(device)
        all_stage2_times.append(time.perf_counter() - epoch_start)

        if (epoch + 1) % cfg.eval_every == 0:
            total_stage2_epoch = cfg.ll_warmup_epochs + epoch + 1
            metrics = evaluate_network(ll_net, True, cfg, device, x_eval, p_exact_eval)
            if metrics.wasserstein < best_w:
                best_w = metrics.wasserstein
                best_epoch = total_stage2_epoch
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in ll_net.state_dict().items()}
            pbar.set_postfix(
                W=f"{metrics.wasserstein:.3e}",
                M=f"{metrics.raw_mass:.4f}",
                E_M=f"{100.0 * metrics.mass_deviation:.2f}%",
                lam_pde=f"{pde_weight:.2f}",
            )
            if (
                epoch + 1 >= cfg.min_physics_epochs_before_stop
                and metrics.wasserstein < cfg.target_w
            ):
                converged_epoch = total_stage2_epoch
                best_metrics = metrics
                best_state = {k: v.detach().cpu().clone() for k, v in ll_net.state_dict().items()}
                break

    sync_device(device)
    stage2_time = time.perf_counter() - stage2_start

    if best_state is not None:
        ll_net.load_state_dict(best_state)
        ll_net.to(device)
    if best_metrics is None:
        best_metrics = evaluate_network(ll_net, True, cfg, device, x_eval, p_exact_eval)

    used_stage2 = converged_epoch if converged_epoch > 0 else max(best_epoch, len(all_stage2_times))
    all_epoch_times = stage1_epoch_times + all_stage2_times
    total_training_time = stage1_time_s + stage2_time
    total_time = total_training_time + (
        data_time_s if cfg.include_data_generation_time else 0.0
    )

    return MethodResult(
        method="Score-based PINN (Ours)",
        converged=converged_epoch > 0,
        work=(
            f"{cfg.score_epochs} + {converged_epoch}"
            if converged_epoch > 0
            else f"{cfg.score_epochs} + 未收敛(best={best_epoch})"
        ),
        average_step_ms=1000.0 * float(np.mean(all_epoch_times)),
        total_time_s=total_time if converged_epoch > 0 else -1.0,
        metrics=best_metrics,
        stage1_iterations=cfg.score_epochs,
        stage2_iterations=used_stage2,
        data_time_s=data_time_s,
        stage1_time_s=stage1_time_s,
        stage2_time_s=stage2_time,
        extra={
            "mass_loss_weight": cfg.mass_loss_weight,
            "warmup_epochs": float(cfg.ll_warmup_epochs),
            "best_stage2_epoch": float(best_epoch),
            "best_w": float(best_w),
            "score_integral_initialization": 0.0,
        },
    )




# =============================================================================
# Corrected Fig. 2 evaluation and plotting
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams.update({
    # Enlarged publication-scale typography for readability after insertion
    # into a two-column manuscript.
    "font.size": 18,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 21,
})


class ExactFOUReference:
    """Fast Fourier-inversion reference with a reusable cosine matrix."""
    def __init__(self, x_grid: np.ndarray, cfg: Config):
        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        self.cfg = cfg
        self.k = np.linspace(0.0, cfg.exact_k_max, cfg.exact_k_num, dtype=np.float64)
        self.dk = float(self.k[1] - self.k[0])
        self.weights = np.ones_like(self.k)
        self.weights[0] = 0.5
        self.weights[-1] = 0.5
        # The expensive trigonometric matrix is built once and reused for all times.
        self.cosine = np.cos(self.x_grid[:, None] * self.k[None, :])

    def pdf(self, t_value: float) -> np.ndarray:
        if t_value <= 1e-14:
            return np.exp(-0.5 * self.x_grid ** 2) / math.sqrt(2.0 * math.pi)
        cfg = self.cfg
        if abs(cfg.theta) < 1e-14:
            a_t = t_value
        else:
            a_t = (1.0 - math.exp(-cfg.alpha * cfg.theta * t_value)) / (
                cfg.alpha * cfg.theta
            )
        gaussian_variance = math.exp(-2.0 * cfg.theta * t_value)
        levy_coefficient = (cfg.sigma ** cfg.alpha) * a_t
        decay = np.exp(
            -0.5 * gaussian_variance * self.k ** 2
            - levy_coefficient * self.k ** cfg.alpha
        )
        pdf = self.cosine @ (decay * self.weights)
        pdf *= self.dk / math.pi
        return np.maximum(pdf, 0.0)


def evaluate_time_series(
    vanilla: VanillaNetwork,
    ll_net: LLNetwork,
    cfg: Config,
    device: torch.device,
    x_grid: np.ndarray,
    time_grid: np.ndarray,
) -> Tuple[List[Dict[str, float]], Dict[float, Dict[str, np.ndarray]]]:
    """Evaluate both solvers against the exact fOU density over physical time."""
    rows: List[Dict[str, float]] = []
    cache: Dict[float, Dict[str, np.ndarray]] = {}
    exact_reference = ExactFOUReference(x_grid, cfg)

    for t_value in tqdm(time_grid, desc="Evaluating corrected Fig. 2", leave=False):
        t_float = float(t_value)
        p_exact_raw = exact_reference.pdf(t_float)
        p_ours_raw = evaluate_network_raw(ll_net, x_grid, t_float, True, device)
        p_vanilla_raw = evaluate_network_raw(vanilla, x_grid, t_float, False, device)

        p_exact_shape = normalize_positive_density(p_exact_raw, x_grid)
        p_ours_shape = normalize_positive_density(p_ours_raw, x_grid)
        p_vanilla_shape = normalize_positive_density(p_vanilla_raw, x_grid)

        w_ours = wasserstein_distance(
            x_grid, x_grid, u_weights=p_ours_shape, v_weights=p_exact_shape
        )
        w_vanilla = wasserstein_distance(
            x_grid, x_grid, u_weights=p_vanilla_shape, v_weights=p_exact_shape
        )

        neg_ours = float(simpson(np.maximum(-p_ours_raw, 0.0), x=x_grid))
        neg_vanilla = float(simpson(np.maximum(-p_vanilla_raw, 0.0), x=x_grid))

        mass_exact = asymptotic_global_mass_from_grid(p_exact_raw, x_grid, cfg.alpha)
        mass_ours = asymptotic_global_mass_from_grid(p_ours_raw, x_grid, cfg.alpha)
        mass_vanilla = asymptotic_global_mass_from_grid(p_vanilla_raw, x_grid, cfg.alpha)

        l2_ours_num = float(simpson((p_ours_shape - p_exact_shape) ** 2, x=x_grid))
        l2_van_num = float(simpson((p_vanilla_shape - p_exact_shape) ** 2, x=x_grid))
        l2_den = float(simpson(p_exact_shape ** 2, x=x_grid))

        row = {
            "time": t_float,
            "W_ours": float(w_ours),
            "W_vanilla": float(w_vanilla),
            "negative_mass_ours": neg_ours,
            "negative_mass_vanilla": neg_vanilla,
            "global_mass_exact_estimate": float(mass_exact),
            "global_mass_ours": float(mass_ours),
            "global_mass_vanilla": float(mass_vanilla),
            "mass_deviation_exact_estimate": float(abs(mass_exact - 1.0)),
            "mass_deviation_ours": float(abs(mass_ours - 1.0)),
            "mass_deviation_vanilla": float(abs(mass_vanilla - 1.0)),
            "L2_rel_ours": math.sqrt(max(l2_ours_num, 0.0) / max(l2_den, 1e-30)),
            "L2_rel_vanilla": math.sqrt(max(l2_van_num, 0.0) / max(l2_den, 1e-30)),
        }
        rows.append(row)
        cache[t_float] = {
            "exact_raw": p_exact_raw,
            "ours_raw": p_ours_raw,
            "vanilla_raw": p_vanilla_raw,
            "exact_shape": p_exact_shape,
            "ours_shape": p_ours_shape,
            "vanilla_shape": p_vanilla_shape,
        }
    return rows, cache

def save_rows_csv(path: str, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows were generated.")
    import csv
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nearest_cached_time(
    cache: Dict[float, Dict[str, np.ndarray]], target: float
) -> Tuple[float, Dict[str, np.ndarray]]:
    time_value = min(cache.keys(), key=lambda value: abs(value - target))
    return time_value, cache[time_value]


def plot_corrected_figure2(
    rows: Sequence[Dict[str, float]],
    cache: Dict[float, Dict[str, np.ndarray]],
    x_grid: np.ndarray,
    cfg: Config,
    output_dir: str,
) -> None:
    """Generate the corrected two-row, six-panel Fig. 2."""
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5), constrained_layout=True)
    density_times = (0.25, 0.50, 0.75)

    row_by_time = {float(row["time"]): row for row in rows}
    for column, target_time in enumerate(density_times):
        time_value, values = nearest_cached_time(cache, target_time)
        metric_time = min(row_by_time.keys(), key=lambda value: abs(value - time_value))
        metric = row_by_time[metric_time]
        ax = axes[0, column]

        # Raw densities are plotted so that scale/mass differences are not hidden.
        ax.plot(x_grid, values["exact_raw"], "k-", linewidth=2.3, label="Exact")
        ax.plot(x_grid, values["ours_raw"], "r-", linewidth=2.0, label="Ours")
        ax.plot(x_grid, values["vanilla_raw"], "b--", linewidth=1.9, label="Vanilla fPINN")
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.set_xlim(-cfg.eval_bound, cfg.eval_bound)
        combined_min = min(
            float(np.min(values["ours_raw"])),
            float(np.min(values["vanilla_raw"])),
            0.0,
        )
        combined_max = max(
            float(np.max(values["exact_raw"])),
            float(np.max(values["ours_raw"])),
            float(np.max(values["vanilla_raw"])),
        )
        lower = min(-0.005, 1.15 * combined_min)
        ax.set_ylim(lower, 1.12 * combined_max)
        ax.set_xlabel("Position $x$")
        ax.set_ylabel("Probability density")
        ax.set_title(f"({chr(97 + column)}) Density at $t={time_value:.2f}$")
        ax.text(
            0.03,
            0.96,
            rf"$W_{{\mathrm{{Ours}}}}={metric['W_ours']:.4f}$" + "\n"
            rf"$W_{{\mathrm{{Vanilla}}}}={metric['W_vanilla']:.4f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
        )
        ax.grid(True, linestyle="--", alpha=0.25)
        if column == 0:
            ax.legend(loc="upper right")

    times = np.array([float(row["time"]) for row in rows])
    w_ours = np.array([float(row["W_ours"]) for row in rows])
    w_vanilla = np.array([float(row["W_vanilla"]) for row in rows])
    neg_ours = np.array([float(row["negative_mass_ours"]) for row in rows])
    neg_vanilla = np.array([float(row["negative_mass_vanilla"]) for row in rows])
    mass_dev_exact = np.array([float(row["mass_deviation_exact_estimate"]) for row in rows])
    mass_dev_ours = np.array([float(row["mass_deviation_ours"]) for row in rows])
    mass_dev_vanilla = np.array([float(row["mass_deviation_vanilla"]) for row in rows])

    ax = axes[1, 0]
    ax.plot(times, w_ours, "r-", linewidth=2.2, marker="o", markersize=3.2, label="Ours")
    ax.plot(times, w_vanilla, "b--", linewidth=2.0, marker="s", markersize=3.0, label="Vanilla fPINN")
    ax.axhline(cfg.target_w, color="0.25", linestyle=":", linewidth=1.5, label="$W=0.1$")
    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("(d) Distributional error")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(times, neg_ours, "r-", linewidth=2.2, marker="o", markersize=3.2, label="Ours")
    ax.plot(times, neg_vanilla, "b--", linewidth=2.0, marker="s", markersize=3.0, label="Vanilla fPINN")
    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel(r"$\int_{-B}^{B}\max(-p,0)\,dx$")
    ax.set_title("(e) Negative probability mass")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(times, 100.0 * mass_dev_ours, "r-", linewidth=2.2, marker="o", markersize=3.2, label="Ours")
    ax.plot(times, 100.0 * mass_dev_vanilla, "b--", linewidth=2.0, marker="s", markersize=3.0, label="Vanilla fPINN")
    ax.plot(
        times,
        100.0 * mass_dev_exact,
        color="0.35",
        linestyle=":",
        linewidth=1.6,
        label="Tail-estimator bias on exact density",
    )
    ax.set_xlabel("Physical time $t$")
    ax.set_ylabel("Global mass deviation (%)")
    ax.set_title("(f) Estimated global probability-mass deviation")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    figure_title = (
        rf"Comparison between the proposed method and the Vanilla fPINN "
        rf"for the fractional OU process ($\alpha={cfg.alpha}$)"
    )
    fig.suptitle(figure_title, fontsize=21)

    png_path = os.path.join(output_dir, "fig2_corrected_alpha_1p5.png")
    pdf_path = os.path.join(output_dir, "fig2_corrected_alpha_1p5.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint_models(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> Tuple[VanillaNetwork, ScoreNetwork, LLNetwork]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get("config", {})
    # Architecture-relevant fields are restored to prevent shape mismatches.
    for name in ("hidden_dim", "num_layers", "alpha", "theta", "sigma", "network_bound"):
        if name in saved_config:
            setattr(cfg, name, saved_config[name])

    vanilla = VanillaNetwork(cfg).to(device)
    score_net = ScoreNetwork(cfg).to(device)
    ll_net = LLNetwork(cfg).to(device)
    vanilla.load_state_dict(checkpoint["vanilla_state_dict"])
    score_net.load_state_dict(checkpoint["score_state_dict"])
    ll_net.load_state_dict(checkpoint["ll_state_dict"])
    vanilla.eval()
    score_net.eval()
    ll_net.eval()
    return vanilla, score_net, ll_net


def run_corrected_fig2(cfg: Config, checkpoint: Optional[str] = None) -> None:
    if cfg.eval_bound != cfg.network_bound:
        raise ValueError("eval_bound and network_bound must be identical for the common tail estimator.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device}")
    print("\n--- Corrected Fig. 2: Ours versus Vanilla fPINN ---")

    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"Loading Table-2 checkpoints from: {checkpoint}")
        vanilla, score_net, ll_net = load_checkpoint_models(checkpoint, cfg, device)
        training_summary = {
            "source": "checkpoint",
            "checkpoint": os.path.abspath(checkpoint),
        }
    else:
        process = FractionalOUProcess(cfg)
        print("Generating the same Stage-I and Stage-II datasets used by the revised Table-2 experiment...")
        data_start = time.perf_counter()
        score_data, collocation_data = generate_training_data(process, cfg)
        data_time = time.perf_counter() - data_start
        print(
            f"Data generation: {data_time:.3f}s; "
            f"score points={len(score_data)}, collocation points={len(collocation_data)}"
        )

        x_target = np.linspace(-cfg.eval_bound, cfg.eval_bound, cfg.eval_grid_n)
        p_exact_target = exact_fou_pdf(x_target, cfg.t_final, cfg)
        vanilla = VanillaNetwork(cfg).to(device)
        score_net = ScoreNetwork(cfg).to(device)
        ll_net = LLNetwork(cfg).to(device)

        print("[1/3] Training Vanilla fPINN with corrected OU drift sign")
        vanilla_result = train_vanilla_network(
            vanilla,
            process,
            collocation_data,
            cfg,
            device,
            x_target,
            p_exact_target,
            data_time,
        )
        print("[2/3] Training Stage-I score network")
        stage1_time, stage1_epoch_times = train_score_network(score_net, score_data, cfg, device)
        print("[3/3] Training Stage-II log-density network")
        ours_result = train_ll_network(
            ll_net,
            score_net,
            process,
            collocation_data,
            cfg,
            device,
            x_target,
            p_exact_target,
            data_time,
            stage1_time,
            stage1_epoch_times,
        )
        training_summary = {
            "source": "trained_in_fig2_script",
            "vanilla_work": vanilla_result.work,
            "vanilla_total_time_s": vanilla_result.total_time_s,
            "vanilla_final_W": vanilla_result.metrics.wasserstein,
            "vanilla_final_mass_deviation": vanilla_result.metrics.mass_deviation,
            "ours_work": ours_result.work,
            "ours_total_time_s": ours_result.total_time_s,
            "ours_final_W": ours_result.metrics.wasserstein,
            "ours_final_mass_deviation": ours_result.metrics.mass_deviation,
        }
        torch.save(
            {
                "config": vars(cfg),
                "vanilla_state_dict": vanilla.state_dict(),
                "score_state_dict": score_net.state_dict(),
                "ll_state_dict": ll_net.state_dict(),
                "training_summary": training_summary,
            },
            os.path.join(cfg.output_dir, "fig2_trained_models.pt"),
        )

    # Include t=0 exactly and enough temporal resolution for smooth physical-time curves.
    x_grid = np.linspace(-cfg.eval_bound, cfg.eval_bound, cfg.figure_grid_n)
    time_grid = np.linspace(0.0, cfg.t_final, cfg.figure_time_points)
    rows, cache = evaluate_time_series(vanilla, ll_net, cfg, device, x_grid, time_grid)

    metrics_path = os.path.join(cfg.output_dir, "fig2_corrected_metrics.csv")
    save_rows_csv(metrics_path, rows)
    plot_corrected_figure2(rows, cache, x_grid, cfg, cfg.output_dir)

    import json
    with open(os.path.join(cfg.output_dir, "fig2_training_summary.json"), "w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=2, ensure_ascii=False)

    final_row = min(rows, key=lambda row: abs(float(row["time"]) - cfg.t_final))
    print("\nCorrected Fig. 2 generated successfully.")
    print(
        f"t=1: W(Ours)={final_row['W_ours']:.6f}, "
        f"W(Vanilla)={final_row['W_vanilla']:.6f}"
    )
    print(
        f"t=1: mass deviation(Ours)={100.0 * final_row['mass_deviation_ours']:.3f}%, "
        f"mass deviation(Vanilla)={100.0 * final_row['mass_deviation_vanilla']:.3f}%"
    )
    print(f"Results saved to: {os.path.abspath(cfg.output_dir)}")


def build_fig2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default="", help="Optional trained_models.pt from the revised Table-2 run.")
    parser.add_argument("--output_dir", default="results_fig2_corrected")
    parser.add_argument("--vanilla_epochs", type=int, default=3000)
    parser.add_argument("--score_epochs", type=int, default=3000)
    parser.add_argument("--ll_warmup_epochs", type=int, default=800)
    parser.add_argument("--ll_epochs", type=int, default=2000)
    parser.add_argument("--min_physics_epochs_before_stop", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--target_w", type=float, default=0.1)
    parser.add_argument("--mass_loss_weight", type=float, default=1.0)
    parser.add_argument("--train_points", type=int, default=50000)
    parser.add_argument("--train_trajectories", type=int, default=800)
    parser.add_argument("--figure_grid_n", type=int, default=801)
    parser.add_argument("--figure_time_points", type=int, default=21)
    parser.add_argument("--quick_test", action="store_true")
    return parser


def main() -> None:
    args = build_fig2_parser().parse_args()
    cfg = Config(
        device=args.device,
        output_dir=args.output_dir,
        vanilla_epochs=args.vanilla_epochs,
        score_epochs=args.score_epochs,
        ll_warmup_epochs=args.ll_warmup_epochs,
        ll_epochs=args.ll_epochs,
        min_physics_epochs_before_stop=args.min_physics_epochs_before_stop,
        eval_every=args.eval_every,
        target_w=args.target_w,
        mass_loss_weight=args.mass_loss_weight,
        train_points=args.train_points,
        train_trajectories=args.train_trajectories,
    )
    # Figure-only settings are attached dynamically to avoid modifying the shared Config definition.
    cfg.figure_grid_n = args.figure_grid_n
    cfg.figure_time_points = args.figure_time_points

    if args.quick_test:
        cfg.vanilla_epochs = 2
        cfg.score_epochs = 2
        cfg.ll_warmup_epochs = 2
        cfg.ll_epochs = 3
        cfg.min_physics_epochs_before_stop = 1
        cfg.eval_every = 1
        cfg.train_points = 512
        cfg.train_trajectories = 40
        cfg.hidden_dim = 16
        cfg.num_layers = 3
        cfg.batch_size = 8
        cfg.gl_terms = 4
        cfg.mass_quad_points = 31
        cfg.exact_k_num = 1500
        cfg.eval_grid_n = 201
        cfg.figure_grid_n = 201
        cfg.figure_time_points = 5
        cfg.output_dir = cfg.output_dir + "_quick"

    run_corrected_fig2(cfg, checkpoint=args.checkpoint or None)


if __name__ == "__main__":
    main()
