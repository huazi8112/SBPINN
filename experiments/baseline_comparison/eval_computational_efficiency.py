"""
不使用 score 积分初始化的两阶段 Score-based PINN 与 P1 Galerkin FEM 对比实验。

主要修正
--------
1. 修正 Vanilla fPINN 漂移项符号，使三种方法求解同一 fractional OU FFP 方程：
       p_t = theta * d(x p)/dx + sigma**alpha * D_Riesz^alpha p.
2. Ours 的质量偏差由未经后处理归一化的原始密度计算，不再人为设为 0。
3. Stage II 不使用 score 积分初始化：先用 score consistency 与质量约束预热，随后平滑引入 FFP 残差。
4. FEM 为真正的连续分片线性 P1 Galerkin FEM：组装质量矩阵、Laplace 刚度矩阵
   和 OU 漂移弱形式矩阵；分数阶扩散由广义 FEM Laplace 算子的分数次幂构造。
5. FEM 自动运行一组 h-dt 细化配置，选取首个满足 W < TARGET_W 的配置。
6. Ours 只有在完成至少指定数量的真实 FFP 物理残差迭代后才允许判定达标，
   因此表 2 的收敛结果不来自预热阶段。
7. Ours 的工作量明确报告为 Stage I + Stage II；Stage II 的预热和物理微调
   计入同一 Stage II 迭代数。
8. 输出 CSV、Markdown 表格和 FEM 收敛记录，便于直接核查。

说明
----
- Wasserstein 距离在公共评价区间 [-EVAL_BOUND, EVAL_BOUND] 上计算；各方法仅在
  计算 W 时归一化，以比较分布形状。
- “最终概率质量守恒偏差”使用原始数值输出。对神经网络，利用论文中的
  alpha-stable 渐近尾部完成公式估算全空间质量：
      M = integral_{-B}^{B} p(x) dx + B/alpha * [p(-B) + p(B)].
  对 FEM，采用扩大计算域上的原始 P1 解积分，域外按齐次 Dirichlet 截断。
- 若将 MASS_LOSS_WEIGHT 设为 0，可评估不带显式质量约束的原始方法。
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
# 真正的 P1 Galerkin FEM
# =============================================================================
def assemble_p1_fem_operators(
    bound: float, n_elements: int, theta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """组装 P1 质量矩阵、Laplace 刚度矩阵和 OU 漂移弱形式矩阵。"""
    if n_elements < 20:
        raise ValueError("FEM 单元数至少为 20。")
    nodes = np.linspace(-bound, bound, n_elements + 1, dtype=np.float64)
    h = float(nodes[1] - nodes[0])
    n_nodes = len(nodes)
    mass_full = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    laplace_full = np.zeros_like(mass_full)
    drift_full = np.zeros_like(mass_full)

    mass_local = h / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    laplace_local = 1.0 / h * np.array([[1.0, -1.0], [-1.0, 1.0]])

    for elem in range(n_elements):
        x_left = float(nodes[elem])
        # 精确积分 int_e x*N1 dx 和 int_e x*N2 dx。
        int_x_n1 = h * (0.5 * x_left + h / 6.0)
        int_x_n2 = h * (0.5 * x_left + h / 3.0)
        derivatives = np.array([[-1.0 / h], [1.0 / h]])
        # theta*d(xp)/dx 的弱形式：-theta*int phi_i' * x * phi_j dx。
        drift_local = -theta * derivatives @ np.array([[int_x_n1, int_x_n2]])

        ids = np.array([elem, elem + 1])
        mass_full[np.ix_(ids, ids)] += mass_local
        laplace_full[np.ix_(ids, ids)] += laplace_local
        drift_full[np.ix_(ids, ids)] += drift_local

    interior = slice(1, -1)
    return (
        nodes,
        mass_full[interior, interior],
        laplace_full[interior, interior],
        drift_full[interior, interior],
    )


def l2_project_initial_gaussian(nodes: np.ndarray, mass: np.ndarray) -> np.ndarray:
    n_elements = len(nodes) - 1
    load = np.zeros(len(nodes), dtype=np.float64)
    xi, weights = np.polynomial.legendre.leggauss(6)
    for elem in range(n_elements):
        a = float(nodes[elem])
        b = float(nodes[elem + 1])
        h = b - a
        xq = 0.5 * (a + b) + 0.5 * h * xi
        n1 = (b - xq) / h
        n2 = (xq - a) / h
        p0 = np.exp(-0.5 * xq ** 2) / math.sqrt(2.0 * math.pi)
        load[elem] += 0.5 * h * np.sum(weights * p0 * n1)
        load[elem + 1] += 0.5 * h * np.sum(weights * p0 * n2)
    return np.linalg.solve(mass, load[1:-1])


def solve_p1_fractional_fem(
    cfg: Config,
    n_elements: int,
    requested_dt: float,
    x_eval: np.ndarray,
    p_exact_eval: np.ndarray,
) -> Tuple[MethodResult, Dict[str, float]]:
    """P1 Galerkin FEM + backward Euler；返回单个 h-dt 配置结果。"""
    total_start = time.perf_counter()
    assembly_start = time.perf_counter()
    nodes, mass, laplace, drift = assemble_p1_fem_operators(
        cfg.fem_bound, n_elements, cfg.theta
    )

    # K v = lambda M v，eigvecs 满足 V^T M V = I。
    eigvals, eigvecs = eigh(laplace, mass, check_finite=False)
    eigvals = np.maximum(eigvals, 0.0)
    mv = mass @ eigvecs
    frac_stiffness = (
        mv * (eigvals ** (cfg.alpha / 2.0))[None, :]
    ) @ mv.T
    weak_generator = drift - (cfg.sigma ** cfg.alpha) * frac_stiffness

    n_steps = max(1, int(math.ceil(cfg.t_final / requested_dt)))
    actual_dt = cfg.t_final / n_steps
    lhs = mass - actual_dt * weak_generator
    lu = lu_factor(lhs, check_finite=False)
    u = l2_project_initial_gaussian(nodes, mass)
    assembly_time = time.perf_counter() - assembly_start

    step_times: List[float] = []
    marching_start = time.perf_counter()
    for _ in range(n_steps):
        step_start = time.perf_counter()
        u = lu_solve(lu, mass @ u, check_finite=False)
        step_times.append(time.perf_counter() - step_start)
    marching_time = time.perf_counter() - marching_start
    total_time = time.perf_counter() - total_start

    p_full = np.concatenate(([0.0], u, [0.0]))
    p_eval = np.interp(x_eval, nodes, p_full, left=0.0, right=0.0)
    fem_global_mass = float(np_trapz(p_full, nodes))
    metrics = compute_metrics(p_eval, p_exact_eval, x_eval, fem_global_mass)

    result = MethodResult(
        method="P1 Galerkin FEM",
        converged=metrics.wasserstein < cfg.target_w,
        work=f"N_h={n_elements - 1}, N_t={n_steps}",
        average_step_ms=1000.0 * float(np.mean(step_times)),
        total_time_s=total_time,
        metrics=metrics,
        extra={
            "elements": float(n_elements),
            "dofs": float(n_elements - 1),
            "time_steps": float(n_steps),
            "requested_dt": float(requested_dt),
            "actual_dt": float(actual_dt),
            "assembly_time_s": float(assembly_time),
            "marching_time_s": float(marching_time),
            "domain_bound": float(cfg.fem_bound),
        },
    )
    row = {
        "elements": n_elements,
        "dofs": n_elements - 1,
        "requested_dt": requested_dt,
        "actual_dt": actual_dt,
        "time_steps": n_steps,
        "wasserstein": metrics.wasserstein,
        "l2_rel": metrics.l2_rel,
        "raw_mass": metrics.raw_mass,
        "mass_deviation": metrics.mass_deviation,
        "negative_mass": metrics.negative_mass,
        "average_step_ms": result.average_step_ms,
        "assembly_time_s": assembly_time,
        "marching_time_s": marching_time,
        "total_time_s": total_time,
        "meets_target": int(result.converged),
    }
    return result, row


def run_fem_convergence_search(
    cfg: Config,
    x_eval: np.ndarray,
    p_exact_eval: np.ndarray,
) -> Tuple[MethodResult, List[Dict[str, float]]]:
    print("\n[4/4] P1 Galerkin FEM 网格–时间步收敛搜索")
    rows: List[Dict[str, float]] = []
    selected: Optional[MethodResult] = None

    for n_elements, dt in cfg.fem_refinements:
        print(f"  FEM: elements={n_elements}, dt={dt:g}")
        result, row = solve_p1_fractional_fem(
            cfg, n_elements, dt, x_eval, p_exact_eval
        )
        rows.append(row)
        print(
            f"    W={result.metrics.wasserstein:.4e}, "
            f"L2={result.metrics.l2_rel:.4e}, "
            f"mass dev={100.0 * result.metrics.mass_deviation:.3f}%, "
            f"total={result.total_time_s:.4f}s"
        )
        if selected is None and result.converged:
            selected = result

    if selected is None:
        # 未达标时使用最细配置，但在表中明确标记未收敛。
        n_elements, dt = cfg.fem_refinements[-1]
        selected, _ = solve_p1_fractional_fem(
            cfg, n_elements, dt, x_eval, p_exact_eval
        )
        selected.converged = False
        selected.work = "未达到目标精度"
        selected.total_time_s = -1.0
    return selected, rows


# =============================================================================
# 输出
# =============================================================================
def result_to_row(result: MethodResult) -> Dict[str, object]:
    return {
        "method": result.method,
        "converged": int(result.converged),
        "work_to_target": result.work,
        "average_step_ms": result.average_step_ms,
        "total_time_to_target_s": result.total_time_s,
        "wasserstein_at_reported_state": result.metrics.wasserstein,
        "raw_global_mass": result.metrics.raw_mass,
        "mass_deviation": result.metrics.mass_deviation,
        "negative_mass_on_eval_domain": result.metrics.negative_mass,
        "relative_L2_shape_error": result.metrics.l2_rel,
        "data_time_s": result.data_time_s,
        "stage1_iterations": result.stage1_iterations,
        "stage2_iterations": result.stage2_iterations,
        "stage1_time_s": result.stage1_time_s,
        "stage2_time_s": result.stage2_time_s,
        **result.extra,
    }


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_time(value: float) -> str:
    if value < 0.0:
        return "无法达到"
    return f"{value:.4f}" if value < 1.0 else f"{value:.2f}"


def print_and_save_table(
    cfg: Config,
    results: Sequence[MethodResult],
    output_path: str,
) -> None:
    lines = [
        f"目标精度：W(t={cfg.t_final:g}) < {cfg.target_w:g}",
        "",
        "| 模型 (Model) | 达到目标精度所需迭代数/自由度–时间步 | 单步平均耗时 (ms) | 达到目标精度总耗时 (s) | 最终概率质量守恒偏差 |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for r in results:
        work = r.work if r.converged else "未收敛 (N/A)"
        total = format_time(r.total_time_s)
        line = (
            f"| {r.method} | {work} | {r.average_step_ms:.3f} | {total} | "
            f"{100.0 * r.metrics.mass_deviation:.3f}% |"
        )
        lines.append(line)

    lines.extend(
        [
            "",
            "注：Wasserstein 距离仅在公共评价区间上对正部归一化后计算；质量偏差使用未经归一化的原始数值解。",
            "Ours 的全空间质量按渐近尾部补全公式估算；FEM 的质量为扩大截断域内 P1 解的积分。",
            "若 mass_loss_weight > 0，则论文方法部分必须明确说明加入了显式全局质量约束。",
        ]
    )
    text = "\n".join(lines)
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# =============================================================================
# 主实验
# =============================================================================
def experiment(cfg: Config) -> None:
    if not (1.0 < cfg.alpha < 2.0):
        raise ValueError("alpha 必须位于 (1,2)。")
    if cfg.eval_bound != cfg.network_bound:
        raise ValueError("当前质量尾部公式要求 eval_bound 与 network_bound 相同。")

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"使用设备: {device}")
    print(f"\n--- Alpha={cfg.alpha} 的公平收敛效率与 FEM 对比实验 ---")

    x_eval = np.linspace(-cfg.eval_bound, cfg.eval_bound, cfg.eval_grid_n)
    print("计算精确 fOU 参考密度...")
    p_exact_eval = exact_fou_pdf(x_eval, cfg.t_final, cfg)
    exact_core_mass = float(simpson(p_exact_eval, x=x_eval))
    exact_tail_completed_mass = asymptotic_global_mass_from_grid(
        p_exact_eval, x_eval, cfg.alpha
    )
    print(
        f"精确解在 [-{cfg.eval_bound:g},{cfg.eval_bound:g}] 上质量="
        f"{exact_core_mass:.8f}; 渐近尾部补全质量={exact_tail_completed_mass:.8f}"
    )

    process = FractionalOUProcess(cfg)
    print("\n生成神经网络所需训练轨迹与采样点...")
    data_start = time.perf_counter()
    score_data, collocation_data = generate_training_data(process, cfg)
    data_time = time.perf_counter() - data_start
    print(
        f"训练数据生成耗时: {data_time:.3f}s, "
        f"score points={len(score_data)}, collocation points={len(collocation_data)}"
    )

    vanilla = VanillaNetwork(cfg).to(device)
    score_net = ScoreNetwork(cfg).to(device)
    ll_net = LLNetwork(cfg).to(device)

    print("\n[1/4] 训练 Vanilla fPINN")
    vanilla_result = train_vanilla_network(
        vanilla,
        process,
        collocation_data,
        cfg,
        device,
        x_eval,
        p_exact_eval,
        data_time,
    )

    print("\n[2/4] 训练 Score-based PINN Stage I")
    stage1_time, stage1_epoch_times = train_score_network(
        score_net, score_data, cfg, device
    )

    print("\n[3/4] 训练 Score-based PINN Stage II")
    ours_result = train_ll_network(
        ll_net,
        score_net,
        process,
        collocation_data,
        cfg,
        device,
        x_eval,
        p_exact_eval,
        data_time,
        stage1_time,
        stage1_epoch_times,
    )

    fem_result, fem_rows = run_fem_convergence_search(
        cfg, x_eval, p_exact_eval
    )

    results = [vanilla_result, ours_result, fem_result]
    comparison_csv = os.path.join(cfg.output_dir, "table2_comparison.csv")
    fem_csv = os.path.join(cfg.output_dir, "fem_convergence.csv")
    table_md = os.path.join(cfg.output_dir, "table2_markdown.md")
    write_csv(comparison_csv, [result_to_row(r) for r in results])
    write_csv(fem_csv, fem_rows)
    print_and_save_table(cfg, results, table_md)

    torch.save(
        {
            "config": vars(cfg),
            "vanilla_state_dict": vanilla.state_dict(),
            "score_state_dict": score_net.state_dict(),
            "ll_state_dict": ll_net.state_dict(),
        },
        os.path.join(cfg.output_dir, "trained_models.pt"),
    )
    print(f"\n结果已保存到: {os.path.abspath(cfg.output_dir)}")
    print(f"  - {comparison_csv}")
    print(f"  - {fem_csv}")
    print(f"  - {table_md}")


def parse_refinements(text: str) -> Tuple[Tuple[int, float], ...]:
    """解析形如 '50:0.02,100:0.01,200:0.005' 的 FEM 细化序列。"""
    pairs: List[Tuple[int, float]] = []
    for block in text.split(","):
        elements, dt = block.split(":")
        pairs.append((int(elements.strip()), float(dt.strip())))
    if not pairs:
        raise ValueError("FEM 细化序列不能为空。")
    return tuple(pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--output_dir", default="results_table2_no_score_integration")
    parser.add_argument("--vanilla_epochs", type=int, default=3000)
    parser.add_argument("--score_epochs", type=int, default=3000)
    parser.add_argument("--ll_warmup_epochs", type=int, default=800)
    parser.add_argument("--ll_epochs", type=int, default=2000)
    parser.add_argument("--score_lr", type=float, default=5e-4)
    parser.add_argument("--ll_warmup_lr", type=float, default=1e-3)
    parser.add_argument("--ll_lr", type=float, default=2e-4)
    parser.add_argument("--ll_pde_ramp_epochs", type=int, default=1500)
    parser.add_argument("--min_physics_epochs_before_stop", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--target_w", type=float, default=0.1)
    parser.add_argument("--mass_loss_weight", type=float, default=1.0)
    parser.add_argument("--score_loss_weight", type=float, default=1.0)
    parser.add_argument("--train_points", type=int, default=50000)
    parser.add_argument("--train_trajectories", type=int, default=800)
    parser.add_argument("--fem_bound", type=float, default=12.0)
    parser.add_argument(
        "--fem_refinements",
        default="50:0.02,100:0.01,200:0.005,300:0.0025",
    )
    parser.add_argument(
        "--exclude_data_time",
        action="store_true",
        help="不把必要训练数据生成耗时计入神经网络端到端总耗时。",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="仅用于检查代码流程，不可用于论文结果。",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        device=args.device,
        output_dir=args.output_dir,
        vanilla_epochs=args.vanilla_epochs,
        score_epochs=args.score_epochs,
        ll_warmup_epochs=args.ll_warmup_epochs,
        ll_epochs=args.ll_epochs,
        score_lr=args.score_lr,
        ll_warmup_lr=args.ll_warmup_lr,
        ll_lr=args.ll_lr,
        ll_pde_ramp_epochs=args.ll_pde_ramp_epochs,
        min_physics_epochs_before_stop=args.min_physics_epochs_before_stop,
        eval_every=args.eval_every,
        target_w=args.target_w,
        mass_loss_weight=args.mass_loss_weight,
        score_loss_weight=args.score_loss_weight,
        train_points=args.train_points,
        train_trajectories=args.train_trajectories,
        fem_bound=args.fem_bound,
        fem_refinements=parse_refinements(args.fem_refinements),
        include_data_generation_time=not args.exclude_data_time,
    )
    if args.quick_test:
        cfg.vanilla_epochs = 2
        cfg.score_epochs = 2
        cfg.ll_warmup_epochs = 2
        cfg.ll_epochs = 3
        cfg.ll_pde_ramp_epochs = 1
        cfg.min_physics_epochs_before_stop = 1
        cfg.eval_every = 1
        cfg.train_points = 512
        cfg.train_trajectories = 40
        cfg.batch_size = 32
        cfg.gl_terms = 8
        cfg.mass_quad_points = 41
        cfg.mass_time_samples = 1
        cfg.eval_grid_n = 201
        cfg.exact_k_num = 2000
        cfg.fem_refinements = ((30, 0.05),)
        print("警告：quick_test 结果仅用于程序联调，不能用于论文。")
    experiment(cfg)


if __name__ == "__main__":
    main()
