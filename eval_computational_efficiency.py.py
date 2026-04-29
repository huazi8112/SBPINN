"""
分数阶Fokker-Planck方程 Score-based 求解器 (包含收敛性与计算效能自动统计)
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings

warnings.filterwarnings('ignore')

import time  # 新增：用于计算效能统计
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 全局字体设置 (Times New Roman & 矢量嵌入)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
})

from scipy.stats import levy_stable, wasserstein_distance
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from torch.optim import Adam
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==========================================
# 全局配置参数
# ==========================================
THETA = 0.5
SIGMA = 1.0
T = 1.0
N_TIME_STEPS = 100


# ==========================================
# 神经网络定义 (保持不变)
# ==========================================
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.05))
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=-1))


class LLNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=-1))


class VanillaNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=-1))


# ==========================================
# 核心计算函数：GL导数 (保持不变)
# ==========================================
def fractional_derivative_GL_with_boundary(func, x, alpha, t_tensor, dx=0.05, domain_bound=5.0):
    n_terms = 80
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append(coeffs[-1] * (k - 1 - alpha) / k)
    coeffs = torch.tensor(coeffs, device=x.device)

    result = torch.zeros_like(x)

    for k in range(n_terms):
        coeff = coeffs[k]

        x_left = x - k * dx
        mask_in_left = x_left.abs() <= domain_bound
        val_left = torch.zeros_like(x_left)

        if mask_in_left.any():
            x_in = x_left[mask_in_left].view(-1, 1)
            t_in = t_tensor[mask_in_left].view(-1, 1)
            val_left[mask_in_left] = func(x_in, t_in).view(-1)

        if (~mask_in_left).any():
            x_out = x_left[~mask_in_left]
            x_b = (domain_bound * torch.sign(x_out)).view(-1, 1)
            t_out = t_tensor[~mask_in_left].view(-1, 1)
            val_b = func(x_b, t_out).view(-1)
            decay = (x_out.abs() / domain_bound).pow(-(1 + alpha))
            val_left[~mask_in_left] = val_b * decay

        x_right = x + k * dx
        mask_in_right = x_right.abs() <= domain_bound
        val_right = torch.zeros_like(x_right)

        if mask_in_right.any():
            x_in = x_right[mask_in_right].view(-1, 1)
            t_in = t_tensor[mask_in_right].view(-1, 1)
            val_right[mask_in_right] = func(x_in, t_in).view(-1)

        if (~mask_in_right).any():
            x_out = x_right[~mask_in_right]
            x_b = (domain_bound * torch.sign(x_out)).view(-1, 1)
            t_out = t_tensor[~mask_in_right].view(-1, 1)
            val_b = func(x_b, t_out).view(-1)
            decay = (x_out.abs() / domain_bound).pow(-(1 + alpha))
            val_right[~mask_in_right] = val_b * decay

        term_val = coeff * (val_left + val_right)
        result += term_val

    cos_val = np.cos(alpha * np.pi / 2)
    if abs(cos_val) < 1e-10:
        factor = 1.0 / (dx ** alpha)
    else:
        factor = -1.0 / (2 * cos_val * (dx ** alpha))

    return result * factor


# ==========================================
# 随机过程与数据生成 (保持不变)
# ==========================================
class FractionalOUProcess:
    def __init__(self, theta=0.5, sigma=1.0, alpha=1.5):
        self.theta = theta
        self.sigma = sigma
        self.alpha = alpha

    def drift(self, x):
        return -self.theta * x

    def diffusion(self, x):
        if isinstance(x, np.ndarray):
            return self.sigma * np.ones_like(x)
        else:
            return self.sigma * torch.ones_like(x)

    def simulate_trajectory(self, x0, T, N, M):
        dt = T / N
        trajectories = np.zeros((M, N + 1))
        trajectories[:, 0] = x0
        scale = dt ** (1.0 / self.alpha)
        levy_increments = levy_stable.rvs(alpha=self.alpha, beta=0, loc=0, scale=scale, size=(M, N))
        for i in range(N):
            x_curr = trajectories[:, i]
            dx = self.drift(x_curr) * dt + self.diffusion(x_curr) * levy_increments[:, i]
            trajectories[:, i + 1] = x_curr + dx
        return trajectories


def generate_training_data(ou_process, T, N, M, tail_ratio=0.3):
    x0 = np.random.normal(0, 1, M)
    trajectories = ou_process.simulate_trajectory(x0, T, N, M)
    all_points = []
    for m in range(M):
        for n in range(N + 1):
            t = n * T / N
            x = trajectories[m, n]
            if abs(x) < 10.0:
                all_points.append([x, t])
                if t < 0.1:
                    for _ in range(4): all_points.append([x, t])
    all_points = np.array(all_points)
    x_values = all_points[:, 0]
    threshold = np.percentile(np.abs(x_values), 70)
    high_mask = np.abs(x_values) <= threshold
    tail_mask = np.abs(x_values) > threshold
    n_total = 50000
    n_tail = int(n_total * tail_ratio)
    n_high = n_total - n_tail
    high_points = all_points[high_mask]
    tail_points = all_points[tail_mask]
    if len(high_points) > n_high:
        idx = np.random.choice(len(high_points), n_high, replace=False)
        high_points = high_points[idx]
    if len(tail_points) > 0:
        replace = len(tail_points) < n_tail
        idx = np.random.choice(len(tail_points), n_tail, replace=replace)
        tail_points = tail_points[idx]
    training_points = np.vstack([high_points, tail_points])
    np.random.shuffle(training_points)
    return training_points, trajectories


# ==========================================
# 工具函数 (计算 PDF)
# ==========================================
def compute_pdf(ll_net, x_grid, t):
    x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    t_t = torch.tensor(np.ones((len(x_grid), 1)) * t, dtype=torch.float32).to(device)
    with torch.no_grad():
        log_p = ll_net(x_t, t_t).cpu().numpy().flatten()
    pdf = np.exp(log_p - log_p.max())
    norm = simpson(pdf, x_grid)
    return pdf / (norm + 1e-8)


# ==========================================
# 动态收敛评估函数 (用于计算效能对比)
# ==========================================
def eval_convergence(net, mc_trajs, is_vanilla=False):
    """评估网络在稳态(t=1.0)的 Wasserstein 距离与质量偏差"""
    x_grid = np.linspace(-6, 6, 200)
    mc_data = mc_trajs[:, -1]  # 提取 t=1.0 的MC数据

    bins = np.linspace(-6, 6, 60)
    hist_vals, _ = np.histogram(mc_data, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if is_vanilla:
        x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
        t_t = torch.tensor(np.ones((len(x_grid), 1)) * 1.0, dtype=torch.float32).to(device)
        with torch.no_grad():
            p_pred = net(x_t, t_t).cpu().numpy().flatten()
        p_clean = np.maximum(0, p_pred)
        norm_v = simpson(p_clean, x_grid) + 1e-8
        v_weights = interp1d(x_grid, p_clean / norm_v, bounds_error=False, fill_value=0)(bin_centers)
        if np.sum(v_weights) <= 1e-10: v_weights = np.ones_like(v_weights) / len(v_weights)
        w_dist = wasserstein_distance(bin_centers, bin_centers, hist_vals, v_weights)
        mass_dev = np.abs(1.0 - simpson(p_pred, x_grid))
    else:
        pdf_ours = compute_pdf(net, x_grid, 1.0)
        ours_weights = interp1d(x_grid, pdf_ours, bounds_error=False, fill_value=0)(bin_centers)
        w_dist = wasserstein_distance(bin_centers, bin_centers, hist_vals, ours_weights)
        mass_dev = 0.0  # 归一化必然为 0

    return w_dist, mass_dev


# ==========================================
# 训练过程 (新增性能统计)
# ==========================================
def train_vanilla_network(vanilla_net, ou_process, training_data, mc_trajs, epochs=3000, batch_size=256, lr=1e-3,
                          alpha=1.5):
    optimizer = Adam(vanilla_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    start_time = time.time()
    epoch_times = []
    converged_epoch = -1
    converged_time = -1.0

    pbar = tqdm(range(epochs), desc="Vanilla训练", leave=False)
    for epoch in pbar:
        ep_start = time.time()

        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx].requires_grad_(True)
        t_batch = t_train[idx].requires_grad_(True)

        p_pred = vanilla_net(x_batch, t_batch)
        dp_dt = torch.autograd.grad(p_pred.sum(), t_batch, create_graph=True)[0]
        dp_dx = torch.autograd.grad(p_pred.sum(), x_batch, create_graph=True)[0]

        drift_term = -ou_process.theta * p_pred - ou_process.theta * x_batch * dp_dx
        diffusion_term = (ou_process.sigma ** alpha) * fractional_derivative_GL_with_boundary(
            vanilla_net, x_batch, alpha, t_batch, dx=0.08, domain_bound=6.0)

        loss_res = ((dp_dt - (drift_term + diffusion_term)) ** 2).mean()

        t0_mask = t_batch < 0.05
        loss_init = 0.0
        if t0_mask.any():
            p0_true = torch.exp(-0.5 * x_batch[t0_mask] ** 2) / np.sqrt(2 * np.pi)
            loss_init = ((p_pred[t0_mask] - p0_true) ** 2).mean()

        loss_pos = 10.0 * torch.mean(torch.relu(-p_pred) ** 2)
        loss_bc = (vanilla_net(torch.tensor([-6.0, 6.0], dtype=torch.float32).to(device).view(-1, 1),
                               (torch.rand(2, 1) * T).to(device)) ** 2).mean()

        loss_total = loss_res + 20.0 * loss_init + 1.0 * loss_bc + loss_pos

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(vanilla_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        ep_end = time.time()
        epoch_times.append(ep_end - ep_start)

        # 动态收敛评估 (每50轮)
        if (epoch + 1) % 50 == 0 and converged_epoch == -1:
            w_dist, _ = eval_convergence(vanilla_net, mc_trajs, is_vanilla=True)
            if w_dist < 0.1:
                converged_epoch = epoch + 1
                converged_time = time.time() - start_time

    avg_ep_time = np.mean(epoch_times) * 1000  # ms
    _, final_mass_dev = eval_convergence(vanilla_net, mc_trajs, is_vanilla=True)

    return converged_epoch, avg_ep_time, converged_time, final_mass_dev


def train_score_network(score_net, training_data, ou_process, epochs=2000, batch_size=256, lr=1e-3):
    optimizer = Adam(score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    start_time = time.time()
    pbar = tqdm(range(epochs), desc="Score训练", leave=False)
    for epoch in pbar:
        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx].requires_grad_(True)
        t_batch = t_train[idx]

        score_pred = score_net(x_batch, t_batch)
        grad_score = torch.autograd.grad(score_pred.sum(), x_batch, create_graph=True)[0]
        loss_ssm = 0.5 * (score_pred ** 2).mean() + grad_score.mean()

        loss_init = 0.0
        t0_mask = t_batch < 0.05
        if t0_mask.any():
            loss_init = ((score_pred[t0_mask] - (-x_batch[t0_mask])) ** 2).mean()

        loss_total = loss_ssm + 5.0 * loss_init
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return time.time() - start_time  # 返回第一阶段总耗时


def train_ll_network(ll_net, score_net, ou_process, training_data, mc_trajs, stage1_time, epochs=3000, batch_size=256,
                     lr=1e-3, alpha=1.5):
    optimizer = Adam(ll_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    def density_func(x, t):
        return torch.exp(ll_net(x, t))

    start_time = time.time()
    epoch_times = []
    converged_epoch = -1
    converged_time = -1.0

    pbar = tqdm(range(epochs), desc="LL训练", leave=False)
    for epoch in pbar:
        ep_start = time.time()

        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx].requires_grad_(True)
        t_batch = t_train[idx].requires_grad_(True)

        q_pred = ll_net(x_batch, t_batch)
        p_pred = torch.exp(q_pred)

        score_ll = torch.autograd.grad(q_pred.sum(), x_batch, create_graph=True)[0]
        dq_dt = torch.autograd.grad(q_pred.sum(), t_batch, create_graph=True)[0]

        with torch.no_grad():
            score_target = score_net(x_batch, t_batch)

        lhs = p_pred * dq_dt
        drift_term = -p_pred * (-ou_process.theta + -ou_process.theta * x_batch * score_ll)
        diffusion_term = (ou_process.sigma ** alpha) * fractional_derivative_GL_with_boundary(
            density_func, x_batch, alpha, t_batch, dx=0.08, domain_bound=6.0)

        residual = lhs - (drift_term + diffusion_term)
        loss_res = (residual ** 2).mean()
        loss_score = ((score_ll - score_target) ** 2).mean()

        loss_init = 0.0
        t0_mask = t_batch < 0.05
        if t0_mask.any():
            q0_true = -0.5 * x_batch[t0_mask] ** 2 - 0.5 * np.log(2 * np.pi)
            loss_init = ((q_pred[t0_mask] - q0_true) ** 2).mean()

        loss_total = loss_res + 0.5 * loss_score + 10.0 * loss_init

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        ep_end = time.time()
        epoch_times.append(ep_end - ep_start)

        # 动态收敛评估 (每50轮)
        if (epoch + 1) % 50 == 0 and converged_epoch == -1:
            w_dist, _ = eval_convergence(ll_net, mc_trajs, is_vanilla=False)
            if w_dist < 0.1:
                converged_epoch = epoch + 1
                # 记录总有效时间: 第一阶段总耗时 + 第二阶段当前耗时
                converged_time = stage1_time + (time.time() - start_time)

    avg_ep_time = np.mean(epoch_times) * 1000  # ms
    _, final_mass_dev = eval_convergence(ll_net, mc_trajs, is_vanilla=False)

    return converged_epoch, avg_ep_time, converged_time, final_mass_dev


def experiment_comparison(alpha=1.5):
    print(f"\n--- 运行计算效能与收敛性对比实验 Alpha = {alpha} ---")
    ou = FractionalOUProcess(theta=THETA, sigma=SIGMA, alpha=alpha)

    print("  正在生成蒙特卡洛基准与网络训练数据...")
    mc_trajs = ou.simulate_trajectory(np.random.normal(0, 1, 10000), T, N_TIME_STEPS, 10000)
    train_data, _ = generate_training_data(ou, T, N_TIME_STEPS, 500)

    score_net = ScoreNetwork().to(device)
    ll_net = LLNetwork().to(device)
    vanilla_net = VanillaNetwork().to(device)

    print("\n[1/3] 开始训练 Vanilla fPINN (基线模型)...")
    v_ep, v_ep_time, v_total_time, v_mass = train_vanilla_network(vanilla_net, ou, train_data, mc_trajs, alpha=alpha)

    print("\n[2/3] 开始训练 Score-Based PINN - 阶段一 (估计分数场)...")
    s1_time = train_score_network(score_net, train_data, ou)

    print("\n[3/3] 开始训练 Score-Based PINN - 阶段二 (重构对数似然)...")
    o_ep, o_ep_time, o_total_time, o_mass = train_ll_network(ll_net, score_net, ou, train_data, mc_trajs,
                                                             stage1_time=s1_time, alpha=alpha)

    # 格式化输出终端 Markdown 表格
    def fmt_time(t): return f"{t:.1f}" if t > 0 else "无法达到"

    def fmt_ep(e): return str(e) if e > 0 else "未收敛 (N/A)"

    print("\n" + "=" * 80)
    print("实验跑分完毕！请将以下 Markdown 表格复制到论文『表 4』中：")
    print("=" * 80)
    print(
        "| 模型 (Model) | 达到目标精度所需迭代数 ($W < 0.1$) | 单步平均耗时 (ms) | 达到目标精度总耗时 (s) | 最终概率质量守恒偏差 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    print(f"| Vanilla fPINN | {fmt_ep(v_ep)} | {v_ep_time:.1f} | {fmt_time(v_total_time)} | {v_mass * 100:.2f}% |")
    print(
        f"| **本文方法 (Ours)** | {fmt_ep(o_ep)} | {o_ep_time:.1f} | {fmt_time(o_total_time)} | **{o_mass * 100:.2f}%** |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    experiment_comparison(alpha=1.5)