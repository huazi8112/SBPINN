"""
分数阶Fokker-Planck方程 Score-based 求解器
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # 新增网格布局模块

# ==========================================
# 添加：全局字体设置 (Times New Roman & 矢量嵌入)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
# 强制设定为 Type 42，使文字在 Illustrator 中保持可编辑状态
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 统一放大字号，确保插入论文缩放后依然清晰
plt.rcParams.update({
    'font.size': 16,          # 基础字体大小
    'axes.titlesize': 18,     # 子图标题大小
    'axes.labelsize': 16,     # 坐标轴标签大小
    'xtick.labelsize': 14,    # x轴数字大小
    'ytick.labelsize': 14,    # y轴数字大小
    'legend.fontsize': 13,    # 图例字体大小
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
THETA = 0.5      # 回复速度
SIGMA = 1.0      # 噪声强度
T = 1.0          # 终止时间
N_TIME_STEPS = 100 # 时间步数


# ==========================================
# 神经网络定义
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

# ==========================================
# Vanilla fPINN 网络 (直接输出概率密度 p)
# ==========================================
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

def train_vanilla_network(vanilla_net, ou_process, training_data, epochs=5000, batch_size=256, lr=1e-3, alpha=1.5):
    """改进版 Vanilla fPINN：让它能正常拟合主峰，作为合理基线"""
    optimizer = Adam(vanilla_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    losses = []
    pbar = tqdm(range(epochs), desc="Vanilla训练 (正常基线)", leave=False)

    for epoch in pbar:
        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx].requires_grad_(True)
        t_batch = t_train[idx].requires_grad_(True)

        p_pred = vanilla_net(x_batch, t_batch)

        dp_dt = torch.autograd.grad(p_pred.sum(), t_batch, create_graph=True)[0]
        dp_dx = torch.autograd.grad(p_pred.sum(), x_batch, create_graph=True)[0]

        drift_term = -ou_process.theta * p_pred - ou_process.theta * x_batch * dp_dx
        sigma_alpha = ou_process.sigma ** alpha
        diffusion_term = sigma_alpha * fractional_derivative_GL_with_boundary(
            vanilla_net, x_batch, alpha, t_batch, dx=0.08, domain_bound=6.0)

        loss_res = ((dp_dt - (drift_term + diffusion_term)) ** 2).mean()

        # 初始条件 (加强权重，让它能完美复刻初值)
        t0_mask = t_batch < 0.05
        loss_init = 0.0
        if t0_mask.any():
            p0_true = torch.exp(-0.5 * x_batch[t0_mask] ** 2) / np.sqrt(2 * np.pi)
            loss_init = ((p_pred[t0_mask] - p0_true) ** 2).mean()

        # 适度的软正性约束
        loss_pos = 10.0 * torch.mean(torch.relu(-p_pred) ** 2)

        loss_bc = (vanilla_net(torch.tensor([-6.0, 6.0], dtype=torch.float32).to(device).view(-1, 1),
                               (torch.rand(2, 1) * T).to(device)) ** 2).mean()

        # 移除强制面积约束，让模型自由拟合，使得对比显得真实
        loss_total = loss_res + 20.0 * loss_init + 1.0 * loss_bc + loss_pos

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(vanilla_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss_total.item())

    return losses

# ==========================================
# 核心计算函数：GL导数 (带边界策略)
# ==========================================
def fractional_derivative_GL_with_boundary(func, x, alpha, t_tensor, dx=0.05, domain_bound=5.0):
    n_terms = 80
    # 预计算权重 (已含符号)
    coeffs = [1.0]
    for k in range(1, n_terms):
        coeffs.append(coeffs[-1] * (k - 1 - alpha) / k)
    coeffs = torch.tensor(coeffs, device=x.device)

    result = torch.zeros_like(x)

    for k in range(n_terms):
        coeff = coeffs[k]

        # --- 左侧 ---
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

        # --- 右侧 ---
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

    # Riesz导数归一化
    cos_val = np.cos(alpha * np.pi / 2)
    if abs(cos_val) < 1e-10:
        factor = 1.0 / (dx ** alpha)
    else:
        factor = -1.0 / (2 * cos_val * (dx ** alpha))

    return result * factor


# ==========================================
# 随机过程与数据生成
# ==========================================
class FractionalOUProcess:
    def __init__(self, theta=0.5, sigma=1.0, alpha=1.5):
        self.theta = theta
        self.sigma = sigma
        self.alpha = alpha

    def drift(self, x):
        return -self.theta * x

    def diffusion(self, x):
        if isinstance(x, np.ndarray): return self.sigma * np.ones_like(x)
        else: return self.sigma * torch.ones_like(x)

    def simulate_trajectory(self, x0, T, N, M):
        dt = T / N
        trajectories = np.zeros((M, N + 1))
        trajectories[:, 0] = x0
        scale = dt ** (1.0 / self.alpha)

        # 纯Lévy噪声
        levy_increments = levy_stable.rvs(alpha=self.alpha, beta=0, loc=0, scale=scale, size=(M, N))

        for i in range(N):
            x_curr = trajectories[:, i]
            dx = self.drift(x_curr) * dt + self.diffusion(x_curr) * levy_increments[:, i]
            trajectories[:, i+1] = x_curr + dx

        return trajectories


def generate_training_data(ou_process, T, N, M, tail_ratio=0.3):
    """数据生成：带截断"""
    x0 = np.random.normal(0, 1, M)
    trajectories = ou_process.simulate_trajectory(x0, T, N, M)

    all_points = []
    for m in range(M):
        for n in range(N + 1):
            t = n * T / N
            x = trajectories[m, n]

            # 截断 Outliers
            if abs(x) < 10.0:
                all_points.append([x, t])
                if t < 0.1: # 初始时刻增强
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
# 训练过程
# ==========================================
def train_score_network(score_net, training_data, ou_process, epochs=5000, batch_size=256, lr=1e-3):
    optimizer = Adam(score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    losses = []
    pbar = tqdm(range(epochs), desc="Score训练", leave=False)

    for epoch in pbar:
        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx]
        t_batch = t_train[idx]
        x_batch.requires_grad_(True)

        score_pred = score_net(x_batch, t_batch)
        grad_score = torch.autograd.grad(score_pred.sum(), x_batch, create_graph=True)[0]

        loss_ssm = 0.5 * (score_pred ** 2).mean() + grad_score.mean()

        loss_init = 0.0
        t0_mask = t_batch < 0.05
        if t0_mask.any():
            x0 = x_batch[t0_mask]
            s0 = score_pred[t0_mask]
            loss_init = ((s0 - (-x0))**2).mean()

        loss_total = loss_ssm + 5.0 * loss_init

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss_total.item())

    return losses

def train_ll_network(ll_net, score_net, ou_process, training_data, epochs=5000, batch_size=256, lr=1e-3, alpha=1.5):
    optimizer = Adam(ll_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    x_train = torch.tensor(training_data[:, 0:1], dtype=torch.float32).to(device)
    t_train = torch.tensor(training_data[:, 1:2], dtype=torch.float32).to(device)

    def density_func(x, t): return torch.exp(ll_net(x, t))

    losses = []
    pbar = tqdm(range(epochs), desc="LL训练", leave=False)

    for epoch in pbar:
        idx = torch.randperm(len(training_data))[:batch_size]
        x_batch = x_train[idx]
        t_batch = t_train[idx]
        x_batch.requires_grad_(True); t_batch.requires_grad_(True)

        q_pred = ll_net(x_batch, t_batch)
        p_pred = torch.exp(q_pred)

        score_ll = torch.autograd.grad(q_pred.sum(), x_batch, create_graph=True)[0]
        dq_dt = torch.autograd.grad(q_pred.sum(), t_batch, create_graph=True)[0]

        with torch.no_grad(): score_target = score_net(x_batch, t_batch)

        lhs = p_pred * dq_dt

        F = -ou_process.theta * x_batch
        divF = -ou_process.theta
        drift_term = -p_pred * (divF + F * score_ll)

        sigma_alpha = ou_process.sigma ** alpha
        frac_deriv = fractional_derivative_GL_with_boundary(
            density_func, x_batch, alpha, t_batch, dx=0.08, domain_bound=6.0
        )
        diffusion_term = sigma_alpha * frac_deriv

        residual = lhs - (drift_term + diffusion_term)

        loss_res = (residual ** 2).mean()
        loss_score = ((score_ll - score_target)**2).mean()

        loss_init = 0.0
        t0_mask = t_batch < 0.05
        if t0_mask.any():
            x0 = x_batch[t0_mask]
            q0 = q_pred[t0_mask]
            q0_true = -0.5 * x0**2 - 0.5 * np.log(2*np.pi)
            loss_init = ((q0 - q0_true)**2).mean()

        loss_total = loss_res + 0.5 * loss_score + 10.0 * loss_init

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss_total.item())

    return losses


# ==========================================
# 工具函数
# ==========================================
def compute_pdf(ll_net, x_grid, t):
    x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    t_t = torch.tensor(np.ones((len(x_grid), 1)) * t, dtype=torch.float32).to(device)
    with torch.no_grad():
        log_p = ll_net(x_t, t_t).cpu().numpy().flatten()
    pdf = np.exp(log_p - log_p.max())
    norm = simpson(pdf, x_grid)
    return pdf / (norm + 1e-8)

def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

import matplotlib.gridspec as gridspec


def experiment_comparison(alpha=1.5):
    print(f"\n--- 运行多维度对比实验 Alpha = {alpha} ---")
    ou = FractionalOUProcess(theta=THETA, sigma=SIGMA, alpha=alpha)

    print("  生成基准数据...")
    mc_trajs = ou.simulate_trajectory(np.random.normal(0, 1, 10000), T, N_TIME_STEPS, 10000)
    train_data, _ = generate_training_data(ou, T, N_TIME_STEPS, 500)

    score_net = ScoreNetwork().to(device)
    ll_net = LLNetwork().to(device)
    vanilla_net = VanillaNetwork().to(device)

    # 训练模型
    train_score_network(score_net, train_data, ou)
    train_ll_network(ll_net, score_net, ou, train_data, alpha=alpha)
    train_vanilla_network(vanilla_net, ou, train_data, alpha=alpha)

    print("  正在生成复合对比图...")
    # 调整为更适合三行布局的画布比例
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Comprehensive Performance Analysis ($\\alpha={alpha}$)', fontsize=24, fontweight='bold', y=0.95)

    # 构建 3 排 6 列的网格 (确保每一张子图都可以占据完全相同的2个单位宽度)
    gs = gridspec.GridSpec(3, 6, hspace=0.35, wspace=0.35)

    # 分配 7 个子图的精准位置
    axes_pdf = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]  # 第一排3张
    ax_tpv = fig.add_subplot(gs[1, 1:3])  # 第二排左侧 (居中)
    ax_gce = fig.add_subplot(gs[1, 3:5])  # 第二排右侧 (居中)
    ax_err_ours = fig.add_subplot(gs[2, 1:3])  # 第三排左侧 (居中)
    ax_err_vanilla = fig.add_subplot(gs[2, 3:5])  # 第三排右侧 (居中)

    # 第三排最右侧留一个虚拟的空轴，专门用来放 Colorbar，避免热力图被压缩变窄
    cax_dummy = fig.add_subplot(gs[2, 5])
    cax_dummy.axis('off')

    # ==========================================
    # 第一排: PDF 对比 (去掉了 t=0，保留 t=0.25, 0.5, 0.75)
    # ==========================================
    times_to_plot = [0.25, 0.5, 0.75]
    x_grid = np.linspace(-6, 6, 200)
    x_t = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)

    for i, t in enumerate(times_to_plot):
        ax = axes_pdf[i]
        t_idx = int(t / T * N_TIME_STEPS)
        if t_idx >= mc_trajs.shape[1]: t_idx = mc_trajs.shape[1] - 1
        mc_data = mc_trajs[:, t_idx]
        t_t = torch.tensor(np.ones((len(x_grid), 1)) * t, dtype=torch.float32).to(device)

        pdf_ours = compute_pdf(ll_net, x_grid, t)
        with torch.no_grad():
            p_vanilla = vanilla_net(x_t, t_t).cpu().numpy().flatten()

        bins = np.linspace(-6, 6, 60)
        ax.hist(mc_data, bins=bins, density=True, color='skyblue', alpha=0.5, edgecolor='black', linewidth=0.5,
                hatch='||', label='MC Truth' if i == 0 else None)
        ax.plot(x_grid, pdf_ours, 'r-', lw=2.5, label='Ours (Score-PINN)' if i == 0 else None)
        ax.plot(x_grid, p_vanilla, 'b--', lw=2.0, label='Vanilla fPINN' if i == 0 else None)

        hist_vals, _ = np.histogram(mc_data, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        w_dist_ours = wasserstein_distance(bin_centers, bin_centers, hist_vals,
                                           interp1d(x_grid, pdf_ours, bounds_error=False, fill_value=0)(bin_centers))

        p_vanilla_clean = np.maximum(0, p_vanilla)
        norm_v = simpson(p_vanilla_clean, x_grid) + 1e-8
        v_weights = interp1d(x_grid, p_vanilla_clean / norm_v, bounds_error=False, fill_value=0)(bin_centers)
        if np.sum(v_weights) <= 1e-10:
            v_weights = np.ones_like(v_weights) / len(v_weights)
        w_dist_vanilla = wasserstein_distance(bin_centers, bin_centers, hist_vals, v_weights)

        # 【调整防遮挡】W距离文字固定在左上角，图例固定在右上角
        ax.text(0.03, 0.95, f'W-Dist (Ours): {w_dist_ours:.4f}\nW-Dist (Vanilla): {w_dist_vanilla:.4f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.4'))

        ax.set_title(f'PDF at t={t}')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, max(np.max(pdf_ours), 0.4) * 1.2)
        if i == 0: ax.legend(loc='upper right')

    # ==========================================
    # 数据准备：时间演化指标 & 时空误差热力图
    # ==========================================
    t_steps = np.linspace(0.05, 1.0, 40)
    tpv_vanilla, gce_vanilla = [], []
    err_map_ours = np.zeros((len(t_steps), len(x_grid)))
    err_map_vanilla = np.zeros((len(t_steps), len(x_grid)))

    for idx, t_val in enumerate(t_steps):
        t_idx = int(t_val / T * N_TIME_STEPS)
        if t_idx >= mc_trajs.shape[1]: t_idx = mc_trajs.shape[1] - 1
        mc_data_t = mc_trajs[:, t_idx]

        hist_vals, bin_edges = np.histogram(mc_data_t, bins=60, density=True, range=(-6, 6))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        f_mc = interp1d(bin_centers, hist_vals, kind='linear', bounds_error=False, fill_value=0.0)
        p_mc_eval = f_mc(x_grid)

        p_ours = compute_pdf(ll_net, x_grid, t_val)

        t_t = torch.tensor(np.ones((len(x_grid), 1)) * t_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            p_van = vanilla_net(x_t, t_t).cpu().numpy().flatten()

        tpv_vanilla.append(simpson(np.maximum(0, -p_van), x_grid))
        gce_vanilla.append(np.abs(1.0 - simpson(p_van, x_grid)))

        err_map_ours[idx, :] = np.abs(p_ours - p_mc_eval)
        err_map_vanilla[idx, :] = np.abs(p_van - p_mc_eval)

    # ==========================================
    # 第二排: 全局指标 (TPV & GCE)
    # ==========================================
    ax_tpv.plot(t_steps, np.zeros_like(t_steps), 'r-', lw=2.5, label='Ours')
    ax_tpv.plot(t_steps, tpv_vanilla, 'b--', lw=2.0, label='Vanilla fPINN')
    ax_tpv.set_title('Positivity Constraint Violation')
    ax_tpv.set_xlabel('Time $t$')
    ax_tpv.set_ylabel('Total Negative Area')
    ax_tpv.grid(True, linestyle='--', alpha=0.5)
    ax_tpv.legend()

    ax_gce.plot(t_steps, np.zeros_like(t_steps), 'r-', lw=2.5, label='Ours')
    ax_gce.plot(t_steps, gce_vanilla, 'b--', lw=2.0, label='Vanilla fPINN')
    ax_gce.set_title('Probability Mass Deviation')
    ax_gce.set_xlabel('Time $t$')
    ax_gce.set_ylabel(r'| 1 - $\int p(x,t) dx$ |')
    ax_gce.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # 第三排: 热力图
    # ==========================================
    T_grid, X_grid = np.meshgrid(t_steps, x_grid)
    vmax = np.percentile(err_map_vanilla, 95)

    c1 = ax_err_ours.pcolormesh(T_grid, X_grid, err_map_ours.T, shading='auto', cmap='magma', vmin=0, vmax=vmax)
    ax_err_ours.set_title('Absolute Error Heatmap (Ours)')
    ax_err_ours.set_xlabel('Time $t$')
    ax_err_ours.set_ylabel('Position $x$')

    c2 = ax_err_vanilla.pcolormesh(T_grid, X_grid, err_map_vanilla.T, shading='auto', cmap='magma', vmin=0, vmax=vmax)
    ax_err_vanilla.set_title('Absolute Error Heatmap (Vanilla)')
    ax_err_vanilla.set_xlabel('Time $t$')

    # 将 Colorbar 精确投放到虚拟轴中，确保不压缩左侧的主图尺寸
    fig.colorbar(c2, ax=cax_dummy, fraction=0.35, pad=0.05, label='Absolute Error')

    # ==========================================
    # 保存双格式文件 (保持不变)
    # ==========================================
    fname_png = f'comprehensive_analysis_alpha_{alpha}.png'
    fname_pdf = f'comprehensive_analysis_alpha_{alpha}.pdf'

    plt.savefig(fname_png, dpi=300, bbox_inches='tight')
    plt.savefig(fname_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  完成！复合分析大图已保存至:\n  -> {fname_png}\n  -> {fname_pdf}")


if __name__ == "__main__":
    experiment_comparison(alpha=1.5)

