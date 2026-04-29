"""
分数阶Fokker-Planck方程 Score-based 求解器 (最终修正版 v4 - 可视化增强)
修正日志：
1. [可视化] 直方图增加条纹(hatch)效果，更加生动
2. [可视化] 锁定X轴范围(-6, 6)，并动态调整Y轴使图像饱满
3. [可视化] 损失函数曲线仅显示平滑后的收敛趋势，移除杂乱的原始数据
4. [核心] 保留了纯Lévy噪声、GL算子修正、概率空间残差等核心算法

日期：2026-01-12
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
import matplotlib.pyplot as plt
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


def plot_spacetime_heatmap(ll_net, T, alpha, save_name):
    """
    绘制时空演化热力图 p(x,t)
    """
    print("  正在生成时空热力图...")

    # 1. 准备高分辨率网格
    t_fine = np.linspace(0, T, 100)
    x_fine = np.linspace(-6, 6, 200)
    X_grid, T_grid = np.meshgrid(x_fine, t_fine)

    # 2. 批量计算网络输出
    X_flat = torch.tensor(X_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    T_flat = torch.tensor(T_grid.reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        q_pred = ll_net(X_flat, T_flat).cpu().numpy().reshape(100, 200)

    # 3. 转换并通过积分归一化
    pdf_surface = np.exp(q_pred)
    for i in range(len(t_fine)):
        norm = simpson(pdf_surface[i, :], x_fine)
        pdf_surface[i, :] /= (norm + 1e-8)

    # 4. 绘图
    fig = plt.figure(figsize=(10, 6))
    # 使用 pcolormesh 绘制热力图，cmap='magma' 或 'viridis' 效果较好
    plt.pcolormesh(T_grid, X_grid, pdf_surface, shading='auto', cmap='magma')

    cbar = plt.colorbar()
    cbar.set_label('Probability Density $p(x,t)$', rotation=270, labelpad=15)

    plt.title(f'Spatiotemporal Evolution ($\\alpha={alpha}$)', fontsize=14)
    plt.xlabel('Time $t$', fontsize=12)
    plt.ylabel('Position $x$', fontsize=12)
    plt.ylim(-6, 6)
    plt.xlim(0, T)

    # 保存
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  热力图已保存至: {save_name}")

# ==========================================
# 实验主程序 (可视化调整版)
# ==========================================
def experiment_comparison(alpha):
    print(f"\n运行实验 Alpha = {alpha} ...")
    ou = FractionalOUProcess(theta=THETA, sigma=SIGMA, alpha=alpha)

    # 1. 生成MC基准
    print("  生成基准数据...")
    mc_trajs = ou.simulate_trajectory(np.random.normal(0, 1, 10000), T, N_TIME_STEPS, 10000)

    # 2. 训练
    print("  训练模型...")
    train_data, _ = generate_training_data(ou, T, N_TIME_STEPS, 500)
    score_net = ScoreNetwork().to(device)
    ll_net = LLNetwork().to(device)

    s_loss = train_score_network(score_net, train_data, ou)
    l_loss = train_ll_network(ll_net, score_net, ou, train_data, alpha=alpha)

    # 3. 计算误差 (移除绘图模块)
    print("  正在计算误差指标...")
    times = [0.0, 0.25, 0.5, 0.75]
    x_grid = np.linspace(-6, 6, 200)
    errors = {'L2': [], 'Linf': [], 'Wasserstein': []}

    for t in times:
        t_idx = int(t / T * N_TIME_STEPS)
        mc_data = mc_trajs[:, t_idx]
        bins = np.linspace(-6, 6, 60)

        pdf = compute_pdf(ll_net, x_grid, t)

        hist_vals, _ = np.histogram(mc_data, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if len(bin_centers) > 0 and np.sum(hist_vals) > 0:
            pdf_interp = interp1d(x_grid, pdf)(bin_centers)
            w_dist = wasserstein_distance(bin_centers, bin_centers, hist_vals, pdf_interp)

            dx = bin_centers[1] - bin_centers[0]
            l2_err = np.sqrt(np.sum((hist_vals - pdf_interp) ** 2 * dx))
            linf_err = np.max(np.abs(hist_vals - pdf_interp))
        else:
            w_dist, l2_err, linf_err = 0.0, 0.0, 0.0

        errors['Wasserstein'].append(w_dist)
        errors['L2'].append(l2_err)
        errors['Linf'].append(linf_err)

    return errors


if __name__ == "__main__":
    import pandas as pd

    alphas = [1.5, 1.6, 1.7, 1.8]
    times = [0.0, 0.25, 0.5, 0.75]
    all_results = {}

    # 全局锁定随机种子，保证单次循环与独立运行的基准状态完全一致
    torch.manual_seed(42)
    np.random.seed(42)

    for a in alphas:
        all_results[a] = experiment_comparison(a)

    rows = []
    metrics = ['L2 Error', 'Linf Error', 'Wasserstein']
    metric_keys = ['L2', 'Linf', 'Wasserstein']

    for m_idx, metric_name in enumerate(metrics):
        for t_idx, t in enumerate(times):
            row = {'Metric': metric_name if t_idx == 0 else '', 'Time': f"t={t:.2f}"}
            for a in alphas:
                row[f'a={a}'] = f"{all_results[a][metric_keys[m_idx]][t_idx]:.4E}"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("experimental_errors_table.csv", index=False)
    print("\n==================================================")
    print("实验一误差表格预览：")
    print(df.to_string(index=False))
    print("==================================================")