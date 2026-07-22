"""Evaluate the nonlinear bistable SBPINN experiment for multiple stability indices.

The implementation retains the probability-space residual, asymptotic exterior
padding, and two-expert reconstruction used for the manuscript results.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance, levy_stable
from scipy.interpolate import interp1d

# 全局配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. 辅助工具
# ==========================================
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# ==========================================
# 1. 网络架构
# ==========================================
class ScoreNetwork(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

class LLNetwork(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

# ==========================================
# 2. GL 分数阶导数 (修复维度Bug版)
# ==========================================
def fractional_derivative_GL_Safe(func_exp_q, x, t, alpha, dx=0.02, limit=5.0):
    """
    func_exp_q: 输入函数，计算的是 e^q(x)
    该函数包含边界渐进填充策略，并修复了维度展平问题
    """
    M = int(limit / dx)
    attr_name = f'coeffs_{alpha}'
    if not hasattr(fractional_derivative_GL_Safe, attr_name):
        coeffs = [1.0]
        for k in range(1, M + 1):
            coeffs.append(coeffs[-1] * (alpha - k + 1) / k)
        setattr(fractional_derivative_GL_Safe, attr_name, torch.tensor(coeffs, device=x.device))

    coeffs = getattr(fractional_derivative_GL_Safe, attr_name)

    val_left = torch.zeros_like(x)
    val_right = torch.zeros_like(x)

    def get_val_with_boundary(coords, t_tensor):
        # 1. 创建掩码
        mask_in = (coords >= -limit) & (coords <= limit)
        out_vals = torch.zeros_like(coords)

        # 2. 处理域内点 (Training Domain)
        if mask_in.any():
            # [关键修正] .view(-1, 1) 恢复二维形状
            x_in = coords[mask_in].view(-1, 1)
            t_in = t_tensor[mask_in].view(-1, 1)

            # 计算并展平回 1D 以匹配 mask 索引
            out_vals[mask_in] = func_exp_q(x_in, t_in).view(-1)

        # 3. 处理域外点 (Boundary Filling)
        if (~mask_in).any():
            x_out_flat = coords[~mask_in] # 1D
            t_out_flat = t_tensor[~mask_in] # 1D

            # 构造边界输入 (二维)
            x_bound_in = (torch.sign(x_out_flat) * limit).view(-1, 1)
            t_bound_in = t_out_flat.view(-1, 1)

            # 计算边界值并展平
            p_bound = func_exp_q(x_bound_in, t_bound_in).view(-1)

            # 计算衰减 (使用 1D 计算)
            decay = (limit / torch.abs(x_out_flat)) ** (1 + alpha)

            out_vals[~mask_in] = p_bound * decay

        return out_vals

    for k in range(M + 1):
        c = coeffs[k] * ((-1)**k)
        x_l = x - k*dx
        val_left += c * get_val_with_boundary(x_l, t)
        x_r = x + k*dx
        val_right += c * get_val_with_boundary(x_r, t)

    coef_riesz = -0.5 / np.cos(alpha * np.pi / 2) / (dx**alpha)
    return coef_riesz * (val_left + val_right)

# ==========================================
# 3. SDE & 数据 (纯 Levy 噪声)
# ==========================================
class BistableSDE:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def drift(self, x):
        return -x**3 + 1.5*(x**2) - 0.54*x

    def simulate(self, x0, T, steps):
        dt = T / steps
        M = x0.shape[0]
        traj = np.zeros((M, steps + 1))
        traj[:, 0] = x0

        scale = dt ** (1.0 / self.alpha)
        # 仅在第一次调用时打印
        print(f"Generating Pure Lévy Noise (Alpha={self.alpha})...")
        levy_noise = levy_stable.rvs(self.alpha, 0, loc=0, scale=scale, size=(M, steps))

        for i in range(steps):
            x = traj[:, i]
            noise = levy_noise[:, i]
            x_safe = np.clip(x, -10, 10)
            d_val = self.drift(x_safe)
            x_new = x + d_val*dt + self.sigma * noise
            traj[:, i+1] = np.clip(x_new, -8, 8)
        return traj

def generate_split_data(sde, T, N_train, M):
    SIM_MULTIPLIER = 10
    N_sim = N_train * SIM_MULTIPLIER
    M_half = M // 2
    x0_L = np.random.normal(-0.2, 0.25, M_half)
    x0_R = np.random.normal(1.0, 0.25, M_half)
    x0 = np.concatenate([x0_L, x0_R])

    traj_high = sde.simulate(x0, T, N_sim)
    traj = traj_high[:, ::SIM_MULTIPLIER]
    traj_L = traj[:M_half, :]
    traj_R = traj[M_half:, :]

    def build_dataset(trajectory):
        data = []
        for i in range(N_train + 1):
            t = i * T / N_train
            x_vals = trajectory[:, i]
            valid_mask = np.isfinite(x_vals) & (np.abs(x_vals) < 5.0)
            x_vals = x_vals[valid_mask]
            if len(x_vals) > 800:
                x_vals = np.random.choice(x_vals, 800, replace=False)
            for x in x_vals:
                data.append([x, t])
        return np.array(data, dtype=np.float32)

    data_L = build_dataset(traj_L)
    data_R = build_dataset(traj_R)
    return data_L, data_R, traj

# ==========================================
# 4. 训练流程 (严格公式匹配)
# ==========================================
def train_expert(name, data, sde, iterations=5000, batch_size=256):
    x_all = torch.tensor(data[:, 0:1], device=DEVICE)
    t_all = torch.tensor(data[:, 1:2], device=DEVICE)
    N_data = x_all.shape[0]

    # --- Phase 1: Score Training ---
    score_net = ScoreNetwork().to(DEVICE)
    opt_s = Adam(score_net.parameters(), lr=1e-3)
    loss_history_s = []

    pbar = tqdm(range(iterations), desc=f"{name}-Score", leave=False)
    for i in pbar:
        idx = torch.randint(0, N_data, (batch_size,), device=DEVICE)
        bx, bt = x_all[idx], t_all[idx]
        bx.requires_grad_(True)

        s = score_net(bx, bt)
        grad_s = torch.autograd.grad(s.sum(), bx, create_graph=True)[0]
        loss = 0.5 * (s**2).mean() + grad_s.mean()

        opt_s.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        opt_s.step()
        if i % 10 == 0: loss_history_s.append(loss.item())

    # --- Phase 2: LL Training ---
    ll_net = LLNetwork().to(DEVICE)
    opt_l = Adam(ll_net.parameters(), lr=1e-3)
    loss_history_l = []

    pbar = tqdm(range(iterations), desc=f"{name}-LL", leave=False)
    for i in pbar:
        idx = torch.randint(0, N_data, (batch_size,), device=DEVICE)
        bx, bt = x_all[idx], t_all[idx]
        bx.requires_grad_(True)
        bt.requires_grad_(True)

        # 1. 变量准备
        q = ll_net(bx, bt)
        exp_q = torch.exp(q)
        s_target = score_net(bx, bt).detach()

        # 2. 时间导数
        dq_dt = torch.autograd.grad(q.sum(), bt, create_graph=True)[0]

        # 3. 空间项
        bx_safe = torch.clamp(bx, -5.0, 5.0)
        f_val = -bx_safe**3 + 1.5*(bx_safe**2) - 0.54*bx_safe
        div_f = -3*(bx_safe**2) + 3*bx_safe - 0.54

        # 4. 分数阶项
        func_exp_q = lambda x, t: torch.exp(ll_net(x, t))
        frac_term_val = fractional_derivative_GL_Safe(func_exp_q, bx, bt, sde.alpha)

        sigma_alpha = sde.sigma ** sde.alpha
        D_alpha_term = sigma_alpha * frac_term_val

        # 5. 残差组装 (Fokker-Planck probability space form)
        # Residual = e^q * dq/dt + e^q * (div_f + f * s) - sigma^alpha * D^alpha[e^q]

        term1 = exp_q * dq_dt
        term2 = exp_q * (div_f + f_val * s_target)
        term3 = D_alpha_term

        residual = term1 + term2 - term3

        # 6. Score Matching 约束
        grad_q = torch.autograd.grad(q.sum(), bx, create_graph=True)[0]
        loss_sm = ((grad_q - s_target)**2).mean()

        loss = (residual**2).mean() + 1.0 * loss_sm

        opt_l.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        opt_l.step()

        if i % 10 == 0: loss_history_l.append(loss.item())

    return score_net, ll_net, loss_history_s, loss_history_l

# ==========================================
# 5. 批处理实验
# ==========================================
def run_experiment(target_alpha):
    SIGMA = 0.3
    T_MAX = 3.0
    ITERATIONS = 5000

    print(f"\n" + "=" * 50)
    print(f"开始运行实验: Alpha = {target_alpha}")
    print("=" * 50)

    sde = BistableSDE(alpha=target_alpha, sigma=SIGMA)
    data_L, data_R, traj_full = generate_split_data(sde, T_MAX, N_train=100, M=8000)

    score_L, ll_L, ls_L, ll_L_hist = train_expert(f"Exp-L(a={target_alpha})", data_L, sde, iterations=ITERATIONS)
    score_R, ll_R, ls_R, ll_R_hist = train_expert(f"Exp-R(a={target_alpha})", data_R, sde, iterations=ITERATIONS)

    # === 计算指标 (移除了画图模块) ===
    print(f"正在计算三种误差指标 (Alpha={target_alpha})...")
    t_evals = [0.0, 1.0, 2.0, 3.0]

    x_grid = np.linspace(-2.5, 3.0, 200)
    x_ts = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    errors = {'L2': [], 'Linf': [], 'Wasserstein': []}

    for idx, t in enumerate(t_evals):
        t_idx = int(t / T_MAX * 100)
        t_idx = min(t_idx, 100)
        mc_data = np.concatenate([traj_full[:, t_idx]])
        mc_data = mc_data[np.abs(mc_data) < 10]

        t_ts = torch.ones_like(x_ts) * t
        with torch.no_grad():
            log_p_L = ll_L(x_ts, t_ts).cpu().numpy().flatten()
            p_L = np.exp(log_p_L - log_p_L.max())
            p_L /= (simpson(p_L, x_grid) + 1e-8)

            log_p_R = ll_R(x_ts, t_ts).cpu().numpy().flatten()
            p_R = np.exp(log_p_R - log_p_R.max())
            p_R /= (simpson(p_R, x_grid) + 1e-8)

        p_total = 0.5 * p_L + 0.5 * p_R

        hist, bin_edges = np.histogram(mc_data, bins=80, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        f_interp = interp1d(bin_centers, hist, bounds_error=False, fill_value=0.0)
        p_mc_interp = f_interp(x_grid)
        p_mc_interp /= (simpson(p_mc_interp, x_grid) + 1e-8)

        w_dist = wasserstein_distance(x_grid, x_grid, p_mc_interp, p_total)
        l2_err = np.sqrt(simpson((p_mc_interp - p_total) ** 2, x=x_grid))
        linf_err = np.max(np.abs(p_mc_interp - p_total))

        errors['Wasserstein'].append(w_dist)
        errors['L2'].append(l2_err)
        errors['Linf'].append(linf_err)

    return errors


if __name__ == "__main__":
    import pandas as pd

    alpha_list = [1.5, 1.6, 1.7, 1.8]
    t_evals = [0.0, 1.0, 2.0, 3.0]
    all_results = {}

    # 全局锁定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    for a in alpha_list:
        all_results[a] = run_experiment(a)

    rows = []
    metrics = ['L2 Error', 'Linf Error', 'Wasserstein']
    metric_keys = ['L2', 'Linf', 'Wasserstein']

    for m_idx, metric_name in enumerate(metrics):
        for t_idx, t in enumerate(t_evals):
            row = {'Metric': metric_name if t_idx == 0 else '', 'Time': f"t={t:.1f}"}
            for a in alpha_list:
                row[f'a={a}'] = f"{all_results[a][metric_keys[m_idx]][t_idx]:.4E}"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("experiment2_errors_table.csv", index=False)

    print("\n==================================================")
    print("实验二误差表格预览：")
    print(df.to_string(index=False))
    print("==================================================")