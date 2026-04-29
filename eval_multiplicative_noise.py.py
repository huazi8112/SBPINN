import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import levy_stable, wasserstein_distance
from scipy.integrate import simpson
from tqdm import tqdm
from scipy.interpolate import interp1d
# ==========================================
# 0. 全局配置
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def smooth_curve(points, factor=0.8):
    if len(points) == 0: return []
    smoothed = []
    for p in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed


SIGMA_CONST = 1.0
T_END = 1.0
DT = 0.001
N_STEPS = int(T_END / DT)
PHYSICAL_LIMIT = 8.0


# ==========================================
# 1. 网络模型
# ==========================================
class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Softplus(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.Softplus(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.Softplus(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class LLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Softplus(),
            nn.Linear(64, 64), nn.Softplus(),
            nn.Linear(64, 64), nn.Softplus(),
            nn.Linear(64, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. 数据与参数
# ==========================================
def drift_func(x): return -x


def diffusion_func(x):
    if isinstance(x, torch.Tensor):
        return 1.0 + 0.5 * torch.sin(x) ** 2
    return 1.0 + 0.5 * np.sin(x) ** 2


def calculate_effective_diffusion_median(x_data_tensor, alpha):
    print(f"Computing Effective Diffusion (Median Strategy for a={alpha})...")
    with torch.no_grad():
        center = torch.median(x_data_tensor)
        g_phys = diffusion_func(center)
        g_pow = torch.pow(g_phys, alpha)
    print(f"  -> Physical g^alpha: {g_pow.item():.4f}")
    return g_pow.item()


def generate_mc_data_stratified(n_samples, alpha):
    print(f"Generating Pure Lévy Noise (Alpha={alpha})...")
    raw_samples = int(n_samples * 1.5)
    x = np.random.normal(0, 0.5, raw_samples)
    curr_x = x
    scale = DT ** (1 / alpha)

    for i in range(N_STEPS):
        f = drift_func(curr_x)
        g = diffusion_func(curr_x)
        noise = levy_stable.rvs(alpha, 0, scale=scale, size=raw_samples)
        curr_x = curr_x + f * DT + g * noise

    final_x_mc = curr_x
    x_vals = final_x_mc
    thresh = np.percentile(np.abs(x_vals), 75)
    mask_core = np.abs(x_vals) <= thresh
    mask_tail = ~mask_core
    data_core = x_vals[mask_core]
    data_tail = x_vals[mask_tail]

    n_target = 10000
    n_c = int(n_target * 0.7)
    n_t = int(n_target * 0.3)

    if len(data_tail) == 0:
        idx_t = []
        n_c = n_target
    else:
        idx_t = np.random.choice(len(data_tail), n_t, replace=True)

    idx_c = np.random.choice(len(data_core), n_c, replace=True)

    if len(idx_t) > 0:
        x_train = np.concatenate([data_core[idx_c], data_tail[idx_t]])
    else:
        x_train = data_core[idx_c]

    t_train = np.full(len(x_train), T_END)
    train_data = np.stack([x_train, t_train], axis=1)
    np.random.shuffle(train_data)

    return train_data, final_x_mc


# ==========================================
# 3. 优化后的 GL 算子 (支持预计算权重)
# ==========================================
def gl_fractional_derivative_optimized(ll_net, x, t, alpha, dx, M, limit, precomputed_weights=None):
    """
    优化版算子：直接接收预计算好的 weights，避免重复循环
    """
    B = x.shape[0]

    # 使用预计算权重
    if precomputed_weights is not None:
        weights = precomputed_weights
    else:
        # Fallback (不推荐)
        weights = torch.zeros(M, device=x.device)
        w = 1.0
        for k in range(M):
            weights[k] = w
            w = w * (1 - (alpha + 1) / (k + 1))
        weights = weights.view(1, M)

    k_vec = torch.arange(M, device=x.device).float().view(1, M)
    x_left_mat = x - k_vec * dx
    x_right_mat = x + k_vec * dx
    t_mat = t.expand(B, M)

    inp_left = torch.stack([x_left_mat.flatten(), t_mat.flatten()], dim=1)
    inp_right = torch.stack([x_right_mat.flatten(), t_mat.flatten()], dim=1)

    def compute_batch_with_padding(inp_tensor):
        x_in = inp_tensor[:, 0:1]
        t_in = inp_tensor[:, 1:2]

        mask_in = (torch.abs(x_in) <= limit)
        x_clamped = torch.clamp(x_in, -limit, limit)
        inp_safe = torch.cat([x_clamped, t_in], dim=1)
        val_raw = torch.exp(ll_net(inp_safe))

        x_sign = torch.sign(x_in)
        inp_bound = torch.cat([x_sign * limit, t_in], dim=1)
        val_bound = torch.exp(ll_net(inp_bound))
        dist_from_origin = torch.abs(x_in)
        decay_factor = (limit / (dist_from_origin + 1e-8)) ** (alpha + 1)
        val_decay = val_bound * decay_factor

        return torch.where(mask_in, val_raw, val_decay)

    val_left = compute_batch_with_padding(inp_left).view(B, M)
    val_right = compute_batch_with_padding(inp_right).view(B, M)

    sum_left = torch.sum(val_left * weights, dim=1, keepdim=True)
    sum_right = torch.sum(val_right * weights, dim=1, keepdim=True)

    riesz_coeff = -1.0 / (2.0 * np.cos(alpha * np.pi / 2.0))
    result = riesz_coeff * (sum_left + sum_right) / (dx ** alpha)
    return result


# ==========================================
# 4. 主流程
# ==========================================
def run_single_alpha(alpha):
    print(f"\n" + "#" * 60)
    print(f"### Running Optimized Hybrid Experiment: Alpha = {alpha} ###")
    print("#" * 60)
    set_seed(42)

    X_train_np, final_x_mc = generate_mc_data_stratified(n_samples=10000, alpha=alpha)
    X_train_full = torch.FloatTensor(X_train_np).to(DEVICE)
    x_coords_only = X_train_full[:, 0:1]

    G_ALPHA_BAR = calculate_effective_diffusion_median(x_coords_only, alpha)

    # --- 预计算 GL 权重 (极速优化) ---
    M_GL = 1000  # 降低到1000以提高速度，精度足够
    print("Pre-computing GL Weights...")
    gl_weights = torch.zeros(M_GL, device=DEVICE)
    w = 1.0
    for k in range(M_GL):
        gl_weights[k] = w
        w = w * (1 - (alpha + 1) / (k + 1))
    gl_weights = gl_weights.view(1, M_GL)  # [1, M]

    # --- Stage 1: Score ---
    print("Stage 1: Training Score...")
    score_net = ScoreNetwork().to(DEVICE)
    opt_score = Adam(score_net.parameters(), lr=2e-3)
    scheduler_score = CosineAnnealingLR(opt_score, T_max=2000)
    loss_history_score = []

    BATCH_SIZE = 512
    pbar_s = tqdm(range(2000), desc=f"Score (a={alpha})", leave=False)
    for epoch in pbar_s:
        idx = torch.randint(0, len(X_train_full), (BATCH_SIZE,))
        batch = X_train_full[idx]
        batch.requires_grad_(True)
        s = score_net(batch)
        grad_s = autograd.grad(s.sum(), batch, create_graph=True)[0]
        div_s = grad_s[:, 0:1]
        loss = 0.5 * (s ** 2) + div_s
        loss = loss.mean()
        opt_score.zero_grad();
        loss.backward();
        opt_score.step();
        scheduler_score.step()
        loss_history_score.append(loss.item())

    score_net.eval()
    for p in score_net.parameters(): p.requires_grad = False

    # --- Stage 2: LL ---
    print("Stage 2: Training LL with Hybrid Sampling...")
    ll_net = LLNetwork().to(DEVICE)
    opt_ll = Adam(ll_net.parameters(), lr=1e-3)
    scheduler_ll = CosineAnnealingLR(opt_ll, T_max=3000)
    loss_history_ll = []

    # Warm-up IC
    for _ in range(800):
        x0 = (torch.rand(BATCH_SIZE, 1) * 6 - 3).to(DEVICE)
        t0 = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
        q0_pred = ll_net(torch.cat([x0, t0], dim=1))
        p0_true = 1.0 / np.sqrt(2 * np.pi * 0.5 ** 2) * np.exp(-0.5 * x0.cpu().numpy() ** 2 / 0.5 ** 2)
        q0_true = torch.FloatTensor(np.log(p0_true + 1e-8)).to(DEVICE)
        loss_init = ((q0_pred - q0_true) ** 2).mean() * 5.0
        opt_ll.zero_grad();
        loss_init.backward();
        opt_ll.step()

    # Split Loss Function
    def compute_loss_split(batch_data, batch_uniform):
        # Part A: 核心区 PDE
        batch_data.requires_grad_(True)
        x_d = batch_data[:, 0:1];
        t_d = batch_data[:, 1:2]

        q_d = ll_net(batch_data)
        p_d = torch.exp(q_d)

        grad_q_full = autograd.grad(q_d.sum(), batch_data, create_graph=True)[0]
        grad_q_x = grad_q_full[:, 0:1];
        q_t = grad_q_full[:, 1:2]

        with torch.no_grad():
            s_val = score_net(batch_data)
            score_target = s_val

        term_time = p_d * q_t
        term_drift = p_d * (1.0 + x_d * s_val)

        # 使用预计算权重
        frac_deriv_p = gl_fractional_derivative_optimized(
            ll_net, x_d, t_d, alpha,
            dx=0.005, M=M_GL, limit=PHYSICAL_LIMIT,
            precomputed_weights=gl_weights  # 传入权重
        )
        term_diff = (SIGMA_CONST ** alpha) * G_ALPHA_BAR * frac_deriv_p

        residual = term_time - (term_drift + term_diff)
        loss_pde = (residual ** 2).mean()
        loss_consistency = ((grad_q_x - score_target) ** 2).mean()

        # Part B: 真空区 Decay
        x_u = batch_uniform[:, 0:1]
        mask_vacuum = torch.abs(x_u) > 3.5
        p_u = torch.exp(ll_net(batch_uniform))
        p_vacuum = p_u[mask_vacuum]
        loss_decay = (p_vacuum ** 2).mean() * 100.0 if len(p_vacuum) > 0 else 0.0

        # Part C: 边界 BC
        B_bound = x_d.shape[0] // 4
        x_bound_abs = torch.FloatTensor(B_bound, 1).uniform_(7.5, PHYSICAL_LIMIT + 0.5).to(DEVICE)
        signs = torch.sign(torch.randn(B_bound, 1).to(DEVICE))
        x_bound = x_bound_abs * signs
        t_bound = torch.rand(B_bound, 1).to(DEVICE) * T_END
        p_bound = torch.exp(ll_net(torch.cat([x_bound, t_bound], dim=1)))
        loss_bc = (p_bound * 10.0).mean()

        return loss_pde, loss_consistency, loss_decay, loss_bc

    # === Main Loop ===
    pbar_l = tqdm(range(3000), desc=f"LL-PDE (a={alpha})", leave=False)
    for epoch in pbar_l:
        # 1. 采样
        idx = torch.randint(0, len(X_train_full), (BATCH_SIZE // 2,))
        batch_data = X_train_full[idx]

        x_uni = (torch.rand(BATCH_SIZE // 2, 1).to(DEVICE) * 2 - 1) * PHYSICAL_LIMIT
        t_uni = torch.rand(BATCH_SIZE // 2, 1).to(DEVICE) * T_END
        batch_uniform = torch.cat([x_uni, t_uni], dim=1)

        loss_pde, loss_score, loss_decay, loss_bc = compute_loss_split(batch_data, batch_uniform)

        # === 修复点：确保 Loss 计算中的 p0_true 是 Tensor ===
        x0 = (torch.rand(BATCH_SIZE, 1) * 6 - 3).to(DEVICE)
        t0 = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
        q0_pred = ll_net(torch.cat([x0, t0], dim=1))

        # 计算 Numpy 值
        p0_val = 1.0 / np.sqrt(2 * np.pi * 0.5 ** 2) * np.exp(-0.5 * x0.cpu().numpy() ** 2 / 0.5 ** 2)
        # 转为 Tensor
        p0_true = torch.FloatTensor(p0_val).to(DEVICE)

        # 现在 torch.log 可以正常工作了
        loss_init = ((q0_pred - torch.log(p0_true + 1e-8)) ** 2).mean()

        total_loss = loss_pde + 2.0 * loss_init + 5.0 * loss_score + loss_decay + 10.0 * loss_bc

        opt_ll.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ll_net.parameters(), 1.0)
        opt_ll.step()
        scheduler_ll.step()

        if epoch % 10 == 0: loss_history_ll.append(total_loss.item())

    # ==========================================
    # 5. 可视化 (多时刻评估 + 损失曲线)
    # ==========================================
    print(f"  Generating multi-time snapshots for a={alpha}...")

    # --- 1. 重新生成一次 MC 数据以获取中间时刻的快照 ---
    # 为了不破坏原有的训练数据结构，我们在这里临时生成一次用于画图的高精度数据
    eval_samples = 20000
    x_eval = np.random.normal(0, 0.5, eval_samples)
    curr_x = x_eval.copy()
    scale = DT ** (1 / alpha)

    # 记录需要保存的时间点步数
    checkpoints = {
        0: int(0.0 / DT),
        1: int(0.25 / DT),
        2: int(0.5 / DT),
        3: int(0.75 / DT),
        4: int(1.0 / DT)
    }
    # 反向映射步数到索引用于存储
    step_to_idx = {v: k for k, v in checkpoints.items()}

    snapshots_mc = {0: x_eval.copy()}  # t=0

    # 运行 SDE 并在特定时刻保存
    for step in range(1, N_STEPS + 1):
        f = drift_func(curr_x)
        g = diffusion_func(curr_x)
        noise = levy_stable.rvs(alpha, 0, scale=scale, size=eval_samples)
        curr_x = curr_x + f * DT + g * noise

        if step in step_to_idx:
            snapshots_mc[step_to_idx[step]] = curr_x.copy()

    # --- 2. 计算指标 (移除了画图模块) ---
    print(f"正在计算三种误差指标 (Alpha={alpha})...")
    time_labels = [0.0, 0.25, 0.5, 0.75, 1.0]

    bound_low, bound_high = np.percentile(snapshots_mc[4], [0.5, 99.5])
    vis_bound = max(abs(bound_low), abs(bound_high)) * 1.3
    vis_bound = min(vis_bound, 8.0)
    x_plot = np.linspace(-vis_bound, vis_bound, 1000)

    errors = {'L2': [], 'Linf': [], 'Wasserstein': []}

    for i, t_val in enumerate(time_labels):
        t_tensor = np.full(1000, t_val)
        inp_plot = torch.FloatTensor(np.stack([x_plot, t_tensor], axis=1)).to(DEVICE)

        with torch.no_grad():
            if t_val == 0:
                p_pred = 1.0 / np.sqrt(2 * np.pi * 0.5 ** 2) * np.exp(-0.5 * x_plot ** 2 / 0.5 ** 2)
            else:
                p_pred = np.exp(ll_net(inp_plot).cpu().numpy().flatten())

        integral = simpson(p_pred, x=x_plot)
        p_pred_norm = p_pred / (integral + 1e-8)

        mc_data = snapshots_mc[i]
        mask_clip = (mc_data >= -vis_bound) & (mc_data <= vis_bound)
        mc_clipped = mc_data[mask_clip]

        if len(mc_clipped) > 0:
            w_dist = wasserstein_distance(mc_clipped, x_plot, u_weights=None, v_weights=p_pred_norm)

            hist, bin_edges = np.histogram(mc_data, bins=100, range=(-vis_bound, vis_bound), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            pdf_mc_interp = interp1d(bin_centers, hist, bounds_error=False, fill_value=0.0)(x_plot)

            l2_err = np.sqrt(simpson((pdf_mc_interp - p_pred_norm) ** 2, x=x_plot))
            linf_err = np.max(np.abs(pdf_mc_interp - p_pred_norm))
        else:
            w_dist, l2_err, linf_err = 0.0, 0.0, 0.0

        errors['Wasserstein'].append(w_dist)
        errors['L2'].append(l2_err)
        errors['Linf'].append(linf_err)

    return errors


if __name__ == "__main__":
    import pandas as pd

    alpha_list = [1.5, 1.6, 1.7, 1.8]
    time_labels = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = {}

    # 全局锁定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    for a in alpha_list:
        all_results[a] = run_single_alpha(a)

    rows = []
    metrics = ['L2 Error', 'Linf Error', 'Wasserstein']
    metric_keys = ['L2', 'Linf', 'Wasserstein']

    for m_idx, metric_name in enumerate(metrics):
        for t_idx, t in enumerate(time_labels):
            row = {'Metric': metric_name if t_idx == 0 else '', 'Time': f"t={t:.2f}"}
            for a in alpha_list:
                row[f'a={a}'] = f"{all_results[a][metric_keys[m_idx]][t_idx]:.4E}"
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("experiment4_errors_table.csv", index=False)

    print("\n==================================================")
    print("实验四误差表格预览：")
    print(df.to_string(index=False))
    print("==================================================")