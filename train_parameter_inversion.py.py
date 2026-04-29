import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from torch.optim import Adam
from tqdm import tqdm

# ==========================================
# 0. 全局配置 (高密度高精度版)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# 真实参数
TRUE_LAMBDA = 0.5
FIXED_SIGMA = 1.0
FIXED_ALPHA = 1.5
T_MAX = 1.0

# 训练配置
UNIFIED_LIMIT = 12.0

# 【关键配置】：dx=0.01 (高精度) + 无 Shift (保证对齐)
GL_DX = 0.01
GL_PADDING = int(UNIFIED_LIMIT / GL_DX) + 100

print(f"=== Experiment 3: High-Precision PINN (Alpha={FIXED_ALPHA}) ===")
print(f"Device: {DEVICE}")


# ==========================================
# 1. 核心算子：标准 GL 分数阶导数 (回归正统)
# ==========================================
def fractional_derivative_GL_asymptotic(func_log_p, x, t, alpha, dx=0.01, padding=300, limit=12.0):
    # 获取 batch_size (修复之前的 Bug)
    batch_size = x.shape[0]

    if not hasattr(fractional_derivative_GL_asymptotic, 'coeffs'):
        coeffs = [1.0]
        for k in range(padding):
            # 标准递推公式
            w_next = coeffs[-1] * (k - alpha) / (k + 1)
            coeffs.append(w_next)
        coeffs = torch.tensor(coeffs, device=x.device, dtype=torch.float32)
        setattr(fractional_derivative_GL_asymptotic, 'coeffs', coeffs)

    w_tensor = getattr(fractional_derivative_GL_asymptotic, 'coeffs')

    # 【回退】：移除 Shift，回归标准 GL
    # shifts = k * dx
    k_indices = torch.arange(0, len(w_tensor), device=x.device)
    shifts = k_indices * dx

    x_flat = x.view(-1, 1)
    t_flat = t.view(-1, 1)

    # 标准采样
    x_left = x_flat - shifts
    x_right = x_flat + shifts

    x_query = torch.cat([x_left.flatten(), x_right.flatten()]).view(-1, 1)
    t_repeated = t_flat.repeat_interleave(len(w_tensor), dim=0)
    t_query = torch.cat([t_repeated, t_repeated]).view(-1, 1)

    # 区分界内界外
    mask_inside = (x_query.abs() <= limit)
    log_p_vals = torch.zeros_like(x_query)

    # 界内：查询网络
    if mask_inside.any():
        x_in = x_query[mask_inside].view(-1, 1)
        t_in = t_query[mask_inside].view(-1, 1)
        log_p_vals[mask_inside] = func_log_p(x_in, t_in).squeeze()

    # 界外：解析填充
    if (~mask_inside).any():
        x_out = x_query[~mask_inside].view(-1, 1)
        t_out = t_query[~mask_inside].view(-1, 1)
        x_bound = torch.sign(x_out) * limit
        with torch.no_grad():
            log_p_bound = func_log_p(x_bound, t_out).squeeze()
        decay = (1 + alpha) * (torch.log(x_out.abs()) - np.log(limit)).squeeze()
        log_p_vals[~mask_inside] = log_p_bound - decay

    p_vals = torch.exp(torch.clamp(log_p_vals, max=20.0))

    total_points = x_left.numel()
    p_left = p_vals[:total_points].view(batch_size, -1)
    p_right = p_vals[total_points:].view(batch_size, -1)

    gl_sum = (p_left * w_tensor).sum(dim=1, keepdim=True) + \
             (p_right * w_tensor).sum(dim=1, keepdim=True)

    coef = -0.5 / np.cos(alpha * np.pi / 2) / (dx ** alpha)
    return coef * gl_sum


# ==========================================
# 2. 网络结构 (SimpleNetwork)
# ==========================================
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


# ==========================================
# 3. 数据生成 (超高密度)
# ==========================================
def generate_levy_noise(alpha, dt, size):
    phi = (np.random.uniform(0, 1, size) - 0.5) * np.pi
    w = -np.log(np.random.uniform(0, 1, size))
    term1 = np.sin(alpha * phi) / (np.cos(phi) ** (1 / alpha))
    term2 = (np.cos((1 - alpha) * phi) / w) ** ((1 - alpha) / alpha)
    return term1 * term2 * (dt ** (1 / alpha))


def get_observed_data():
    print("Generating Synthetic Data...")
    dt = 0.001
    n_steps = int(T_MAX / dt)

    # 【关键升级】：40000 粒子，支撑 0.01 的细网格
    n_particles = 40000

    x = np.zeros((n_particles, 1))
    x[:, 0] = np.random.normal(0, 1.0, n_particles)

    data = []
    for i in tqdm(range(n_steps)):
        levy = generate_levy_noise(FIXED_ALPHA, dt, (n_particles, 1))
        dx = -TRUE_LAMBDA * x * dt + FIXED_SIGMA * levy
        x = x + dx
        if i % 20 == 0 and i > 0:
            mask = np.abs(x) < UNIFIED_LIMIT
            x_valid = x[mask]
            if len(x_valid) > 0:
                t_valid = np.ones_like(x_valid) * (i * dt)
                data.append(np.stack([x_valid, t_valid], axis=1))

    data = np.concatenate(data, axis=0)
    # 采样更多点进行训练
    idx = np.random.choice(len(data), 60000, replace=False)
    X = torch.tensor(data[idx, 0:1], dtype=torch.float32).to(DEVICE)
    T = torch.tensor(data[idx, 1:2], dtype=torch.float32).to(DEVICE)
    return X, T


# ==========================================
# 4. Stage 1: Score Matching
# ==========================================
def stage1_learn_score(X_train, T_train):
    print("\n>>> Stage 1: Score Matching")
    score_net = SimpleNetwork().to(DEVICE)
    opt = Adam(score_net.parameters(), lr=1e-3)

    for ep in tqdm(range(2000)):
        idx = torch.randint(0, len(X_train), (2048,))  # 增大 Batch
        x = X_train[idx].requires_grad_(True)
        t = T_train[idx].requires_grad_(True)

        s = score_net(x, t)

        epsilon = torch.randn_like(s)
        grad_s_eps = torch.autograd.grad(torch.sum(s * epsilon), x, create_graph=True)[0]
        div_s = torch.sum(grad_s_eps * epsilon, dim=1, keepdim=True)

        loss = 0.5 * torch.mean(s ** 2) + torch.mean(div_s)

        opt.zero_grad()
        loss.backward()
        opt.step()

    for p in score_net.parameters(): p.requires_grad = False
    return score_net


# ==========================================
# 5. Stage 2: Density Integration
# ==========================================
def stage2_integrate_score(X_train, T_train, teacher_net):
    print("\n>>> Stage 2: Density Refinement")

    ll_net = SimpleNetwork().to(DEVICE)
    opt = Adam(ll_net.parameters(), lr=1e-3)

    x_grid = torch.linspace(-UNIFIED_LIMIT, UNIFIED_LIMIT, 300).view(-1, 1).to(DEVICE)
    dx_grid = (2 * UNIFIED_LIMIT) / 299

    for ep in tqdm(range(3000)):
        idx = torch.randint(0, len(X_train), (2048,))
        x = X_train[idx].requires_grad_(True)
        t = T_train[idx].requires_grad_(True)

        q = ll_net(x, t)
        s_pred = torch.autograd.grad(q.sum(), x, create_graph=True)[0]

        with torch.no_grad():
            s_target = teacher_net(x, t)

        loss_grad = torch.mean((s_pred - s_target) ** 2)

        # 归一化约束
        rand_t_val = np.random.uniform(0.1, T_MAX)
        t_grid = torch.ones_like(x_grid) * rand_t_val
        q_grid = ll_net(x_grid, t_grid)
        log_Z = torch.logsumexp(q_grid, dim=0) + np.log(dx_grid)
        loss_norm = torch.mean(log_Z ** 2)

        loss = loss_grad + 0.1 * loss_norm

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            print(f"Ep {ep}: Grad={loss_grad.item():.6f}, Norm={loss_norm.item():.4f}")

    return ll_net


# ==========================================
# 6. Stage 3: PINN Inversion
# ==========================================
def stage3_pinn_optimization(ll_net, X_train, T_train):
    print("\n>>> Stage 3: PINN Optimization (High Precision)")

    # 检查 LogZ
    with torch.no_grad():
        x_grid = torch.linspace(-UNIFIED_LIMIT, UNIFIED_LIMIT, 2000).view(-1, 1).to(DEVICE)
        dx_grid = (2 * UNIFIED_LIMIT) / 1999
        t_mid = torch.ones_like(x_grid) * 0.5
        q_grid = ll_net(x_grid, t_mid)
        log_Z = (torch.logsumexp(q_grid, dim=0) + np.log(dx_grid)).item()
    print(f"Global LogZ Check: {log_Z:.4f}")

    raw_lambda = nn.Parameter(torch.tensor([2.0]).to(DEVICE))
    optimizer = Adam([raw_lambda], lr=1e-2)
    history = []

    mask = X_train.abs() < UNIFIED_LIMIT
    x_c = X_train[mask].view(-1, 1).clone().requires_grad_(True)
    t_c = T_train[mask].view(-1, 1).clone().requires_grad_(True)

    if len(x_c) > 5000:
        idx = torch.randperm(len(x_c))[:5000]
        x_c = x_c[idx]
        t_c = t_c[idx]

    print("Pre-computing derivatives...")

    def normalized_log_p_func(x, t):
        return ll_net(x, t)

    with torch.no_grad():
        q_c = ll_net(x_c, t_c)
        p_c = torch.exp(torch.clamp(q_c, max=10.0)).detach()

    q_c_graph = ll_net(x_c, t_c)
    s_c = torch.autograd.grad(q_c_graph.sum(), x_c, create_graph=True)[0].detach()
    dt_q = torch.autograd.grad(q_c_graph.sum(), t_c, create_graph=True)[0].detach()

    # GL 计算 (dx=0.01 + Standard)
    with torch.no_grad():
        frac_term = fractional_derivative_GL_asymptotic(
            normalized_log_p_func,
            x_c, t_c, FIXED_ALPHA,
            dx=GL_DX, padding=GL_PADDING, limit=UNIFIED_LIMIT
        ).detach()

    x_c_loop = x_c.detach()

    print("Start optimization loop...")

    for ep in tqdm(range(2000)):
        current_lam = torch.nn.functional.softplus(raw_lambda)

        lhs = p_c * dt_q
        rhs_drift = current_lam * p_c * (1 + x_c_loop * s_c)
        rhs_diff = (FIXED_SIGMA ** FIXED_ALPHA) * frac_term

        residual = lhs - (rhs_drift + rhs_diff)
        loss_pde = torch.mean(residual ** 2)

        optimizer.zero_grad()
        loss_pde.backward()
        optimizer.step()

        history.append(current_lam.item())

        if ep % 200 == 0:
            print(f"Ep {ep}: Lam={current_lam.item():.4f}, PDE Loss={loss_pde.item():.4e}")

    return history


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    X_train, T_train = get_observed_data()
    score_net_model = stage1_learn_score(X_train, T_train)
    ll_net = stage2_integrate_score(X_train, T_train, score_net_model)
    param_history = stage3_pinn_optimization(ll_net, X_train, T_train)

    final_lambda = param_history[-1]

    print("\n" + "=" * 40)
    print(f"True Lambda      : {TRUE_LAMBDA}")
    print(f"Estimated Lambda : {final_lambda:.4f}")
    error = abs(final_lambda - TRUE_LAMBDA) / TRUE_LAMBDA * 100
    print(f"Relative Error   : {error:.2f}%")
    print("=" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(param_history, label='Estimated Lambda', linewidth=2)
    plt.axhline(TRUE_LAMBDA, color='red', linestyle='--', label='True Lambda')
    plt.title(f'Inverse Problem ($\\alpha={FIXED_ALPHA}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    fname_png = "Exp3_Final_Precision.png"
    fname_pdf = "Exp3_Final_Precision.pdf"

    # 保存高分辨率 PNG
    plt.savefig(fname_png, dpi=600, bbox_inches='tight')
    # 紧接着保存矢量图 PDF
    plt.savefig(fname_pdf, format='pdf', bbox_inches='tight')

    print(f"✓ 结果已保存至 {fname_png} 和 {fname_pdf}")