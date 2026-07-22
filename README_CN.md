# SBPINN：Lévy 噪声驱动系统的分数阶 Score-Based PINN 求解框架

本仓库与论文 **A Fractional Score-Based PINN Solver Framework for Lévy Noise-Driven Systems** 对应，包含论文全部数值实验代码、最终结果表以及二维周期边界和外部 Dirichlet 边界实验的复现实验配置。

## 目录与论文章节对应关系

- `experiments/linear_fou/`：第 3.1 节，一维分数阶 OU 精确解验证；
- `experiments/baseline_comparison/`：第 3.2 节，Vanilla fPINN 与 Galerkin FEM 对比；
- `experiments/two_dimensional_periodic/`：第 3.3.2 节，二维非线性周期边界实验；
- `experiments/two_dimensional_dirichlet/`：第 3.3.3 节，二维外部 Dirichlet killed-Lévy 实验；
- `experiments/bistable/`：第 3.4 节，双稳态多模态实验；
- `experiments/parameter_inversion/`：第 3.5 节，参数反演；
- `experiments/multiplicative_noise/`：第 3.6 节，乘性噪声实验；
- `results/paper_tables/`：论文 Tables 1–6 的最终数值。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python .\scripts\check_environment.py
```

正式二维实验建议使用 CUDA GPU。代码仓库未包含大体积轨迹、检查点和 `.npz` 稠密解文件，但已保留最终表格、配置和核心复现实验代码。各实验文件夹内均有独立运行说明。
