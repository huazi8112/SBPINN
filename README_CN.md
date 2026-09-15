# Score-Based Fractional PINNs for Lévy Noise-Driven Systems

本仓库是论文 **Score-Based Fractional PINNs for Lévy Noise-Driven Systems** 的公开代码整理版，只保留最终稿对应的实验代码、紧凑结果数据和复现说明；历史诊断版本、失效表格、旧 median 主方法、模型检查点和大体积中间结果均未放入公开包。

## 论文章节与代码对应

- `experiments/linear_fou/`：3.1，一维 fOU 精确参考验证；
- `experiments/boundary_padding/`：3.1.3，域外渐近填充误差量化；
- `experiments/score_ablation/`：3.2，score-guided homotopy 与 no-score 消融；
- `experiments/two_dimensional_periodic/`：3.3.2，二维周期边界；
- `experiments/two_dimensional_dirichlet/`：3.3.3，二维 killed-Lévy 外部 Dirichlet；
- `experiments/score_fpinn_comparison/`：3.3.4，与 Score-fPINN 的直接比较；
- `experiments/bistable/`：3.4，双稳态双 expert；
- `experiments/parameter_inversion/`：3.5，参数反演；
- `experiments/multiplicative_noise/`：3.6，transformed-u 乘性 Lévy 噪声；
- `results/paper_tables/`：最终稿 Tables 1–8 的 CSV；
- `results/source_data/`：上述表格对应的紧凑源数据；
- `results/figures/`：代表性主文/补充材料图片。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts\check_environment.py
```

完整二维、多随机种子及乘性噪声正式运行建议使用 CUDA GPU。所有脚本均避免写死本机绝对路径。

## 最小检查

```powershell
python scripts\syntax_check.py
python experiments\linear_fou\train_linear_fou.py --device cpu --alphas 1.5 --quick --no_save_model
python experiments\score_ablation\run_score_stage_ablation.py --device cpu --quick --methods homotopy,no_score
```

更完整的复现命令见英文 `README.md` 以及各实验目录下的 `README.md`。

## 上传 GitHub

本压缩包解压后的 `SBPINN_GitHub_ready` 目录可以直接作为仓库内容。不要再把原始 `archive/`、`.idea/`、`revision_results/`、`.pt`、`.npz` 等历史/大体积文件复制回仓库。
