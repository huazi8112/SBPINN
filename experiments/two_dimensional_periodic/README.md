# Section 3.3.2 — Two-dimensional periodic benchmark

The experiment uses the nonlinear, non-gradient torus system with `alpha=1.5` and `noise_scale=0.34`. The reference, SBPINN, and each periodic Q1 FEM mesh are run in isolated processes.

Quick CPU software test:

```bash
python run_nonlinear_torus_benchmark_isolated.py --quick --device cpu --output_dir results_periodic
```

Publication configuration:

```bash
python run_nonlinear_torus_benchmark_isolated.py \
  --device cuda \
  --alpha 1.5 \
  --noise_scale 0.34 \
  --fem_meshes "12,16,24,32,40,48,64,80,96" \
  --fem_precheck "128,160" \
  --fem_memory_budget_gb 16 \
  --fem_time_budget_s 3600 \
  --output_dir results_nonlinear_torus_levy_v2 \
  --resume
```
