# Section 3.3.3 — Exterior-Dirichlet killed-Lévy benchmark

The experiment uses a homogeneous exterior Dirichlet condition on `(0,1)^2`, a hard boundary envelope in the SBPINN representation, and a structured mass-lumped P1 FEM with dense nonlocal quadrature.

Quick CPU software test:

```bash
python run_dirichlet_benchmark_isolated.py --quick --device cpu --output_dir results_dirichlet
```

Main benchmark:

```bash
python run_dirichlet_benchmark_isolated.py \
  --device cuda \
  --alpha 1.5 \
  --noise_scale 0.34 \
  --fem_meshes "12,16,24,32,40,48,56,64,80,96,112" \
  --fem_precheck "128" \
  --fem_memory_budget_gb 16 \
  --fem_time_budget_s 3600 \
  --output_dir results_dirichlet_killed_levy \
  --resume
```

The optional validation-patch runner reproduces the refined reference and FEM checks used to obtain the final manuscript table:

```bash
python run_dirichlet_validation_patches.py --help
```
