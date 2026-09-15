# Score-Based Fractional PINNs for Lévy Noise-Driven Systems

Code and compact numerical results for the SBPINN manuscript. The framework uses a two-stage strategy: Stage I learns score information from stochastic trajectories, and Stage II reconstructs a non-negative density under the fractional Fokker-Planck physics with a gradual score-to-physics transition.

## Repository structure

| Manuscript section | Directory | Main experiment |
|---|---|---|
| Sec. 3.1 | `experiments/linear_fou/` | 1D fractional Ornstein-Uhlenbeck benchmark |
| Sec. 3.1.3 | `experiments/boundary_padding/` | exterior-padding error diagnostic |
| Sec. 3.2 | `experiments/score_ablation/` | score-guided homotopy vs no-score ablation |
| Sec. 3.3.2 | `experiments/two_dimensional_periodic/` | 2D nonlinear periodic benchmark + Q1 FEM |
| Sec. 3.3.3 | `experiments/two_dimensional_dirichlet/` | 2D killed-Levy benchmark + P1 FEM |
| Sec. 3.3.4 | `experiments/score_fpinn_comparison/` | controlled Score-fPINN comparison |
| Sec. 3.4 | `experiments/bistable/` | two-expert bistable reconstruction |
| Sec. 3.5 | `experiments/parameter_inversion/` | mean-reversion parameter inversion |
| Sec. 3.6 | `experiments/multiplicative_noise/` | transformed-u state-dependent Levy noise |

The compact manuscript tables and source CSV/JSON files are under `results/`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py
```

A CUDA-enabled GPU is recommended for the full 2D and multi-seed experiments. CPU quick tests are available for several scripts.

## Quick validation

```bash
python scripts/syntax_check.py
python experiments/linear_fou/train_linear_fou.py --device cpu --alphas 1.5 --quick --no_save_model
python experiments/score_ablation/run_score_stage_ablation.py --device cpu --quick --methods homotopy,no_score
python experiments/two_dimensional_periodic/run_nonlinear_torus_benchmark_isolated.py --quick --device cpu --output_dir outputs/periodic_quick
python experiments/two_dimensional_dirichlet/run_dirichlet_benchmark_isolated.py --quick --device cpu --output_dir outputs/dirichlet_quick
python experiments/multiplicative_noise/run_multiplicative_transformed_u.py --quick --device cpu --output_dir outputs/multiplicative_quick
```

## Main reproduction commands

### 1D fOU

```bash
python experiments/linear_fou/train_linear_fou.py --device cuda --alphas 1.5,1.6,1.7,1.8 --output_dir outputs/linear_fou
```

### Boundary-padding diagnostic

```bash
python experiments/boundary_padding/eval_boundary_padding.py --output_dir outputs/boundary_padding
```

### Score-stage ablation

```bash
python experiments/score_ablation/run_score_stage_ablation.py --device cuda --alpha 1.5 --seeds 42,123,2024,2025,2026 --methods homotopy,no_score --no_score_budget same_stage2_updates --output_dir outputs/score_ablation_matched_updates
python experiments/score_ablation/run_score_stage_ablation.py --device cuda --alpha 1.5 --seeds 42,123,2024,2025,2026 --methods homotopy,no_score --no_score_budget same_gl --output_dir outputs/score_ablation_matched_gl
```

### 2D benchmarks

Use the isolated-process runners documented in the corresponding experiment folders.

### Score-fPINN comparison

Run the SBPINN counterpart with:

```bash
python run_sbpinn_fou_benchmark.py --device cuda --alpha 1.75 --seed 42 --output_dir outputs/sbpinn_alpha175_seed42
```

and Score-fPINN with:

```bash
python experiments/score_fpinn_comparison/run_score_fpinn_fou.py --device cuda --alpha 1.75 --seed 42 --output_dir outputs/score_fpinn_alpha175_seed42
```

Repeat for seeds 42, 123, and 2024.

### Bistable and parameter inversion

```bash
cd experiments/bistable
python train_bistable_system.py
python eval_bistable_system.py

cd ../parameter_inversion
python train_parameter_inversion.py
```

### State-dependent multiplicative noise

```bash
python experiments/multiplicative_noise/run_multiplicative_transformed_u.py --device cuda --alpha 1.5 --beta 0.5 --seed 42 --output_dir outputs/multiplicative_beta05_seed42
```

Repeat `beta=0.5` for seeds 42, 123, and 2024. The stronger state-dependence check uses `--beta 1.0 --seed 42`.

## Numerical realizations of the fractional operator

The code follows the problem-dependent treatments used in the manuscript:

- one-dimensional truncated whole-space problems: standard Grünwald-Letnikov stencils with asymptotic exterior values;
- two-dimensional periodic problem: periodic spectral realization;
- killed process with homogeneous exterior Dirichlet condition: zero-extended spectral realization;
- multiplicative noise: transformed variable `u=a(x)p` with the spatially varying coefficient retained in the nonlocal term.

## Reproducibility policy

The public release excludes large neural-network checkpoints, dense trajectory arrays, and temporary pilot outputs. Compact source data, final quantitative tables, representative figures, experiment configurations, and all active scripts required to regenerate the results are included.

## Citation

If you use this code, please cite the associated manuscript:

> H. Xue, J. Zhang, Z. Wang, H. Wang, *Score-Based Fractional PINNs for Lévy Noise-Driven Systems*.

## License

MIT License. See `LICENSE`.
