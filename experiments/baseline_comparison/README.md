# Section 3.2 — Baseline comparison and computational work

Figure 2 / Vanilla fPINN comparison:

```bash
python benchmark_vanilla_fpinn.py --device auto --output_dir results_fig2
```

Quick smoke test:

```bash
python benchmark_vanilla_fpinn.py --device cpu --quick_test
```

Table 2 / Galerkin FEM comparison:

```bash
python eval_computational_efficiency.py --device auto --output_dir results_table2
```
