# Section 3.3.4 — Score-fPINN comparison

This folder contains the controlled one-dimensional fOU implementation used for the direct density-reconstruction comparison with Score-fPINN at alpha=1.75.

Example run:

```bash
python experiments/score_fpinn_comparison/run_score_fpinn_fou.py \
  --device cuda --alpha 1.75 --seed 42 \
  --output_dir outputs/score_fpinn_alpha175_seed42
```

Repeat with seeds `42`, `123`, and `2024`. The SBPINN counterpart is generated with `run_sbpinn_fou_benchmark.py` at repository root.
