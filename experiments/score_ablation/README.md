# Section 3.2 — Score-guided training ablation

The paired ablation compares the full score-guided homotopy SBPINN with a no-score log-density GL-PINN under two budget controls.

Publication runs:

```bash
python experiments/score_ablation/run_score_stage_ablation.py \
  --device cuda --alpha 1.5 --seeds 42,123,2024,2025,2026 \
  --methods homotopy,no_score --no_score_budget same_stage2_updates \
  --output_dir outputs/score_ablation_matched_updates

python experiments/score_ablation/run_score_stage_ablation.py \
  --device cuda --alpha 1.5 --seeds 42,123,2024,2025,2026 \
  --methods homotopy,no_score --no_score_budget same_gl \
  --output_dir outputs/score_ablation_matched_gl
```

Quick CPU test:

```bash
python experiments/score_ablation/run_score_stage_ablation.py --device cpu --quick --methods homotopy,no_score
```
