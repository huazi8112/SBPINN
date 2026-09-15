# Section 3.6 — State-dependent multiplicative Levy noise

The manuscript result uses the transformed variable `u=a(x)p`, with `a(x)=|g(x)|^alpha`, and trains `r=log u`. The physical density is recovered as `p=u/a` before evaluation.

Main transformed-u run:

```bash
python experiments/multiplicative_noise/run_multiplicative_transformed_u.py \
  --device cuda --alpha 1.5 --beta 0.5 --seed 42 \
  --output_dir outputs/multiplicative_beta05_seed42
```

Repeat the `beta=0.5` run for seeds `42`, `123`, and `2024`. The stronger state-dependence check uses `--beta 1.0 --seed 42`.

`run_multiplicative_noise_revision.py` is retained as the operator/approximation diagnostic used during method validation; it is not the main transformed-u training entry point.
