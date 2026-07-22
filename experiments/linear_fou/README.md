# Section 3.1 — Linear fractional OU process

Run the publication configuration:

```bash
python train_linear_fou.py --device auto --alphas 1.5,1.6,1.7,1.8
```

Quick CPU smoke test:

```bash
python train_linear_fou.py --device cpu --alphas 1.5 --quick --no_save_model
```

Generate the table-only evaluation:

```bash
python eval_linear_fou.py
```
