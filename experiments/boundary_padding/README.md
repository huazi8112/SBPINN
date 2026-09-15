# Exterior-padding diagnostic (fOU)

This script quantifies the finite-range error of the algebraic exterior extension used by the one-dimensional standard-GL stencil. Exact exterior values are obtained independently from characteristic-function inversion.

```bash
python experiments/boundary_padding/eval_boundary_padding.py --output_dir outputs/boundary_padding
```

The reported diagnostics include exterior relative L1/L2 errors, exterior mass error, full-domain GL-operator error, and the boundary-strip GL error.
