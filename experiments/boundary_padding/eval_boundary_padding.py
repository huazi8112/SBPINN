"""fOU asymptotic boundary-padding diagnostic.

This standalone diagnostic matches the revised 1-D fOU benchmark parameters:
  theta=0.5, sigma=1, L=6, standard GL h=0.08, M=80,
and compares the asymptotic padding p_pad(x,t)=p_exact(sign(x)L,t)*(L/|x|)^(1+alpha)
against the exact characteristic-function inversion on the full exterior range
actually queried by the GL stencil, L < |x| <= L+(M-1)h = 12.32.

It also isolates the induced GL-operator error by using exact density values at
all interior stencil points and changing only the exterior treatment:
  oracle: exact characteristic-function inversion outside [-L,L]
  padding: asymptotic exterior values outside [-L,L]

No neural-network training or Monte Carlo simulation is involved.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson


THETA = 0.5
SIGMA = 1.0
L = 6.0
GL_DX = 0.08
GL_TERMS = 80
ALPHAS = (1.5, 1.6, 1.7, 1.8)
TIMES = (0.0, 0.25, 0.50, 0.75)


def exact_fou_pdf(
    x_grid: np.ndarray,
    t: float,
    alpha: float,
    k_max: float = 100.0,
    k_num: int = 16000,
    chunk_size: int = 256,
) -> np.ndarray:
    """Exact transient fOU density by the same characteristic inversion as Sec. 3.1."""
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if t <= 1.0e-14:
        return np.exp(-0.5 * x_grid ** 2) / math.sqrt(2.0 * math.pi)

    a_t = (1.0 - math.exp(-alpha * THETA * t)) / (alpha * THETA)
    gaussian_variance = math.exp(-2.0 * THETA * t)
    levy_coefficient = SIGMA ** alpha * a_t

    k = np.linspace(0.0, k_max, k_num, dtype=np.float64)
    decay = np.exp(
        -0.5 * gaussian_variance * k ** 2
        - levy_coefficient * k ** alpha
    )
    pdf = np.empty_like(x_grid)
    for start in range(0, len(x_grid), chunk_size):
        block = x_grid[start:start + chunk_size, None]
        integrand = np.cos(block * k[None, :]) * decay[None, :]
        pdf[start:start + chunk_size] = (
            np.trapezoid(integrand, k, axis=1) / math.pi
        )
    return np.maximum(pdf, 0.0)


def padding_from_exact_anchor(x: np.ndarray, t: float, alpha: float,
                              k_max: float, k_num: int) -> np.ndarray:
    """Asymptotic padding using the exact boundary anchor to isolate model error."""
    x = np.asarray(x, dtype=np.float64)
    p_L = float(exact_fou_pdf(np.array([L]), t, alpha, k_max, k_num)[0])
    return p_L * (L / np.abs(x)) ** (1.0 + alpha)


def gl_coefficients(alpha: float, n_terms: int) -> np.ndarray:
    coeff = [1.0]
    for k in range(1, n_terms):
        coeff.append(coeff[-1] * (k - 1.0 - alpha) / k)
    return np.asarray(coeff, dtype=np.float64)


def gl_padding_error(alpha: float, t: float, k_max: float, k_num: int) -> Dict[str, float]:
    x_center = np.linspace(-L, L, int(round(2.0 * L / GL_DX)) + 1)
    shifts = np.arange(GL_TERMS, dtype=np.float64) * GL_DX
    left = x_center[:, None] - shifts[None, :]
    right = x_center[:, None] + shifts[None, :]

    # Because x_center and shifts are commensurate with h, only 309 unique
    # auxiliary coordinates are needed over [-12.32,12.32].
    unique_x = np.unique(np.round(np.concatenate([left.ravel(), right.ravel()]), 12))
    exact_unique = exact_fou_pdf(unique_x, t, alpha, k_max, k_num)
    lookup = {round(float(x), 12): float(p) for x, p in zip(unique_x, exact_unique)}
    p_L = float(exact_fou_pdf(np.array([L]), t, alpha, k_max, k_num)[0])

    def evaluate(points: np.ndarray, use_padding: bool) -> np.ndarray:
        flat = points.ravel()
        vals = np.empty_like(flat)
        for i, x in enumerate(flat):
            if (not use_padding) or abs(x) <= L + 1.0e-12:
                vals[i] = lookup[round(float(x), 12)]
            else:
                vals[i] = p_L * (L / abs(x)) ** (1.0 + alpha)
        return vals.reshape(points.shape)

    coeff = gl_coefficients(alpha, GL_TERMS)
    factor = -1.0 / (
        2.0 * math.cos(alpha * math.pi / 2.0) * GL_DX ** alpha
    )
    op_oracle = factor * np.sum(
        coeff[None, :] * (evaluate(left, False) + evaluate(right, False)), axis=1
    )
    op_pad = factor * np.sum(
        coeff[None, :] * (evaluate(left, True) + evaluate(right, True)), axis=1
    )
    diff = op_pad - op_oracle

    def rel_l2(mask: np.ndarray) -> float:
        return float(np.linalg.norm(diff[mask]) / max(np.linalg.norm(op_oracle[mask]), 1.0e-300))

    full = np.ones_like(x_center, dtype=bool)
    core4 = np.abs(x_center) <= 4.0 + 1.0e-12
    core5 = np.abs(x_center) <= 5.0 + 1.0e-12
    boundary = np.abs(x_center) > 4.0
    idx = int(np.argmax(np.abs(diff)))
    return {
        "alpha": alpha,
        "time": t,
        "gl_rel_l2_full": rel_l2(full),
        "gl_rel_l2_core_absx_le_4": rel_l2(core4),
        "gl_rel_l2_core_absx_le_5": rel_l2(core5),
        "gl_rel_l2_boundary_4_lt_absx_le_6": rel_l2(boundary),
        "gl_max_abs_error": float(np.max(np.abs(diff))),
        "gl_max_abs_error_x": float(x_center[idx]),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_profile_plot(outdir: Path, alpha: float, t: float, k_max: float, k_num: int) -> None:
    x_max = L + (GL_TERMS - 1) * GL_DX
    x = np.linspace(L, x_max, 1201)
    exact = exact_fou_pdf(x, t, alpha, k_max, k_num)
    pad = padding_from_exact_anchor(x, t, alpha, k_max, k_num)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.plot(x, exact, linestyle="--", linewidth=2.3, label="Exact inversion")
    ax.plot(x, pad, linestyle="-", linewidth=2.0, label="Asymptotic padding")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|x|$")
    ax.set_ylabel(r"$p(x,t)$")
    ax.set_title(rf"$\alpha={alpha:.1f},\ t={t:.2f}$")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    tag = f"alpha{str(alpha).replace('.', 'p')}_t{str(t).replace('.', 'p')}"
    fig.savefig(outdir / f"padding_profile_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="revision_results/01_fou/padding_diagnostic")
    parser.add_argument("--exact_k_max", type=float, default=100.0)
    parser.add_argument("--exact_k_num", type=int, default=16000)
    parser.add_argument("--ext_grid_n", type=int, default=2001)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    x_max = L + (GL_TERMS - 1) * GL_DX
    x_ext = np.linspace(L, x_max, args.ext_grid_n)

    density_rows: List[Dict[str, float]] = []
    gl_rows: List[Dict[str, float]] = []
    for alpha in ALPHAS:
        for t in TIMES:
            exact = exact_fou_pdf(x_ext, t, alpha, args.exact_k_max, args.exact_k_num)
            pad = padding_from_exact_anchor(x_ext, t, alpha, args.exact_k_max, args.exact_k_num)
            diff = pad - exact
            exact_mass = 2.0 * float(simpson(exact, x=x_ext))
            pad_mass = 2.0 * float(simpson(pad, x=x_ext))
            rel_l1 = 2.0 * float(simpson(np.abs(diff), x=x_ext)) / max(exact_mass, 1e-300)
            rel_l2 = math.sqrt(2.0 * float(simpson(diff ** 2, x=x_ext))) / max(
                math.sqrt(2.0 * float(simpson(exact ** 2, x=x_ext))), 1e-300
            )
            mass_abs = abs(pad_mass - exact_mass)
            mass_rel = mass_abs / max(exact_mass, 1e-300)
            max_abs = float(np.max(np.abs(diff)))
            p_L = float(exact[0])

            sample_x = np.array([8.0, 10.0, x_max])
            sample_exact = exact_fou_pdf(sample_x, t, alpha, args.exact_k_max, args.exact_k_num)
            sample_pad = padding_from_exact_anchor(sample_x, t, alpha, args.exact_k_max, args.exact_k_num)
            ratios = sample_pad / np.maximum(sample_exact, 1e-300)

            density_rows.append({
                "alpha": alpha,
                "time": t,
                "exterior_x_min": L,
                "exterior_x_max": x_max,
                "exterior_rel_l1": rel_l1,
                "exterior_rel_l2": rel_l2,
                "exterior_exact_mass": exact_mass,
                "exterior_padding_mass": pad_mass,
                "exterior_mass_relative_error": mass_rel,
                "exterior_mass_absolute_error_total_probability": mass_abs,
                "max_abs_density_error": max_abs,
                "max_abs_density_error_over_boundary_density": max_abs / max(p_L, 1e-300),
                "padding_over_exact_at_absx_8": float(ratios[0]),
                "padding_over_exact_at_absx_10": float(ratios[1]),
                "padding_over_exact_at_absx_12p32": float(ratios[2]),
            })
            gl_rows.append(gl_padding_error(alpha, t, args.exact_k_max, args.exact_k_num))
            print(
                f"alpha={alpha:.1f}, t={t:.2f}: ext relL2={rel_l2:.4%}, "
                f"abs mass err={mass_abs:.3e}, GL full relL2={gl_rows[-1]['gl_rel_l2_full']:.4%}"
            )

    write_csv(outdir / "boundary_padding_density_error_all.csv", density_rows)
    write_csv(outdir / "boundary_padding_density_error_reported_times.csv", [r for r in density_rows if r["time"] > 0])
    write_csv(outdir / "boundary_padding_t0_control.csv", [r for r in density_rows if r["time"] == 0])
    write_csv(outdir / "boundary_padding_gl_error.csv", gl_rows)

    pos_density = [r for r in density_rows if r["time"] > 0]
    pos_gl = [r for r in gl_rows if r["time"] > 0]
    summary = [{
        "n_alpha_time_cases": len(pos_density),
        "mean_exterior_rel_l1": float(np.mean([r["exterior_rel_l1"] for r in pos_density])),
        "max_exterior_rel_l1": float(np.max([r["exterior_rel_l1"] for r in pos_density])),
        "mean_exterior_rel_l2": float(np.mean([r["exterior_rel_l2"] for r in pos_density])),
        "max_exterior_rel_l2": float(np.max([r["exterior_rel_l2"] for r in pos_density])),
        "mean_exterior_mass_abs_error_total_probability": float(np.mean([r["exterior_mass_absolute_error_total_probability"] for r in pos_density])),
        "max_exterior_mass_abs_error_total_probability": float(np.max([r["exterior_mass_absolute_error_total_probability"] for r in pos_density])),
        "mean_gl_rel_l2_full": float(np.mean([r["gl_rel_l2_full"] for r in pos_gl])),
        "max_gl_rel_l2_full": float(np.max([r["gl_rel_l2_full"] for r in pos_gl])),
        "mean_gl_rel_l2_boundary_4_lt_absx_le_6": float(np.mean([r["gl_rel_l2_boundary_4_lt_absx_le_6"] for r in pos_gl])),
        "max_gl_rel_l2_boundary_4_lt_absx_le_6": float(np.max([r["gl_rel_l2_boundary_4_lt_absx_le_6"] for r in pos_gl])),
    }]
    write_csv(outdir / "boundary_padding_summary.csv", summary)

    # Two compact log-scale profile figures: representative and worst reported case.
    make_profile_plot(outdir, 1.5, 0.25, args.exact_k_max, args.exact_k_num)
    make_profile_plot(outdir, 1.8, 0.75, args.exact_k_max, args.exact_k_num)

    # Characteristic-inversion convergence check at two endpoints of difficulty.
    conv_rows = []
    for alpha, t in ((1.5, 0.25), (1.8, 0.75)):
        x = np.linspace(L, x_max, 501)
        base = exact_fou_pdf(x, t, alpha, args.exact_k_max, args.exact_k_num)
        fine = exact_fou_pdf(x, t, alpha, 120.0, 32000)
        conv_rows.append({
            "alpha": alpha,
            "time": t,
            "baseline_kmax": args.exact_k_max,
            "baseline_knum": args.exact_k_num,
            "fine_kmax": 120.0,
            "fine_knum": 32000,
            "relative_l2_difference": float(np.linalg.norm(base - fine) / np.linalg.norm(fine)),
            "max_abs_difference_over_peak": float(np.max(np.abs(base - fine)) / np.max(fine)),
        })
    write_csv(outdir / "characteristic_inversion_convergence.csv", conv_rows)

    # Markdown report.
    s = summary[0]
    t0_gl = max(r["gl_rel_l2_full"] for r in gl_rows if r["time"] == 0)
    t0_mass = max(r["exterior_exact_mass"] for r in density_rows if r["time"] == 0)
    report = f"""# fOU boundary-padding diagnostic for Reviewer #3 Comment 3

## Protocol
- Revised fOU parameters: theta={THETA}, sigma={SIGMA}, L={L}.
- Standard GL: h={GL_DX}, M={GL_TERMS}, hence the entire exterior interval actually queried is {L}<|x|<={x_max:.2f}.
- Exact reference: characteristic-function inversion used in the revised Section 3.1 (k_max={args.exact_k_max:g}, k_num={args.exact_k_num}).
- Padding error is isolated with an **exact boundary anchor** p_exact(±L,t), so neural boundary error is not mixed into the test.
- Reported benchmark times: t=0.25,0.50,0.75 for alpha=1.5,1.6,1.7,1.8 (12 cases).
- A t=0 Gaussian-initial-condition control is also retained rather than omitted.

## Main numerical findings (t>0)
- Exterior relative L1 error: mean {100*s['mean_exterior_rel_l1']:.2f}%, max {100*s['max_exterior_rel_l1']:.2f}%.
- Exterior relative L2 error: mean {100*s['mean_exterior_rel_l2']:.2f}%, max {100*s['max_exterior_rel_l2']:.2f}%.
- Absolute error in probability mass over the entire GL-queried exterior interval: mean {100*s['mean_exterior_mass_abs_error_total_probability']:.4f} percentage points, max {100*s['max_exterior_mass_abs_error_total_probability']:.4f} percentage points.
- Induced standard-GL operator relative L2 error over the full central domain [-6,6]: mean {100*s['mean_gl_rel_l2_full']:.4f}%, max {100*s['max_gl_rel_l2_full']:.4f}%.
- Even in the boundary strip 4<|x|<=6, the induced operator error is mean {100*s['mean_gl_rel_l2_boundary_4_lt_absx_le_6']:.3f}%, max {100*s['max_gl_rel_l2_boundary_4_lt_absx_le_6']:.3f}%.

## t=0 control
At t=0 the initial density is Gaussian, so an alpha-stable algebraic tail is not an asymptotically correct relative model. The relative exterior-density error is therefore large. However, the exact probability mass in 6<|x|<=12.32 is only {t0_mass:.3e}; the maximum induced full-domain GL relative L2 error across alpha is {t0_gl:.3e}. Thus the failure of the heavy-tail relative asymptotic at exactly t=0 has negligible absolute/operator impact for L=6.

## Interpretation
The padding is not pointwise exact: at positive times it overestimates the finite-range exterior density by a moderate amount that grows with alpha and t. The key numerical observation is that this discrepancy occurs in a low-mass region and is strongly attenuated in the GL weighted sum. Therefore the manuscript should call Eq. (29) an **asymptotically motivated exterior approximation**, not claim exact or rigorous exterior reconstruction.
"""
    (outdir / "boundary_padding_report.md").write_text(report, encoding="utf-8")
    print(f"\nResults written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
