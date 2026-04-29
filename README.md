# SBPINN (Fractional Score-Based PINN Solver Framework)

## 1. System Requirements

* **Python** 3.8+ (for neural network training, evaluation, and scientific visualization)
* **Hardware**: CUDA-enabled GPU is highly recommended for accelerating fractional operator evaluation.
  * Required packages: `torch` (PyTorch), `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`

---

## 2. Overview

SBPINN is a computational framework for solving fractional Fokker-Planck (FFP) equations and modeling complex anomalous diffusion systems driven by $\alpha$-stable Lévy noise. 

The pipeline integrates:

* Data-driven Sliced Score Matching (SSM)
* Log-likelihood reconstruction under physical constraints
* Grünwald-Letnikov (GL) discrete approximation for nonlocal Riesz derivatives
* Asymptotic boundary padding strategy to eliminate truncation singularities
* Adaptive physical weighting mechanism (via $e^q$ multiplier) to ensure stable convergence
* High-fidelity, publication-ready academic visualizations (Nature/Cell/SIAM styling standards using Times New Roman).

The framework is validated on four distinct dynamical scenarios: **Linear fOU Process**, **Nonlinear Bistable System**, **Parameter Inversion Problem**, and **Multiplicative Noise-Driven System**.

---

## 3. Repository Structure

```text
SBPINN/
│
├── 1.Linear_fOU_Process/             # Scripts for linear fractional OU process
├── 2.Bistable_System/                # Scripts for nonlinear bimodal distributions
├── 3.Parameter_Inversion/            # Scripts for inverse parameter identification
├── 4.Multiplicative_Noise/           # Scripts for state-dependent noise systems
├── 5.Benchmarks/                     # Baseline comparison and efficiency evaluation
└── README.md                         # Project documentation
```

---

## 4. File Descriptions

### 4.1 Linear Fractional OU Process (`1.Linear_fOU_Process/`)

**Scripts:**

| File | Description |
| :--- | :--- |
| `train_linear_fou.py` | Main script for training the two-stage model on the linear fOU process and generating PDF evolution figures. |
| `eval_linear_fou.py` | Evaluates the model across different stability indices ($\alpha$) and outputs quantitative error metrics (L2, Linf, Wasserstein). |

### 4.2 Nonlinear Bistable System (`2.Bistable_System/`)

**Scripts:**

| File | Description |
| :--- | :--- |
| `train_bistable_system.py` | Implements the expert divide-and-conquer strategy to avoid mode collapse in bimodal systems and generates visual results. |
| `eval_bistable_system.py` | Evaluates the superposed bimodal distribution predictions and calculates long-tail structural errors. |

### 4.3 Parameter Inversion (`3.Parameter_Inversion/`)

**Scripts:**

| File | Description |
| :--- | :--- |
| `train_parameter_inversion.py` | Executes the three-stage optimization pipeline to dynamically invert the unknown restoring coefficient ($\lambda$) from synthetic observational data. |

### 4.4 Multiplicative Noise-Driven System (`4.Multiplicative_Noise/`)

**Scripts:**

| File | Description |
| :--- | :--- |
| `train_multiplicative_noise.py` | Incorporates the spatial median approximation strategy to decouple spatially varying coefficients from the nonlocal Riesz operator. |
| `eval_multiplicative_noise.py` | Generates comprehensive error tables (`.csv`) evaluating the method's accuracy under multiplicative Lévy noise. |

### 4.5 Benchmarks & Efficiency (`5.Benchmarks/`)

**Scripts:**

| File | Description |
| :--- | :--- |
| `benchmark_vanilla_fpinn.py` | Compares SBPINN against traditional soft-penalty Vanilla fPINN, generating absolute error heatmaps and global physical constraint analyses. |
| `eval_computational_efficiency.py`| Tracks and outputs average execution time, convergence iteration counts, and mass conservation deviation metrics. |

---

## 5. Execution Pipeline

The framework is highly modular. You can run the independent scripts for each experimental scenario. For a standard evaluation of a specific dynamical system, follow this general workflow:

### Step 1: Data Generation & Stage I Training
**Action:** Run `train_*.py` scripts (e.g., `train_linear_fou.py`).
**Process:** * Simulates stochastic trajectories using the Euler-Maruyama scheme.
* Executes Sliced Score Matching (SSM) to estimate the empirical score field.
* Saves the frozen score network weights as prior knowledge.

### Step 2: Stage II Log-Likelihood Learning
**Action:** Continues automatically within the `train_*.py` scripts.
**Process:**
* Initializes the log-likelihood network.
* Evaluates fractional ODE residuals using the hybrid AD-GL computational strategy.
* Applies the asymptotic boundary padding and algebraic residual reconstruction for numerical stability.

### Step 3: Quantitative Evaluation
**Action:** Run `eval_*.py` scripts (e.g., `eval_linear_fou.py`).
**Process:**
* Compares network predictions against large-scale Monte Carlo ground truth.
* Computes $L^2$ error, $L^{\infty}$ error, and Wasserstein distance.
* Generates `experimental_errors_table.csv` for statistical reporting.

### Step 4: Academic Visualization
**Action:** Handled by the training/evaluation scripts.
**Process:**
* Outputs high-resolution figures (`.pdf` and `.png`).
* Figures follow professional standards: Times New Roman font, no background shading, and specific color palettes (Nature/SIAM style).
* For bimodal systems, plots individual expert fits alongside the assembled global density.

---

## 6. Notes

* **Alpha ($\alpha$) Parameter**: This refers to the stability index of the Lévy process ($0 < \alpha < 2$). It determines the rate of power-law decay in the heavy-tailed distributions and the intensity of jump discontinuities.
* **Fixed GL Weights**: Note that the Grünwald-Letnikov (GL) weights are fixed for calculation once the fractional order is set and are not dynamically adjustable during training.
* **Numerical Stability**: The framework utilizes an exponential multiplier ($e^q$) in the physical residual reconstruction. This is critical for eliminating numerical singularities and preventing gradient explosion in low-probability tail regions.
* **Diagram Standards**: All generated visualizations adhere to professional journal standards (Nature/SIAM). Specific textual labels (e.g., "80% High Density") and background shading have been removed to ensure a clean, academic aesthetic.
* **Boundary Handling**: The asymptotic boundary padding strategy is applied at the spatial truncation threshold (e.g., $|x| = 5.0$ or $8.0$) to handle the contradiction between finite domains and infinite-domain fractional dynamics.
* **Output**: All high-resolution figures (`.pdf`, `.png`) and quantitative result tables (`.csv`) are saved directly to the root directory of the project upon script completion.