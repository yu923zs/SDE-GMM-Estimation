# Robust CF-GMM Estimation for CTS-OU Models

> **Authorship note.** Companion code for the paper below, produced during my
> master's studies. The code was written and debugged with the assistance of AI
> coding tools; my own contributions are the study design, estimator
> specification, parameter choices, validation of the reported results, and
> manuscript writing. The pre-computed outputs in `results/` are the ones used
> in the paper.

Companion code for the paper:

"Robust characteristic-function GMM estimation of classical tempered stable
Ornstein--Uhlenbeck models" (submitted to *Computational Economics*, special
issue "Financial Markets and Nonlinear Dynamics").

## Overview

The repository implements three two-step GMM estimators for a classical
tempered stable (CTS) Lévy-driven Ornstein--Uhlenbeck model:

| Method | Description |
|--------|-------------|
| CF-GMM | Standard two-step CF-GMM with identity-then-optimal weighting. |
| Huber-CF-GMM | Marginal trimming (3.5-MAD) plus Huber-weighted covariance for the step-2 weight matrix. |
| PC-Huber-CF-GMM | Huber weighting plus a principal-component spectral floor (1e-3 × λ_max) on the weight-matrix eigenvalues. |

The 22-dimensional moment vector combines 4 dynamic moments (residual,
residual × lag, centered squared residual, centered squared residual × lag)
with 18 stationary characteristic-function moments (real and imaginary parts
at 9 frequencies).

## Repository structure

```
SDE-GMM-Estimation/
├── src/
│   ├── cts_ou.py        # model, simulation, three estimators
│   ├── run_mc.py        # Monte Carlo experiments
│   ├── empirical.py     # VIX full-sample, shock sensitivity, rolling window
│   ├── se_calc.py       # asymptotic sandwich standard errors (VIX)
│   ├── se_mc_check.py   # MC validation of the sandwich SEs
│   ├── j_test.py        # Hansen J overidentification test (VIX)
│   ├── sens_grid.py     # sensitivity to alpha and lambda
│   ├── gen_fig1.py      # Figure 1 (MC RMSE)
│   └── gen_fig23.py     # Figures 2-3 (VIX series, rolling sigma, shock sensitivity)
├── data/
│   └── vix_fred.csv     # FRED VIXCLS, 1990-01-02 to 2026-09-21
├── results/             # pre-computed outputs
│   ├── mc_results.json
│   ├── vix_stage1.json
│   ├── vix_results.json
│   ├── rolling_ckpt.json
│   ├── se_vix.json
│   ├── j_test.json
│   └── sens_grid.json
├── figures/             # paper figures (PDF)
│   ├── fig1.pdf
│   ├── fig2.pdf
│   └── fig3.pdf
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## Data

VIX daily close (VIXCLS) is retrieved from the Federal Reserve Bank of St.
Louis (FRED). The included `data/vix_fred.csv` spans 1990-01-02 to
2026-09-21 (9278 observations, Δ = 1/252). The series is publicly available
at https://fred.stlouisfed.org/series/VIXCLS.

## Reproducing the results

All scripts are run from the `src/` directory.

### 1. Monte Carlo (Tables 1-3, Figure 1)

```bash
cd src
# 4 scenarios × 4 contamination mechanisms = 16 cells, 300 replications each
for s in 0 1 2 3; do for c in 0 1 2 3; do python run_mc.py $s $c; done; done
# generate Figure 1
python gen_fig1.py
```

Parameters: α = 1.5, λ = 1, n = 600, Δ = 1/12, R = 300 replications.
Scenarios S1-S4 and contamination mechanisms (clean, additive outliers,
block shift, scale contamination) are defined in `run_mc.py`.

### 2. VIX empirical analysis (Tables 4 and 6, Figures 2-3)

```bash
cd src
python empirical.py     # full-sample, shock sensitivity, rolling window
python gen_fig23.py     # Figures 2-3
```

### 3. Standard errors and J test (Table 6)

```bash
python se_calc.py       # sandwich SEs for the VIX full-sample fits
python j_test.py        # Hansen J overidentification test (df = 19)
```

### 4. Sensitivity to fixed tail parameters (Table 5)

```bash
python sens_grid.py     # alpha in {1.2, 1.5, 1.8} × lambda in {0.5, 1, 2}
```

### 5. SE validation (appendix)

```bash
python se_mc_check.py   # 100-rep MC comparing asymptotic SE with empirical dispersion
```

## Mapping to the paper

| Script | Paper output |
|--------|-------------|
| run_mc.py | Tables 1-3 |
| empirical.py | Table 4, Table 6 (rolling window) |
| se_calc.py | Table 6 (standard errors) |
| j_test.py | Table 6 (Hansen J test) |
| sens_grid.py | Table 5 |
| se_mc_check.py | Appendix (SE validation) |
| gen_fig1.py | Figure 1 |
| gen_fig23.py | Figures 2-3 |

## Environment

```bash
pip install -r requirements.txt
```

Python 3.10+ with NumPy, pandas, SciPy, matplotlib, statsmodels.

### Docker

```bash
docker build -t sde-gmm .
docker run -it sde-gmm
# inside the container (WORKDIR /app/src):
python run_mc.py 0 0
```

## License

MIT
