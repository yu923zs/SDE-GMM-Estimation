# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:00:29 2026

@author: 86183
"""

# -*- coding: utf-8 -*-
"""
One-file code for CTS Levy driven OU model empirical figures/tables.

Model:
    dX_t = -kappa (X_t - mu) dt + sigma dL_t

Methods:
    1. GMM
    2. Huber-GMM
    3. PC-CF-RGMM

Input:
    VIX_Processed_For_TW_GMM.csv

Output:
    results_vix_cts_ou/
        figures/
        tables/

Author: your name
Date: 2026-05-13
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gamma
from scipy import stats
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# ============================================================
# 0. User settings
# ============================================================

DATA_PATH = "VIX_Processed_For_TW_GMM.csv"

# 建议使用 log(VIX) 作为 OU 状态变量
USE_LOG = True

# 日度数据。若按年化时间，也可设 H = 1/252。
# 这里为了和你原文模拟 h=1 保持一致，默认 h=1。
H = 1.0

# 输出目录
BASE_DIR = os.path.abspath(".")
OUT_DIR = os.path.join(BASE_DIR, "results_vix_cts_ou_stage3_final")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# 三种方法
METHODS = ["GMM", "Huber-GMM", "PC-CF-RGMM"]
PARAM_NAMES = ["kappa", "mu", "sigma", "alpha", "lambda"]

# 估计设置
HUBER_C = 2.5
CF_FREQS = np.array([0.15, 0.30, 0.50, 0.75, 1.00, 1.35, 1.75, 2.25, 3.00])
RHO_CF = 1.0
PC_TAU = 0.90

PC_RIDGE = 1e-4
PC_EIG_FLOOR = 1e-3
PC_INV_CAP = 200.0
PC_WEIGHT_SHRINKAGE = 0.30

MAXITER = 800
N_STARTS = 12
RANDOM_SEED = 20260513

# 样本外预测比例
TEST_RATIO = 0.20

# 滚动窗口设置
ROLLING_WINDOW = 500
ROLLING_STEP = 60
RUN_ROLLING = True

# 如果数据样本较短，自动调小滚动窗口
MIN_ROLLING_WINDOW = 250

np.random.seed(RANDOM_SEED)

# ============================================================
# 1. Plot style
# ============================================================

def set_plot_style():
    plt.rcParams["figure.figsize"] = (10, 5.5)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.30
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False

set_plot_style()

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Figure saved] {path}")

# ============================================================
# 2. Load VIX data
# ============================================================

def detect_date_column(df):
    candidates = ["observation_date", "Date", "date", "DATE", "Time", "time", "datetime", "Datetime"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def detect_value_column(df):
    """
    自动识别 VIX 数值列。
    优先选择列名中含 VIX、Close、value 的数值列。
    若找不到，则选择最后一个数值列。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        # 尝试把非日期列转成数值
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    priority_keywords = ["VIX", "vix", "Close", "close", "Adj Close", "value", "Value", "PX_LAST"]
    for key in priority_keywords:
        for c in numeric_cols:
            if key in c:
                return c

    if len(numeric_cols) == 0:
        raise ValueError("No numeric column found in CSV. Please check VIX data file.")

    return numeric_cols[-1]

def load_vix_data(path, use_log=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find data file: {path}")

    df = pd.read_csv(path)

    date_col = detect_date_column(df)
    value_col = detect_value_column(df)

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    else:
        df["IndexDate"] = np.arange(len(df))
        date_col = "IndexDate"

    vix = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.DataFrame({
        "date": df[date_col],
        "VIX": vix
    })

    out = out.dropna()
    out = out[out["VIX"] > 0].copy()
    out = out.reset_index(drop=True)

    if use_log:
        out["X"] = np.log(out["VIX"])
        x_label = "log(VIX)"
    else:
        out["X"] = out["VIX"]
        x_label = "VIX"

    print("=" * 80)
    print("[Data loaded]")
    print(f"File: {path}")
    print(f"Date column: {date_col}")
    print(f"Value column: {value_col}")
    print(f"Observations: {len(out)}")
    print(f"Input variable X: {x_label}")
    print("=" * 80)

    return out, x_label

# ============================================================
# 3. CTS characteristic exponent and OU innovation CF
# ============================================================

def cts_C_standardized(alpha, lam):
    """
    C such that Var(L_1)=1:
        Var(L_1) = 2 C Gamma(2-alpha) lambda^{alpha-2}
    """
    return 1.0 / (2.0 * gamma(2.0 - alpha) * lam ** (alpha - 2.0))

def cts_exponent(u, alpha, lam):
    """
    Symmetric Classical Tempered Stable characteristic exponent:
        Psi_L(u) =
        +C Gamma(-alpha) [ (lambda-iu)^alpha + (lambda+iu)^alpha - 2 lambda^alpha ]

    注意 alpha=1 附近 Gamma(-alpha) 有奇点，因此估计时避开 alpha=1。
    """
    u = np.asarray(u, dtype=float)
    C = cts_C_standardized(alpha, lam)
    val = C * gamma(-alpha) * (
        (lam - 1j * u) ** alpha
        + (lam + 1j * u) ** alpha
        - 2.0 * lam ** alpha
    )
    # FIXED: CTS exponent sign (per Kim et al. 2008, Rosinski 2007, Grabchak 2020)
    return val

def ou_innovation_cf(u, theta, h=1.0, quad_nodes=40):
    """
    OU innovation CF:
        phi_h(u) = exp{ int_0^h Psi_L(u sigma exp(-kappa r)) dr }
    """
    kappa, mu, sigma, alpha, lam = theta

    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    scalar_input = np.isscalar(u) or np.ndim(u) == 0

    nodes, weights = np.polynomial.legendre.leggauss(quad_nodes)
    r_nodes = 0.5 * h * (nodes + 1.0)
    r_weights = 0.5 * h * weights

    arg = u_arr[:, None] * sigma * np.exp(-kappa * r_nodes[None, :])
    psi_vals = cts_exponent(arg, alpha, lam)
    integral_vals = np.sum(psi_vals * r_weights[None, :], axis=1)
    phi_vals = np.exp(integral_vals)

    if scalar_input:
        return phi_vals[0]
    return phi_vals



def sanity_check_cts():
    """
    Lightweight pre-run CTS sanity check.
    If this fails, the script stops before any simulation/estimation.
    """
    # 1. psi(0) == 0
    v0 = cts_exponent(0, alpha=0.7, lam=1.0)
    assert abs(v0) < 1e-12, f"psi(0) failed: {v0}"
    print("[sanity 1/5] psi(0) == 0 OK")

    # 2. Re(psi(u)) <= 0 for real u
    for u in [0.1, 1.0, 5.0]:
        val = cts_exponent(u, 0.7, 1.0)
        assert np.real(val) <= 1e-12, f"Re(psi({u})) > 0: {val}"
    print("[sanity 2/5] Re(psi(u)) <= 0 OK")

    # 3. |exp(psi(u))| <= 1
    for u in [0.1, 1.0, 5.0]:
        val = np.exp(cts_exponent(u, 0.7, 1.0))
        assert abs(val) <= 1 + 1e-10, f"|exp(psi({u}))| > 1: {val}"
    print("[sanity 3/5] |exp(psi(u))| <= 1 OK")

    # 4. ou_innovation_cf(0) == 1
    theta = np.array([0.5, 0.0, 1.0, 0.7, 1.0], dtype=float)
    val = ou_innovation_cf(0.0, theta, h=0.01)
    assert abs(val - 1.0) < 1e-12, f"ou_innovation_cf(0) failed: {val}"
    print("[sanity 4/5] ou_innovation_cf(0) == 1 OK")

    # 5. CF decay.
    # NOTE: h=0.01 at u=100 is too short and can falsely fail.
    # We use h=1.0 for a reliable smoke check.
    val = ou_innovation_cf(100.0, theta, h=1.0)
    assert abs(val) < 1e-3, f"CF decay failed: |phi_h(100)|={abs(val)}"
    print("[sanity 5/5] CF decay OK")

    print("✅ CTS exponent sanity check PASSED")




def append_timing_log_external(msg):
    """
    Append an auditable timing/event line.
    This is intentionally lightweight and independent from internal checkpoint logic.
    """
    try:
        path = os.path.join(RESULT_DIR, "timing_log.txt")
    except Exception:
        try:
            path = os.path.join(OUT_DIR, "timing_log.txt")
        except Exception:
            path = "timing_log.txt"

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(msg).rstrip() + "\n")
    except Exception:
        pass


# ============================================================
# 4. Moment functions
# ============================================================

def residuals_and_z(x, theta, h=1.0):
    """
    Discrete OU:
        X_j = mu + exp(-kappa h)(X_{j-1}-mu) + delta_j
    """
    kappa, mu, sigma, alpha, lam = theta
    phi = np.exp(-kappa * h)

    x_lag = x[:-1]
    x_now = x[1:]

    mean = mu + phi * (x_lag - mu)
    delta = x_now - mean

    vh = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * h)) / (2.0 * kappa)
    if vh <= 0 or not np.isfinite(vh):
        z = np.full_like(delta, np.nan)
    else:
        z = delta / np.sqrt(vh)

    return delta, z, x_lag

def dynamic_moments(x, theta, h=1.0):
    kappa, mu, sigma, alpha, lam = theta
    delta, z, x_lag = residuals_and_z(x, theta, h)

    x_scale = sigma / np.sqrt(2.0 * kappa)
    if x_scale <= 1e-8 or not np.isfinite(x_scale):
        x_scale = np.nanstd(x_lag)
    if x_scale <= 1e-8 or not np.isfinite(x_scale):
        x_scale = 1.0

    xc = (x_lag - mu) / x_scale

    m1 = z
    m2 = z * xc
    m3 = z ** 2 - 1.0
    m4 = (z ** 2 - 1.0) * xc

    return np.column_stack([m1, m2, m3, m4])

def radial_huber(y, c=2.5):
    """
    Row-wise radial Huber transform.
    """
    y = np.asarray(y, dtype=float)
    norm = np.linalg.norm(y, axis=1)
    scale = np.ones_like(norm)
    mask = norm > c
    scale[mask] = c / norm[mask]
    return y * scale[:, None]

def robust_dynamic_moments(x, theta, h=1.0, c=2.5):
    """
    Robust version of dynamic moments.
    使用 MAD 标准化后做径向 Huber 截断。
    """
    d = dynamic_moments(x, theta, h)
    if not np.all(np.isfinite(d)):
        return d

    med = np.nanmedian(d, axis=0)
    mad = np.nanmedian(np.abs(d - med), axis=0)
    scale = 1.4826 * mad
    scale[scale < 1e-8] = 1.0

    y = d / scale
    yh = radial_huber(y, c=c)
    return yh

def cf_moments(x, theta, h=1.0, freqs=CF_FREQS):
    """
    Standardized characteristic-function moments.
    Use z = delta / sqrt(v_h), and theoretical CF:
        phi_z(u) = phi_eta(u / sqrt(v_h)).
    """
    kappa, mu, sigma, alpha, lam = theta
    delta, z, x_lag = residuals_and_z(x, theta, h)

    freqs = np.asarray(freqs, dtype=float)

    vh = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * h)) / (2.0 * kappa)
    if vh <= 0 or not np.isfinite(vh):
        return np.full((len(delta), 2 * len(freqs)), np.nan)

    sd_h = np.sqrt(vh)

    phi_vals = ou_innovation_cf(
        freqs / sd_h,
        theta,
        h=h
    )

    cols = []
    for k, u in enumerate(freqs):
        cols.append(np.cos(u * z) - np.real(phi_vals[k]))
        cols.append(np.sin(u * z) - np.imag(phi_vals[k]))

    return np.column_stack(cols)

def combined_moments(x, theta, method):
    c = cf_moments(x, theta, H, CF_FREQS)

    if method == "GMM":
        d = dynamic_moments(x, theta, H)
        return np.column_stack([d, RHO_CF * c])

    if method == "Huber-GMM":
        d = robust_dynamic_moments(x, theta, H, HUBER_C)
        return np.column_stack([d, RHO_CF * c])

    if method == "PC-CF-RGMM":
        d = robust_dynamic_moments(x, theta, H, HUBER_C)
        return np.column_stack([d, RHO_CF * c])

    raise ValueError(f"Unknown method: {method}")

# ============================================================
# 5. Estimation
# ============================================================

def ar1_initial_guess(x):
    """
    AR(1)-based initial values.
    """
    x_lag = x[:-1]
    x_now = x[1:]

    Xmat = np.column_stack([np.ones_like(x_lag), x_lag])
    beta = np.linalg.lstsq(Xmat, x_now, rcond=None)[0]
    c, phi = beta[0], beta[1]

    phi = float(np.clip(phi, 0.001, 0.999))
    kappa = -np.log(phi) / H
    mu = c / (1.0 - phi)

    resid = x_now - (c + phi * x_lag)
    sd_resid = np.std(resid, ddof=1)

    denom = np.sqrt((1.0 - np.exp(-2.0 * kappa * H)) / (2.0 * kappa))
    sigma = sd_resid / max(denom, 1e-8)

    # tail 参数初值
    alpha = 0.75
    lam = 2.0

    theta0 = np.array([kappa, mu, sigma, alpha, lam], dtype=float)
    return theta0

def parameter_bounds(x):
    """
    根据实际数据自动设置参数边界。
    """
    x = np.asarray(x, dtype=float)
    x_mean = np.mean(x)
    x_std = np.std(x, ddof=1)
    dx_std = np.std(np.diff(x), ddof=1)

    mu_lo = x_mean - 1.5 * x_std
    mu_hi = x_mean + 1.5 * x_std

    sigma_lo = max(1e-4, 0.02 * dx_std)
    sigma_hi = max(1.0, 20.0 * dx_std)

    bounds = [
        (0.001, 0.50),       # kappa
        (mu_lo, mu_hi),      # mu
        (sigma_lo, sigma_hi),# sigma
        (0.20, 1.80),        # alpha
        (0.20, 10.00),       # lambda
    ]
    return bounds

def theta_valid(theta, bounds):
    theta = np.asarray(theta, dtype=float)

    if not np.all(np.isfinite(theta)):
        return False

    kappa, mu, sigma, alpha, lam = theta

    if kappa <= 0 or sigma <= 0 or lam <= 0:
        return False
    if not (0.0 < alpha < 2.0):
        return False

    # 避开 alpha=1 的数值奇点
    if abs(alpha - 1.0) < 0.015:
        return False

    for val, (lo, hi) in zip(theta, bounds):
        if val < lo or val > hi:
            return False

    return True

def soft_penalty(theta, bounds):
    """
    边界软惩罚，避免估计值贴边。
    """
    penalty = 0.0

    for val, (lo, hi) in zip(theta, bounds):
        width = hi - lo
        margin = 0.03 * width

        if val - lo < margin:
            penalty += ((margin - (val - lo)) / margin) ** 4

        if hi - val < margin:
            penalty += ((margin - (hi - val)) / margin) ** 4

    # alpha=1 附近软惩罚
    alpha = theta[3]
    dist = abs(alpha - 1.0)
    if dist < 0.04:
        penalty += ((0.04 - dist) / 0.04) ** 4

    return 1e3 * penalty

def objective(theta, x, method, bounds, W=None):
    theta = np.asarray(theta, dtype=float)

    if not theta_valid(theta, bounds):
        return 1e12

    try:
        G = combined_moments(x, theta, method)

        if not np.all(np.isfinite(G)):
            return 1e12

        gbar = np.mean(G, axis=0)

        if W is None:
            val = float(gbar @ gbar)
        else:
            val = float(gbar @ W @ gbar)

        if not np.isfinite(val):
            return 1e12

        return val + soft_penalty(theta, bounds)

    except Exception:
        return 1e12

def make_initial_guesses(theta0, bounds, n_starts=N_STARTS):
    guesses = []

    tail_grid = [
        (0.45, 0.60),
        (0.60, 0.80),
        (0.75, 1.20),
        (0.75, 2.00),
        (1.20, 3.00),
        (1.30, 4.00),
    ]

    for a0, l0 in tail_grid:
        g = theta0.copy()
        g[3] = a0
        g[4] = l0

        for i, (lo, hi) in enumerate(bounds):
            g[i] = np.clip(g[i], lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))

        if abs(g[3] - 1.0) < 0.05:
            g[3] = 0.90 if g[3] < 1.0 else 1.10

        guesses.append(g)

    n_extra = max(0, n_starts - len(guesses))

    for _ in range(n_extra):
        scale = np.array([
            np.random.normal(1.0, 0.25),
            np.random.normal(1.0, 0.10),
            np.random.normal(1.0, 0.25),
            np.random.normal(1.0, 0.20),
            np.random.normal(1.0, 0.30),
        ])
        g = theta0 * scale

        for i, (lo, hi) in enumerate(bounds):
            g[i] = np.clip(g[i], lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))

        if abs(g[3] - 1.0) < 0.05:
            g[3] = 0.90 if g[3] < 1.0 else 1.10

        guesses.append(g)

    return guesses

def compute_pc_weight(x, theta_prelim, method):
    """
    Safe PC-regularized weighting matrix.
    """
    G = combined_moments(x, theta_prelim, method)

    if not np.all(np.isfinite(G)):
        return np.eye(G.shape[1]), np.nan

    Gc = G - np.mean(G, axis=0)
    q = Gc.shape[1]

    S = (Gc.T @ Gc) / Gc.shape[0]
    S = S + PC_RIDGE * np.eye(q)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = np.sum(eigvals)
    if total <= 0 or not np.isfinite(total):
        return np.eye(q), np.nan

    ratio = np.cumsum(eigvals) / total
    r = int(np.searchsorted(ratio, PC_TAU) + 1)
    r = max(1, min(r, q))

    P = eigvecs[:, :r]
    L = eigvals[:r]

    floor = PC_EIG_FLOOR * np.max(eigvals)
    L_safe = np.maximum(L, floor)

    inv_L = 1.0 / L_safe
    inv_L = np.minimum(inv_L, PC_INV_CAP)

    W_pc = P @ np.diag(inv_L) @ P.T

    W = (1.0 - PC_WEIGHT_SHRINKAGE) * W_pc + PC_WEIGHT_SHRINKAGE * np.eye(q)

    if not np.all(np.isfinite(W)):
        return np.eye(q), r

    return W, r

def estimate_method(x, method):
    """
    Estimate one method.
    """
    x = np.asarray(x, dtype=float)
    bounds = parameter_bounds(x)
    theta0 = ar1_initial_guess(x)
    guesses = make_initial_guesses(theta0, bounds, N_STARTS)

    best_res = None
    best_val = np.inf

    # Stage 1: identity weight
    for g0 in guesses:
        res = minimize(
            objective,
            g0,
            args=(x, method, bounds, None),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": MAXITER, "ftol": 1e-9}
        )

        val = res.fun if np.isfinite(res.fun) else np.inf
        if val < best_val:
            best_val = val
            best_res = res

    if best_res is None:
        return {
            "method": method,
            "success": False,
            "theta": np.full(5, np.nan),
            "objective": np.nan,
            "pc_rank": np.nan,
            "message": "failed"
        }

    theta_prelim = best_res.x
    pc_rank = np.nan

    final_res = best_res

    # Stage 2 for PC-CF-RGMM
    if method == "PC-CF-RGMM":
        try:
            W_pc, pc_rank = compute_pc_weight(x, theta_prelim, method)
        except Exception:
            W_pc = None
            pc_rank = np.nan

        best_res2 = None
        best_val2 = np.inf

        for g0 in [theta_prelim] + guesses:
            res2 = minimize(
                objective,
                g0,
                args=(x, method, bounds, W_pc),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": MAXITER, "ftol": 1e-9}
            )

            val2 = res2.fun if np.isfinite(res2.fun) else np.inf
            if val2 < best_val2:
                best_val2 = val2
                best_res2 = res2

        if best_res2 is not None and np.isfinite(best_val2):
            if best_val2 <= 10.0 * best_val:
                final_res = best_res2
            else:
                final_res = best_res
        else:
            final_res = best_res

    return {
        "method": method,
        "success": bool(final_res.success),
        "theta": final_res.x,
        "objective": float(final_res.fun),
        "pc_rank": pc_rank,
        "message": str(final_res.message)
    }

def estimate_all_methods(x):
    results = []
    for m in METHODS:
        print(f"[Estimating] {m} ...")
        res = estimate_method(x, m)
        results.append(res)

        print(
            f"  success={res['success']}, "
            f"theta={np.round(res['theta'], 6)}, "
            f"objective={res['objective']:.6g}"
        )

    rows = []
    for res in results:
        row = {
            "method": res["method"],
            "success": res["success"],
            "objective": res["objective"],
            "pc_rank": res["pc_rank"],
            "message": res["message"]
        }
        for i, p in enumerate(PARAM_NAMES):
            row[p] = res["theta"][i]
        rows.append(row)

    df_params = pd.DataFrame(rows)
    return results, df_params

# ============================================================
# 6. Forecasting
# ============================================================

def one_step_forecast(x, theta):
    kappa, mu, sigma, alpha, lam = theta
    phi = np.exp(-kappa * H)
    return mu + phi * (x[:-1] - mu)

def forecast_evaluation(x_train, x_test, theta):
    """
    使用训练集最后一个点作为测试递推起点。
    对测试期做一步预测：
        E[X_{t+1}|X_t] = mu + phi (X_t - mu)
    """
    kappa, mu, sigma, alpha, lam = theta
    phi = np.exp(-kappa * H)

    x_all = np.concatenate([[x_train[-1]], x_test])
    pred = mu + phi * (x_all[:-1] - mu)
    actual = x_all[1:]

    err = actual - pred

    out = {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MeanError": np.mean(err),
        "StdError": np.std(err, ddof=1)
    }

    return pred, actual, err, out

# ============================================================
# 7. Tables
# ============================================================

def descriptive_statistics(df, x_label):
    x = df["X"].values
    dx = np.diff(x)

    rows = []

    for name, arr in [(x_label, x), (f"Delta {x_label}", dx)]:
        row = {
            "series": name,
            "n": len(arr),
            "mean": np.mean(arr),
            "std": np.std(arr, ddof=1),
            "min": np.min(arr),
            "q05": np.quantile(arr, 0.05),
            "median": np.median(arr),
            "q95": np.quantile(arr, 0.95),
            "max": np.max(arr),
            "skewness": stats.skew(arr),
            "kurtosis": stats.kurtosis(arr, fisher=False),
            "acf1": acf(arr, nlags=1, fft=False)[1] if len(arr) > 3 else np.nan
        }
        rows.append(row)

    desc = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_descriptive_statistics.csv")
    desc.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")
    return desc

def save_parameter_table(df_params):
    path = os.path.join(TAB_DIR, "table_parameter_estimates.csv")
    df_params.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    # LaTeX version
    tex_path = os.path.join(TAB_DIR, "table_parameter_estimates.tex")
    cols = ["method"] + PARAM_NAMES + ["objective", "pc_rank"]
    tex = df_params[cols].to_latex(
        index=False,
        float_format=lambda z: f"{z:.6f}",
        caption="Parameter estimates for VIX empirical application",
        label="tab:vix_parameter_estimates",
        escape=False
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[LaTeX table saved] {tex_path}")

def save_forecast_table(rows):
    df = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_forecast_errors.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    tex_path = os.path.join(TAB_DIR, "table_forecast_errors.tex")
    tex = df.to_latex(
        index=False,
        float_format=lambda z: f"{z:.6f}",
        caption="One-step out-of-sample forecast errors",
        label="tab:vix_forecast_errors",
        escape=False
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[LaTeX table saved] {tex_path}")

    return df

def save_forecast_train_parameter_table(rows):
    df = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_forecast_train_parameter_estimates.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")
    return df

def save_forecast_details_table(forecast_details, use_log=True):
    rows = []

    for method, det in forecast_details.items():
        dates = det["date"]
        pred = det["pred"]
        actual = det["actual"]
        err = det["error"]

        for i in range(len(actual)):
            row = {
                "method": method,
                "date": dates[i],
                "actual_X": actual[i],
                "pred_X": pred[i],
                "error_X_actual_minus_pred": err[i],
            }

            if use_log:
                row["actual_VIX"] = np.exp(actual[i])
                row["pred_VIX"] = np.exp(pred[i])
                row["error_VIX_actual_minus_pred"] = row["actual_VIX"] - row["pred_VIX"]

            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_oos_forecast_details.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    summary_rows = []
    for method, g in df.groupby("method"):
        row = {
            "method": method,
            "n": len(g),
            "actual_X_mean": g["actual_X"].mean(),
            "pred_X_mean": g["pred_X"].mean(),
            "mean_error_X_actual_minus_pred": g["error_X_actual_minus_pred"].mean(),
            "actual_X_min": g["actual_X"].min(),
            "actual_X_max": g["actual_X"].max(),
            "pred_X_min": g["pred_X"].min(),
            "pred_X_max": g["pred_X"].max(),
        }

        if use_log:
            row["actual_VIX_mean"] = g["actual_VIX"].mean()
            row["pred_VIX_mean"] = g["pred_VIX"].mean()
            row["mean_error_VIX_actual_minus_pred"] = g["error_VIX_actual_minus_pred"].mean()

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(TAB_DIR, "table_oos_forecast_details_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {summary_path}")

    return df, df_summary

def save_rolling_table(rolling_df):
    path = os.path.join(TAB_DIR, "table_rolling_estimates.csv")
    rolling_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    stab_rows = []
    for method, g in rolling_df.groupby("method"):
        row = {"method": method, "n_windows": len(g)}
        for p in PARAM_NAMES:
            row[f"{p}_mean"] = g[p].mean()
            row[f"{p}_std"] = g[p].std(ddof=1)
            row[f"{p}_min"] = g[p].min()
            row[f"{p}_max"] = g[p].max()
        stab_rows.append(row)

    stab = pd.DataFrame(stab_rows)
    stab_path = os.path.join(TAB_DIR, "table_rolling_stability.csv")
    stab.to_csv(stab_path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {stab_path}")

    return stab

# ============================================================
# 8. Figures
# ============================================================

def fig_vix_series(df, x_label):
    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df["VIX"], lw=1.0)
    plt.title("VIX level series")
    plt.xlabel("Date")
    plt.ylabel("VIX")
    savefig("fig_01_vix_level_series.png")

    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df["X"], lw=1.0, color="darkblue")
    plt.title(f"{x_label} series used in estimation")
    plt.xlabel("Date")
    plt.ylabel(x_label)
    savefig("fig_02_model_input_series.png")

def fig_distribution_diagnostics(df, x_label):
    x = df["X"].values
    dx = np.diff(x)

    plt.figure(figsize=(11, 4.8))
    plt.hist(x, bins=40, density=True, alpha=0.70, color="steelblue", edgecolor="white")
    grid = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(grid, stats.norm.pdf(grid, np.mean(x), np.std(x, ddof=1)), "r--", lw=2)
    plt.title(f"Histogram of {x_label}")
    plt.xlabel(x_label)
    plt.ylabel("Density")
    savefig("fig_03_histogram_input.png")

    plt.figure(figsize=(11, 4.8))
    stats.probplot(dx, dist="norm", plot=plt)
    plt.title(f"Normal QQ plot of Delta {x_label}")
    savefig("fig_04_qqplot_delta_input.png")

    nlags = min(40, len(x) // 4)
    acf_x = acf(x, nlags=nlags, fft=False)
    acf_dx = acf(dx, nlags=nlags, fft=False)

    plt.figure(figsize=(11, 4.8))
    plt.stem(range(len(acf_x)), acf_x, basefmt=" ")
    plt.title(f"ACF of {x_label}")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    savefig("fig_05_acf_input.png")

    plt.figure(figsize=(11, 4.8))
    plt.stem(range(len(acf_dx)), acf_dx, basefmt=" ")
    plt.title(f"ACF of Delta {x_label}")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    savefig("fig_06_acf_delta_input.png")

def fig_fitted_values(df, results, x_label):
    x = df["X"].values
    date = df["date"].values

    plt.figure(figsize=(12, 5.5))
    plt.plot(date[1:], x[1:], label="Observed", color="black", lw=1.0)

    for res in results:
        theta = res["theta"]
        pred = one_step_forecast(x, theta)
        plt.plot(date[1:], pred, lw=1.0, label=res["method"])

    plt.title(f"In-sample one-step fitted values: {x_label}")
    plt.xlabel("Date")
    plt.ylabel(x_label)
    plt.legend()
    savefig("fig_07_insample_fitted_values.png")

def fig_forecast_errors(df, df_forecast, forecast_details, x_label):
    plt.figure(figsize=(11, 5))
    methods = df_forecast["method"].tolist()
    rmse_vals = df_forecast["RMSE"].values
    mae_vals = df_forecast["MAE"].values

    x_pos = np.arange(len(methods))
    width = 0.35

    plt.bar(x_pos - width / 2, rmse_vals, width=width, label="RMSE")
    plt.bar(x_pos + width / 2, mae_vals, width=width, label="MAE")
    plt.xticks(x_pos, methods)
    plt.ylabel("Forecast error")
    plt.title("Out-of-sample one-step forecast error comparison")
    plt.legend()
    savefig("fig_08_oos_forecast_errors_bar.png")

    plt.figure(figsize=(12, 5.5))
    for method, det in forecast_details.items():
        plt.plot(det["date"], det["error"], lw=1.0, label=method)

    plt.axhline(0.0, color="black", linestyle="--", lw=1)
    plt.title(f"Out-of-sample forecast errors: {x_label}")
    plt.xlabel("Date")
    plt.ylabel("Forecast error")
    plt.legend()
    savefig("fig_09_oos_forecast_errors_timeseries.png")

def run_rolling_estimation(df, x_label):
    x = df["X"].values
    dates = df["date"].values
    n = len(x)

    window = ROLLING_WINDOW
    if n < window + 50:
        window = max(MIN_ROLLING_WINDOW, int(0.60 * n))

    if n < window + 20:
        print("[Rolling skipped] Sample too short for rolling estimation.")
        return None

    rows = []
    starts = list(range(0, n - window + 1, ROLLING_STEP))

    print("=" * 80)
    print(f"[Rolling estimation] window={window}, step={ROLLING_STEP}, windows={len(starts)}")
    print("=" * 80)

    for s in starts:
        e = s + window
        x_win = x[s:e]
        date_end = dates[e - 1]

        for method in METHODS:
            print(f"  Rolling window end={date_end}, method={method}")
            try:
                res = estimate_method(x_win, method)
                row = {
                    "window_start": s,
                    "window_end": e - 1,
                    "date_end": date_end,
                    "method": method,
                    "success": res["success"],
                    "objective": res["objective"],
                    "pc_rank": res["pc_rank"]
                }
                for i, p in enumerate(PARAM_NAMES):
                    row[p] = res["theta"][i]
                rows.append(row)
            except Exception as err:
                row = {
                    "window_start": s,
                    "window_end": e - 1,
                    "date_end": date_end,
                    "method": method,
                    "success": False,
                    "objective": np.nan,
                    "pc_rank": np.nan
                }
                for p in PARAM_NAMES:
                    row[p] = np.nan
                rows.append(row)

    rolling_df = pd.DataFrame(rows)
    rolling_df["date_end"] = pd.to_datetime(rolling_df["date_end"], errors="coerce")

    rolling_stab = save_rolling_table(rolling_df)
    return rolling_df

def fig_rolling_estimates(rolling_df):
    if rolling_df is None or len(rolling_df) == 0:
        return

    # 论文中重点展示 kappa, alpha, lambda 即可，图不要太多
    key_params = ["kappa", "alpha", "lambda"]

    for p in key_params:
        plt.figure(figsize=(12, 5.5))

        for method in METHODS:
            sub = rolling_df[(rolling_df["method"] == method) & (rolling_df["success"] == True)]
            if len(sub) == 0:
                continue
            plt.plot(sub["date_end"], sub[p], marker="o", ms=3, lw=1.2, label=method)

        plt.title(f"Rolling-window estimates of {p}")
        plt.xlabel("Window end date")
        plt.ylabel(p)
        plt.legend()
        savefig(f"fig_10_rolling_{p}.png")

    # alpha-lambda 联合散点图
    plt.figure(figsize=(8, 6))
    for method in METHODS:
        sub = rolling_df[(rolling_df["method"] == method) & (rolling_df["success"] == True)]
        if len(sub) == 0:
            continue
        plt.scatter(sub["alpha"], sub["lambda"], s=35, alpha=0.75, label=method)

    plt.xlabel("alpha")
    plt.ylabel("lambda")
    plt.title("Rolling estimates: alpha-lambda stability")
    plt.legend()
    savefig("fig_11_rolling_alpha_lambda_scatter.png")

# ============================================================
# 8.5 Additional diagnostics for PC-CF-RGMM advantages
# ============================================================

def theta_from_param_row(row):
    return np.array(
        [row["kappa"], row["mu"], row["sigma"], row["alpha"], row["lambda"]],
        dtype=float
    )

def make_theta_dict(param_df):
    """
    Convert parameter table to {method: theta}.
    """
    out = {}
    for _, row in param_df.iterrows():
        method = row["method"]
        out[method] = theta_from_param_row(row)
    return out

def pc_weight_diagnostics_single(x, theta, method, sample_name="full"):
    """
    Diagnose covariance matrix of moment conditions and PC truncation effect.
    This is post-estimation diagnostic using the final theta.
    """
    G = combined_moments(x, theta, method)

    if not np.all(np.isfinite(G)):
        return {
            "sample": sample_name,
            "method": method,
            "q_moments": np.nan,
            "raw_min_eig": np.nan,
            "raw_max_eig": np.nan,
            "raw_condition_number": np.nan,
            "ridge_min_eig": np.nan,
            "ridge_max_eig": np.nan,
            "ridge_condition_number": np.nan,
            "selected_rank_tau90": np.nan,
            "explained_ratio_selected": np.nan,
            "discarded_eig_share": np.nan,
            "n_small_eigs_1e-6": np.nan,
            "n_small_eigs_1e-4": np.nan,
        }

    Gc = G - np.mean(G, axis=0)
    q = Gc.shape[1]

    S_raw = (Gc.T @ Gc) / Gc.shape[0]
    eig_raw = np.linalg.eigvalsh(S_raw)
    eig_raw = np.sort(eig_raw)[::-1]

    raw_max = eig_raw[0]
    raw_min = eig_raw[-1]

    if raw_min > 1e-14:
        raw_cond = raw_max / raw_min
    else:
        raw_cond = np.inf

    S_ridge = S_raw + PC_RIDGE * np.eye(q)
    eig_ridge = np.linalg.eigvalsh(S_ridge)
    eig_ridge = np.sort(eig_ridge)[::-1]

    ridge_max = eig_ridge[0]
    ridge_min = eig_ridge[-1]
    ridge_cond = ridge_max / ridge_min if ridge_min > 1e-14 else np.inf

    total = np.sum(eig_ridge)
    if total > 0 and np.isfinite(total):
        cum_ratio = np.cumsum(eig_ridge) / total
        r = int(np.searchsorted(cum_ratio, PC_TAU) + 1)
        r = max(1, min(r, q))
        explained = cum_ratio[r - 1]
        discarded_share = 1.0 - explained
    else:
        r = np.nan
        explained = np.nan
        discarded_share = np.nan

    out = {
        "sample": sample_name,
        "method": method,
        "q_moments": q,
        "raw_min_eig": raw_min,
        "raw_max_eig": raw_max,
        "raw_condition_number": raw_cond,
        "ridge_min_eig": ridge_min,
        "ridge_max_eig": ridge_max,
        "ridge_condition_number": ridge_cond,
        "selected_rank_tau90": r,
        "explained_ratio_selected": explained,
        "discarded_eig_share": discarded_share,
        "n_small_eigs_1e-6": int(np.sum(eig_raw < 1e-6)),
        "n_small_eigs_1e-4": int(np.sum(eig_raw < 1e-4)),
    }

    return out

def save_pc_weight_diagnostics(df_params_full, df_params_train, x_full, x_train):
    """
    Save PC/covariance diagnostics for full sample and train sample.
    """
    rows = []

    theta_full = make_theta_dict(df_params_full)
    theta_train = make_theta_dict(df_params_train)

    for method in METHODS:
        rows.append(
            pc_weight_diagnostics_single(
                x_full,
                theta_full[method],
                method,
                sample_name="full"
            )
        )

    for method in METHODS:
        rows.append(
            pc_weight_diagnostics_single(
                x_train,
                theta_train[method],
                method,
                sample_name="train"
            )
        )

    df_diag = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_pc_weight_diagnostics.csv")
    df_diag.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    return df_diag

def fig_pc_eigen_spectrum(x, theta, method="PC-CF-RGMM", sample_name="full"):
    """
    Plot eigenvalue spectrum of moment covariance matrix.
    """
    G = combined_moments(x, theta, method)
    if not np.all(np.isfinite(G)):
        return

    Gc = G - np.mean(G, axis=0)
    S = (Gc.T @ Gc) / Gc.shape[0]
    S = S + PC_RIDGE * np.eye(S.shape[0])

    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]

    total = np.sum(eigvals)
    ratio = np.cumsum(eigvals) / total
    r = int(np.searchsorted(ratio, PC_TAU) + 1)

    plt.figure(figsize=(10, 5.5))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, marker="o", lw=1.5)
    plt.axvline(r, color="red", linestyle="--", lw=1.5, label=f"PC rank = {r}")
    plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue, log scale")
    plt.title(f"Moment covariance eigenvalue spectrum: {method}, {sample_name}")
    plt.legend()
    savefig("fig_14_pc_eigen_spectrum.png")

def theoretical_standardized_cf(freqs, theta):
    """
    Model-implied CF of standardized residual z.
    This follows the same normalization as cf_moments().
    """
    kappa, mu, sigma, alpha, lam = theta

    vh = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * H)) / (2.0 * kappa)
    if vh <= 0 or not np.isfinite(vh):
        return np.full(len(freqs), np.nan + 1j * np.nan)

    sd_h = np.sqrt(vh)

    phi_vals = ou_innovation_cf(
        np.asarray(freqs, dtype=float) / sd_h,
        theta,
        h=H
    )

    return phi_vals

def cf_fit_metrics_single(x, theta, method, sample_name="full", eval_freqs=None):
    """
    Compare empirical CF of standardized residuals with model-implied CF.
    """
    if eval_freqs is None:
        # More dense than estimation frequencies
        eval_freqs = np.array(
            [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.75,
             0.90, 1.00, 1.20, 1.35, 1.50, 1.75, 2.00, 2.25,
             2.50, 2.75, 3.00],
            dtype=float
        )

    delta, z, x_lag = residuals_and_z(x, theta, H)
    z = z[np.isfinite(z)]

    if len(z) == 0:
        return {
            "sample": sample_name,
            "method": method,
            "n_resid": 0,
            "cf_mse": np.nan,
            "cf_mae": np.nan,
            "cf_max_abs_error": np.nan,
            "cf_low_freq_mse": np.nan,
            "cf_high_freq_mse": np.nan,
        }

    emp_cf = np.array([np.mean(np.exp(1j * u * z)) for u in eval_freqs])
    model_cf = theoretical_standardized_cf(eval_freqs, theta)

    abs_err = np.abs(emp_cf - model_cf)
    sq_err = abs_err ** 2

    low_mask = eval_freqs <= 1.0
    high_mask = eval_freqs > 1.0

    out = {
        "sample": sample_name,
        "method": method,
        "n_resid": len(z),
        "cf_mse": float(np.mean(sq_err)),
        "cf_mae": float(np.mean(abs_err)),
        "cf_max_abs_error": float(np.max(abs_err)),
        "cf_low_freq_mse": float(np.mean(sq_err[low_mask])),
        "cf_high_freq_mse": float(np.mean(sq_err[high_mask])),
    }

    return out

def save_cf_fit_diagnostics(df_params_full, df_params_train, x_full, x_train, x_test):
    """
    Save CF fit metrics for full, train, and test samples.
    For test sample, include x_train[-1] as lagged initial value.
    """
    rows = []

    theta_full = make_theta_dict(df_params_full)
    theta_train = make_theta_dict(df_params_train)

    # Full-sample CF fit uses full-sample estimates
    for method in METHODS:
        rows.append(
            cf_fit_metrics_single(
                x_full,
                theta_full[method],
                method,
                sample_name="full"
            )
        )

    # Train-sample CF fit uses train estimates
    for method in METHODS:
        rows.append(
            cf_fit_metrics_single(
                x_train,
                theta_train[method],
                method,
                sample_name="train"
            )
        )

    # Test-sample CF fit uses train estimates
    x_test_with_lag = np.concatenate([[x_train[-1]], x_test])
    for method in METHODS:
        rows.append(
            cf_fit_metrics_single(
                x_test_with_lag,
                theta_train[method],
                method,
                sample_name="test"
            )
        )

    df_cf = pd.DataFrame(rows)

    path = os.path.join(TAB_DIR, "table_cf_fit_diagnostics.csv")
    df_cf.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    return df_cf

def fig_cf_mse_comparison(df_cf):
    """
    Bar plot of CF MSE by sample and method.
    """
    if df_cf is None or len(df_cf) == 0:
        return

    samples = df_cf["sample"].unique().tolist()

    plt.figure(figsize=(11, 5.5))

    width = 0.25
    x_pos = np.arange(len(samples))

    for i, method in enumerate(METHODS):
        vals = []
        for s in samples:
            sub = df_cf[(df_cf["sample"] == s) & (df_cf["method"] == method)]
            vals.append(sub["cf_mse"].iloc[0] if len(sub) else np.nan)

        plt.bar(
            x_pos + (i - 1) * width,
            vals,
            width=width,
            label=method
        )

    plt.xticks(x_pos, samples)
    plt.ylabel("CF MSE")
    plt.title("Empirical characteristic function fitting error")
    plt.legend()
    savefig("fig_13_cf_mse_comparison.png")

def residual_diagnostics_single(x, theta, method, sample_name="full"):
    delta, z, x_lag = residuals_and_z(x, theta, H)
    z = z[np.isfinite(z)]

    if len(z) == 0:
        return {
            "sample": sample_name,
            "method": method,
            "n_resid": 0,
            "z_mean": np.nan,
            "z_std": np.nan,
            "z_skew": np.nan,
            "z_kurtosis": np.nan,
            "z_q01": np.nan,
            "z_q05": np.nan,
            "z_q95": np.nan,
            "z_q99": np.nan,
            "tail_rate_abs_gt_2": np.nan,
            "tail_rate_abs_gt_3": np.nan,
            "max_abs_z": np.nan,
        }

    return {
        "sample": sample_name,
        "method": method,
        "n_resid": len(z),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1)),
        "z_skew": float(stats.skew(z)),
        "z_kurtosis": float(stats.kurtosis(z, fisher=False)),
        "z_q01": float(np.quantile(z, 0.01)),
        "z_q05": float(np.quantile(z, 0.05)),
        "z_q95": float(np.quantile(z, 0.95)),
        "z_q99": float(np.quantile(z, 0.99)),
        "tail_rate_abs_gt_2": float(np.mean(np.abs(z) > 2.0)),
        "tail_rate_abs_gt_3": float(np.mean(np.abs(z) > 3.0)),
        "max_abs_z": float(np.max(np.abs(z))),
    }

def save_residual_diagnostics(df_params_full, df_params_train, x_full, x_train, x_test):
    rows = []

    theta_full = make_theta_dict(df_params_full)
    theta_train = make_theta_dict(df_params_train)

    for method in METHODS:
        rows.append(
            residual_diagnostics_single(
                x_full,
                theta_full[method],
                method,
                sample_name="full"
            )
        )

    for method in METHODS:
        rows.append(
            residual_diagnostics_single(
                x_train,
                theta_train[method],
                method,
                sample_name="train"
            )
        )

    x_test_with_lag = np.concatenate([[x_train[-1]], x_test])
    for method in METHODS:
        rows.append(
            residual_diagnostics_single(
                x_test_with_lag,
                theta_train[method],
                method,
                sample_name="test"
            )
        )

    df_resid = pd.DataFrame(rows)
    path = os.path.join(TAB_DIR, "table_residual_diagnostics.csv")
    df_resid.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    return df_resid

def save_extreme_forecast_errors(df_forecast_details):
    """
    Decompose OOS forecast errors into normal and extreme movement periods.

    Extreme dates are defined using actual_X daily changes in the OOS period.
    """
    df = df_forecast_details.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Use one method to define actual movement because actual_X is the same
    base = df[df["method"] == METHODS[0]].copy()
    base = base.sort_values("date")
    base["dX"] = base["actual_X"].diff()
    base["abs_dX"] = np.abs(base["dX"])

    q90 = base["abs_dX"].quantile(0.90)
    q95 = base["abs_dX"].quantile(0.95)

    base["group_all"] = True
    base["group_extreme10"] = base["abs_dX"] >= q90
    base["group_extreme5"] = base["abs_dX"] >= q95
    base["group_spike_up"] = base["dX"] >= q90
    base["group_spike_down"] = base["dX"] <= -q90

    group_cols = [
        "group_all",
        "group_extreme10",
        "group_extreme5",
        "group_spike_up",
        "group_spike_down",
    ]

    group_map = base[["date"] + group_cols].copy()
    df = df.merge(group_map, on="date", how="left")

    rows = []

    for method in METHODS:
        sub_method = df[df["method"] == method].copy()

        for gcol in group_cols:
            sub = sub_method[sub_method[gcol] == True]

            if len(sub) == 0:
                row = {
                    "method": method,
                    "group": gcol.replace("group_", ""),
                    "n": 0,
                    "MAE_X": np.nan,
                    "RMSE_X": np.nan,
                    "MeanError_X": np.nan,
                    "MAE_VIX": np.nan,
                    "RMSE_VIX": np.nan,
                    "MeanError_VIX": np.nan,
                }
            else:
                err_x = sub["error_X_actual_minus_pred"].values
                err_vix = sub["error_VIX_actual_minus_pred"].values \
                    if "error_VIX_actual_minus_pred" in sub.columns else np.full(len(sub), np.nan)

                row = {
                    "method": method,
                    "group": gcol.replace("group_", ""),
                    "n": len(sub),
                    "MAE_X": float(np.mean(np.abs(err_x))),
                    "RMSE_X": float(np.sqrt(np.mean(err_x ** 2))),
                    "MeanError_X": float(np.mean(err_x)),
                    "MAE_VIX": float(np.mean(np.abs(err_vix))),
                    "RMSE_VIX": float(np.sqrt(np.mean(err_vix ** 2))),
                    "MeanError_VIX": float(np.mean(err_vix)),
                }

            rows.append(row)

    df_extreme = pd.DataFrame(rows)

    path = os.path.join(TAB_DIR, "table_extreme_forecast_errors.csv")
    df_extreme.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    return df_extreme

def fig_extreme_forecast_rmse(df_extreme):
    """
    Plot RMSE_X for all/extreme groups.
    """
    if df_extreme is None or len(df_extreme) == 0:
        return

    groups = ["all", "extreme10", "extreme5", "spike_up", "spike_down"]
    groups = [g for g in groups if g in df_extreme["group"].unique()]

    plt.figure(figsize=(12, 5.5))

    width = 0.25
    x_pos = np.arange(len(groups))

    for i, method in enumerate(METHODS):
        vals = []
        for g in groups:
            sub = df_extreme[
                (df_extreme["method"] == method) &
                (df_extreme["group"] == g)
            ]
            vals.append(sub["RMSE_X"].iloc[0] if len(sub) else np.nan)

        plt.bar(
            x_pos + (i - 1) * width,
            vals,
            width=width,
            label=method
        )

    plt.xticks(x_pos, groups, rotation=20)
    plt.ylabel("RMSE of X")
    plt.title("Out-of-sample forecast RMSE during extreme movement periods")
    plt.legend()
    savefig("fig_15_extreme_forecast_rmse.png")

def save_rolling_jump_stability(rolling_df):
    """
    Additional rolling stability: adjacent-window parameter jumps.
    """
    if rolling_df is None or len(rolling_df) == 0:
        return None

    rows = []

    for method in METHODS:
        sub = rolling_df[
            (rolling_df["method"] == method) &
            (rolling_df["success"] == True)
        ].copy()

        sub = sub.sort_values("date_end")

        row = {
            "method": method,
            "n_windows": len(sub)
        }

        for p in PARAM_NAMES:
            vals = sub[p].values.astype(float)
            vals = vals[np.isfinite(vals)]

            if len(vals) <= 1:
                row[f"{p}_mean_abs_jump"] = np.nan
                row[f"{p}_rmse_jump"] = np.nan
                row[f"{p}_max_abs_jump"] = np.nan
            else:
                diff = np.diff(vals)
                row[f"{p}_mean_abs_jump"] = float(np.mean(np.abs(diff)))
                row[f"{p}_rmse_jump"] = float(np.sqrt(np.mean(diff ** 2)))
                row[f"{p}_max_abs_jump"] = float(np.max(np.abs(diff)))

        rows.append(row)

    df_jump = pd.DataFrame(rows)

    path = os.path.join(TAB_DIR, "table_rolling_jump_stability.csv")
    df_jump.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table saved] {path}")

    return df_jump

def fig_rolling_std_comparison(rolling_stability_df):
    """
    Bar plot for rolling parameter standard deviations.
    """
    if rolling_stability_df is None or len(rolling_stability_df) == 0:
        return

    params = PARAM_NAMES
    x_pos = np.arange(len(params))
    width = 0.25

    plt.figure(figsize=(12, 5.5))

    for i, method in enumerate(METHODS):
        sub = rolling_stability_df[rolling_stability_df["method"] == method]
        if len(sub) == 0:
            continue

        vals = [sub[f"{p}_std"].iloc[0] for p in params]

        plt.bar(
            x_pos + (i - 1) * width,
            vals,
            width=width,
            label=method
        )

    plt.xticks(x_pos, params)
    plt.ylabel("Rolling-window standard deviation")
    plt.title("Rolling parameter stability comparison")
    plt.legend()
    savefig("fig_12_rolling_std_comparison.png")

# ============================================================
# 9. Main pipeline
# ============================================================

def main():
    sanity_check_cts()
    append_timing_log_external('SCRIPT_START')

    print("=" * 80)
    print("CTS Levy driven OU model: VIX empirical figure/table generator | GENERATED STAGE 3")
    print("=" * 80)

    # 1. Load data
    df, x_label = load_vix_data(DATA_PATH, USE_LOG)
    print("[Stage 3] VIX full sample observations used:", len(df))
    x = df["X"].values

    # 2. Descriptive statistics and preliminary figures
    descriptive_statistics(df, x_label)
    fig_vix_series(df, x_label)
    fig_distribution_diagnostics(df, x_label)

    # 3. Full-sample estimation
    print("=" * 80)
    print("[Full-sample estimation]")
    print("=" * 80)
    results, df_params = estimate_all_methods(x)
    save_parameter_table(df_params)

    # 4. In-sample fitted values
    fig_fitted_values(df, results, x_label)

    # 5. Out-of-sample one-step forecast evaluation
    n = len(x)
    n_test = int(np.floor(TEST_RATIO * n))
    n_train = n - n_test

    x_train = x[:n_train]
    x_test = x[n_train:]

    date_test = df["date"].values[n_train:]

    print("=" * 80)
    print(f"[Out-of-sample forecasting] train={len(x_train)}, test={len(x_test)}")
    print("=" * 80)

    forecast_rows = []
    forecast_details = {}
    forecast_train_param_rows = []

    for method in METHODS:
        print(f"[Forecast estimation] {method}")
        res_train = estimate_method(x_train, method)
        theta_train = res_train["theta"]

        pred, actual, err, metrics = forecast_evaluation(x_train, x_test, theta_train)

        row = {
            "method": method,
            "success": res_train["success"],
            "train_objective": res_train["objective"],
            **metrics
        }
        forecast_rows.append(row)

        # 保存用于 OOS forecast 的训练集参数
        param_row = {
            "method": method,
            "success": res_train["success"],
            "train_objective": res_train["objective"],
            "pc_rank": res_train["pc_rank"],
            "message": res_train["message"],
        }

        for i, p in enumerate(PARAM_NAMES):
            param_row[p] = theta_train[i]

        kappa, mu, sigma, alpha, lam = theta_train
        param_row["phi"] = np.exp(-kappa * H)

        forecast_train_param_rows.append(param_row)

        forecast_details[method] = {
            "date": date_test,
            "pred": pred,
            "actual": actual,
            "error": err
        }

    df_forecast = save_forecast_table(forecast_rows)
    df_forecast_train_params = save_forecast_train_parameter_table(forecast_train_param_rows)
    df_forecast_details, df_forecast_details_summary = save_forecast_details_table(
        forecast_details,
        use_log=USE_LOG
    )

    fig_forecast_errors(df, df_forecast, forecast_details, x_label)

    # 5.5 Additional diagnostics: PC, CF, residuals, extreme forecast errors
    print("=" * 80)
    print("[Additional diagnostics for PC-CF-RGMM advantages]")
    print("=" * 80)

    df_pc_diag = save_pc_weight_diagnostics(
        df_params_full=df_params,
        df_params_train=df_forecast_train_params,
        x_full=x,
        x_train=x_train
    )

    df_cf_diag = save_cf_fit_diagnostics(
        df_params_full=df_params,
        df_params_train=df_forecast_train_params,
        x_full=x,
        x_train=x_train,
        x_test=x_test
    )

    df_resid_diag = save_residual_diagnostics(
        df_params_full=df_params,
        df_params_train=df_forecast_train_params,
        x_full=x,
        x_train=x_train,
        x_test=x_test
    )

    df_extreme = save_extreme_forecast_errors(df_forecast_details)

    fig_cf_mse_comparison(df_cf_diag)
    fig_extreme_forecast_rmse(df_extreme)

    # PC eigen spectrum based on full-sample PC-CF-RGMM estimate
    try:
        theta_full_dict = make_theta_dict(df_params)
        fig_pc_eigen_spectrum(
            x,
            theta_full_dict["PC-CF-RGMM"],
            method="PC-CF-RGMM",
            sample_name="full"
        )
    except Exception as err:
        print(f"[Warning] PC eigen spectrum failed: {err}")

    # 6. Rolling-window estimation
    if RUN_ROLLING:
        rolling_df = run_rolling_estimation(df, x_label)

        if rolling_df is not None:
            # 新增：计算滚动窗口相邻跳跃稳定性
            df_jump = save_rolling_jump_stability(rolling_df)

            # 因为 run_rolling_estimation() 内部已经保存了 table_rolling_stability.csv，
            # 所以这里只需要把它读回来，用来画滚动标准差对比图
            rolling_stab_path = os.path.join(TAB_DIR, "table_rolling_stability.csv")

            if os.path.exists(rolling_stab_path):
                rolling_stab = pd.read_csv(rolling_stab_path)
                fig_rolling_std_comparison(rolling_stab)

        # 原来的滚动参数路径图
        fig_rolling_estimates(rolling_df)

    else:
        print("[Rolling estimation skipped] RUN_ROLLING=False")

    print("=" * 80)
    print("[Done]")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"Tables saved in: {TAB_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()