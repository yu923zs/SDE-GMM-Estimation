# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:13:28 2026

@author: 86183
"""

# -*- coding: utf-8 -*-
"""
Monte Carlo simulation and figure/table generation for

Classical Tempered Stable Levy driven Ornstein--Uhlenbeck model

Methods:
    1. GMM
    2. Huber-GMM
    3. PC-CF-RGMM

Features:
    - checkpoint / resume
    - automatic table generation
    - automatic figure generation
    - robust optimization with multiple initial values
    - characteristic-function based moment estimation

Author: your name
Date: 2026-05-13
"""

import os
import sys
import time
import pickle
import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gamma
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ============================================================
# 0. Global configuration
# ============================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RESULT_DIR = os.path.join(BASE_DIR, "results_stage3_final")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_stage3_final")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")

for d in [RESULT_DIR, TABLE_DIR, FIGURE_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. User settings
# ============================================================

@dataclass
class SimulationConfig:
    # ---------- run size ----------
    n_obs: int = 1000
    n_rep: int = 100              # stage3 tuning; final use 200
    h: float = 1.0
    seed: int = 20260513

    # ---------- optimization ----------
    maxiter: int = 800
    ftol: float = 1e-9

    # ---------- robustification ----------
    huber_c: float = 2.5

    # Standardized-CF frequencies.
    # These are frequencies for z = eta_h / sqrt(v_h), not raw eta_h.
    cf_freqs: tuple = (0.15, 0.30, 0.50, 0.75, 1.00, 1.35, 1.75, 2.25, 3.00)

    # CF weight. Since all methods now include CF moments,
    # use 1.0 first; if CF dominates too much, test 0.7.
    rho_cf: float = 1.0

    # PC regularization
    pc_tau: float = 0.90
    pc_ridge: float = 1e-4
    pc_eig_floor: float = 1e-3
    pc_inv_cap: float = 200.0
    pc_weight_shrinkage: float = 0.30

    n_starts: int = 12

    checkpoint_every: int = 1
    fresh_start: bool = True

    # Soft boundary penalty
    use_boundary_penalty: bool = True
    boundary_penalty_strength: float = 1e2
    boundary_margin_ratio: float = 0.03
    boundary_tol_ratio: float = 0.01
    alpha_pole_margin: float = 0.04
    alpha_pole_penalty_strength: float = 1e3

    # Sampling grid for tuning
    innovation_grid_nx: int = 1501
    innovation_grid_nu: int = 1000
    cf_quad_nodes: int = 40
    innovation_xmax_multiplier: float = 10.0
    only_postprocess: bool = True

CFG = SimulationConfig()

# ============================================================
# 2. Parameter scenarios and contamination scenarios
# ============================================================

PARAM_SCENARIOS = {
   "S1_baseline": {
        "kappa": 0.05,
        "mu": 20.0,
        "sigma": 1.0,
        "alpha": 0.75,
        "lambda": 2.0,
        "description": "baseline"
    },
    "S2_heavy_tail": {
        "kappa": 0.05,
        "mu": 20.0,
        "sigma": 1.0,
        "alpha": 0.60,
        "lambda": 0.80,
        "description": "heavy tail"
    },
    "S3_light_tail": {
        "kappa": 0.05,
        "mu": 20.0,
        "sigma": 1.0,
        "alpha": 1.30,
        "lambda": 4.0,
        "description": "light tail"
    },
    "S4_fast_reversion": {
        "kappa": 0.15,
        "mu": 20.0,
        "sigma": 1.0,
        "alpha": 0.75,
        "lambda": 2.0,
        "description": "fast mean reversion"
    },
}

CONTAMINATION_SCENARIOS = {
    "C0_clean": {
        "type": "none",
        "epsilon": 0.00,
        "b": 0.0,
        "block_len": 0
    },
    "C1_random_1pct": {
        "type": "random",
        "epsilon": 0.01,
        "b": 8.0,
        "block_len": 0
    },
    "C2_random_3pct": {
        "type": "random",
        "epsilon": 0.03,
        "b": 8.0,
        "block_len": 0
    },
    "C3_block": {
        "type": "block",
        "epsilon": 0.03,
        "b": 8.0,
        "block_len": 30
    },
    "C4_tail": {
        "type": "tail",
        "epsilon": 0.03,
        "b": 8.0,
        "block_len": 0
    },
}

METHODS = ["GMM", "Huber-GMM", "PC-CF-RGMM"]

PARAM_NAMES = ["kappa", "mu", "sigma", "alpha", "lambda"]

# ============================================================
# 3. Utility functions
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)

def theta_to_array(theta_dict):
    return np.array([
        theta_dict["kappa"],
        theta_dict["mu"],
        theta_dict["sigma"],
        theta_dict["alpha"],
        theta_dict["lambda"]
    ], dtype=float)

def array_to_theta(theta):
    return {
        "kappa": float(theta[0]),
        "mu": float(theta[1]),
        "sigma": float(theta[2]),
        "alpha": float(theta[3]),
        "lambda": float(theta[4])
    }

def ensure_finite(x, large=1e12):
    if not np.all(np.isfinite(x)):
        return large
    return x

def safe_norm(x):
    val = np.linalg.norm(x)
    if not np.isfinite(val):
        return 1e12
    return val

def save_pickle(obj, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def current_time_string():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# 4. Classical tempered stable characteristic exponent
# ============================================================

def cts_C_standardized(alpha, lam):
    """
    Standardization constant C such that Var(L_1)=1:
        Var(L_1) = 2 C Gamma(2-alpha) lambda^{alpha-2}
    """
    return 1.0 / (2.0 * gamma(2.0 - alpha) * lam ** (alpha - 2.0))

def cts_exponent(u, alpha, lam):
    """
    Symmetric Classical Tempered Stable characteristic exponent.

    Psi_L(u) =
        + C Gamma(-alpha) [ (lambda - i u)^alpha
                            + (lambda + i u)^alpha
                            - 2 lambda^alpha ]

    Note:
        alpha should not be exactly 1 due to Gamma(-alpha) pole.
        In numerical optimization we avoid alpha too close to 1.
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
    Characteristic function of OU innovation:
        eta_h = sigma int_0^h exp(-kappa r) dL_r

    phi_h(u;theta) =
        exp{ int_0^h Psi_L(u sigma exp(-kappa r)) dr }

    This version supports both scalar u and vector u.
    It avoids the broadcasting problem of scipy.integrate.fixed_quad.
    """
    kappa, mu, sigma, alpha, lam = theta

    # Convert u to at least 1D array
    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    scalar_input = np.isscalar(u) or np.ndim(u) == 0

    # Gauss-Legendre nodes and weights on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(quad_nodes)

    # Transform nodes from [-1, 1] to [0, h]
    r_nodes = 0.5 * h * (nodes + 1.0)
    r_weights = 0.5 * h * weights

    # Shape:
    # u_arr[:, None] has shape (m, 1)
    # r_nodes[None, :] has shape (1, quad_nodes)
    # argument has shape (m, quad_nodes)
    arg = (
        u_arr[:, None]
        * sigma
        * np.exp(-kappa * r_nodes[None, :])
    )

    psi_vals = cts_exponent(arg, alpha, lam)

    # Integrate over r dimension
    integral_vals = np.sum(
        psi_vals * r_weights[None, :],
        axis=1
    )

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
# 5. Innovation sampler through numerical CDF inversion
# ============================================================

class InnovationSampler:
    """
    Numerical sampler for OU innovation distribution using
    Gil-Pelaez inversion:

        F(x) = 1/2 - 1/pi int_0^U Im( exp(-i u x) phi(u) / u ) du

    This is slower than exact samplers but very general.
    The CDF grid is precomputed once for each parameter scenario.
    """

    def __init__(self, theta, h=1.0, nx=4001, nu=2500,
                 xmax_multiplier=10.0, quad_nodes=40):
        self.theta = np.asarray(theta, dtype=float)
        self.h = h
        self.nx = nx
        self.nu = nu
        self.xmax_multiplier = xmax_multiplier
        self.quad_nodes = quad_nodes

        self.x_grid = None
        self.cdf_grid = None
        self.inv_cdf = None

        self._build()

    def _innovation_sd(self):
        kappa, mu, sigma, alpha, lam = self.theta
        return sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * self.h)) / (2.0 * kappa))

    def _build(self):
        sd = self._innovation_sd()
        xmax = self.xmax_multiplier * sd

        x_grid = np.linspace(-xmax, xmax, self.nx)
        u_grid = np.linspace(1e-6, 80.0 / max(sd, 1e-8), self.nu)

        phi_u = ou_innovation_cf(
            u_grid, self.theta, h=self.h, quad_nodes=self.quad_nodes
        )

        # Matrix shape: nx x nu
        # For memory control: do in blocks
        cdf = np.empty_like(x_grid)
        block = 200

        for start in range(0, self.nx, block):
            end = min(start + block, self.nx)
            x_block = x_grid[start:end]
            expo = np.exp(-1j * np.outer(x_block, u_grid))
            integrand = np.imag(expo * phi_u / u_grid)
            integral = np.trapz(integrand, u_grid, axis=1)
            cdf[start:end] = 0.5 - integral / np.pi

        # Numerical cleanup
        cdf = np.maximum.accumulate(cdf)
        cdf = np.clip(cdf, 0.0, 1.0)

        # Ensure full support endpoints
        cdf[0] = 0.0
        cdf[-1] = 1.0

        # Remove duplicate CDF values for interpolation
        unique_cdf, unique_idx = np.unique(cdf, return_index=True)
        unique_x = x_grid[unique_idx]

        if unique_cdf[0] > 0.0:
            unique_cdf = np.insert(unique_cdf, 0, 0.0)
            unique_x = np.insert(unique_x, 0, x_grid[0])
        if unique_cdf[-1] < 1.0:
            unique_cdf = np.append(unique_cdf, 1.0)
            unique_x = np.append(unique_x, x_grid[-1])

        self.x_grid = x_grid
        self.cdf_grid = cdf
        self.inv_cdf = interp1d(
            unique_cdf,
            unique_x,
            kind="linear",
            bounds_error=False,
            fill_value=(x_grid[0], x_grid[-1])
        )

    def sample(self, n):
        u = np.random.uniform(size=n)
        return self.inv_cdf(u)

# ============================================================
# 6. Simulate OU path and contaminate data
# ============================================================

def simulate_ou_path(theta, n_obs, h, sampler):
    """
    Discrete OU recursion:
        X_j = mu + exp(-kappa h)(X_{j-1}-mu) + eta_j
    """
    kappa, mu, sigma, alpha, lam = theta
    phi = np.exp(-kappa * h)

    x = np.empty(n_obs + 1)
    x[0] = mu

    innovations = sampler.sample(n_obs)

    for j in range(1, n_obs + 1):
        x[j] = mu + phi * (x[j - 1] - mu) + innovations[j - 1]

    return x

def contaminate_path(x, contamination):
    """
    Contaminate increments and reconstruct path.
    """
    ctype = contamination["type"]
    eps = contamination["epsilon"]
    b = contamination["b"]
    block_len = contamination.get("block_len", 0)

    if ctype == "none" or eps <= 0:
        return x.copy()

    x_new = x.copy()
    dx = np.diff(x_new)
    n = len(dx)
    sd_dx = np.std(dx)

    if sd_dx <= 0 or not np.isfinite(sd_dx):
        sd_dx = 1.0

    m = max(1, int(np.floor(eps * n)))

    if ctype == "random":
        idx = np.random.choice(n, size=m, replace=False)
        signs = np.random.choice([-1.0, 1.0], size=m)
        dx[idx] += b * sd_dx * signs

    elif ctype == "block":
        L = block_len if block_len > 0 else m
        L = min(L, n)
        start = np.random.randint(0, n - L + 1)
        sign = np.random.choice([-1.0, 1.0])
        dx[start:start + L] += sign * b * sd_dx

    elif ctype == "tail":
        idx = np.argsort(np.abs(dx))[-m:]
        dx[idx] += b * sd_dx * np.sign(dx[idx])

    else:
        raise ValueError(f"Unknown contamination type: {ctype}")

    x_new[1:] = x_new[0] + np.cumsum(dx)
    return x_new

# ============================================================
# 7. Moment functions
# ============================================================

def residuals_and_z(x, theta, h):
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

def dynamic_moments(x, theta, h):
    """
    Low-order dynamic moments with scaled instruments.

    The old version used (x_lag - mu) directly, which can dominate
    the moment scale. We normalize the state instrument by an
    approximate stationary scale.
    """
    kappa, mu, sigma, alpha, lam = theta
    delta, z, x_lag = residuals_and_z(x, theta, h)

    # Approximate stationary scale of OU component.
    # Since L has unit variance and innovation is scaled by sigma,
    # Var(X) approximately sigma^2 / (2 kappa).
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

def radial_huber(y, c):
    """
    y: n x q
    radial Huber transform row-wise.
    """
    norms = np.linalg.norm(y, axis=1)
    scale = np.ones_like(norms)
    mask = norms > c
    scale[mask] = c / norms[mask]
    return y * scale[:, None]

def robust_dynamic_moments(x, theta, h, c):
    """
    Robustified dynamic moments.

    Important change:
    We do NOT subtract the sample median from the moments.
    Subtracting a theta-dependent sample median changes the population
    moment condition and can distort the GMM objective.

    We only scale columns and apply radial Huber shrinkage.
    """
    d = dynamic_moments(x, theta, h)
    if not np.all(np.isfinite(d)):
        return d

    med = np.nanmedian(d, axis=0)
    mad = np.nanmedian(np.abs(d - med), axis=0)
    scale = 1.4826 * mad
    scale[scale < 1e-6] = 1.0

    y = d / scale
    y_h = radial_huber(y, c)

    return y_h

def cf_moments(x, theta, h, freqs, quad_nodes):
    """
    Standardized characteristic-function moments.

    Instead of using raw innovation residual delta, we use

        z = delta / sqrt(v_h),

    where

        v_h = sigma^2 (1 - exp(-2 kappa h)) / (2 kappa).

    Then the theoretical CF is

        phi_z(u; theta) = phi_eta(u / sqrt(v_h); theta).

    This stabilizes the frequency scale and improves identification of
    alpha and lambda.
    """
    kappa, mu, sigma, alpha, lam = theta
    delta, z, x_lag = residuals_and_z(x, theta, h)
    freqs = np.asarray(freqs, dtype=float)

    vh = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * h)) / (2.0 * kappa)
    if vh <= 0 or not np.isfinite(vh):
        return np.full((len(delta), 2 * len(freqs)), np.nan)

    sd_h = np.sqrt(vh)

    # CF of standardized residual z:
    # E exp(i u z) = phi_eta(u / sd_h)
    phi_vals = ou_innovation_cf(
        freqs / sd_h,
        theta,
        h=h,
        quad_nodes=quad_nodes
    )

    cols = []
    for k, u in enumerate(freqs):
        empirical_real = np.cos(u * z)
        empirical_imag = np.sin(u * z)

        theo_real = np.real(phi_vals[k])
        theo_imag = np.imag(phi_vals[k])

        cols.append(empirical_real - theo_real)
        cols.append(empirical_imag - theo_imag)

    return np.column_stack(cols)

def combined_moments(x, theta, method, cfg):
    """
    Return n x q moment matrix according to method.

    Stage3 design:
        GMM        = dynamic moments + standardized CF moments
        Huber-GMM  = robust dynamic moments + standardized CF moments
        PC-CF-RGMM = robust dynamic moments + standardized CF moments
                     with PC-regularized weight matrix in stage 2

    This makes alpha and lambda identifiable for all three methods.
    """
    c = cf_moments(
        x, theta, cfg.h,
        cfg.cf_freqs,
        cfg.cf_quad_nodes
    )

    if method == "GMM":
        d = dynamic_moments(x, theta, cfg.h)
        return np.column_stack([d, cfg.rho_cf * c])

    elif method == "Huber-GMM":
        d = robust_dynamic_moments(x, theta, cfg.h, cfg.huber_c)
        return np.column_stack([d, cfg.rho_cf * c])

    elif method == "PC-CF-RGMM":
        d = robust_dynamic_moments(x, theta, cfg.h, cfg.huber_c)
        return np.column_stack([d, cfg.rho_cf * c])

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# 8. Objective functions and estimation
# ============================================================

def parameter_bounds():
    """
    Parameter bounds:
        kappa  in [0.005, 0.50]
        mu     in [1, 80]
        sigma  in [0.10, 5.00]
        alpha  in [0.20, 1.80]
        lambda in [0.20, 10.00]
    """
    return [
        (0.005, 0.50),   # kappa
        (1.0, 80.0),     # mu
        (0.10, 5.00),    # sigma
        (0.20, 1.80),    # alpha
        (0.20, 10.00),    # lambda
    ]

def boundary_soft_penalty(theta, cfg):
    """
    Soft penalty for estimates close to parameter boundaries.

    The penalty is zero when theta is away from boundaries.
    It increases smoothly when theta enters the boundary margin region.

    This prevents the optimizer from choosing artificial boundary solutions.
    """
    if not cfg.use_boundary_penalty:
        return 0.0

    theta = np.asarray(theta, dtype=float)
    bounds = parameter_bounds()

    penalty = 0.0

    for val, (lo, hi) in zip(theta, bounds):
        width = hi - lo
        margin = cfg.boundary_margin_ratio * width

        # Distance to lower and upper boundary
        dist_low = val - lo
        dist_high = hi - val

        # Penalize lower boundary region
        if dist_low < margin:
            ratio = (margin - dist_low) / margin
            penalty += ratio ** 4

        # Penalize upper boundary region
        if dist_high < margin:
            ratio = (margin - dist_high) / margin
            penalty += ratio ** 4

    return cfg.boundary_penalty_strength * penalty

def alpha_pole_soft_penalty(theta, cfg):
    """
    Additional soft penalty for alpha close to 1.

    The CTS exponent contains Gamma(-alpha), which is singular at alpha=1.
    Instead of a hard discontinuous rejection only, this smooth penalty helps
    the optimizer avoid the singular region.
    """
    alpha = float(theta[3])
    dist = abs(alpha - 1.0)

    if dist >= cfg.alpha_pole_margin:
        return 0.0

    ratio = (cfg.alpha_pole_margin - dist) / cfg.alpha_pole_margin
    return cfg.alpha_pole_penalty_strength * ratio ** 4

def is_near_boundary(theta, cfg):
    """
    Check whether final estimate is too close to any parameter boundary.
    Used to mark artificial boundary solutions.
    """
    theta = np.asarray(theta, dtype=float)
    bounds = parameter_bounds()

    for val, (lo, hi) in zip(theta, bounds):
        width = hi - lo
        tol = cfg.boundary_tol_ratio * width

        if val - lo < tol:
            return True
        if hi - val < tol:
            return True

    return False

def theta_valid(theta):
    kappa, mu, sigma, alpha, lam = theta

    if not np.all(np.isfinite(theta)):
        return False

    if kappa <= 0 or sigma <= 0 or lam <= 0:
        return False

    if not (0.0 < alpha < 2.0):
        return False

    # Hard exclusion very close to alpha = 1
    # because Gamma(-alpha) has a pole at alpha = 1.
    if abs(alpha - 1.0) < 0.015:
        return False

    # Check hard bounds
    bounds = parameter_bounds()
    for val, (lo, hi) in zip(theta, bounds):
        if val < lo or val > hi:
            return False

    return True

def moment_objective(theta, x, method, cfg, W=None):
    """
    GMM objective with soft boundary penalty.

    Objective =
        moment distance
        + soft boundary penalty
        + alpha pole penalty
    """
    theta = np.asarray(theta, dtype=float)

    if not theta_valid(theta):
        return 1e12

    try:
        G = combined_moments(x, theta, method, cfg)

        if not np.all(np.isfinite(G)):
            return 1e12

        gbar = np.mean(G, axis=0)

        if W is None:
            moment_val = float(gbar @ gbar)
        else:
            moment_val = float(gbar @ W @ gbar)

        if not np.isfinite(moment_val):
            return 1e12

        # Soft penalties
        penalty_boundary = boundary_soft_penalty(theta, cfg)
        penalty_alpha_pole = alpha_pole_soft_penalty(theta, cfg)

        val = moment_val + penalty_boundary + penalty_alpha_pole

        if not np.isfinite(val):
            return 1e12

        return val

    except Exception:
        return 1e12

def ar1_initial_guess(x, h):
    """
    Data-driven AR(1)-based initial guess for kappa, mu, sigma.
    Tail parameters are initialized separately by grid.
    """
    x_lag = x[:-1]
    x_now = x[1:]

    X = np.column_stack([np.ones_like(x_lag), x_lag])

    try:
        beta = np.linalg.lstsq(X, x_now, rcond=None)[0]
        a, b = beta[0], beta[1]
    except Exception:
        a, b = np.nan, np.nan

    if not np.isfinite(b):
        b = 0.95

    b = float(np.clip(b, 0.20, 0.995))

    kappa0 = -np.log(b) / h
    kappa0 = float(np.clip(kappa0, 0.005, 0.50))

    if abs(1.0 - b) > 1e-6 and np.isfinite(a):
        mu0 = a / (1.0 - b)
    else:
        mu0 = np.nanmean(x)

    mu0 = float(np.clip(mu0, 1.0, 80.0))

    resid = x_now - (a + b * x_lag)
    var_resid = np.nanvar(resid)

    denom = 1.0 - np.exp(-2.0 * kappa0 * h)
    if denom <= 1e-8 or not np.isfinite(var_resid):
        sigma0 = 1.0
    else:
        sigma0 = np.sqrt(var_resid * 2.0 * kappa0 / denom)

    sigma0 = float(np.clip(sigma0, 0.10, 5.00))

    return np.array([kappa0, mu0, sigma0, 0.75, 1.50], dtype=float)

def initial_guesses(x, true_theta, cfg):
    """
    Multiple starts.

    Stage3:
        - use AR(1)-based estimates for kappa, mu, sigma;
        - use a tail-parameter grid for alpha and lambda;
        - add perturbed starts around true_theta only as numerical
          stabilization in Monte Carlo, but do not include exact true_theta.
    """
    bounds = parameter_bounds()
    guesses = []

    base = ar1_initial_guess(x, cfg.h)

    # Tail grid covering heavy, baseline, and light tail cases.
    tail_grid = [
        (0.45, 0.60),
        (0.60, 0.80),
        (0.75, 1.20),
        (0.75, 2.00),
        (1.20, 3.00),
        (1.30, 4.00),
    ]

    for a0, l0 in tail_grid:
        g = base.copy()
        g[3] = a0
        g[4] = l0
        guesses.append(g)

    # Add perturbed starts around true theta, but not exact true theta.
    true_theta = np.asarray(true_theta, dtype=float)

    n_extra = max(0, cfg.n_starts - len(guesses))
    for _ in range(n_extra):
        perturb = np.array([
            np.random.normal(1.0, 0.15),
            np.random.normal(1.0, 0.08),
            np.random.normal(1.0, 0.20),
            np.random.normal(1.0, 0.15),
            np.random.normal(1.0, 0.20),
        ])
        g = true_theta * perturb

        for i, (lo, hi) in enumerate(bounds):
            width = hi - lo
            margin = 0.05 * width
            g[i] = np.clip(g[i], lo + margin, hi - margin)

        if abs(g[3] - 1.0) < 0.04:
            g[3] = 0.90 if g[3] < 1.0 else 1.10

        guesses.append(g)

    # Clip all guesses and remove alpha close to pole.
    clean_guesses = []
    for g in guesses:
        g = np.asarray(g, dtype=float)
        for i, (lo, hi) in enumerate(bounds):
            width = hi - lo
            margin = 0.03 * width
            g[i] = np.clip(g[i], lo + margin, hi - margin)

        if abs(g[3] - 1.0) < 0.04:
            g[3] = 0.90 if g[3] < 1.0 else 1.10

        clean_guesses.append(g)

    return clean_guesses

def compute_pc_weight(x, theta_prelim, method, cfg):
    """
    Safe PC-regularized weight matrix.

    Improvements over stage1:
        1. covariance ridge;
        2. eigenvalue floor;
        3. inverse-eigenvalue cap;
        4. shrinkage toward identity.

    This avoids excessive amplification of noisy low-variance moment
    directions, which was one likely reason for lambda overestimation.
    """
    G = combined_moments(x, theta_prelim, method, cfg)

    if not np.all(np.isfinite(G)):
        return np.eye(G.shape[1])

    Gc = G - np.mean(G, axis=0)
    q = Gc.shape[1]

    S = (Gc.T @ Gc) / Gc.shape[0]
    S = S + cfg.pc_ridge * np.eye(q)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = np.sum(eigvals)
    if total <= 0 or not np.isfinite(total):
        return np.eye(q)

    cumratio = np.cumsum(eigvals) / total
    r = int(np.searchsorted(cumratio, cfg.pc_tau) + 1)
    r = max(1, min(r, q))

    P_r = eigvecs[:, :r]
    L_r = eigvals[:r]

    # floor and cap inverse eigenvalues
    floor = cfg.pc_eig_floor * np.max(eigvals)
    L_safe = np.maximum(L_r, floor)

    inv_L = 1.0 / L_safe
    inv_L = np.minimum(inv_L, cfg.pc_inv_cap)

    W_pc = P_r @ np.diag(inv_L) @ P_r.T

    # shrink toward identity for stability
    W = (1.0 - cfg.pc_weight_shrinkage) * W_pc + cfg.pc_weight_shrinkage * np.eye(q)

    if not np.all(np.isfinite(W)):
        return np.eye(q)

    return W

def estimate_method(x, method, true_theta, cfg):
    """
    Estimate parameters for one method.
    """
    bounds = parameter_bounds()
    guesses = initial_guesses(x, true_theta, cfg)

    start_time = time.time()

    best_res = None
    best_val = np.inf

    # Stage 1: identity weight
    for g0 in guesses:
        res = minimize(
            moment_objective,
            g0,
            args=(x, method, cfg, None),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": cfg.maxiter,
                "ftol": cfg.ftol,
            }
        )

        val = res.fun if np.isfinite(res.fun) else np.inf

        if val < best_val:
            best_val = val
            best_res = res

    if best_res is None:
        return {
            "success": False,
            "theta_hat": np.full(5, np.nan),
            "objective": np.nan,
            "message": "stage1 failed",
            "runtime": time.time() - start_time
        }

    theta_prelim = best_res.x

    best_res2 = None
    best_val2 =np.inf

    # Stage 2 for PC-CF-RGMM: PC weight
    if method == "PC-CF-RGMM":
        try:
            W_pc = compute_pc_weight(x, theta_prelim, method, cfg)
        except Exception:
            W_pc = None
            
        for g0 in [theta_prelim] + guesses:
            res2 = minimize(
                moment_objective,
                g0,
                args=(x, method, cfg, W_pc),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": cfg.maxiter,
                    "ftol": cfg.ftol,
                }
            )

            val2 = res2.fun if np.isfinite(res2.fun) else np.inf

            if val2 < best_val2:
                best_val2 = val2
                best_res2 = res2

    # Safeguard: if PC-weighted stage is numerically much worse,
    # fall back to identity-weight stage.
    if best_res2 is not None and np.isfinite(best_val2):
        if best_val2 <= 10.0 * best_val:
            final_res = best_res2
        else:
            final_res = best_res
    else:
        final_res = best_res

    theta_hat = final_res.x if final_res is not None else np.full(5, np.nan)
    success_raw = bool(final_res.success) if final_res is not None else False
    objective = float(final_res.fun) if final_res is not None else np.nan
    message = str(final_res.message) if final_res is not None else "failed"

    # Detect artificial boundary solution
    boundary_flag = False
    if final_res is not None and np.all(np.isfinite(theta_hat)):
        boundary_flag = is_near_boundary(theta_hat, cfg)
        
    # Treat boundary solution as non-successful
    success = success_raw and (not boundary_flag)

    if boundary_flag:
        message = message + " | boundary_solution_detected"

    return {
        "success": success,
        "theta_hat": theta_hat,
        "objective": objective,
        "message": message,
        "runtime": time.time() - start_time,
        "boundary_flag": boundary_flag,
        "success_raw": success_raw
    }
    
# ============================================================
# 9. Checkpoint handling
# ============================================================

def initialize_checkpoint(cfg):
    if cfg.fresh_start:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        raw_path = os.path.join(RESULT_DIR, "raw_results.csv")
        if os.path.exists(raw_path):
            os.remove(raw_path)

    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = load_pickle(CHECKPOINT_FILE)
        print(f"[Resume] Loaded checkpoint from {CHECKPOINT_FILE}")
        return checkpoint

    checkpoint = {
        "created_at": current_time_string(),
        "updated_at": current_time_string(),
        "completed_jobs": set(),
        "results": []
    }
    save_pickle(checkpoint, CHECKPOINT_FILE)
    print(f"[Init] Created new checkpoint at {CHECKPOINT_FILE}")
    return checkpoint

def job_key(scenario_name, contamination_name, rep, method):
    return f"{scenario_name}|{contamination_name}|rep{rep}|{method}"

def append_result_to_csv(result_dict):
    raw_path = os.path.join(RESULT_DIR, "raw_results.csv")
    df = pd.DataFrame([result_dict])
    if not os.path.exists(raw_path):
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(raw_path, index=False, mode="a", header=False, encoding="utf-8-sig")

# ============================================================
# 10. Main simulation loop
# ============================================================

def run_simulations(cfg):
    set_seed(cfg.seed)

    checkpoint = initialize_checkpoint(cfg)

    # Precompute innovation samplers for each parameter scenario
    print("\n[Step] Precomputing innovation samplers...")
    samplers = {}

    for sname, spara in PARAM_SCENARIOS.items():
        theta_true = theta_to_array(spara)
        print(f"  Building sampler for {sname} ...")
        sampler = InnovationSampler(
            theta_true,
            h=cfg.h,
            nx=cfg.innovation_grid_nx,
            nu=cfg.innovation_grid_nu,
            xmax_multiplier=cfg.innovation_xmax_multiplier,
            quad_nodes=cfg.cf_quad_nodes
        )
        samplers[sname] = sampler

    print("[Done] Innovation samplers ready.\n")

    total_jobs = (
        len(PARAM_SCENARIOS)
        * len(CONTAMINATION_SCENARIOS)
        * cfg.n_rep
        * len(METHODS)
    )

    done_jobs = len(checkpoint["completed_jobs"])
    print(f"[Info] Total jobs: {total_jobs}")
    print(f"[Info] Already completed: {done_jobs}")

    job_counter = done_jobs

    for sname, spara in PARAM_SCENARIOS.items():
        theta_true = theta_to_array(spara)
        sampler = samplers[sname]

        for cname, cpara in CONTAMINATION_SCENARIOS.items():

            for rep in range(cfg.n_rep):

                # Simulate once per scenario-contamination-rep
                # Then estimate all methods on same data.
                # Need deterministic seed for reproducibility.
                local_seed = (
                    cfg.seed
                    + 100000 * list(PARAM_SCENARIOS.keys()).index(sname)
                    + 10000 * list(CONTAMINATION_SCENARIOS.keys()).index(cname)
                    + rep
                )
                np.random.seed(local_seed)

                # If all methods of this rep completed, skip simulation
                all_done = all(
                    job_key(sname, cname, rep, m) in checkpoint["completed_jobs"]
                    for m in METHODS
                )
                if all_done:
                    continue

                x_clean = simulate_ou_path(theta_true, cfg.n_obs, cfg.h, sampler)
                x_obs = contaminate_path(x_clean, cpara)

                for method in METHODS:
                    key = job_key(sname, cname, rep, method)

                    if key in checkpoint["completed_jobs"]:
                        continue

                    print(
                        f"[{current_time_string()}] "
                        f"Running {key} "
                        f"({job_counter + 1}/{total_jobs})"
                    )

                    try:
                        est = estimate_method(x_obs, method, theta_true, cfg)
                    except Exception as e:
                        est = {
                            "success": False,
                            "theta_hat": np.full(5, np.nan),
                            "objective": np.nan,
                            "message": f"exception: {e}",
                            "runtime": np.nan
                        }

                    theta_hat = est["theta_hat"]

                    result = {
                       "scenario": sname,
                       "scenario_desc": spara["description"],
                       "contamination": cname,
                       "contamination_type": cpara["type"],
                       "epsilon": cpara["epsilon"],
                       "b": cpara["b"],
                       "rep": rep,
                       "method": method,
                       "success": est["success"],
                       "success_raw": est.get("success_raw", est["success"]),
                       "boundary_flag": est.get("boundary_flag", False),
                       "objective": est["objective"],
                       "message": est["message"],
                       "runtime": est["runtime"],
                     }   

                    for i, pname in enumerate(PARAM_NAMES):
                        result[f"true_{pname}"] = theta_true[i]
                        result[f"hat_{pname}"] = theta_hat[i]
                        result[f"err_{pname}"] = theta_hat[i] - theta_true[i]

                    checkpoint["results"].append(result)
                    checkpoint["completed_jobs"].add(key)
                    checkpoint["updated_at"] = current_time_string()

                    append_result_to_csv(result)

                    job_counter += 1

                    if job_counter % cfg.checkpoint_every == 0:
                        save_pickle(checkpoint, CHECKPOINT_FILE)

    save_pickle(checkpoint, CHECKPOINT_FILE)

    print("\n[Done] All simulations finished.")
    print(f"[Saved] Raw results: {os.path.join(RESULT_DIR, 'raw_results.csv')}")
    print(f"[Saved] Checkpoint: {CHECKPOINT_FILE}")

# ============================================================
# 11. Summary tables
# ============================================================

def _to_bool_series(s):
    """
    Convert a pandas Series to boolean safely.
    Handles bool, numeric, and string values.
    """
    if s.dtype == bool:
        return s

    if np.issubdtype(s.dtype, np.number):
        return s.astype(float) != 0.0

    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])

def load_raw_results():
    raw_path = os.path.join(RESULT_DIR, "raw_results.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No raw_results.csv found at {raw_path}. "
            f"Run simulations first."
        )

    df = pd.read_csv(raw_path)

    # Normalize boolean columns after reading from CSV
    for col in ["success", "success_raw", "boundary_flag"]:
        if col in df.columns:
            df[col] = _to_bool_series(df[col])

    # Remove possible duplicated jobs caused by interruption between
    # appending raw_results.csv and saving checkpoint.pkl.
    key_cols = ["scenario", "contamination", "rep", "method"]
    if all(c in df.columns for c in key_cols):
        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
        after = len(df)
        if after < before:
            print(f"[Info] Dropped duplicated raw result rows: {before - after}")

    return df

def make_summary_tables(df):
    """
    Generate summary tables:
        - bias
        - rmse
        - mae
        - sd
        - fail rate
        - average runtime
    """
    df = df.copy()

    for pname in PARAM_NAMES:
        df[f"abs_err_{pname}"] = np.abs(df[f"err_{pname}"])
        df[f"sq_err_{pname}"] = df[f"err_{pname}"] ** 2

    group_cols = [
        "scenario",
        "scenario_desc",
        "contamination",
        "contamination_type",
        "epsilon",
        "b",
        "method"
    ]

    rows = []

    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_total"] = len(g)
        row["success_rate"] = np.mean(g["success"].astype(float))
        row["fail_rate"] = 1.0 - row["success_rate"]
        row["avg_runtime"] = np.nanmean(g["runtime"])
        
        if "boundary_flag" in g.columns:
            row["boundary_rate"] = np.mean(g["boundary_flag"].astype(float))
        else:
            row["boundary_rate"] = np.nan
                
        if "success_raw" in g.columns:
            row["raw_success_rate"] = np.mean(g["success_raw"].astype(float))
        else:
            row["raw_success_rate"] = row["success_rate"]
                        
        g_success = g[g["success"] == True]

        for pname in PARAM_NAMES:
            err = g_success[f"err_{pname}"].dropna()
            hat = g_success[f"hat_{pname}"].dropna()

            row[f"bias_{pname}"] = np.mean(err) if len(err) > 0 else np.nan
            row[f"rmse_{pname}"] = np.sqrt(np.mean(err ** 2)) if len(err) > 0 else np.nan
            row[f"mae_{pname}"] = np.mean(np.abs(err)) if len(err) > 0 else np.nan
            row[f"sd_{pname}"] = np.std(hat, ddof=1) if len(hat) > 1 else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(RESULT_DIR, "summary_results.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Separate tables
    bias_cols = group_cols + [f"bias_{p}" for p in PARAM_NAMES]
    rmse_cols = group_cols + [f"rmse_{p}" for p in PARAM_NAMES]
    mae_cols = group_cols + [f"mae_{p}" for p in PARAM_NAMES]
    sd_cols = group_cols + [f"sd_{p}" for p in PARAM_NAMES]
    fail_cols = group_cols + [
    "raw_success_rate",
    "success_rate",
    "fail_rate",
    "boundary_rate",
    "avg_runtime"
    ]

    summary[bias_cols].to_csv(
        os.path.join(TABLE_DIR, "table_bias.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    summary[rmse_cols].to_csv(
        os.path.join(TABLE_DIR, "table_rmse.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    summary[mae_cols].to_csv(
        os.path.join(TABLE_DIR, "table_mae.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    summary[sd_cols].to_csv(
        os.path.join(TABLE_DIR, "table_sd.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    summary[fail_cols].to_csv(
        os.path.join(TABLE_DIR, "table_fail_runtime.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # Pivot RMSE tables by method for each parameter
    for pname in PARAM_NAMES:
        pivot = summary.pivot_table(
            index=["scenario", "contamination"],
            columns="method",
            values=f"rmse_{pname}"
        ).reset_index()

        pivot.to_csv(
            os.path.join(TABLE_DIR, f"pivot_rmse_{pname}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Pivot Bias tables by method for each parameter
    for pname in PARAM_NAMES:
        pivot = summary.pivot_table(
            index=["scenario", "contamination"],
            columns="method",
            values=f"bias_{pname}"
        ).reset_index()

        pivot.to_csv(
            os.path.join(TABLE_DIR, f"pivot_bias_{pname}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print(f"[Saved] Summary table: {summary_path}")
    print(f"[Saved] Tables in: {TABLE_DIR}")

    return summary

# ============================================================
# 12. Figures
# ============================================================

def set_plot_style():
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_rmse_bars(summary):
    """
    Bar plots of RMSE for each parameter and each scenario.
    """
    for pname in PARAM_NAMES:
        for scenario in summary["scenario"].unique():
            sub = summary[summary["scenario"] == scenario].copy()

            labels = sub["contamination"].unique()
            x = np.arange(len(labels))
            width = 0.25

            plt.figure(figsize=(12, 6))

            for i, method in enumerate(METHODS):
                vals = []
                for c in labels:
                    tmp = sub[
                        (sub["contamination"] == c)
                        & (sub["method"] == method)
                    ]
                    vals.append(tmp[f"rmse_{pname}"].values[0] if len(tmp) > 0 else np.nan)

                plt.bar(
                    x + (i - 1) * width,
                    vals,
                    width=width,
                    label=method
                )

            plt.xticks(x, labels, rotation=30, ha="right")
            plt.ylabel(f"RMSE of {pname}")
            plt.title(f"RMSE comparison for {pname} - {scenario}")
            plt.legend()

            path = os.path.join(
                FIGURE_DIR,
                f"rmse_bar_{scenario}_{pname}.png"
            )
            savefig(path)

def plot_bias_heatmaps(summary):
    """
    Simple heatmap of bias by method and contamination.
    """
    for pname in PARAM_NAMES:
        for scenario in summary["scenario"].unique():
            sub = summary[summary["scenario"] == scenario]

            pivot = sub.pivot_table(
                index="method",
                columns="contamination",
                values=f"bias_{pname}"
            )

            plt.figure(figsize=(11, 4))
            im = plt.imshow(pivot.values, aspect="auto", cmap="coolwarm")

            plt.colorbar(im, label=f"Bias of {pname}")
            plt.xticks(
                np.arange(len(pivot.columns)),
                pivot.columns,
                rotation=30,
                ha="right"
            )
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            plt.title(f"Bias heatmap for {pname} - {scenario}")

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        plt.text(
                            j, i, f"{val:.3f}",
                            ha="center", va="center",
                            color="black", fontsize=8
                        )

            path = os.path.join(
                FIGURE_DIR,
                f"bias_heatmap_{scenario}_{pname}.png"
            )
            savefig(path)

def plot_boxplots(df):
    """
    Boxplots of estimation errors for each parameter.
    """
    df_success = df[df["success"] == True].copy()

    for pname in PARAM_NAMES:
        for scenario in df_success["scenario"].unique():
            for contamination in df_success["contamination"].unique():

                sub = df_success[
                    (df_success["scenario"] == scenario)
                    & (df_success["contamination"] == contamination)
                ]

                if len(sub) == 0:
                    continue

                data = []
                labels = []

                for method in METHODS:
                    tmp = sub[sub["method"] == method][f"err_{pname}"].dropna()
                    if len(tmp) > 0:
                        data.append(tmp.values)
                        labels.append(method)

                if len(data) == 0:
                    continue

                plt.figure(figsize=(9, 5))
                plt.boxplot(data, labels=labels, showfliers=True)
                plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
                plt.ylabel(f"Estimation error of {pname}")
                plt.title(f"Error boxplot: {pname}, {scenario}, {contamination}")

                path = os.path.join(
                    FIGURE_DIR,
                    f"boxplot_{scenario}_{contamination}_{pname}.png"
                )
                savefig(path)

def plot_contamination_sensitivity(summary):
    """
    For random contamination levels, plot RMSE against epsilon.
    """
    random_sub = summary[
        summary["contamination_type"].isin(["none", "random"])
    ].copy()

    if len(random_sub) == 0:
        return

    for pname in PARAM_NAMES:
        for scenario in random_sub["scenario"].unique():
            sub = random_sub[random_sub["scenario"] == scenario]

            plt.figure(figsize=(9, 5))

            for method in METHODS:
                tmp = sub[sub["method"] == method].sort_values("epsilon")
                if len(tmp) == 0:
                    continue
                plt.plot(
                    tmp["epsilon"],
                    tmp[f"rmse_{pname}"],
                    marker="o",
                    label=method
                )

            plt.xlabel("Contamination ratio epsilon")
            plt.ylabel(f"RMSE of {pname}")
            plt.title(f"Contamination sensitivity: {pname}, {scenario}")
            plt.legend()

            path = os.path.join(
                FIGURE_DIR,
                f"sensitivity_{scenario}_{pname}.png"
            )
            savefig(path)

def plot_fail_rates(summary):
    """
    Failure rate comparison.
    """
    for scenario in summary["scenario"].unique():
        sub = summary[summary["scenario"] == scenario]

        labels = sub["contamination"].unique()
        x = np.arange(len(labels))
        width = 0.25

        plt.figure(figsize=(12, 5))

        for i, method in enumerate(METHODS):
            vals = []
            for c in labels:
                tmp = sub[
                    (sub["contamination"] == c)
                    & (sub["method"] == method)
                ]
                vals.append(tmp["fail_rate"].values[0] if len(tmp) > 0 else np.nan)

            plt.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                label=method
            )

        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Failure rate")
        plt.title(f"Optimization failure rate - {scenario}")
        plt.legend()

        path = os.path.join(FIGURE_DIR, f"fail_rate_{scenario}.png")
        savefig(path)

# ============================================================
# 12B. Paper-quality figures for simulation section
# ============================================================

SCENARIO_ORDER = [
    "S1_baseline",
    "S2_heavy_tail",
    "S3_light_tail",
    "S4_fast_reversion"
]

CONTAM_ORDER = [
    "C0_clean",
    "C1_random_1pct",
    "C2_random_3pct",
    "C3_block",
    "C4_tail"
]

SCENARIO_LABEL = {
    "S1_baseline": "S1：基准情形",
    "S2_heavy_tail": "S2：重尾情形",
    "S3_light_tail": "S3：轻尾情形",
    "S4_fast_reversion": "S4：快速均值回复"
}

CONTAM_LABEL = {
    "C0_clean": "无污染",
    "C1_random_1pct": "1%随机污染",
    "C2_random_3pct": "3%随机污染",
    "C3_block": "3%块状污染",
    "C4_tail": "3%尾部污染"
}

METHOD_COLOR = {
    "GMM": "#4C72B0",
    "Huber-GMM": "#55A868",
    "PC-CF-RGMM": "#C44E52"
}

PAPER_FIGURE_DIR = os.path.join(FIGURE_DIR, "paper_figures")
os.makedirs(PAPER_FIGURE_DIR, exist_ok=True)

def set_paper_plot_style():
    """
    论文正文图形样式：
    1. 支持中文标题和坐标轴标签；
    2. 增大字号，避免 PDF 中标题显示不清；
    3. 保存为较高分辨率图片。
    """
    import matplotlib
    from matplotlib import font_manager

    # 常见中文字体候选，按系统自动选择
    chinese_font_candidates = [
        "Microsoft YaHei",      # Windows
        "SimHei",               # Windows
        "SimSun",               # Windows
        "Noto Sans CJK SC",     # Linux
        "Source Han Sans SC",   # Adobe 思源黑体
        "PingFang SC",          # macOS
        "Heiti SC",             # macOS
        "Arial Unicode MS"
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = None
    for font in chinese_font_candidates:
        if font in available_fonts:
            chosen_font = font
            break

    if chosen_font is None:
        print("[Warning] 未找到常见中文字体，中文可能显示为方框。建议安装 SimHei 或 Noto Sans CJK SC。")
        chosen_font = "DejaVu Sans"

    plt.rcParams["font.family"] = chosen_font
    plt.rcParams["font.sans-serif"] = [chosen_font]
    plt.rcParams["axes.unicode_minus"] = False

    # PDF/PS 中嵌入 TrueType 字体，避免 LaTeX 或查看器中字体发虚
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # 数学符号字体
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # 整体字号调大
    plt.rcParams["font.size"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 16

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.30
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["figure.dpi"] = 150

def save_paper_fig(name):
    """
    同时保存 PDF 和 PNG。
    兼容 constrained_layout 和 tight_layout。
    如果图中已经启用 constrained_layout 或已经创建 colorbar，
    则不再强制调用 tight_layout，避免 Matplotlib 布局引擎冲突。
    """
    pdf_path = os.path.join(PAPER_FIGURE_DIR, name + ".pdf")
    png_path = os.path.join(PAPER_FIGURE_DIR, name + ".png")

    fig = plt.gcf()

    # 尝试 tight_layout；如果和 constrained_layout/colorbar 冲突，就跳过
    try:
        fig.tight_layout(pad=1.2)
    except RuntimeError as e:
        print(f"[Layout warning] {name}: 跳过 tight_layout，原因：{e}")
    except Exception as e:
        print(f"[Layout warning] {name}: tight_layout 失败，原因：{e}")

    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"[Paper figure saved] {pdf_path}")
    print(f"[Paper figure saved] {png_path}")

def plot_paper_rmse_sigma_heatmap(summary):
    """
    图 1：
    不同模拟场景下尺度参数 σ 的 RMSE 热力图。
    """
    df = summary.copy()
    df["log_rmse_sigma"] = np.log10(df["rmse_sigma"] + 1e-12)

    vmin = df["log_rmse_sigma"].min()
    vmax = df["log_rmse_sigma"].max()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.5, 4.8),
        sharey=True,
        constrained_layout=True
    )

    method_title = {
        "GMM": "GMM",
        "Huber-GMM": "Huber-GMM",
        "PC-CF-RGMM": "PC-CF-RGMM"
    }

    for ax, method in zip(axes, METHODS):
        sub = df[df["method"] == method]

        pivot_log = sub.pivot_table(
            index="scenario",
            columns="contamination",
            values="log_rmse_sigma"
        ).loc[SCENARIO_ORDER, CONTAM_ORDER]

        pivot_raw = sub.pivot_table(
            index="scenario",
            columns="contamination",
            values="rmse_sigma"
        ).loc[SCENARIO_ORDER, CONTAM_ORDER]

        im = ax.imshow(
            pivot_log.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(method_title[method], pad=10)
        ax.set_xticks(np.arange(len(CONTAM_ORDER)))
        ax.set_xticklabels(
            [CONTAM_LABEL[c] for c in CONTAM_ORDER],
            rotation=35,
            ha="right"
        )

        ax.set_yticks(np.arange(len(SCENARIO_ORDER)))
        ax.set_yticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER])

        for i in range(len(SCENARIO_ORDER)):
            for j in range(len(CONTAM_ORDER)):
                val = pivot_raw.iloc[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black"
                )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("log10(RMSEσ)", fontsize=13)

    # 图题改为中文
    fig.suptitle("不同模拟场景下尺度参数 σ 的 RMSE", y=1.04)

    save_paper_fig("sim_fig_01_rmse_sigma_heatmap")

def plot_paper_mu_block_bar(summary):
    """
    图 2：
    块状污染下长期均值参数 μ 的 RMSE。
    """
    sub = summary[summary["contamination"] == "C3_block"].copy()

    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.24

    for k, method in enumerate(METHODS):
        vals = []
        for s in SCENARIO_ORDER:
            val = sub[
                (sub["scenario"] == s) &
                (sub["method"] == method)
            ]["rmse_mu"].values[0]
            vals.append(val)

        ax.bar(
            x + (k - 1) * width,
            vals,
            width=width,
            label=method,
            color=METHOD_COLOR[method],
            edgecolor="black",
            linewidth=0.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABEL[s] for s in SCENARIO_ORDER],
        rotation=18,
        ha="right"
    )
    ax.set_ylabel("μ 的 RMSE")
    ax.set_title("块状污染下长期均值参数 μ 的 RMSE", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save_paper_fig("sim_fig_02_rmse_mu_block_bar")

def plot_paper_sigma_improvement(summary):
    """
    图 3：
    PC-CF-RGMM 相对于 GMM 在 σ 估计上的 RMSE 改善比例。
    """
    selected_contams = [
        "C2_random_3pct",
        "C3_block",
        "C4_tail"
    ]

    rows = []

    for s in SCENARIO_ORDER:
        for c in selected_contams:
            gmm = summary[
                (summary["scenario"] == s) &
                (summary["contamination"] == c) &
                (summary["method"] == "GMM")
            ]["rmse_sigma"].values[0]

            pc = summary[
                (summary["scenario"] == s) &
                (summary["contamination"] == c) &
                (summary["method"] == "PC-CF-RGMM")
            ]["rmse_sigma"].values[0]

            improvement = 100.0 * (1.0 - pc / gmm)

            rows.append({
                "scenario": s,
                "contamination": c,
                "improvement": improvement
            })

    imp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.0, 5.4))

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.25

    colors = {
        "C2_random_3pct": "#8172B3",
        "C3_block": "#CCB974",
        "C4_tail": "#64B5CD"
    }

    for k, c in enumerate(selected_contams):
        vals = []
        for s in SCENARIO_ORDER:
            val = imp_df[
                (imp_df["scenario"] == s) &
                (imp_df["contamination"] == c)
            ]["improvement"].values[0]
            vals.append(val)

        ax.bar(
            x + (k - 1) * width,
            vals,
            width=width,
            label=CONTAM_LABEL[c],
            color=colors[c],
            edgecolor="black",
            linewidth=0.5
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SCENARIO_LABEL[s] for s in SCENARIO_ORDER],
        rotation=18,
        ha="right"
    )
    ax.set_ylabel("相对于 GMM 的 RMSE 改善比例（%）")
    ax.set_title("PC-CF-RGMM 相对于 GMM 在 σ 估计上的 RMSE 改善", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    save_paper_fig("sim_fig_03_sigma_improvement_pc_over_gmm")

def plot_paper_lambda_tradeoff(summary):
    """
    图 4：
    λ 参数估计中的稳健性—效率权衡。
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12.2, 7.8),
        sharex=True,
        constrained_layout=True
    )
    axes = axes.ravel()

    for ax, s in zip(axes, SCENARIO_ORDER):
        sub_s = summary[summary["scenario"] == s]
        x = np.arange(len(CONTAM_ORDER))

        for method in METHODS:
            vals = []
            for c in CONTAM_ORDER:
                val = sub_s[
                    (sub_s["contamination"] == c) &
                    (sub_s["method"] == method)
                ]["rmse_lambda"].values[0]
                vals.append(val)

            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.9,
                markersize=4.5,
                label=method,
                color=METHOD_COLOR[method]
            )

        ax.set_title(SCENARIO_LABEL[s], pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONTAM_LABEL[c] for c in CONTAM_ORDER],
            rotation=30,
            ha="right"
        )
        ax.set_ylabel("λ 的 RMSE")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04)
    )

    fig.suptitle("λ 参数估计中的稳健性—效率权衡", y=1.08)

    save_paper_fig("sim_fig_04_lambda_tradeoff")

def plot_paper_runtime_bar(summary):
    """
    图 5：
    不同估计方法的平均运行时间。
    """
    avg_runtime = summary.groupby("method")["avg_runtime"].mean().reindex(METHODS)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    bars = ax.bar(
        np.arange(len(METHODS)),
        avg_runtime.values,
        color=[METHOD_COLOR[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("平均运行时间（秒）")
    ax.set_title("不同估计方法的计算成本", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, avg_runtime.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    save_paper_fig("sim_fig_05_runtime_bar")

def plot_paper_success_runtime_scatter(summary):
    """
    图 6：
    成功率与运行时间的关系。
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    for method in METHODS:
        sub = summary[summary["method"] == method]

        ax.scatter(
            sub["avg_runtime"],
            sub["success_rate"],
            s=55,
            alpha=0.78,
            label=method,
            color=METHOD_COLOR[method],
            edgecolor="black",
            linewidth=0.4
        )

    ax.set_xlabel("平均运行时间（秒）")
    ax.set_ylabel("成功率")
    ax.set_ylim(0.80, 1.02)
    ax.set_title("数值稳定性与计算成本", pad=12)
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    save_paper_fig("sim_fig_06_success_runtime_scatter")

def generate_paper_figures(summary):
    """
    生成第 5 节论文正文使用的高质量图。
    """
    set_paper_plot_style()

    plot_paper_rmse_sigma_heatmap(summary)
    plot_paper_mu_block_bar(summary)
    plot_paper_sigma_improvement(summary)
    plot_paper_lambda_tradeoff(summary)
    plot_paper_runtime_bar(summary)
    plot_paper_success_runtime_scatter(summary)

    print(f"[Saved] Paper-quality figures in: {PAPER_FIGURE_DIR}")

def generate_all_figures(df, summary):
    """
    Generate both exploratory figures and paper-quality figures.
    """

    # 1. Original exploratory figures
    set_plot_style()

    plot_rmse_bars(summary)
    plot_bias_heatmaps(summary)
    # plot_boxplots(df)  # 可选：图太多，论文正文不建议生成
    plot_contamination_sensitivity(summary)
    plot_fail_rates(summary)

    print(f"[Saved] Exploratory figures in: {FIGURE_DIR}")

    # 2. Paper-quality summary figures
    generate_paper_figures(summary)

# ============================================================
# 13. LaTeX table helper
# ============================================================

def generate_latex_tables(summary):
    """
    Generate selected LaTeX tables for direct inclusion.
    """
    latex_dir = os.path.join(TABLE_DIR, "latex")
    os.makedirs(latex_dir, exist_ok=True)

    # Example: RMSE table for alpha and lambda
    for pname in ["alpha", "lambda", "kappa", "sigma"]:
        pivot = summary.pivot_table(
            index=["scenario", "contamination"],
            columns="method",
            values=f"rmse_{pname}"
        ).reset_index()

        tex = pivot.to_latex(
            index=False,
            float_format=lambda x: f"{x:.4f}",
            caption=f"RMSE comparison for {pname}",
            label=f"tab:rmse_{pname}",
            escape=False
        )

        with open(os.path.join(latex_dir, f"rmse_{pname}.tex"), "w", encoding="utf-8") as f:
            f.write(tex)

    # Failure rate table
    fail_table = summary.pivot_table(
        index=["scenario", "contamination"],
        columns="method",
        values="fail_rate"
    ).reset_index()

    tex = fail_table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption="Optimization failure rates",
        label="tab:failure_rates",
        escape=False
    )

    with open(os.path.join(latex_dir, "failure_rates.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[Saved] LaTeX tables in: {latex_dir}")

# ============================================================
# 14. Postprocessing pipeline
# ============================================================

def postprocess():
    print("\n[Step] Loading raw results...")
    df = load_raw_results()

    print("[Step] Making summary tables...")
    summary = make_summary_tables(df)

    print("[Step] Generating figures...")
    generate_all_figures(df, summary)

    print("[Step] Generating LaTeX tables...")
    generate_latex_tables(summary)

    print("\n[Done] Postprocessing complete.")

# ============================================================
# 15. Main entry
# ============================================================

def main():
    sanity_check_cts()
    append_timing_log_external('SCRIPT_START')

    print("=" * 80)
    print("Monte Carlo for CTS Levy driven OU model | GENERATED STAGE 3 FINAL")
    print("Methods: GMM, Huber-GMM, PC-CF-RGMM")
    print("=" * 80)
    print(f"Start time: {current_time_string()}")
    print(f"Results directory: {RESULT_DIR}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print("=" * 80)

    if CFG.only_postprocess:
        postprocess()
        return

    run_simulations(CFG)
    postprocess()

    print("=" * 80)
    print(f"End time: {current_time_string()}")
    print("=" * 80)

if __name__ == "__main__":
    main()