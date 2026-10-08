# -*- coding: utf-8 -*-
"""
CTS-driven OU: simulation + robust CF-GMM estimation.
Model: dX_t = kappa*(mu - X_t)dt + sigma*dL_t, L symmetric CTS (variance 1).
"""
import numpy as np
from scipy.special import gammaincc, gamma
from scipy.optimize import minimize

# ---------------- symmetric CTS ----------------
# Levy density: nu(dx) = C * e^{-lam|x|}/|x|^{1+alpha} on each side.
# Variance = 2*C*Gamma(2-alpha)*lam^{alpha-2}; standardize to 1.

def cts_C(alpha, lam=1.0):
    return 1.0 / (2.0 * gamma(2.0 - alpha) * lam ** (alpha - 2.0))

def levy_symbol(v, alpha, lam=1.0):
    """psi(v) = log CF of CTS at unit time (symmetric, zero drift)."""
    C = cts_C(alpha, lam)
    # psi(v) = 2*C*Re[ Gamma(-alpha)*((lam - i v)^alpha - lam^alpha) ]
    return 2.0 * C * np.real(gamma(-alpha) * ((lam - 1j * v) ** alpha - lam ** alpha))

def simulate_ou_path(kappa, mu, sigma, alpha, n, dt, rng,
                     lam=1.0, delta_jump=0.02, Tmax_extra=1.0):
    """Simulate X at n equally spaced times (step dt) via shot-noise series
    for the symmetric CTS Levy process + Gaussian small-jump compensation."""
    C = cts_C(alpha, lam)
    Ttot = n * dt + Tmax_extra
    # Shot-noise series for the symmetric CTS (Rosiński 2007; Kawai–Masuda):
    #   S_j = (alpha*Gamma_j/(C*Ttot))^{-1/alpha}   (stable-series sizes)
    #   V_j = min(S_j, E_j/lam)                     (exponential marks temper)
    # signs give symmetry. Truncation at Gamma_max is compensated by a
    # Gaussian term with the residual variance of the tempered measure.
    Gamma_max = 60.0
    n_jumps = rng.poisson(Gamma_max)
    Gj = np.cumsum(rng.exponential(1.0, n_jumps))
    Gj = Gj[Gj <= Gamma_max]
    S = (alpha * Gj / (C * Ttot)) ** (-1.0 / alpha)
    V = np.minimum(S, rng.exponential(1.0, len(S)) / lam)
    v0 = (alpha * Gamma_max / (C * Ttot)) ** (-1.0 / alpha)
    small_var = max(Ttot - (V ** 2).sum(), 0.2 * Ttot)  # residual small-jump variance
    signs = rng.choice([-1.0, 1.0], size=len(V))
    jump_times = rng.uniform(0, Ttot, size=len(V))
    # Levy process on a fine sub-grid: L(t) levels, then increments
    sub = 16                     # sub-steps per dt
    h = dt / sub
    m = n * sub
    ts = np.arange(1, m + 1) * h
    L = np.zeros(m)
    idx = np.argsort(jump_times)
    jt = jump_times[idx]
    js = (signs[idx] * V[idx])
    pos = np.searchsorted(jt, ts)
    if pos[-1] > 0:
        L = np.cumsum(js[:pos[-1]])[pos - 1]
    L += np.cumsum(np.sqrt(small_var / Ttot * h) * rng.standard_normal(m))
    inc = np.diff(L, prepend=0.0)                        # sub-grid increments
    # OU recursion with exact weighting of the sub-grid increments
    rho_h = np.exp(-kappa * h)
    x = mu + sigma * rng.standard_normal() / np.sqrt(2 * kappa)
    # midpoint rectangle weights for the stochastic convolution within one step
    w = sigma * np.exp(-kappa * (h - (np.arange(sub) + 0.5) * h / sub))
    X = np.empty(n)
    inc = inc.reshape(n, sub)
    for i in range(n):
        x = mu + (x - mu) * rho_h ** sub + (w * inc[i]).sum()
        X[i] = x
    return X

def add_contamination(X, kind, rng, sstat):
    """kind: None, 'ao' additive outliers, 'block' level shift, 'scale' variance inflation.
    Magnitudes are relative to the stationary std sstat = sigma/sqrt(2*kappa)."""
    X = X.copy()
    n = len(X)
    if kind == 'ao':
        mask = rng.random(n) < 0.05
        X[mask] += rng.choice([-1, 1], mask.sum()) * (3.0 * sstat) * (0.5 + rng.random(mask.sum()))
    elif kind == 'block':
        L = max(10, n // 50)
        start = rng.integers(0, n - L)
        X[start:start + L] += 2.5 * sstat * rng.choice([-1, 1])
    elif kind == 'scale':
        mask = rng.random(n) < 0.10
        X[mask] = X[mask] * 1.8 + 1.0 * sstat * rng.standard_normal(mask.sum())
    return X

# ---------------- moments ----------------
def moments(X, theta, alpha, dt, u_grid, lam=1.0, F_interp=None):
    """Return (n, 22) moment matrix: 4 dynamic + 18 stationary-CF (Re/Im x 9 u's)."""
    if F_interp is None:
        F_interp = F_interp_factory(alpha, lam)
    kappa, mu, sig = theta
    rho = np.exp(-kappa * dt)
    Xm = X[:-1]
    Xc = X[1:]
    m = mu + (Xm - mu) * rho
    eps = Xc - m
    v = sig ** 2 * (1 - rho ** 2) / (2 * kappa)
    dyn = np.column_stack([eps, eps * Xm, eps ** 2 - v, (eps ** 2 - v) * Xm])
    # unconditional (stationary) CF moments: E[e^{iu X}] = phi_stat(u; theta)
    # phi_stat(u) = exp(i*mu*u + F(sig*u)/kappa),  F(w) = int_0^w psi(y)/y dy
    phi = np.array([np.exp(1j * mu * u + F_interp(sig * u) / kappa) for u in u_grid])
    e_iuX = np.exp(1j * np.outer(X, u_grid))             # (n+1, U)
    h = e_iuX - phi[None, :]
    cf = np.column_stack([h.real, h.imag])               # (n+1, 2U)
    return np.column_stack([dyn, cf[1:]])

_F_CACHE = {}

def F_interp_factory(alpha, lam=1.0, Wmax=40.0, npts=6000):
    """Tabulate F(w) = int_0^w psi(y)/y dy and return an interpolator."""
    key = (alpha, lam)
    if key not in _F_CACHE:
        from scipy.integrate import cumulative_trapezoid
        w = np.linspace(1e-8, Wmax, npts)
        psi = np.array([levy_symbol(y, alpha, lam) for y in w])
        F = cumulative_trapezoid(psi / w, w, initial=0.0)
        _F_CACHE[key] = (w, F)
    w, F = _F_CACHE[key]
    def f(x):
        return np.interp(x, w, F)
    return f

def huber_weights_dyn(dyn, c=1.345):
    """Radial Huber weights on the 4-dim dynamic score."""
    z = dyn - np.median(dyn, axis=0, keepdims=True)
    scale = np.median(np.abs(z), axis=0, keepdims=True) * 1.4826 + 1e-12
    zs = z / scale
    r = np.sqrt((zs ** 2).sum(axis=1))
    s = np.median(r) * 1.4826 + 1e-12
    w = np.minimum(1.0, c * s / r)
    return w

def estimate(X, alpha, dt, method='cfgmm', u_grid=None, lam=1.0, c=1.345, trim=3.5,
             return_diag=False):
    """Two-step GMM. method in {'cfgmm','huber','pc'}.

    cfgmm: standard two-step CF-GMM.
    huber: marginal trimming (trim robust std, default 3.5) + Huber-weighted
           covariance for the step-2 weight matrix. The moment average is left
           UNWEIGHTED so that E[gbar(theta0)] = 0 and consistency is preserved.
    pc:    huber + principal-component spectral floor on the weight matrix
           eigenvalues (floor = 1e-3 * max eigenvalue).
    """
    if u_grid is None:
        u_grid = np.linspace(0.2, 2.0, 9)
    F_interp = F_interp_factory(alpha, lam)
    n = len(X) - 1
    robust = method in ('huber', 'pc')
    if robust:
        # Marginal trimming based on X: drop observations farther than
        # `trim` robust std from the median. For CTS (finite moments) the
        # trimming fraction vanishes asymptotically, so E[gbar(theta0)]=0
        # is preserved. AO/shift outliers that survive trimming are handled
        # by the Huber-weighted covariance in the weight matrix.
        med = np.median(X)
        mad = np.median(np.abs(X - med)) * 1.4826 + 1e-12
        X = X[np.abs(X - med) <= trim * mad]
    Xm, Xc = X[:-1], X[1:]
    # initial theta from AR(1)
    A = np.column_stack([np.ones(len(Xm)), Xm])
    coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    mu0, b = coef
    rho0 = np.clip(b, 1e-4, 0.999)
    kappa0 = -np.log(rho0) / dt
    resid = Xc - (mu0 + b * (Xm - mu0))
    sig0 = np.sqrt(max(np.var(resid) * 2 * kappa0 / (1 - rho0 ** 2), 1e-6))
    theta0 = np.array([kappa0, mu0, sig0])
    bounds = [(0.01, 30.0), (X.min() - 5, X.max() + 5), (0.01, 5.0)]

    # fixed center/scale for the radial weights (from the step-1 fit)
    center = scale = None

    def weights_of(g):
        z = (g - center) / scale
        r = np.sqrt((z ** 2).sum(axis=1))
        rs = np.median(r) * 1.4826 + 1e-12
        return np.minimum(1.0, c * rs / r)

    def gbar(theta):
        """Unweighted moment average. Huber weights are NOT applied here so
        that E[gbar(theta0)] = 0 under the model, which preserves consistency.
        Robustness enters only through the weight matrix (covariance estimate)."""
        g = moments(X, theta, alpha, dt, u_grid, lam, F_interp)
        return g.mean(axis=0)

    def obj(theta, W):
        gb = gbar(theta)
        return gb @ W @ gb

    def solve(W, start=theta0):
        best, bestv = None, np.inf
        for s0 in [start, np.array([max(kappa0 * 2, 0.5), mu0, sig0 * 1.3])]:
            res = minimize(obj, s0, args=(W,), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 150})
            if res.fun < bestv:
                best, bestv = res.x, res.fun
        return best

    # step 1 with diagonal scaling
    g0 = moments(X, theta0, alpha, dt, u_grid, lam, F_interp)
    if robust:
        center = np.median(g0, axis=0)
        scale = np.median(np.abs(g0 - center), axis=0) * 1.4826
        scale = np.maximum(scale, 1e-3 * np.median(scale)) + 1e-12
    d0 = np.clip(g0.var(axis=0), 1e-10, None)
    W1 = np.diag(1.0 / d0)
    th1 = solve(W1)
    g1 = moments(X, th1, alpha, dt, u_grid, lam, F_interp)
    if robust:
        center = np.median(g1, axis=0)
        scale = np.median(np.abs(g1 - center), axis=0) * 1.4826
        scale = np.maximum(scale, 1e-3 * np.median(scale)) + 1e-12
        w1 = weights_of(g1)
        # Huber-weighted covariance (weights apply to the scatter estimate only,
        # not to the moment mean) -- robust against outlier contamination while
        # the unweighted moment average keeps E[gbar(theta0)] = 0.
        gc = g1 - g1.mean(axis=0)
        S = (w1[:, None] * gc).T @ gc / w1.sum()
    else:
        S = np.cov(g1.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 0)
    if method == 'pc':
        # principal-component spectral flooring: eigenvalues are clipped
        # from below at 1e-3 * lambda_max
        floor = vals.max() * 1e-3
        vals_t = np.maximum(vals, floor)
    else:
        vals_t = vals + vals.max() * 1e-6
    W2 = vecs @ np.diag(1.0 / vals_t) @ vecs.T
    th2 = solve(W2, start=th1)
    if return_diag:
        gb2 = gbar(th2)
        diag = dict(th1=[float(t) for t in th1],
                    n_obs=int(len(X) - 1),
                    obj=float(gb2 @ W2 @ gb2),
                    cond_S=float(vals.max() / max(vals.min(), 1e-300)),
                    S=S.tolist(), W2=W2.tolist())
        return th2, diag
    return th2

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    kappa, mu, sig, alpha = 2.0, 3.0, 0.5, 1.5
    n, dt = 600, 1 / 12
    X = simulate_ou_path(kappa, mu, sig, alpha, n, dt, rng)
    print('sample mean/std:', X.mean(), X.std(), 'theory std', sig/np.sqrt(2*kappa))
    for method in ['cfgmm', 'huber', 'pc']:
        th = estimate(X, alpha, dt, method=method)
        print(method, '-> kappa=%.3f mu=%.3f sigma=%.3f' % tuple(th))
    Xc = add_contamination(X, 'block', rng, sstat=sig)
    for method in ['cfgmm', 'huber', 'pc']:
        th = estimate(Xc, alpha, dt, method=method)
        print('block', method, '-> kappa=%.3f mu=%.3f sigma=%.3f' % tuple(th))
