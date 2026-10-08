# -*- coding: utf-8 -*-
"""VIX empirical analysis."""
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_ou import simulate_ou_path, estimate, moments, F_interp_factory

ALPHA, DT = 1.5, 1 / 252
METHODS = ['cfgmm', 'huber', 'pc']

def load_vix():
    csv = '../data/vix_fred.csv' if os.path.exists('../data/vix_fred.csv') else 'vix_fred.csv'
    df = pd.read_csv(csv)
    df['v'] = pd.to_numeric(df['VIXCLS'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    v = df['v'].values
    X = np.log(v)
    dates = pd.to_datetime(df['observation_date'])
    return X, dates, v

def desc_stats(v, X):
    from scipy import stats
    d = dict(n=len(v), mean=float(v.mean()), std=float(v.std()),
             min=float(v.min()), max=float(v.max()),
             log_mean=float(X.mean()), log_std=float(X.std()),
             log_skew=float(stats.skew(X)), log_kurt=float(stats.kurtosis(X)),
             start=str(pd.Timestamp('1990-01-02').date()), end='2026-09-21')
    return d

def one_step_rmse(X, theta):
    kappa, mu, sig = theta
    rho = np.exp(-kappa * DT)
    pred = mu + (X[:-1] - mu) * rho
    err = X[1:] - pred
    return float(np.sqrt(np.mean(err ** 2)))

def rolling_task(args):
    i, X, W = args
    xw = X[i:i + W]
    out = {}
    for m in METHODS:
        try:
            out[m] = [float(t) for t in estimate(xw, ALPHA, DT, method=m)]
        except Exception:
            out[m] = [float('nan')] * 3
    return i, out

def shock_task(args):
    seed, X, full_cache = args
    r = np.random.default_rng(seed)
    sstat = full_cache['huber']['theta'][2] / np.sqrt(2 * full_cache['huber']['theta'][0])
    base = {m: full_cache[m]['theta'][2] for m in METHODS}
    i0 = r.integers(252, len(X) - 252)
    sgn = r.choice([-1.0, 1.0])
    Xs = X.copy()
    Xs[i0:i0 + 5] += sgn * 3.0 * sstat
    out = {}
    for m in METHODS:
        try:
            th = estimate(Xs, ALPHA, DT, method=m)
            out[m] = abs(th[2] - base[m]) / base[m]
        except Exception:
            out[m] = np.nan
    return out

def main():
    X, dates, v = load_vix()
    print('n =', len(X), flush=True)
    if os.path.exists('vix_stage1.json'):
        res = json.load(open('vix_stage1.json'))
    else:
        res = {'desc': desc_stats(v, X)}

    # ---- full-sample estimation ----
    if 'full' not in res:
        full = {}
        for m in METHODS:
            th = estimate(X, ALPHA, DT, method=m)
            full[m] = dict(theta=[float(t) for t in th], rmse=one_step_rmse(X, th))
            print('full', m, full[m], flush=True)
        res['full'] = full
        # condition number of S at the cfgmm fit (22 moment conditions)
        F = F_interp_factory(ALPHA)
        u_grid = np.linspace(0.2, 2.0, 9)
        g = moments(X, np.array(full['cfgmm']['theta']), ALPHA, DT, u_grid, 1.0, F)
        S = np.cov(g.T)
        ev = np.linalg.eigvalsh(S)
        res['cond_S'] = dict(max=float(ev.max()), min=float(ev.min()), cond=float(ev.max() / max(ev.min(), 1e-300)))
        print('cond(S):', res['cond_S'], flush=True)
        with open('vix_stage1.json', 'w') as f:
            json.dump(res, f)
    full = res['full']

    # ---- artificial shocks (parallel) ----
    if 'shock' not in res:
        with Pool(min(10, os.cpu_count() - 1)) as pool:
            shock_list = pool.map(shock_task, [(11 + 97 * r, X, full) for r in range(60)])
        shock_res = {m: [s[m] for s in shock_list] for m in METHODS}
        res['shock'] = {m: dict(mean=float(np.nanmean(shock_res[m])), q90=float(np.nanquantile(shock_res[m], 0.9)),
                                med=float(np.nanmedian(shock_res[m]))) for m in METHODS}
        print('shock:', res['shock'], flush=True)
        with open('vix_stage1.json', 'w') as f:
            json.dump(res, f)

    # ---- rolling-window sigma (checkpointed, resumable) ----
    W, step = 504, 21
    idxs = list(range(0, len(X) - W, step))
    print('rolling windows:', len(idxs), flush=True)
    CKPT = 'rolling_ckpt.json'
    ckpt = {}
    if os.path.exists(CKPT):
        ckpt = json.load(open(CKPT))
    todo = [i for i in idxs if str(i) not in ckpt]
    print('todo:', len(todo), flush=True)
    if todo:
        with Pool(min(10, os.cpu_count() - 1)) as pool:
            for done, (i, out) in enumerate(pool.imap_unordered(rolling_task, [(i, X, W) for i in todo])):
                ckpt[str(i)] = out
                if done % 25 == 0:
                    json.dump(ckpt, open(CKPT, 'w'))
                    print('rolling', done, '/', len(todo), flush=True)
        json.dump(ckpt, open(CKPT, 'w'))
    roll_partial = {i: ckpt[str(i)] for i in idxs}
    roll_out = {}
    for m in METHODS:
        sig_path = np.array([roll_partial[i][m][2] for i in idxs], dtype=float)
        kap_path = np.array([roll_partial[i][m][0] for i in idxs], dtype=float)
        roll_out[m] = dict(
            dates=[str(dates[i + W - 1].date()) for i in idxs],
            sigma=[None if np.isnan(s) else float(s) for s in sig_path],
            kappa=[None if np.isnan(s) else float(s) for s in kap_path],
            sigma_std=float(np.nanstd(sig_path)), sigma_iqr=float(np.nanpercentile(sig_path, 75) - np.nanpercentile(sig_path, 25)))
    res['rolling'] = roll_out
    print('rolling sigma std:', {m: round(roll_out[m]['sigma_std'], 4) for m in METHODS}, flush=True)
    print('rolling sigma IQR:', {m: round(roll_out[m]['sigma_iqr'], 4) for m in METHODS}, flush=True)

    with open('vix_results.json', 'w') as f:
        json.dump(res, f)
    print('saved vix_results.json', flush=True)

roll_partial = {}

if __name__ == '__main__':
    main()
