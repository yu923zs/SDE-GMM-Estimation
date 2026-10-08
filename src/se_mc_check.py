# -*- coding: utf-8 -*-
"""Validate the asymptotic sandwich SEs against MC sampling dispersion.
Simulates CTS-OU (kappa=2, mu=3, sigma=0.5, alpha=1.5, lam=1, dt=1/252,
n=1500) for 100 reps, fits the huber estimator, and compares the mean
asymptotic SE with the empirical std of the estimates (+ 95% CI coverage).
"""
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_ou import simulate_ou_path, estimate
from se_calc import se_of, make_gbar
from empirical import DT

K, MU, SIG, N, R = 2.0, 3.0, 0.5, 1500, 100


def rep(seed):
    rng = np.random.default_rng(seed)
    X = simulate_ou_path(K, MU, SIG, 1.5, N, DT, rng)
    try:
        th, diag = estimate(X, 1.5, DT, method='huber', return_diag=True)
        W, S = np.array(diag['W2']), np.array(diag['S'])
        se, _ = se_of(np.asarray(th), make_gbar(X), W, S, diag['n_obs'])
        return list(th), list(se)
    except Exception:
        return None


def main():
    with Pool(min(10, os.cpu_count() - 1)) as pool:
        out = [r for r in pool.map(rep, [7000 + i for i in range(R)]) if r]
    th = np.array([o[0] for o in out])
    se = np.array([o[1] for o in out])
    n_ok = len(out)
    emp_std = th.std(axis=0, ddof=1)
    mean_se = se.mean(axis=0)
    bias = th.mean(axis=0) - np.array([K, MU, SIG])
    cover = ((th - 1.96 * se <= [K, MU, SIG]) &
             (th + 1.96 * se >= [K, MU, SIG])).mean(axis=0)
    names = ['kappa', 'mu', 'sigma']
    truth = [K, MU, SIG]
    print(f'reps used: {n_ok}/{R}, n={N}, method=huber')
    print(f"{'par':<7}{'truth':>8}{'bias':>9}{'emp_std':>9}{'mean_SE':>9}"
          f"{'ratio':>7}{'cov95':>8}")
    for j, nm in enumerate(names):
        print(f'{nm:<7}{truth[j]:>8.3f}{bias[j]:>9.3f}{emp_std[j]:>9.4f}'
              f'{mean_se[j]:>9.4f}{mean_se[j]/emp_std[j]:>7.2f}{cover[j]:>8.2f}')


if __name__ == '__main__':
    main()
