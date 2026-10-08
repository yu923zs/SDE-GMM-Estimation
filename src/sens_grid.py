# -*- coding: utf-8 -*-
"""Sensitivity of VIX full-sample estimates to the fixed tail index alpha and
tempering parameter lambda. Grid: alpha in {1.2,1.5,1.8} x lambda in {0.5,1,2}
x 3 estimators, everything else identical to the paper's full-sample stage
(DT=1/252, same u_grid, same trimming). Writes sens_grid.json + prints table.
"""
import json
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_ou import estimate
from empirical import load_vix, one_step_rmse, DT, METHODS

ALPHAS = [1.2, 1.5, 1.8]
LAMS = [0.5, 1.0, 2.0]


def task(args):
    alpha, lam, m, X = args
    try:
        th = estimate(X, alpha, DT, method=m, lam=lam)
        return dict(alpha=alpha, lam=lam, method=m,
                    theta=[float(t) for t in th],
                    rmse=one_step_rmse(X, th), error=None)
    except Exception as e:
        return dict(alpha=alpha, lam=lam, method=m, theta=None,
                    rmse=None, error=f'{type(e).__name__}: {e}')


def main():
    X, dates, v = load_vix()
    print('n =', len(X), flush=True)
    tasks = [(a, l, m, X) for a in ALPHAS for l in LAMS for m in METHODS]
    print('tasks:', len(tasks), flush=True)
    with Pool(min(10, os.cpu_count() - 1)) as pool:
        results = pool.map(task, tasks)

    out = {'alphas': ALPHAS, 'lambdas': LAMS, 'methods': METHODS,
           'dt': DT, 'grid': {}}
    for r in results:
        key = f"a{r['alpha']}_l{r['lam']}"
        out['grid'].setdefault(key, {})[r['method']] = {
            'theta': r['theta'], 'rmse': r['rmse'], 'error': r['error']}

    # ---- printable table ----
    hdr = f"{'alpha':>5} {'lam':>4} | {'method':<16} {'kappa':>8} {'mu':>8} {'sigma':>8} {'rmse1':>8}"
    print(hdr)
    print('-' * len(hdr))
    for a in ALPHAS:
        for l in LAMS:
            key = f'a{a}_l{l}'
            for m in METHODS:
                e = out['grid'][key][m]
                if e['theta'] is None:
                    print(f'{a:>5} {l:>4} | {m:<16} FAILED: {e["error"]}')
                else:
                    k, mu, s = e['theta']
                    print(f'{a:>5} {l:>4} | {m:<16} {k:>8.3f} {mu:>8.3f} {s:>8.3f} {e["rmse"]:>8.4f}')

    with open('sens_grid.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('saved sens_grid.json', flush=True)


if __name__ == '__main__':
    main()
