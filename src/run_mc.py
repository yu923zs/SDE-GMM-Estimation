# -*- coding: utf-8 -*-
"""Monte Carlo study: 4 scenarios x (clean + 3 contamination) x 3 estimators.
Usage: python run_mc.py <scen_idx> <contam_idx>   -> runs one cell, appends to mc_results.json
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from cts_ou import simulate_ou_path, add_contamination, estimate

SCEN = [
    {'name': 'S1', 'kappa': 1.0, 'mu': 3.0, 'sig': 0.3},
    {'name': 'S2', 'kappa': 2.0, 'mu': 3.0, 'sig': 0.5},
    {'name': 'S3', 'kappa': 4.0, 'mu': 3.3, 'sig': 0.6},
    {'name': 'S4', 'kappa': 2.5, 'mu': 2.7, 'sig': 0.8},
]
CONTAM = ['clean', 'ao', 'block', 'scale']
METHODS = ['cfgmm', 'huber', 'pc']
ALPHA, DT, N, R = 1.5, 1 / 12, 600, 300
OUT = 'mc_results.json'

def run_rep(args):
    seed, kappa, mu, sig, contam = args
    rng = np.random.default_rng(seed)
    X = simulate_ou_path(kappa, mu, sig, ALPHA, N, DT, rng)
    if contam != 'clean':
        X = add_contamination(X, contam, rng, sstat=sig / np.sqrt(2 * kappa))
    out = np.full((len(METHODS), 3), np.nan)
    for j, mth in enumerate(METHODS):
        try:
            out[j] = estimate(X, ALPHA, DT, method=mth)
        except Exception:
            pass
    return out

def main():
    si, ci = int(sys.argv[1]), int(sys.argv[2])
    sc, contam = SCEN[si], CONTAM[ci]
    key = f"{sc['name']}|{contam}"
    t0 = time.time()
    tasks = [(20260923 + 1000003 * (si * len(CONTAM) + ci) + r,
              sc['kappa'], sc['mu'], sc['sig'], contam) for r in range(R)]
    with Pool(min(10, os.cpu_count() - 1)) as pool:
        ests = np.array(pool.map(run_rep, tasks))
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    data.setdefault('meta', {'scen': SCEN, 'contam': CONTAM, 'methods': METHODS,
                             'alpha': ALPHA, 'dt': DT, 'n': N, 'R': R})
    data.setdefault('results', {})[key] = ests.tolist()
    with open(OUT, 'w') as f:
        json.dump(data, f)
    print(f'{key}: done in {time.time()-t0:.0f}s, fail rates',
          np.isnan(ests).mean(axis=0).round(3).tolist())

if __name__ == '__main__':
    main()
