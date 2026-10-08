# -*- coding: utf-8 -*-
"""Hansen J overidentification test for the VIX full-sample fits.
22 moment conditions, 3 parameters -> df = 19.
J is computed as n * (minimized second-step objective), i.e. with the same
weight matrix used in estimation. theta is re-estimated (deterministic), and
cross-checked against the stored vix_stage1.json values.
Writes j_test.json.
"""
import json
import os
import sys
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_ou import estimate
from empirical import load_vix, DT, METHODS

DF = 22 - 3


def main():
    X, dates, v = load_vix()
    stored = json.load(open('vix_stage1.json'))['full']
    out = {'df': DF, 'methods': {}}
    for m in METHODS:
        th, diag = estimate(X, 1.5, DT, method=m, return_diag=True)
        J = diag['n_obs'] * diag['obj']
        p = float(chi2.sf(J, DF))
        st = np.array(stored[m]['theta'])
        maxdiff = float(np.max(np.abs(np.asarray(th) - st)))
        # effective-rank variant: pseudo-inverse of S ignoring the floored
        # directions (informative for the near-singular cfgmm covariance)
        out['methods'][m] = dict(
            theta=[float(t) for t in th], J=float(J), p=p,
            n_obs=diag['n_obs'], cond_S=diag['cond_S'],
            maxdiff_vs_stored=maxdiff)
        print(f"{m}: J={J:.2f}  p={p:.4g}  n={diag['n_obs']}  "
              f"cond(S)={diag['cond_S']:.2e}  maxdiff_vs_stored={maxdiff:.2e}",
              flush=True)
    with open('j_test.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('saved j_test.json', flush=True)


if __name__ == '__main__':
    main()
