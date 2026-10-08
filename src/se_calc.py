# -*- coding: utf-8 -*-
"""Asymptotic standard errors for the VIX full-sample two-step GMM fits.
Sandwich form: V = (G'WG)^-1 (G'W S W G) (G'WG)^-1 / n, with G the numerical
Jacobian of the (trimmed) moment average, W the second-step weight matrix and
S the covariance actually used in estimation. Writes se_vix.json.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cts_ou import estimate, moments, F_interp_factory
from empirical import load_vix, DT, METHODS

ALPHA, LAM = 1.5, 1.0


def make_gbar(X):
    """gbar(theta) with the same 3.5-MAD trimming as inside estimate()."""
    med = np.median(X)
    mad = np.median(np.abs(X - med)) * 1.4826 + 1e-12
    Xt = X[np.abs(X - med) <= 3.5 * mad]
    F = F_interp_factory(ALPHA, LAM)
    ug = np.linspace(0.2, 2.0, 9)

    def gbar(theta):
        return moments(Xt, theta, ALPHA, DT, ug, LAM, F).mean(axis=0)
    return gbar


def se_of(th, gbar, W, S, n):
    G = np.empty((22, 3))
    for j in range(3):
        h = max(1e-5, 1e-4 * abs(th[j]))
        e = np.zeros(3)
        e[j] = h
        G[:, j] = (gbar(th + e) - gbar(th - e)) / (2 * h)
    A = G.T @ W @ G
    B = G.T @ W @ S @ W @ G
    V = np.linalg.inv(A) @ B @ np.linalg.inv(A) / n
    return np.sqrt(np.diag(V)), float(np.linalg.cond(A))


def main():
    X, dates, v = load_vix()
    out = {}
    for m in METHODS:
        th, diag = estimate(X, ALPHA, DT, method=m, return_diag=True)
        W = np.array(diag['W2'])
        S = np.array(diag['S'])
        se, cond_A = se_of(np.asarray(th), make_gbar(X), W, S, diag['n_obs'])
        out[m] = dict(theta=[float(t) for t in th],
                      se=[float(s) for s in se],
                      tstat=[float(t / s) for t, s in zip(th, se)],
                      cond_GWG=cond_A, n_obs=diag['n_obs'])
        with np.printoptions(precision=4, suppress=True):
            print(m, 'theta=', np.asarray(th), '\n  se=', se,
                  ' t=', np.asarray(out[m]['tstat']),
                  ' cond(GWG)=%.2e' % cond_A, flush=True)
    with open('se_vix.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('saved se_vix.json', flush=True)


if __name__ == '__main__':
    main()
