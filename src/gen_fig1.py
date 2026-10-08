# -*- coding: utf-8 -*-
"""Generate latex_source/figures/fig1.pdf from mc_results.json:
RMSE of sigma (left) and kappa (right), averaged over the four scenarios,
under four contamination mechanisms, log scale."""
import json
import numpy as np
import matplotlib.pyplot as plt

mc = json.load(open('../results/mc_results.json'))
scen = mc['meta']['scen']
res = mc['results']
truth = {s['name']: [s['kappa'], s['mu'], s['sig']] for s in scen}
contam = ['clean', 'ao', 'block', 'scale']
methods = ['cfgmm', 'huber', 'pc']
MNAME = {'cfgmm': 'CF-GMM', 'huber': 'Huber-CF-GMM', 'pc': 'PC-Huber-CF-GMM'}
CNAME = {'clean': 'Clean', 'ao': 'Additive outliers', 'block': 'Block shift',
         'scale': 'Scale contamination'}
COLOR = {'cfgmm': '#8C9BAA', 'huber': '#C0704A', 'pc': '#3E6B8C'}


def mean_rmse(ct, m, par):
    vals = []
    for s in scen:
        E = np.array(res['%s|%s' % (s['name'], ct)])
        err = E - np.array(truth[s['name']])
        vals.append(np.sqrt((err ** 2).mean(axis=0))[m if False else methods.index(m), par])
    return float(np.mean(vals))


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, par, title in [(axes[0], 2, r'RMSE of $\sigma$'),
                       (axes[1], 0, r'RMSE of $\kappa$')]:
    x = np.arange(len(contam))
    w = 0.26
    for j, m in enumerate(methods):
        rmses = [mean_rmse(ct, m, par) for ct in contam]
        ax.bar(x + (j - 1) * w, rmses, width=w, label=MNAME[m], color=COLOR[m],
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CNAME[c] for c in contam], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig('../figures/fig1.pdf', bbox_inches='tight')
print('fig1.pdf saved')
