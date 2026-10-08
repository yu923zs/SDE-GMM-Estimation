# -*- coding: utf-8 -*-
"""Generate latex_source/figures/fig2.pdf and fig3.pdf from vix_results.json."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

methods = ['cfgmm', 'huber', 'pc']
MNAME = {'cfgmm': 'CF-GMM', 'huber': 'Huber-CF-GMM', 'pc': 'PC-Huber-CF-GMM'}
COLOR = {'cfgmm': '#8C9BAA', 'huber': '#C0704A', 'pc': '#3E6B8C'}
OUT = '../figures'

vix = json.load(open('../results/vix_results.json'))
df = pd.read_csv('../data/vix_fred.csv' if os.path.exists('../data/vix_fred.csv') else 'vix_fred.csv')
df['v'] = pd.to_numeric(df['VIXCLS'], errors='coerce')
df = df.dropna()
dates_v = pd.to_datetime(df['observation_date'])
logv = np.log(df['v'].values)

# ---------- Figure 2: log VIX + rolling sigma paths ----------
fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=False,
                         gridspec_kw={'height_ratios': [1, 1]})
ax = axes[0]
ax.plot(dates_v, logv, lw=0.5, color='#3E6B8C')
ax.set_ylabel(r'$\log(\mathrm{VIX})$')
ax.set_title('Log VIX series (1990--2026)', fontsize=11)
ax.grid(alpha=0.3)

ax = axes[1]
for m in methods:
    rd = pd.to_datetime(vix['rolling'][m]['dates'])
    s = np.array(vix['rolling'][m]['sigma'], dtype=float)
    ax.plot(rd, s, lw=1.2, label=MNAME[m], color=COLOR[m])
ax.set_ylabel(r'Rolling-window $\hat{\sigma}$')
ax.set_title(r'Rolling-window (504 trading days) $\sigma$ estimates', fontsize=11)
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '/fig2.pdf', bbox_inches='tight')
plt.close(fig)

# ---------- Figure 3: shock sensitivity ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
vals = [vix['shock'][m]['mean'] * 100 for m in methods]
bars = ax.bar([MNAME[m] for m in methods], vals,
              color=[COLOR[m] for m in methods], edgecolor='white', width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f'{v:.1f}%',
            ha='center', fontsize=10)
ax.set_ylabel(r'Mean relative change of $\hat{\sigma}$ (%)')
ax.set_title(r'Sensitivity of $\hat{\sigma}$ to artificial shocks', fontsize=11)
ax.set_ylim(0, max(vals) * 1.2)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + '/fig3.pdf', bbox_inches='tight')
plt.close(fig)
print('fig2.pdf and fig3.pdf saved')
