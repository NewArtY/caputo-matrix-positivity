"""Candidate figures of the neural benchmarks (B1, B2, B6, B7, surrogate).

Reads only figures/data/Neural_*.csv and tables/data/Table_*.csv written by scripts/aggregate_neural.py and writes
vector PDFs to figures/candidates/. Usage: python scripts/plot_neural.py [--preview <dir for PNG copies>]

Colour: identity is carried by at most three hues of one validated categorical order (blue, orange, aqua) together
with a marker shape or line style, so that every figure also reads in grey-scale print; a fourth series is drawn in a
neutral grey with its own line style; the classical reference (L1 time stepper, least squares, exact solution) is
always drawn in ink. Magnitude on a grid uses a single-hue ramp.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, 'figures', 'data'); TD = os.path.join(ROOT, 'tables', 'data'); OUT = os.path.join(ROOT, 'figures', 'candidates')
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3 (validated on a white surface)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'font.size': 8,
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'axes.linewidth': 0.6,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'grid.linestyle': '-', 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 300})
W2 = 190 / 25.4
MAPS = ['unc', 'herm', 'lin', 'factor', 'add']
MAP_LABEL = {'unc': 'uncon-\nstrained', 'herm': 'Hermitian', 'lin': 'linear\nHerm., tr=1', 'factor': 'PSD\nfactor', 'add': 'PSD\nadditive',
             'gate2': 'PSD\ntwo-gate', 'gate2b': 'PSD\nblock gate', 'bloch': 'Bloch\nhead'}
MAP_SHORT = {'unc': 'uncon-\nstrained', 'herm': 'Herm.', 'lin': 'linear\ntr = 1', 'factor': 'PSD\nfactor', 'add': 'PSD\nadditive',
             'gate2': 'PSD\ntwo-gate', 'gate2b': 'PSD\nblock', 'bloch': 'Bloch'}
STATE_LABEL = {'mixed': 'mixed', 'nearly_pure': 'nearly pure', 'pure': 'pure'}


def read(name, folder=FD):
    p = os.path.join(folder, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})      # no date: identical bytes on regeneration
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True); fig.savefig(os.path.join(PREVIEW, name + '.png'), bbox_inches='tight', dpi=170)
    plt.close(fig); print('wrote', name + '.pdf')


def panel(ax, s, dx=-0.14):
    ax.text(dx, 1.05, f'({s})', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def strip(ax, x, vals, color, marker, filled=True, width=0.09):
    """Per-seed values as small dots and their median as a larger marker with a surface ring."""
    vals = np.asarray(vals, dtype=float); off = np.linspace(-width, width, len(vals)) if len(vals) > 1 else [0.0]
    ax.plot(x + np.asarray(off), vals, ls='none', marker=marker, ms=2.4, mfc=color if filled else 'white', mec=color, mew=0.5, alpha=0.55)
    ax.plot([x], [np.median(vals)], ls='none', marker=marker, ms=6.2, mfc=color if filled else 'white', mec='white' if filled else color,
            mew=0.9, zorder=4)


# ---------------------------------------------------------------------------------------------- B2

def fig_b2_factorial():
    runs = read('Neural_B2_runs.csv'); st = read('Neural_B2_stepper.csv')
    if runs is None:
        return
    runs = runs[(runs.stage == 'main') & (runs.status == 'ok') & runs['map'].isin(MAPS)]
    series = [('soft-tuned', 'soft IC, tuned weight', BLUE, 's'), ('hard-exp', r'hard IC, gate $1-e^{-\beta\tau}$', ORANGE, '^'),
              ('hard-pow', r'hard IC, gate $1-e^{-\beta\tau^{\alpha}}$', AQUA, 'o')]
    alphas = sorted(runs.alpha.unique()); states = [s for s in STATE_LABEL if s in set(runs.state)]
    fig, axs = plt.subplots(len(states), len(alphas), figsize=(W2, 2.05 * len(states) + 0.5), sharex=True, sharey='col', squeeze=False)
    for i, s in enumerate(states):
        for j, a in enumerate(alphas):
            ax = axs[i, j]; d = runs[(runs.state == s) & (runs.alpha == a)]
            for k, m in enumerate(MAPS):
                for q, (ig, _, c, mk) in enumerate(series):
                    v = d[(d['map'] == m) & (d.ic_gate == ig)].rel_fro.values
                    if len(v):
                        strip(ax, k + 0.27 * (q - 1), v, c, mk)
            f = st[(st.alpha == a) & (st.state == s) & (st.N == 128)].rel_fro.iloc[0]
            ax.axhline(f, color=INK, lw=1.0); ax.set_yscale('log'); ax.grid(axis='x', visible=False)
            if j == len(alphas) - 1 or True:
                ax.annotate('L1 stepper', (len(MAPS) - 0.55, f), xytext=(0, 2.5), textcoords='offset points', ha='right', va='bottom', fontsize=6, color=INK)
            if i == 0:
                ax.set_title(rf'$\alpha={a}$')
            if j == 0:
                ax.set_ylabel(f'{STATE_LABEL[s]}\nrel. Frobenius error')
            ax.set_xticks(range(len(MAPS))); ax.set_xticklabels([MAP_SHORT[m] for m in MAPS], fontsize=6)
    h = [Line2D([], [], ls='none', marker=mk, ms=6, mfc=c, mec='white', label=lab) for _, lab, c, mk in series]
    h.append(Line2D([], [], color=INK, lw=1.0, label='implicit L1 time stepper, same mesh'))
    fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.035, 1, 1)); save(fig, 'Neural_B2_factorial')


def fig_b2_effects(metric='rel_fro', name='Neural_B2_effects',
                   xlabel='ratio of relative Frobenius errors, paired by seed and initial state (values below 1 favour the first variant)'):
    p = read('Table_B2_paired.csv', TD)
    if p is None:
        return
    p = p[(p.states == 'all') & (p.alpha != 'all') & (p.metric == metric)].copy(); p['alpha'] = p.alpha.astype(float)
    first = lambda r: r.a.split('/')[0]
    groups = [('gate: pow vs exp', r'(a) gate $1-e^{-\beta\tau^\alpha}$ relative to $1-e^{-\beta\tau}$, hard IC', first),
              ('initial condition: hard(exp) vs soft(tuned)', r'(b) hard IC with gate $1-e^{-\beta\tau}$ relative to tuned soft IC', first),
              ('map vs unconstrained (hard-pow)', r'(c) output map relative to unconstrained output (hard IC, $\tau^\alpha$ gate)', first),
              ('map vs map (hard-pow)', r'(d) PSD map relative to linear head (hard IC, $\tau^\alpha$ gate)',
               lambda r: r.a.split('/')[0] if r.b.startswith('lin/') else None)]
    alphas = sorted(p.alpha.unique()); cols = dict(zip(alphas, (BLUE, ORANGE, AQUA))); mks = dict(zip(alphas, ('s', '^', 'o')))
    short = {'unc': 'unconstrained', 'herm': 'Hermitian', 'lin': 'linear, Herm., tr=1', 'factor': 'PSD factor', 'add': 'PSD additive'}
    fig, axs = plt.subplots(2, 2, figsize=(0.98 * W2, 4.0), sharex=True)
    for ax, (g, title, lab) in zip(axs.ravel(), groups):
        d = p[p.comparison == g].copy(); d['lab'] = d.apply(lab, axis=1); d = d[d.lab.notna() & d.lab.isin(MAPS)]
        labs = [m for m in MAPS if m in set(d.lab)]
        for y, m in enumerate(labs):
            for q, a in enumerate(alphas):
                r = d[(d.lab == m) & (d.alpha == a)]
                if len(r):
                    r = r.iloc[0]; yy = y + 0.24 * (q - 1)
                    ax.plot([r.min_ratio, r.max_ratio], [yy, yy], color=cols[a], lw=0.8, alpha=0.6)
                    ax.plot([r.gmean_ratio_a_over_b], [yy], ls='none', marker=mks[a], ms=5.5, mfc=cols[a], mec='white', mew=0.8, zorder=3)
        ax.axvline(1.0, color=INK, lw=0.8); ax.set_xscale('log'); ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([short[m] for m in labs], fontsize=6.5); ax.set_ylim(len(labs) - 0.5, -0.5)
        ax.grid(axis='y', visible=False); ax.set_title(title, fontsize=7, loc='left')
    fig.supxlabel(xlabel, fontsize=7.5)
    h = [Line2D([], [], ls='none', marker=mks[a], ms=5.5, mfc=cols[a], mec='white', label=rf'$\alpha={a}$') for a in alphas]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.13), title='geometric mean (marker) and range (line) over 30 pairs',
               title_fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); save(fig, name)


def fig_b2_effects_optimisation():
    fig_b2_effects('node_rms_fro_to_l1', 'Neural_B2_effects_distance_to_L1',
                   'ratio of the distances to the L1 solution at the collocation nodes, paired by seed and initial state')


def fig_b2_layer():
    c = read('Neural_B2_error_curves.csv')
    if c is None:
        return
    alphas = sorted(c.alpha.unique()); fig, axs = plt.subplots(2, len(alphas), figsize=(W2, 4.3), sharex='col', squeeze=False)
    series = [('factor/hard-pow', r'PSD factor, $\tau^\alpha$ gate', BLUE, '-'), ('factor/hard-exp', r'PSD factor, gate $1-e^{-\beta\tau}$', ORANGE, '--'),
              ('lin/hard-pow', r'linear head, $\tau^\alpha$ gate', AQUA, '-.'), ('gate2b/hard-pow', r'PSD block gate, $\tau^\alpha$ gate', MUTED, ':')]
    for j, a in enumerate(alphas):
        d = c[(c.alpha == a) & (c.state == 'pure')]; lo = 1e-12 if a < 0.5 else (1e-8 if a < 0.8 else 1e-5)
        for cell, lab, col, ls in series:
            g = d[(d.cell == cell) & (d.tau >= lo)]
            if len(g):
                axs[0, j].fill_between(g.tau, g.err_fro_q25, g.err_fro_q75, color=col, alpha=0.12, lw=0)
                axs[0, j].loglog(g.tau, g.err_fro_median, ls, color=col, label=lab)
                axs[1, j].loglog(g.tau, g.min_eig_median.clip(lower=1e-30), ls, color=col, label=lab)
        g = d[(d.cell == 'lin/hard-pow') & (d.tau >= lo)]
        if len(g):
            axs[1, j].loglog(g.tau, g.min_eig_exact.clip(lower=1e-30), '-', color=INK, lw=1.0, label='exact')
        axs[0, j].set_title(rf'$\alpha={a}$, pure initial state'); axs[1, j].set_xlabel(r'$\tau$'); axs[1, j].set_ylim(bottom=1e-14)
    axs[0, 0].set_ylabel(r'$\|\hat\rho-\rho\|_F$ (median, quartiles)'); axs[1, 0].set_ylabel(r'smallest eigenvalue')
    h, l = axs[1, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    for ax, s in zip(axs.ravel(), 'abcdef'):
        panel(ax, s)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'Neural_B2_initial_layer')


# ---------------------------------------------------------------------------------------------- B6

def fig_b6():
    runs = read('Neural_B6_runs.csv'); st = read('Neural_B6_stepper.csv'); c = read('Neural_B6_curves.csv')
    if runs is None:
        return
    runs = runs[(runs.stage == 'main') & (runs.status == 'ok') & (runs.gate == 'pow')]
    cases = [(s, a) for s in sorted(runs.system.unique()) for a in sorted(runs.alpha.unique())]
    maps = ['unc', 'herm', 'lin', 'factor', 'add', 'gate2', 'gate2b']
    fig, axs = plt.subplots(3, len(cases), figsize=(0.98 * W2, 6.3), squeeze=False)
    name = {'qutrit': 'qutrit ($d=3$)', 'two_qubit': 'two qubits ($d=4$)'}
    for j, (s, a) in enumerate(cases):
        d = runs[(runs.system == s) & (runs.alpha == a)]; ax = axs[0, j]
        for k, m in enumerate(maps):
            v = d[d['map'] == m]
            if len(v):
                psd = m in ('factor', 'add', 'gate2', 'gate2b')
                strip(ax, k, v.rel_fro.values, BLUE if psd else ORANGE, 'o' if psd else '^')
        f = st[(st.system == s) & (st.alpha == a) & (st.N == 128)].rel_fro.iloc[0]
        ax.axhline(f, color=INK, lw=1.0); ax.grid(axis='x', visible=False)
        ax.set_yscale('log' if d.rel_fro.max() / min(d.rel_fro.min(), f) > 5 else 'linear')      # narrow ranges read better on a linear axis
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{100 * v:.3g}%')); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.annotate('L1 stepper', (len(maps) - 0.55, f), xytext=(0, 2.5), textcoords='offset points', ha='right', va='bottom', fontsize=6)
        ax.set_xticks(range(len(maps))); ax.set_xticklabels([])
        ax.set_title(rf'{name.get(s, s)}, $\alpha={a}$')
        ax = axs[1, j]
        for k, m in enumerate(maps):
            v = d[d['map'] == m]
            if len(v):
                psd = m in ('factor', 'add', 'gate2', 'gate2b')
                strip(ax, k, v.min_eig.values, BLUE if psd else ORANGE, 'o' if psd else '^')
        ax.axhline(0.0, color=INK, lw=0.8); ax.set_yscale('symlog', linthresh=1e-5); ax.grid(axis='x', visible=False)
        ax.set_xticks(range(len(maps)))
        ax.set_xticklabels([MAP_SHORT[m].replace('-\n', '').replace('\n', ' ') for m in maps], fontsize=5.8, rotation=40, ha='right')
        ax = axs[2, j]; g = c[(c.system == s) & (c.alpha == a) & (c.tau >= 1e-6)]
        for cell, lab, col, ls in [('lin/hard-pow', 'linear head: median', ORANGE, '-'), ('factor/hard-pow', 'PSD factor: median', BLUE, '--')]:
            h = g[g.cell == cell]
            if len(h):
                ax.plot(h.tau, h.min_eig_median, ls, color=col, label=lab)
                if cell.startswith('lin'):
                    ax.plot(h.tau, h.min_eig_min, ':', color=col, lw=1.0, label='linear head: minimum over seeds')
        h = g[g.cell == 'lin/hard-pow']
        if len(h):
            ax.plot(h.tau, h.min_eig_exact, '-', color=INK, lw=1.0, label='exact')
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1e-5); ax.axhline(0.0, color=MUTED, lw=0.5); ax.set_xlabel(r'$\tau$')
    axs[0, 0].set_ylabel('rel. Frobenius error'); axs[1, 0].set_ylabel(r'$\min_\tau\lambda_{\min}$ per run'); axs[2, 0].set_ylabel(r'$\lambda_{\min}(\tau)$')
    hd = [Line2D([], [], ls='none', marker='^', ms=6, mfc=ORANGE, mec='white', label='no positivity constraint'),
          Line2D([], [], ls='none', marker='o', ms=6, mfc=BLUE, mec='white', label='positive semidefinite by construction'),
          Line2D([], [], color=INK, lw=1.0, label='L1 time stepper (top), zero (middle), exact (bottom)')]
    h, l = axs[2, 0].get_legend_handles_labels()
    fig.legend(handles=hd, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.025), fontsize=6.5)
    fig.legend(h, ['panel (c): ' + x for x in l], loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    for ax, s in zip(axs[:, 0], 'abc'):
        panel(ax, s, -0.3)
    fig.tight_layout(rect=(0, 0.055, 1, 1)); save(fig, 'Neural_B6_positivity')


# ---------------------------------------------------------------------------------------------- B7

def fig_b7():
    e = read('Neural_B7_estimates.csv'); T = read('Table_B7_estimates.csv', TD)
    if e is None:
        return
    e = e[(e.design == 'xz') & (e.status == 'ok')]; noises = [n for n in sorted(e.noise.unique())]
    pinns = sorted(m for m in e.method.unique() if m.startswith('PINN'))[:3]
    methods = [('NLS-L1 N128', 'least squares, L1 stepper', INK, 'D')] + [(m, m.replace('PINN', 'network,'), c, mk)
                                                                         for m, c, mk in zip(pinns, (BLUE, ORANGE, AQUA), ('o', '^', 's'))]
    pars = [('alpha', r'$\hat\alpha$'), ('Gdown', r'$\hat\Gamma_\downarrow$'), ('Gphi', r'$\hat\Gamma_\phi$')]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7))
    for ax, (p, lab) in zip(axs, pars):
        tv = T[f'{p}_true'].iloc[0]; ax.axhline(tv, color=INK, lw=0.8)
        for i, n in enumerate(noises):
            crb = T[(T.design == 'xz') & (T.noise_sd == n)][f'{p}_crb_sd'].iloc[0]
            ax.fill_between([i - 0.45, i + 0.45], tv - crb, tv + crb, color=MUTED, alpha=0.13, lw=0)
            for q, (m, _, c, mk) in enumerate(methods):
                v = e[(e.noise == n) & (e.method == m)][p + '_hat'].values
                if len(v):
                    strip(ax, i + 0.22 * (q - 1.5), v, c, mk, width=0.07)
        ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:g}' for n in noises]); ax.set_xlabel(r'noise s.d. $\sigma$')
        ax.set_ylabel(lab); ax.grid(axis='x', visible=False)
    h = [Line2D([], [], ls='none', marker=mk, ms=6, mfc=c, mec='white', label=lab) for _, lab, c, mk in methods]
    h += [Line2D([], [], color=INK, lw=0.8, label='true value'), Line2D([], [], color=MUTED, lw=5, alpha=0.25, label=r'$\pm$ Cram$\acute{\rm e}$r-Rao s.d.')]
    fig.legend(handles=h, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=6.3)
    for ax, s in zip(axs, 'abc'):
        panel(ax, s, -0.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'Neural_B7_estimates')


# ---------------------------------------------------------------------------------------------- B1

def fig_b1():
    inv = read('Neural_B1_inverse_runs.csv')
    if inv is None:
        return
    d = inv[(inv.stage == 'main') & (inv.noise == 0) & (inv.status == 'ok')]; alphas = sorted(d.alpha_true.unique())
    cells = [('plain', 'exp', ORANGE, '^', True), ('singular', 'exp', ORANGE, '^', False), ('plain', 'pow', BLUE, 'o', True), ('singular', 'pow', BLUE, 'o', False)]
    lab = {('plain', 'exp'): r'input $\tau$, gate $1-e^{-\beta\tau}$', ('singular', 'exp'): r'input $(\tau,\tau^\alpha)$, gate $1-e^{-\beta\tau}$',
           ('plain', 'pow'): r'input $\tau$, gate $1-e^{-\beta\tau^\alpha}$', ('singular', 'pow'): r'input $(\tau,\tau^\alpha)$, gate $1-e^{-\beta\tau^\alpha}$'}
    fig, axs = plt.subplots(1, len(alphas) + 1, figsize=(W2, 2.8))
    for ax, a in zip(axs, alphas):
        g = d[d.alpha_true == a]
        for k, (i, gt, c, mk, fl) in enumerate(cells):
            v = g[(g.input == i) & (g.gate == gt)]
            if len(v):
                strip(ax, k, (v.alpha_hat - a).values, c, mk, fl)
        for col, ls, name in (('alpha_joint', '-', 'discrete optimum of the training objective'), ('alpha_res', ':', 'residual-optimal order of the exact solution')):
            if len(g):
                ax.axhline(g[col].iloc[0] - a, color=INK, lw=0.9, ls=ls)
        ax.axhline(0.0, color=MUTED, lw=0.5); ax.set_yscale('symlog', linthresh=1e-3); ax.set_title(rf'$\alpha_{{\rm true}}={a}$')
        ax.set_xticks(range(4)); ax.set_xticklabels(['', '', '', '']); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel(r'$\hat\alpha-\alpha_{\rm true}$ (noise-free data)')
    m = read('Table_B1_design.csv', TD); ax = axs[-1]
    if m is not None:
        m = m[(m['mode'] == 'inverse') & (m.noise_sd == 0)]
        for i, gt, c, mk, fl in cells:
            g = m[(m.input == i) & (m.gate == gt) & (m.alpha_true == 0.5)].sort_values('mesh_r')
            if len(g):
                ax.plot(g.mesh_r, g.bias.abs(), '-' if fl else '--', color=c, marker=mk, mfc=c if fl else 'white', mec=c, ms=4.5, lw=0.9)
        ax.set_yscale('log'); ax.set_xlabel('mesh grading $r$'); ax.set_ylabel(r'$|{\rm mean}\,\hat\alpha-\alpha_{\rm true}|$')
        ax.set_title(r'$\alpha_{\rm true}=0.5$')
    h = [Line2D([], [], ls='none', marker=mk, ms=6, mfc=c if fl else 'white', mec=c, label=lab[(i, gt)]) for i, gt, c, mk, fl in cells]
    h += [Line2D([], [], color=INK, lw=0.9, label='discrete optimum of the training objective'),
          Line2D([], [], color=INK, lw=0.9, ls=':', label='residual-optimal order for the exact solution')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=6.3)
    for ax, s in zip(axs, 'abcd'):
        panel(ax, s, -0.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'Neural_B1_design')


# ---------------------------------------------------------------------------------------------- surrogate

def fig_surrogate():
    g = read('Neural_surrogate_grid.csv'); T = read('Table_surrogate.csv', TD); st = read('Neural_surrogate_stepper.csv')
    if g is None:
        return
    methods = sorted(g.method.unique()); nh = len(methods) + 1
    fig = plt.figure(figsize=(W2, 2.55))
    outer = fig.add_gridspec(1, 2, width_ratios=[0.62 * nh + 0.1, 1.0], wspace=0.5)
    left = outer[0].subgridspec(1, nh + 1, width_ratios=[1.0] * nh + [0.07], wspace=0.12)
    med = g.groupby(['method', 'alpha', 'Gdown']).rel_fro.median().reset_index()
    s128 = st[st.N == 128]; vmin = min(med.rel_fro.min(), s128.rel_fro.min()); vmax = max(med.rel_fro.max(), s128.rel_fro.max())
    panels = [(m.replace('surrogate add', 'surrogate, PSD additive').replace('surrogate lin', 'surrogate, linear head'), med[med.method == m])
              for m in methods] + [('L1 stepper, $N=128$', s128)]
    axs = []
    for k, (m, d) in enumerate(panels):
        ax = fig.add_subplot(left[0, k]); axs.append(ax)
        pv = d.pivot_table(index='Gdown', columns='alpha', values='rel_fro')
        im = ax.pcolormesh(pv.columns.values, pv.index.values, pv.values, cmap=SEQ, norm=LogNorm(vmin, vmax), shading='nearest',
                           edgecolors='white', linewidth=0.8)
        ax.set_title(f'({"abc"[k]}) {m}', fontsize=6.5, loc='left'); ax.set_xlabel(r'$\alpha$'); ax.grid(False)
        if k:
            ax.set_yticklabels([])
    axs[0].set_ylabel(r'$\Gamma_\downarrow$')
    cax = fig.add_subplot(left[0, nh]); cb = fig.colorbar(im, cax=cax); cb.set_label('rel. Frobenius error (median over seeds)', fontsize=6.5)
    cb.outline.set_visible(False)
    ax = fig.add_subplot(outer[1]); axs.append(ax); Q = np.logspace(0, 8, 200)
    sur = T[T.method.str.startswith('surrogate') & (T.status == 'ok')]
    for (m, d), c, ls in zip(sur.groupby('method'), (BLUE, ORANGE), ('-', '--')):
        ax.loglog(Q, d.train_wall_s.median() + Q * d.inference_ms_per_query.median() * 1e-3, ls, color=c,
                  label=m.replace('surrogate add', 'surrogate, PSD additive').replace('surrogate lin', 'surrogate, linear head'))
    stp = T[T.method.str.startswith('l1_stepper')]
    for N, ls in (('32', ':'), ('64', '-.'), ('128', '-')):
        r = stp[stp.method == f'l1_stepper N={N}']
        if len(r):
            ax.loglog(Q, Q * r.stepper_ms_per_query.iloc[0] * 1e-3, ls=ls, color=INK, lw=1.0, label=f'L1 stepper, $N={N}$')
    ax.set_xlabel('number of queries (trajectories)'); ax.set_ylabel('total time (s): training + queries'); ax.legend(fontsize=5.6, loc='upper left')
    ax.set_title(f'({"abcd"[nh]}) cost against the number of queries', fontsize=6.5, loc='left')
    save(fig, 'Neural_surrogate')


if __name__ == '__main__':
    for f in (fig_b2_factorial, fig_b2_effects, fig_b2_effects_optimisation, fig_b2_layer, fig_b6, fig_b7, fig_b1, fig_surrogate):
        f()
