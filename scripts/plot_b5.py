"""Candidate figures of the conformation-tensor benchmark B5.

Reads only figures/data/B5_*.csv written by scripts/aggregate_b5.py and writes vector PDFs to figures/candidates/.
Usage: python scripts/plot_b5.py [--preview <dir for PNG copies>]

Colour: at most three hues of the categorical order used in the other figures (blue, orange, aqua), each paired
with a marker shape or line style; exact and classical reference solutions are drawn in ink; the status colour red
marks loss of positive definiteness and always comes with its own marker or label.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, 'figures', 'data'); OUT = os.path.join(ROOT, 'figures', 'candidates')
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'          # categorical slots 1-3
CRIT = '#d03b3b'                                             # status: loss of positive definiteness
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif', 'font.size': 8,
                     'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.4, 'lines.markersize': 4.5, 'axes.linewidth': 0.6,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'grid.linestyle': '-', 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'savefig.dpi': 300})
W2 = 190 / 25.4
VARIANT_ORDER = ['sym-fro', 'sym-comp', 'sym-rel', 'factor-fro', 'factor-nat', 'anchor-fro', 'logm-fro', 'logm-nat',
                 'bfactor-fro', 'bfactor-nat', 'blogm-nat']
VLABEL = {'sym-fro': 'unconstrained / Frobenius', 'sym-comp': 'unconstrained / component-scaled',
          'sym-rel': 'unconstrained / relative', 'logm-nat': 'exp(Psi) / affine-invariant',
          'blogm-nat': 'bounded-trace exp / affine-invariant'}
VSHORT = {'sym-fro': 'sym, Fro', 'sym-comp': 'sym, comp', 'sym-rel': 'sym, rel', 'factor-fro': r'$TT^{\top}$, Fro',
          'factor-nat': r'$TT^{\top}$, nat', 'anchor-fro': 'anchor, Fro', 'logm-fro': 'exp, Fro', 'logm-nat': 'exp, nat'}


def read(name):
    p = os.path.join(FD, name)
    return pd.read_csv(p, float_precision='round_trip') if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})      # no date: identical bytes on regeneration
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True); fig.savefig(os.path.join(PREVIEW, name + '.png'), bbox_inches='tight', dpi=170)
    plt.close(fig); print('wrote', name + '.pdf')


def panel(ax, s, dx=-0.14):
    ax.text(dx, 1.05, f'({s})', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


# ---------------------------------------------------------------------------------------------- forward

def fig_forward_errors():
    """Relative Frobenius error of every run (one mark per seed), by output map / residual norm; values above 30 at 30."""
    runs = read('B5_forward_runs.csv'); st = read('B5_forward_stepper.csv')
    if runs is None:
        return
    runs = runs[runs.budget_factor == 1.0]
    cases = sorted({(f, w) for f, w in zip(runs.flow, runs.Wi)}); frames = sorted(runs.frame.unique())
    variants = [v for v in VARIANT_ORDER if v in set(runs.variant)]
    for al in sorted(runs.alpha.unique()):
        fig, axs = plt.subplots(len(frames), len(cases), figsize=(W2, 2.0 * len(frames) + 0.75), sharey=True, sharex=True, squeeze=False)
        for i, fr in enumerate(frames):
            for j, (flow, Wi) in enumerate(cases):
                ax = axs[i, j]; d = runs[(runs.flow == flow) & (runs.Wi == Wi) & (runs.alpha == al) & (runs.frame == fr)]
                for k, v in enumerate(variants):
                    rr = d[d.variant == v].sort_values('seed'); e = np.minimum(rr.rel_fro.values, 30.0); bad = rr.indefinite.values.astype(bool)
                    jit = np.linspace(-0.18, 0.18, max(len(rr), 1))
                    ok_e, bad_e = e[~bad], e[bad]
                    ax.plot(k + jit[:len(ok_e)], ok_e, ls='none', marker='o', ms=3.2, mfc=BLUE, mec='white', mew=0.4, zorder=3)
                    ax.plot(k + jit[len(ok_e):len(ok_e) + len(bad_e)], bad_e, ls='none', marker='x', ms=3.6, mec=CRIT, mew=0.9, zorder=4)
                s = st[(st.flow == flow) & (st.Wi == Wi) & (st.alpha == al) & (st.frame == fr)]
                if len(s):
                    ax.axhline(s.rel_fro.iloc[0], color=INK, lw=1.0, zorder=2)
                ax.set_yscale('log'); ax.set_ylim(1e-4, 60); ax.grid(axis='x', visible=False)
                ax.set_xticks(range(len(variants))); ax.set_xticklabels([VSHORT[v] for v in variants], fontsize=5.5, rotation=90)
                if i == 0:
                    ax.set_title(f'{flow}, Wi = {Wi:g}', fontsize=7)
                if j == 0:
                    ax.set_ylabel(f'{fr} frame\nrel. Frobenius error')
        h = [Line2D([], [], ls='none', marker='o', ms=4.5, mfc=BLUE, mec='white', label='run without negative eigenvalue beyond round-off'),
             Line2D([], [], ls='none', marker='x', ms=4.5, mec=CRIT, mew=1.0, label='run indefinite at some test time'),
             Line2D([], [], color=INK, lw=1.0, label='implicit L1 stepper, same mesh')]
        fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.4); save(fig, f'B5_forward_errors_a{al:g}')


def fig_forward_min_eig():
    """Smallest eigenvalue along the trajectory, all seeds, for two representative cases."""
    c = read('B5_forward_min_eig_curves.csv')
    if c is None:
        return
    reps = list(dict.fromkeys(zip(c.flow, c.Wi, c.alpha, c.frame)))
    show = [('sym-fro', BLUE, '-'), ('sym-comp', ORANGE, '--'), ('logm-nat', AQUA, '-.')]
    fig, axs = plt.subplots(1, len(reps), figsize=(W2, 2.6), squeeze=False)
    for ax, (flow, Wi, al, fr), s in zip(axs[0], reps, 'ab'):
        d = c[(c.flow == flow) & (c.Wi == Wi) & (c.alpha == al) & (c.frame == fr)]
        for v, col, ls in show:
            for _, g in d[d.variant == v].groupby('seed'):
                ax.plot(g.tau, g.min_eig, color=col, ls=ls, lw=0.9)
        g = d[d.variant == 'exact']
        ax.plot(g.tau, g.min_eig, color=INK, lw=1.6, zorder=5)
        ax.axhline(0.0, color=CRIT, lw=0.8)
        ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=0.1); ax.set_xlabel(r'$\tau=t/T$')
        ax.set_title(rf'{flow}, Wi = {Wi:g}, $\alpha={al:g}$, {fr} frame', fontsize=7); panel(ax, s, -0.12)
    axs[0, 0].set_ylabel(r'smallest eigenvalue of $X(\tau)$')
    h = [Line2D([], [], color=INK, lw=1.6, label='exact')] + [Line2D([], [], color=col, ls=ls, lw=0.9, label=VLABEL[v]) for v, col, ls in show]
    h.append(Line2D([], [], color=CRIT, lw=0.8, label='zero (loss of positive definiteness below)'))
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, 'B5_forward_min_eig')


# ---------------------------------------------------------------------------------------------- inverse

def fig_inverse():
    """Estimates of the order and of the relaxation time, one mark per fit, with the Cramer-Rao band."""
    e = read('B5_inverse_estimates.csv'); crb = read('B5_inverse_crb.csv')
    if e is None:
        return
    nls = [m for m in ('nls128', 'nls512') if m in set(e.method)]; nets = [v for v in VARIANT_ORDER if v in set(e.method)]
    allm = nls + nets
    short = {'nls128': 'NLS, $N=128$', 'nls512': 'NLS, $N=512$'}; short.update({v: 'net: ' + VSHORT[v] for v in nets})
    cells = [(a, n) for a in sorted(e.alpha_true.unique()) for n in sorted(e.noise.unique())]
    fig, axs = plt.subplots(2, len(cells), figsize=(0.98 * W2, 4.6), squeeze=False)
    for col, (al, nz) in enumerate(cells):
        d = e[(e.alpha_true == al) & (e.noise == nz)]
        for row, ylab in ((0, r'$\hat\alpha-\alpha$'), (1, r'$\hat\lambda/\lambda-1$')):
            ax = axs[row, col]
            for k, m in enumerate(allm):
                g = d[d.method == m].sort_values('seed')
                if not len(g):
                    continue
                y = (g.alpha_hat - al).values if row == 0 else (g.ell_hat / g.ell_true - 1.0).values
                isn = m in nls
                ax.plot(k + np.linspace(-0.2, 0.2, len(y)), y, ls='none', marker='o' if isn else 's', ms=3.0, mfc=BLUE if isn else ORANGE,
                        mec='white', mew=0.3, zorder=3)
                ax.plot([k - 0.3, k + 0.3], [y.mean()] * 2, color=INK, lw=1.2, zorder=4)
            if crb is not None and nz > 0:
                cr = crb[(crb.alpha_true == al) & (crb.noise == nz)]
                if len(cr):
                    sd = cr.sd_alpha.iloc[0] if row == 0 else cr.sd_log_ell.iloc[0]
                    ax.axhspan(-sd, sd, color=MUTED, alpha=0.13, lw=0, zorder=1)
            ax.axhline(0, color=MUTED, lw=0.6, zorder=2); ax.grid(axis='x', visible=False)
            ax.set_xticks(range(len(allm))); ax.set_xticklabels([short[m] for m in allm] if row == 1 else [], fontsize=5.5, rotation=90)
            ax.tick_params(axis='y', labelsize=6)
            if row == 0:
                ax.set_title(rf'$\alpha={al:g}$, noise {100 * nz:g} %', fontsize=7)
            if col == 0:
                ax.set_ylabel(ylab)
    h = [Line2D([], [], ls='none', marker='o', ms=4.5, mfc=BLUE, mec='white', label='least squares through the implicit L1 stepper'),
         Line2D([], [], ls='none', marker='s', ms=4.5, mfc=ORANGE, mec='white', label=r'network with trainable $(\alpha,\lambda)$'),
         Line2D([], [], color=INK, lw=1.2, label='mean'), Patch(color=MUTED, alpha=0.2, label=r'$\pm$ Cram$\acute{\rm e}$r-Rao s.d.')]
    fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1), w_pad=0.3); save(fig, 'B5_inverse_estimates')


# ---------------------------------------------------------------------------------------------- FENE-P

def fig_fenep():
    """Stretch and smallest eigenvalue in start-up of uniaxial extension (FENE-P), all seeds of three variants."""
    c = read('B5_fenep_curves.csv')
    if c is None:
        return
    show = [('sym-rel', BLUE, '-'), ('sym-comp', ORANGE, '--'), ('blogm-nat', AQUA, '-.')]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    for v, col, ls in show:
        for _, g in c[c.variant == v].groupby('seed'):
            top = v == 'blogm-nat'
            axs[0].plot(g.tau, g.C11, color=col, ls=ls, lw=1.2 if top else 0.9, zorder=6 if top else 3)
            axs[1].plot(g.tau, g.min_eig, color=col, ls=ls, lw=1.2 if top else 0.9, zorder=6 if top else 3)
    g = c[c.variant == 'reference']
    axs[0].plot(g.tau, g.C11, color=INK, lw=1.6, zorder=5); axs[1].plot(g.tau, g.min_eig, color=INK, lw=1.6, zorder=5)
    axs[1].axhline(0.0, color=CRIT, lw=0.8)
    for ax, s in zip(axs, 'ab'):
        ax.set_xscale('log'); ax.set_xlim(1e-4, 1); ax.set_xlabel(r'$\tau=t/T$'); panel(ax, s, -0.13)
    axs[0].set_ylabel(r'stretch $C_{11}$'); axs[1].set_ylabel(r'smallest eigenvalue of $X(\tau)$')
    axs[1].set_yscale('symlog', linthresh=0.05); axs[1].set_yticks([-100, -10, -1, -0.1, 0, 0.1, 1])
    Wi, al = c.Wi.iloc[0], c.alpha.iloc[0]
    fig.suptitle(rf'FENE-P, uniaxial extension, Wi = {Wi:g}, $\alpha={al:g}$, $b=50$', fontsize=7.5, y=0.99)
    h = [Line2D([], [], color=INK, lw=1.6, label='reference')] + [Line2D([], [], color=col, ls=ls, lw=1.0, label=VLABEL[v]) for v, col, ls in show]
    h.append(Line2D([], [], color=CRIT, lw=0.8, label='zero'))
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.98)); save(fig, 'B5_fenep_trajectories')


# ---------------------------------------------------------------------------------------------- model

def fig_model():
    """Largest and smallest eigenvalue of the exact conformation tensor in three start-up flows."""
    m = read('B5_model_eigenvalues.csv')
    if m is None:
        return
    cases = list(dict.fromkeys(zip(m.flow, m.Wi, m.Theta)))
    fig, axs = plt.subplots(1, len(cases), figsize=(W2, 2.5))
    sty = [(1.0, INK, ':'), (0.9, BLUE, '-'), (0.6, ORANGE, '--')]
    for ax, (flow, Wi, Th), s in zip(axs, cases, 'abc'):
        for al, col, ls in sty:
            g = m[(m.flow == flow) & (m.Wi == Wi) & (m.alpha == al)]
            ax.plot(g.tau, g.eig_max, color=col, ls=ls); ax.plot(g.tau, g.eig_min, color=col, ls=ls)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\tau=t/T$')
        ax.set_title(rf'{flow}, Wi = {Wi:g}, $T={Th:g}\,\lambda$', fontsize=7); panel(ax, s, -0.2)
    axs[0].set_ylabel('extreme eigenvalues of $C$')
    axs[0].legend(handles=[Line2D([], [], color=col, ls=ls, label=rf'$\alpha={al:g}$') for al, col, ls in sty], loc='upper left')
    fig.tight_layout(w_pad=0.8); save(fig, 'B5_model_eigenvalues')


if __name__ == '__main__':
    for f in (fig_model, fig_forward_errors, fig_forward_min_eig, fig_inverse, fig_fenep):
        f()
