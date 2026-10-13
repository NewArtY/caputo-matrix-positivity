"""Candidate figures of the deterministic benchmarks (E1-E5, Dirac B3, B4 and tail): vector PDF, TrueType fonts embedded.

  python scripts/plot_deterministic.py [E|SOE|B3|current|tail|maps|all] [--preview <dir for PNG copies>]

Reads only tables/data/Table_*.csv and figures/data/FigDet_* written by scripts/aggregate_deterministic.py and
writes figures/candidates/FigDet_*.pdf. Typography as in the other plotting scripts (DejaVu Serif, 8/7 pt, widths
90 mm and 190 mm). Colours: a colour-vision-deficiency-checked set --
  nominal series (at most three hues per panel): blue, orange, aqua; a fourth series is drawn in neutral ink;
  ordered series (fractional order alpha): one blue ramp, light to dark with increasing alpha;
  maps: one-hue sequential blue ramp;
every series additionally differs by line style or marker, so no panel depends on colour alone.
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TD = os.path.join(ROOT, 'tables', 'data'); FD = os.path.join(ROOT, 'figures', 'data'); OUT = os.path.join(ROOT, 'figures', 'candidates')
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'      # nominal slots, fixed order
RAMP3 = ['#6da7ec', '#256abf', '#0d366b']                                    # ordered: light -> dark
RAMP4 = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
SEQ = ['#eef5fe', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.2, 'lines.markersize': 4,
                     'axes.linewidth': 0.6, 'savefig.dpi': 300, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4,
                     'ytick.minor.width': 0.4, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
                     'axes.axisbelow': True, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
W1, W2 = 90 / 25.4, 190 / 25.4


def panel(ax, s, x=-0.2):
    ax.text(x, 1.05, f'({s})', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', color=INK)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None})      # no date: identical bytes on regeneration
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True); fig.savefig(os.path.join(PREVIEW, name + '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig); print('wrote figures/candidates/' + name + '.pdf')


def guide(ax, x0, x1, y0, slope, label, dx=1.0, dy=1.0):
    """Reference power law through (x0, y0) with the given slope (neutral, dotted)."""
    x = np.array([x0, x1]); ax.plot(x, y0 * (x / x0) ** slope, ':', color=MUTED, lw=0.8, zorder=1)
    ax.text(x1 * dx, y0 * (x1 / x0) ** slope * dy, label, color=INK2, fontsize=6, va='center', ha='left')


def key(color, ls='-', marker=None, mfc=None, label='', lw=1.2):
    return Line2D([], [], color=color, ls=ls, marker=marker, mfc=color if mfc is None else mfc, mec=color, label=label, lw=lw)


# ====================================================================================================== E1-E3
def fig_E():
    T = pd.read_csv(os.path.join(TD, 'Table_E1_E3.csv'))
    alphas = [0.3, 0.5, 0.8]; col = dict(zip(alphas, RAMP3)); mk = dict(zip(alphas, ['o', 's', '^']))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55))
    spec = [('E3', 'max_abs_err', r'max. nodal error  $\max_n|u(t_n)-u^n|$', 'a'), ('E3', 'final_abs_err', r'error at the final time  $|u(T)-u^N|$', 'b'),
            ('E1T', 'weighted_max_trunc', r'weighted truncation error  $\max_n t_n^{\alpha}|R^n|$', 'c')]
    for ax, (exp, q, yl, pl) in zip(axs, spec):
        for a in alphas:
            for mesh, ls, fill in (('graded', '-', True), ('uniform', '--', False)):
                g = T[(T.experiment == exp) & (T.quantity == q) & (T.alpha == a) & (T.mesh == mesh)].sort_values('N')
                ax.loglog(g.N, g.value, ls, color=col[a], marker=mk[a], mfc=col[a] if fill else 'white', mec=col[a], mew=0.9)
        if pl == 'c':
            for a in alphas:                                   # unweighted maximum: does not converge
                g = T[(T.experiment == 'E1T') & (T.quantity == 'max_abs_trunc') & (T.alpha == a) & (T.mesh == 'graded')].sort_values('N')
                ax.loglog(g.N, g.value, '-', color=col[a], lw=0.7)
            ax.text(40, 0.42, r'unweighted $\max_n|R^n|$ (both meshes)', fontsize=6, color=INK2, va='bottom')
            ax.set_ylim(2e-7, 2.0)
        ax.set_xlabel('number of time steps $N$'); ax.set_ylabel(yl); panel(ax, pl, -0.24)
        ax.set_xlim(12 if exp == 'E1T' else 24, 6000); ax.grid(True, which='major')
    guide(axs[0], 256, 4096, 3e-5, -1.7, r'$N^{-1.7}$', 1.08); guide(axs[0], 256, 4096, 0.09, -0.3, r'$N^{-0.3}$', 1.08)
    guide(axs[1], 256, 4096, 3.5e-4, -1.0, r'$N^{-1}$', 1.08); guide(axs[1], 256, 4096, 3e-6, -1.7, r'$N^{-1.7}$', 1.08)
    guide(axs[2], 256, 4096, 3e-5, -1.7, r'$N^{-1.7}$', 1.08)
    h = [key(col[a], '-', mk[a], label=fr'$\alpha={a}$') for a in alphas] + [key(INK2, '-', 'o', label='graded, $r=(2-\\alpha)/\\alpha$'), key(INK2, '--', 'o', 'white', 'uniform')]
    axs[0].legend(handles=h, loc='lower left', ncol=1, handlelength=2.6, borderaxespad=0.3)
    fig.tight_layout(w_pad=1.0); save(fig, 'FigDet_E_verification')


# ====================================================================================================== SOE
def fig_SOE():
    t = pd.read_csv(os.path.join(TD, 'Table_SOE.csv')); k = pd.read_csv(os.path.join(TD, 'Table_SOE_kernel.csv'))
    res = pd.read_csv(os.path.join(TD, 'Table_SOE_residual.csv')); dch = pd.read_csv(os.path.join(TD, 'Table_SOE_dirac_chunk.csv'))
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.7)); axs = axs.ravel()
    # (a) number of modes versus tolerance: quadratic growth
    ax = axs[0]; al3 = [0.3, 0.6, 0.9]; col = dict(zip(al3, RAMP3)); mk = dict(zip(al3, ['o', 's', '^']))
    g0 = k[(k.window == 'generic') & (k.criterion == 'relative') & (k['T'] == 1.0) & (k.T_over_delta == 1e3)]
    for a in al3:
        g = g0[g0.alpha == a].sort_values('tol', ascending=False); L = np.log10(1 / g.tol.values)
        ax.plot(L, g.M, 'o', color=col[a], marker=mk[a], ls='none'); c = np.polyfit(L, g.M.values, 2); x = np.linspace(4, 12, 50)
        ax.plot(x, np.polyval(c, x), '-', color=col[a], lw=1.0, label=fr'$\alpha={a}$')
    ax.set_xlabel(r'$\log_{10}(1/\varepsilon_{\rm SOE})$'); ax.set_ylabel('number of modes $M$'); ax.legend(loc='upper left', title='quadratic fits', title_fontsize=6.5)
    ax.set_title(r'$T/\delta=10^{3}$', fontsize=7, color=INK2); panel(ax, 'a')
    # (b) realised relative error over target: legacy (absolute) versus relative criterion
    ax = axs[1]; g0 = k[(k.window == 'generic') & (k.tol == 1e-10) & (k.T_over_delta == 1e4)]
    for crit, c, lab in (('absolute', ORANGE, 'first version (absolute lower cut)'), ('relative', BLUE, 'this version (relative criterion)')):
        for Tw, ls, m, fill in ((60.0, '-', 'o', True), (1.0, '--', 's', False)):
            g = g0[(g0.criterion == crit) & (g0['T'] == Tw)].sort_values('alpha')
            ax.semilogy(g.alpha, g.rel_err_over_tol, ls, color=c, marker=m, mfc=c if fill else 'white', mec=c)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=':'); ax.text(0.31, 1.25, 'target', fontsize=6, color=INK2)
    ax.set_xlabel(r'fractional order $\alpha$'); ax.set_ylabel(r'realised rel. kernel error / $\varepsilon_{\rm SOE}$')
    ax.legend(handles=[key(ORANGE, '-', 'o', label='first version, $T=60$'), key(ORANGE, '--', 's', 'white', 'first version, $T=1$'),
                       key(BLUE, '-', 'o', label='this version, $T=60$'), key(BLUE, '--', 's', 'white', 'this version, $T=1$')], loc='upper left')
    ax.set_title(r'$\varepsilon_{\rm SOE}=10^{-10}$, $T/\delta=10^{4}$', fontsize=7, color=INK2); panel(ax, 'b')
    # (c) wall time versus N
    sty = {'full_uniform_tab': (BLUE, '-', 'o', True, 'full history, uniform'), 'soe_uniform': (BLUE, '--', 'o', False, 'SOE, uniform'),
           'full_graded_row': (ORANGE, '-', 's', True, 'full history, graded'), 'soe_graded': (ORANGE, '--', 's', False, 'SOE, graded')}
    ax = axs[2]
    g = t[t.solver == 'full_uniform_row'].sort_values('N'); ax.loglog(g.N, g.wall_median_s, '-', color=MUTED, lw=0.8, marker='x', ms=3)
    ax.annotate('full history, uniform,\nweights recomputed\nat every step', xy=(g.N.values[-2], g.wall_median_s.values[-2]), xytext=(60, 3.0), fontsize=5.5, color=INK2, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    for s, (c, ls, m, fill, lab) in sty.items():
        g = t[t.solver == s].sort_values('N')
        ax.errorbar(g.N, g.wall_median_s, yerr=[g.wall_median_s - g.wall_min_s, g.wall_max_s - g.wall_median_s], fmt=ls, color=c, marker=m, mfc=c if fill else 'white', mec=c,
                    elinewidth=0.6, capsize=1.5, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('number of time steps $N$'); ax.set_ylabel('wall time (s)'); panel(ax, 'c')
    # (d), (e) work-precision
    for ax, q, yl, pl in ((axs[3], 'err_max_all_nodes', 'max. error over all nodes', 'd'), (axs[4], 'err_final_time', 'error at the final time', 'e')):
        for s, (c, ls, m, fill, lab) in sty.items():
            g = t[t.solver == s].sort_values('N'); ax.loglog(g.wall_median_s, g[q], ls, color=c, marker=m, mfc=c if fill else 'white', mec=c, label=lab)
        ax.set_xlabel('wall time (s)'); ax.set_ylabel(yl); panel(ax, pl)
    axs[3].legend(loc='lower left', title='solver styles in (c)-(e)', title_fontsize=6.5)
    # (f) when does SOE pay off: time(full)/time(SOE) versus N
    ax = axs[5]
    u = t[t.solver == 'soe_uniform'].sort_values('N'); gq = t[t.solver == 'soe_graded'].sort_values('N'); d = dch[dch.solver == 'soe_uniform'].sort_values('N'); r = res[res['mode'] == 'soe'].sort_values('N')
    ax.loglog(u.N, u.time_ratio_full_over_this, '-', color=BLUE, marker='o', label='Bloch system, uniform mesh')
    ax.loglog(gq.N, gq.time_ratio_full_over_this, '-', color=ORANGE, marker='s', label='Bloch system, graded mesh')
    ax.loglog(d.N, d.time_ratio_full_over_soe, '-', color=AQUA, marker='^', label='Dirac, 256 wave vectors')
    ax.loglog(r.N, r.time_ratio_full_over_soe, ':', color=INK2, marker='D', label='residual operator (torch)')
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=':'); ax.text(9e4, 1.15, 'SOE faster', fontsize=6, color=INK2, va='bottom', ha='right'); ax.text(9e4, 0.62, 'full history faster', fontsize=6, color=INK2, va='top', ha='right')
    ax.set_ylim(0.004, 40); ax.set_xlabel('number of time steps $N$'); ax.set_ylabel('time(full history) / time(SOE)'); ax.legend(loc='lower right'); panel(ax, 'f')
    fig.tight_layout(w_pad=0.8, h_pad=1.2); save(fig, 'FigDet_SOE')


# ====================================================================================================== B3
def fig_B3():
    tr = pd.read_csv(os.path.join(FD, 'FigDet_B3_trajectories.csv')); cv = pd.read_csv(os.path.join(FD, 'FigDet_B3_convergence.csv'))
    al = [0.8, 0.9, 0.95, 1.0]; col = dict(zip(al, RAMP4)); lss = dict(zip(al, ['-.', '--', ':', '-']))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
    for ax, a0, pl in ((axs[0], 0.2, 'a'), (axs[1], 1.0, 'b')):
        for a in al:
            g = tr[(tr.a0 == a0) & (tr.alpha == a)]; ax.plot(g.t, g.nc, lss[a], color=col[a], label=fr'$\alpha={a:g}$')
        ax.set_xlim(-20, 30); ax.set_xlabel(r'$\omega_0(t-t_c)$'); ax.set_ylabel(r'conduction population $n_c(\mathbf{0},t)$'); ax.set_title(fr'$a_0={a0:g}$', fontsize=7, color=INK2); panel(ax, pl)
    axs[0].legend(loc='upper left')
    ax = axs[2]
    for a, m in ((0.95, 'o'), (0.8, 's')):
        g = cv[(cv.a0 == 0.2) & (cv.alpha == a)]
        gl = g[g.method == 'L1-soe'].sort_values('N'); gp = g[g.method == 'PI-trapezoid'].sort_values('N')
        ax.loglog(gl.N, gl.abs_err_vs_reference, '-', color=BLUE, marker=m, mfc=BLUE if a == 0.95 else 'white', mec=BLUE)
        ax.loglog(gp.N, gp.abs_err_vs_reference, '--', color=ORANGE, marker=m, mfc=ORANGE if a == 0.95 else 'white', mec=ORANGE)
    guide(ax, 4096, 65536, 6e-2, -1.05, r'$N^{-1.05}$', 1.1); guide(ax, 1024, 16384, 1.2e-6, -2.0, r'$N^{-2}$', 1.0, 0.25)
    ax.set_ylim(1e-14, 1.0); ax.set_xlim(700, 4e5)
    ax.set_xlabel('number of time steps $N$'); ax.set_ylabel(r'$|n_c(t_f)-n_c^{\rm ref}|$'); ax.set_title(r'$a_0=0.2$', fontsize=7, color=INK2)
    ax.legend(handles=[key(BLUE, '-', 'o', label=r'implicit L1, $\alpha=0.95$'), key(BLUE, '-', 's', 'white', r'implicit L1, $\alpha=0.8$'),
                       key(ORANGE, '--', 'o', label=r'product trapezoid, $\alpha=0.95$'), key(ORANGE, '--', 's', 'white', r'product trapezoid, $\alpha=0.8$')], loc='lower left')
    panel(ax, 'c'); fig.tight_layout(w_pad=0.9); save(fig, 'FigDet_Dirac_B3')


# ====================================================================================================== current
def fig_current():
    tr = pd.read_csv(os.path.join(FD, 'FigDet_B4_current_traces.csv')); cut = pd.read_csv(os.path.join(FD, 'FigDet_B4_cutoff.csv')); B4 = pd.read_csv(os.path.join(TD, 'Table_Dirac_B4.csv'))
    al = [0.8, 0.9, 1.0]; col = dict(zip(al, RAMP3)); lss = dict(zip(al, ['-.', '--', '-'])); mk = dict(zip(al, ['^', 's', 'o']))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9)); axs = axs.ravel()
    # (a) the two definitions at alpha = 1, zeta = 0, for two cutoffs
    ax = axs[0]
    for kc, ls in ((3.0, '-'), (8.0, '--')):
        g = tr[(tr.zeta == 0.0) & (tr.alpha == 1.0) & (tr.kappa_c == kc)]
        ax.plot(g.t, g.Jx_initref, ls, color=ORANGE); ax.plot(g.t, g.Jx_band, ls, color=BLUE)
    ax.set_xlim(-20, 20); ax.set_xlabel(r'$\omega_0(t-t_c)$'); ax.set_ylabel(r'$J_x$  ($e\omega_0^2/v_F$)')
    ax.legend(handles=[key(ORANGE, '-', label=r'${\rm tr}[\hat j(\rho-\rho(t_0))]$, $\kappa_c=3$'), key(ORANGE, '--', label=r'same, $\kappa_c=8$'),
                       key(BLUE, '-', label=r'${\rm tr}[\hat j(\rho-P_-(t))]$, $\kappa_c=3$'), key(BLUE, '--', label=r'same, $\kappa_c=8$')], loc='upper left', ncol=2, columnspacing=1.0, handlelength=2.2)
    ax.set_ylim(-3.3, 4.6); ax.set_title(r'$\alpha=1$, $\zeta=0$', fontsize=7, color=INK2); panel(ax, 'a', -0.16)
    # (b) band-referenced current, cutoff extrapolated, for the three orders
    ax = axs[1]
    for a in al[::-1]:
        g = tr[(tr.zeta == 0.0) & (tr.alpha == a) & (np.isinf(tr.kappa_c))]; ax.plot(g.t, g.Jx_band, lss[a], color=col[a], label=fr'$\alpha={a:g}$')
    ax.set_xlim(-20, 20); ax.set_xlabel(r'$\omega_0(t-t_c)$'); ax.set_ylabel(r'$J_x$  ($e\omega_0^2/v_F$)'); ax.legend(loc='upper left')
    ax.set_title(r'$\zeta=0$, reference $P_-(t)$, $\kappa_c\to\infty$', fontsize=7, color=INK2); panel(ax, 'b', -0.16)
    # (c) peak current versus 1/kappa_c
    ax = axs[2]
    for a in al[::-1]:
        for z, fill, ls, comp in ((0.0, True, '-', 'x'), (1.0, False, '--', 'x')):
            g = cut[(cut.zeta == z) & (cut.alpha == a) & (cut.kappa_c >= 3)].sort_values('kappa_c')
            ax.plot(1 / g.kappa_c, g[f'peak_J{comp}_band'], ls, color=col[a], marker=mk[a], mfc=col[a] if fill else 'white', mec=col[a], ms=3.5)
            e = B4[(B4.zeta == z) & (B4.alpha == a) & np.isinf(B4.kappa_c)]
            ax.errorbar([0.0], e[f'peak_J{comp}_band'], yerr=e[f'peak_J{comp}_band_err'], fmt=mk[a], color=col[a], mfc=col[a] if fill else 'white', mec=col[a], ms=5, capsize=2, elinewidth=0.7, clip_on=False, zorder=5)
    ax.set_xlim(-0.012, 0.35); ax.set_xlabel(r'inverse cutoff $1/\kappa_c$'); ax.set_ylabel(r'peak current $\max_t|J_x|$  ($e\omega_0^2/v_F$)')
    ax.legend(handles=[key(col[a], '-', mk[a], label=fr'$\alpha={a:g}$') for a in al[::-1]] + [key(INK2, '-', 'o', label=r'linear ($\zeta=0$)'), key(INK2, '--', 'o', 'white', r'circular ($\zeta=1$)')],
              loc='upper center', ncol=3, columnspacing=1.0, handlelength=2.2)
    ax.set_ylim(0.115, 0.385)
    ax.text(0.004, 0.118, r'$\kappa_c\to\infty$', fontsize=6, color=INK2, rotation=90, va='bottom')
    panel(ax, 'c', -0.16)
    # (d) intraband and interband parts at kappa_c = 32
    ax = axs[3]
    for a, ls in ((1.0, '-'), (0.8, '--')):
        g = tr[(tr.zeta == 0.0) & (tr.alpha == a) & (tr.kappa_c == 32.0)]
        ax.plot(g.t, g.Jx_intra, ls, color=BLUE); ax.plot(g.t, g.Jx_band - g.Jx_intra, ls, color=ORANGE)
    ax.set_xlim(-20, 20); ax.set_xlabel(r'$\omega_0(t-t_c)$'); ax.set_ylabel(r'$J_x$  ($e\omega_0^2/v_F$)')
    ax.legend(handles=[key(BLUE, '-', label=r'intraband, $\alpha=1$'), key(BLUE, '--', label=r'intraband, $\alpha=0.8$'), key(ORANGE, '-', label=r'interband, $\alpha=1$'),
                       key(ORANGE, '--', label=r'interband, $\alpha=0.8$')], loc='upper left', ncol=2, columnspacing=1.0)
    ax.set_ylim(-0.36, 0.5)
    ax.set_title(r'$\zeta=0$, $\kappa_c=32$', fontsize=7, color=INK2); panel(ax, 'd', -0.16)
    fig.tight_layout(w_pad=1.2, h_pad=1.2); save(fig, 'FigDet_Dirac_current')


# ====================================================================================================== tail
def fig_tail():
    T = pd.read_csv(os.path.join(TD, 'Table_Dirac_tail.csv')); cut = pd.read_csv(os.path.join(FD, 'FigDet_B4_cutoff.csv'))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9)); axs = axs.ravel()
    al = [0.8, 0.9]; col = {0.8: RAMP3[0], 0.9: RAMP3[1], 1.0: RAMP3[2]}; mk = {0.8: '^', 0.9: 's', 1.0: 'o'}
    # (a) K^2 n_c along rays against the analytical constants
    ax = axs[0]; R = T[T.study == 'rays']
    for a in al:
        for (z, ph, xi, ls, fill) in ((1.0, 0.0, 1, '-', True), (1.0, 0.0, -1, '--', False), (0.0, 90.0, 1, '-.', True)):
            g = R[(R.zeta == z) & (R.phi_deg == ph) & (R.xi == xi) & (R.alpha == a) & (R.quantity == 'K^2 n_c')].sort_values('K')
            if not len(g):
                continue
            ax.semilogx(g.K, g.value, ls, color=col[a], marker=mk[a], mfc=col[a] if fill else 'white', mec=col[a])
            ax.plot([55, 80], [g.analytic.values[0]] * 2, '-', color=INK, lw=1.4, solid_capstyle='butt')
    ax.set_xlim(1.7, 82); ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64']); ax.minorticks_off()
    ax.set_xlabel(r'$|\boldsymbol{\kappa}|$'); ax.set_ylabel(r'$|\boldsymbol{\kappa}|^2 n_c(\boldsymbol{\kappa},t_f)$')
    ax.legend(handles=[key(col[0.8], '-', '^', label=r'$\alpha=0.8$'), key(col[0.9], '-', 's', label=r'$\alpha=0.9$'), key(INK2, '-', 'o', label=r'circular, valley $K$'),
                       key(INK2, '--', 'o', 'white', r"circular, valley $K'$"), key(INK2, '-.', 'o', label=r'linear, $\boldsymbol{\kappa}\perp\mathbf{e}_1$'), key(INK, '-', label=r'analytical $C_\alpha$', lw=1.4)],
              loc='upper right', ncol=2, columnspacing=1.0)
    ax.set_ylim(0.07, 0.36); panel(ax, 'a', -0.16)
    # (b) exponents: -2 (alpha<1), -4 (alpha<1 along the polarisation axis; alpha=1), -6 (alpha=1 along the axis)
    ax = axs[1]; A1 = T[T.study == 'alpha1']
    def nc_of(g):
        return np.array([float(s.split('n_c=')[1].split(';')[0]) for s in g.note])
    series = [(R[(R.zeta == 0.0) & (R.phi_deg == 90.0) & (R.xi == 1) & (R.alpha == 0.8)], col[0.8], '-', '^', True),
              (R[(R.zeta == 0.0) & (R.phi_deg == 0.0) & (R.xi == 1) & (R.alpha == 0.8)], col[0.8], '--', '^', False),
              (A1[(A1.zeta == 0.0) & (A1.phi_deg == 90.0) & (A1.xi == 1)], col[1.0], '-', 'o', True),
              (A1[(A1.zeta == 0.0) & (A1.phi_deg == 0.0) & (A1.xi == 1)], col[1.0], '--', 'o', False)]
    for g, c, ls, m, fill in series:
        g = g.sort_values('K')
        if len(g):
            ax.loglog(g.K, nc_of(g), ls, color=c, marker=m, mfc=c if fill else 'white', mec=c)
    guide(ax, 16, 64, 2.2e-3, -2, r'$-2$', 1.1); guide(ax, 16, 64, 1.5e-8, -4, r'$-4$', 1.1); guide(ax, 16, 64, 1.6e-11, -6, r'$-6$', 1.1)
    ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64']); ax.minorticks_off(); ax.set_xlim(1.7, 110)
    ax.set_xlabel(r'$|\boldsymbol{\kappa}|$'); ax.set_ylabel(r'$n_c(\boldsymbol{\kappa},t_f)$, linear polarisation')
    ax.legend(handles=[key(col[0.8], '-', '^', label=r'$\alpha=0.8$, $\boldsymbol{\kappa}\perp\mathbf{e}_1$'), key(col[0.8], '--', '^', 'white', r'$\alpha=0.8$, $\boldsymbol{\kappa}\parallel\mathbf{e}_1$'),
                       key(col[1.0], '-', 'o', label=r'$\alpha=1$, $\boldsymbol{\kappa}\perp\mathbf{e}_1$'), key(col[1.0], '--', 'o', 'white', r'$\alpha=1$, $\boldsymbol{\kappa}\parallel\mathbf{e}_1$')], loc='lower left')
    panel(ax, 'b', -0.16)
    # (c) total density and valley contrast versus the cutoff
    ax = axs[2]
    for a in (1.0, 0.9, 0.8):
        g = cut[(cut.zeta == 1.0) & (cut.alpha == a)].sort_values('kappa_c')
        ax.semilogx(g.kappa_c, g.N_tot, '-', color=col[a], marker=mk[a], mfc=col[a], mec=col[a], ms=3.5)
        ax.semilogx(g.kappa_c, g.dN, '--', color=col[a], marker=mk[a], mfc='white', mec=col[a], ms=3.5)
        s = T[(T.study == 'log_slope') & (T.zeta == 1.0) & (T.alpha == a) & (T.quantity.str.contains('between 16 and 32'))]
        if len(s):
            x = np.array([9.0, 32.0]); y1 = g[g.kappa_c == 32.0].N_tot.values[0]
            ax.plot(x, y1 + s.analytic.values[0] * np.log(x / 32.0) + 0.012, ':', color=MUTED, lw=0.9)
    ax.text(13.0, 0.215, r'dotted: slope $2\langle C_\alpha\rangle/\pi$', fontsize=6, color=INK2)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32']); ax.minorticks_off()
    ax.set_xlabel(r'cutoff $\kappa_c$'); ax.set_ylabel(r'density at $t_f$  ($(\omega_0/v_F)^2$)')
    ax.legend(handles=[key(col[a], '-', mk[a], label=fr'$\alpha={a:g}$') for a in (1.0, 0.9, 0.8)] + [key(INK2, '-', 'o', label=r"$N_K+N_{K'}$"), key(INK2, '--', 'o', 'white', r"$N_K-N_{K'}$")],
              loc='upper left', ncol=2, columnspacing=1.0)
    ax.set_title(r'circular polarisation ($\zeta=1$)', fontsize=7, color=INK2); panel(ax, 'c', -0.16)
    # (d) dependence on the final time and on the dissipation
    ax = axs[3]; fp = os.path.join(FD, 'FigDet_tail_constant_vs_tf.csv')
    if os.path.exists(fp):
        c = pd.read_csv(fp)
        for a in al:
            g = c[c.alpha == a].sort_values('tf')
            ax.loglog(g.tf + 30, g.C_paper, '-', color=col[a]); ax.loglog(g.tf + 30, g.C_no_dissipation, '--', color=col[a])
        F = T[(T.study == 'final_time') & (T.K == 32.0)]
        for a in al:
            g = F[F.alpha == a]; ax.loglog(g.t_f + 30, g.value, mk[a], color=col[a], mfc='white', mec=col[a], ms=5, ls='none')
        ax.set_xlabel(r'window length $\omega_0(t_f-t_0)$'); ax.set_ylabel(r'$C_\alpha(t_f)=\lim|\boldsymbol{\kappa}|^2n_c$')
        ax.legend(handles=[key(col[0.8], '-', label=r'$\alpha=0.8$'), key(col[0.9], '-', label=r'$\alpha=0.9$'), key(INK2, '-', label='analytical, rates of the benchmark'),
                           key(INK2, '--', label=r'analytical, $\bar\Gamma_\downarrow=\bar\Gamma_\phi=0$'), key(INK2, 'none', 'o', 'white', r'simulation, $|\boldsymbol{\kappa}|=32$')], loc='lower left')
        ax.set_xticks([60, 120, 240, 480]); ax.set_xticklabels(['60', '120', '240', '480']); ax.minorticks_off()
        ax.set_title(r'circular polarisation, $\boldsymbol{\kappa}\parallel x$', fontsize=7, color=INK2)
    panel(ax, 'd', -0.16)
    fig.tight_layout(w_pad=1.2, h_pad=1.2); save(fig, 'FigDet_Dirac_tail')


# ====================================================================================================== maps
def fig_maps():
    z = np.load(os.path.join(FD, 'FigDet_B4_maps_zeta1.npz')); kx = z['kappa_axis']; km = 8.0
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ)
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.86, 3.0), sharey=True)
    vmax = max(z['nc_K_alpha1.0'].max(), z['nc_K_alpha0.8'].max())
    for ax, a, pl in ((axs[0], 1.0, 'a'), (axs[1], 0.8, 'b')):
        M = z[f'nc_K_alpha{a}']
        im = ax.imshow(np.maximum(M.T, 1e-6), origin='lower', extent=[-km, km, -km, km], cmap=cmap, norm=LogNorm(vmin=1e-5, vmax=vmax), interpolation='nearest', rasterized=True)
        ax.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], '-', color=INK, lw=0.6); ax.grid(False)
        ax.set_xlabel(r'$\kappa_x$'); ax.set_title(fr'valley $K$, $\zeta=1$, $\alpha={a:g}$', fontsize=7, color=INK2); panel(ax, pl, -0.1)
        ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([-8, -4, 0, 4, 8])
    axs[0].set_ylabel(r'$\kappa_y$'); axs[0].text(-2.8, 3.2, r'$\kappa_c=3$', fontsize=6, color=INK)
    fig.tight_layout(w_pad=0.6)
    cb = fig.colorbar(im, ax=axs, fraction=0.035, pad=0.02); cb.set_label(r'$n_c(\boldsymbol{\kappa},t_f)$'); cb.outline.set_linewidth(0.4)
    save(fig, 'FigDet_Dirac_maps')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'all'
    todo = dict(E=[fig_E], SOE=[fig_SOE], B3=[fig_B3], current=[fig_current], tail=[fig_tail], maps=[fig_maps], all=[fig_E, fig_SOE, fig_B3, fig_current, fig_tail, fig_maps])[what]
    for f in todo:
        try:
            f()
        except FileNotFoundError as e:
            print('skipped', f.__name__, '(missing input:', e.filename, ')')
