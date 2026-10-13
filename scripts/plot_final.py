"""Figures of the article: figures/final/Figure_01.pdf ... Figure_07.pdf.

Reads only figures/data/*.csv and tables/data/*.csv written by scripts/aggregate.py; no number is typed in.
Usage: python scripts/plot_final.py [1 2 ... 7] [--preview <dir for PNG copies>]

Format of every figure: one row of panels, 165 mm x 44 mm, vector PDF with embedded TrueType fonts (DejaVu Sans,
axis labels 8 pt, tick labels and legends 7 pt), panel labels (a), (b), ... inside the axes, no titles.
Colours (Okabe-Ito palette), the same in all figures: L1 time stepper, exact and reference solutions black;
unconstrained output grey; linear head blue; factor map vermilion; additive map green; block-gate map purple;
exponential map orange. In the figures of the Dirac model the orders alpha = 1, 0.9, 0.8 are black, blue and
vermilion. Every series also differs by position, marker shape, fill or line style, so that no figure depends on
colour alone and all of them read in grey-scale print. Magnitude on a grid uses a grey ramp.
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, 'figures', 'data'); TD = os.path.join(ROOT, 'tables', 'data'); OUT = os.path.join(ROOT, 'figures', 'final')
PREVIEW = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
W_MM, H_MM = 165.0, 44.0                       # size of every figure
TOP, BOT = 4.9, 8.3                            # margins above (legend row) and below (tick labels, axis label) the panels, mm
BLACK, GREY, BLUE, VERMILION, GREEN, PURPLE, ORANGE = '#000000', '#4D4D4D', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00'
LIGHT, GUIDE, RULE = '#8C8C8C', '#9A9A9A', '#E3E3E3'
SEQ = LinearSegmentedColormap.from_list('grey_seq', ['#F4F4F4', '#C9C9C9', '#8F8F8F', '#555555', '#141414'])
MAPS = ['unc', 'lin', 'factor', 'add', 'gate2b']
MAP_COLOR = {'unc': GREY, 'lin': BLUE, 'factor': VERMILION, 'add': GREEN, 'gate2b': PURPLE}
MAP_TICK = {'unc': 'unc.', 'lin': 'lin.', 'factor': 'fac.', 'add': 'add.', 'gate2b': 'block'}
ORDER = {1.0: (BLACK, 'o', '-'), 0.9: (BLUE, 's', '-'), 0.8: (VERMILION, '^', '-')}      # Dirac model: colour and marker of the order alpha
DASH, DASHDOT, DOT = (0, (4.5, 1.8)), (0, (5.5, 1.6, 1.2, 1.6)), (0, (1.1, 1.3))
DASHDOTDOT = (0, (5.5, 1.4, 1.2, 1.4, 1.2, 1.4))
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'font.size': 7,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.0, 'lines.markersize': 3.4, 'axes.linewidth': 0.5,
    'axes.edgecolor': BLACK, 'axes.labelcolor': BLACK, 'text.color': BLACK, 'xtick.color': BLACK, 'ytick.color': BLACK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'xtick.minor.size': 1.3, 'ytick.minor.size': 1.3,
    'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8, 'axes.labelpad': 1.8, 'axes.grid': False, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.handlelength': 2.3,
    'legend.handletextpad': 0.5, 'legend.columnspacing': 1.5, 'legend.borderaxespad': 0.0, 'legend.borderpad': 0.0,
    'legend.labelspacing': 0.25, 'savefig.dpi': 300})
PLAIN = FuncFormatter(lambda v, _: f'{v:g}')


def read(name, folder=FD):
    return pd.read_csv(os.path.join(folder, name), float_precision='round_trip')


def new_fig():
    return plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))


def axes(fig, x, w, bottom=BOT, top=TOP):
    """Axes with the left edge at x and the width w (mm); the panels of a figure share bottom and top margins."""
    return fig.add_axes([x / W_MM, bottom / H_MM, w / W_MM, (H_MM - bottom - top) / H_MM])


def tag(ax, letter, text='', box=False):
    """Panel label in the upper left corner inside the axes, followed by a short description of the panel."""
    kw = dict(bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85)) if box else {}
    t = ax.text(0.025, 0.965, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontsize=8, fontweight='bold', zorder=30, **kw)
    if text:
        ax.annotate(text, xy=(1, 0.5), xycoords=t, xytext=(2.5, 0), textcoords='offset points', ha='left', va='center', fontsize=7, zorder=30, **kw)


def legend_row(fig, groups, centers=None, gap=5.0):
    """Legend entries in one row in the margin above the panels. groups: list of (header or None, handles, legend options).
    Without centers the groups are packed with the given gap (mm) and centred on the page; with centers (mm from the left
    edge) each group is centred there and shifted only as far as needed to stay on the page and clear of its neighbours."""
    y = 1.0 - 0.9 / H_MM; px = fig.dpi / 25.4; page = fig.bbox.width; items = []
    for header, handles, kw in groups:
        leg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, y), ncol=len(handles), **kw)
        items.append((leg, fig.text(0.0, 0.5, header, ha='left', va='baseline') if header else None))
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    wh = [t.get_window_extent(r).width + 1.2 * px if t else 0.0 for _, t in items]; wl = [leg.get_window_extent(r).width for leg, _ in items]
    w = [a + b for a, b in zip(wh, wl)]; n = len(w)
    if centers is None:
        total = sum(w) + gap * px * (n - 1); x = [(page - total) / 2 + sum(w[:i]) + gap * px * i for i in range(n)]
    else:
        x = [c * px - wi / 2 for c, wi in zip(centers, w)]
        for i in range(n):
            x[i] = max(x[i], 1.0 * px if i == 0 else x[i - 1] + w[i - 1] + 3.0 * px)
        for i in reversed(range(n)):
            x[i] = min(x[i], page - 1.0 * px - w[i] if i == n - 1 else x[i + 1] - 3.0 * px - w[i])
    if x[0] < 0.99 * px or any(x[i] + w[i] > (x[i + 1] if i + 1 < n else page) + 1e-6 for i in range(n)):
        raise RuntimeError('the legend row does not fit on the page')
    for (leg, t), xi, whi in zip(items, x, wh):
        leg.set_bbox_to_anchor(((xi + whi) / page, y), transform=fig.transFigure)
    fig.canvas.draw()
    for (leg, t), xi in zip(items, x):
        if t:
            base = leg.get_texts()[0]; yb = base.get_transform().transform(base.get_position())[1]
            t.set_position((xi / page, yb / fig.bbox.height))


def legend_top(fig, handles, **kw):
    """One centred row of legend entries in the margin above the panels."""
    legend_row(fig, [(None, handles, kw)])


def loglim(lo, hi, head=0.14, foot=0.04):
    """Limits of a logarithmic axis that leave the fractions head and foot of its length free above and below the data."""
    span = math.log10(hi / lo) / (1.0 - head - foot)
    return 10 ** (math.log10(lo) - foot * span), 10 ** (math.log10(hi) + head * span)


def face(color, fill='full'):
    """Marker faces: 'none' open, 'left' half filled, 'full' filled."""
    if fill == 'none':
        return dict(mfc='white', mec=color)
    return dict(mfc=color, mec=color, fillstyle=fill, markerfacecoloralt='white')


def decades(ax, axis, lo, hi, step=1):
    """Major ticks at 10^k for k = lo, lo + step, ..., hi and unlabelled minor ticks at the decades in between."""
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(FixedLocator([10.0 ** k for k in range(lo, hi + 1, step)]))
    a.set_minor_locator(FixedLocator([10.0 ** k for k in range(lo, hi + 1)])); a.set_minor_formatter(NullFormatter())


def check_text(fig, n):
    """Layout check: no text may leave the page, and no two texts may overlap (tick labels of neighbouring panels, legends)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); page = fig.bbox; shown = list(fig.texts) + [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        shown += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label] + (list(ax.get_legend().get_texts()) if ax.get_legend() else [])
        for axis in (ax.xaxis, ax.yaxis):                                    # labels of the ticks inside the view limits only
            lo, hi = sorted(axis.get_view_interval()); eps = 1e-9 * max(abs(lo), abs(hi))
            shown += [lab for tick in axis.get_major_ticks() + axis.get_minor_ticks() if lo - eps <= tick.get_loc() <= hi + eps for lab in (tick.label1, tick.label2)]
    texts = [(t.get_text(), t.get_window_extent(r)) for t in shown if t.get_visible() and t.get_text().strip()]
    texts = [(s, b) for s, b in texts if b.width > 0 and b.height > 0]
    out = [s for s, b in texts if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > page.width + 0.5 or b.y1 > page.height + 0.5]
    hit = [(s, u) for i, (s, b) in enumerate(texts) for u, c in texts[i + 1:]
           if min(b.x1, c.x1) - max(b.x0, c.x0) > 0.6 and min(b.y1, c.y1) - max(b.y0, c.y0) > 0.6]
    if out or hit:
        raise RuntimeError(f'Figure {n}: text outside the page: {out}; overlapping texts: {hit}')


def save(fig, n):
    os.makedirs(OUT, exist_ok=True); name = f'Figure_{n:02d}'; check_text(fig, n)
    fig.savefig(os.path.join(OUT, name + '.pdf'), metadata={'CreationDate': None})      # no date: identical bytes on regeneration
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True); fig.savefig(os.path.join(PREVIEW, name + '.png'), dpi=300)
    plt.close(fig); print(f'wrote figures/final/{name}.pdf')


# ------------------------------------------------------------------------------------- Figure 1: factorial study (B2)
IC = [('soft-tuned', 's', 'soft'), ('hard-exp', '^', r'hard, $g_1$'), ('hard-pow', 'o', r'hard, $g_\alpha$')]
STATES = [('mixed', 'none', DOT, 0.8, 'mixed'), ('nearly_pure', 'left', DASH, 1.3, 'nearly pure'), ('pure', 'full', '-', 0.5, 'pure')]      # fill, line of the stepper


def figure_1():
    """Relative error of every run and median per initial state, by output map and initial condition; one panel per order."""
    runs = read('Neural_B2_runs.csv'); st = read('Neural_B2_stepper.csv'); st = st[st.N == 128]
    runs = runs[(runs.stage == 'main') & (runs.status == 'ok') & (runs.feature == 'plain') & runs['map'].isin(MAPS)]
    alphas = sorted(runs.alpha.unique()); lo, hi = loglim(100 * min(runs.rel_fro.min(), st.rel_fro.min()), 100 * runs.rel_fro.max(), head=0.13)
    fig = new_fig(); x0, gap = 11.0, 2.0; w = (W_MM - x0 - 1.0 - gap * (len(alphas) - 1)) / len(alphas)
    for j, a in enumerate(alphas):
        ax = axes(fig, x0 + j * (w + gap), w, bottom=5.2); d = runs[runs.alpha == a]
        for k, m in enumerate(MAPS):
            for q, (ig, mk, _) in enumerate(IC):
                for s, (state, fill, _, _, _) in enumerate(STATES):
                    v = 100 * d[(d['map'] == m) & (d.ic_gate == ig) & (d.state == state)].rel_fro.values
                    if len(v):
                        x = k + 0.29 * (q - 1) + 0.088 * (s - 1)
                        ax.plot(np.full(len(v), x), v, ls='none', marker=mk, ms=1.7, mew=0.3, alpha=0.85, zorder=3, **face(MAP_COLOR[m], fill))
                        ax.plot([x], [np.median(v)], ls='none', marker=mk, ms=3.7, mew=0.55, zorder=4 + s, **face(MAP_COLOR[m], fill))
        for state, _, ls, lw, _ in STATES:          # the values for the nearly pure and the pure state almost coincide: thick dashes on a thin line
            ax.axhline(100 * st[(st.alpha == a) & (st.state == state)].rel_fro.iloc[0], color=BLACK, lw=lw, ls=ls, zorder=2)
        for k in range(1, len(MAPS)):
            ax.axvline(k - 0.5, color=RULE, lw=0.5, zorder=1)
        ax.set_yscale('log'); ax.set_ylim(lo, hi); ax.set_xlim(-0.5, len(MAPS) - 0.5); ax.yaxis.set_major_formatter(PLAIN)
        ax.set_xticks(range(len(MAPS))); ax.set_xticklabels([MAP_TICK[m] for m in MAPS]); ax.tick_params(axis='x', length=0)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r'$e_F$ (%)')
        tag(ax, 'abc'[j], rf'$\alpha={a:g}$')
    shapes = [Line2D([], [], ls='none', marker=mk, ms=3.7, mfc=BLACK, mec=BLACK, label=lab) for _, mk, lab in IC]
    fills = [Line2D([], [], color=BLACK, lw=lw, ls=ls, marker='o', ms=3.7, mew=0.55, label=lab, **face(BLACK, fill)) for _, fill, ls, lw, lab in STATES]
    legend_row(fig, [('initial condition:', shapes, dict(handlelength=1.0, handletextpad=0.3, columnspacing=1.2)),      # marker shape
                     ('initial state:', fills, dict(handlelength=3.0, columnspacing=1.2))], gap=8.0)                  # fill, line of the stepper
    save(fig, 1)


# ------------------------------------------------------------------------------------- Figure 2: initial layer (B2)
LAYER = [('factor/hard-exp', VERMILION, DASH, r'factor, $g_1$'), ('factor/hard-pow', VERMILION, '-', r'factor, $g_\alpha$'),
         ('lin/hard-pow', BLUE, DASHDOT, r'linear head, $g_\alpha$'), ('gate2b/hard-pow', PURPLE, DOT, r'block gate, $g_\alpha$'),
         ('add/hard-pow', GREEN, DASHDOTDOT, r'additive, $g_\alpha$')]
TAU_MIN = 1e-8
LW = {False: 1.0, True: 1.4}          # dotted lines are drawn wider


def figure_2():
    """Error and smallest eigenvalue in the initial layer for the pure initial state."""
    c = read('Neural_B2_error_curves.csv'); c = c[(c.state == 'pure') & (c.tau >= TAU_MIN * (1 - 1e-9))]
    fig = new_fig(); w = (W_MM - 12.0 - 13.0 - 6.0 - 3.4) / 3
    axs = [axes(fig, 12.0, w), axes(fig, 12.0 + w + 13.0, w), axes(fig, 12.0 + 2 * w + 13.0 + 6.0, w)]
    d = c[c.alpha == 0.6]; ax = axs[0]
    for cell, col, ls, _ in LAYER:
        g = d[d.cell == cell]
        ax.fill_between(g.tau, g.err_fro_q25, g.err_fro_q75, color=col, alpha=0.16, lw=0); ax.loglog(g.tau, g.err_fro_median, color=col, ls=ls, lw=LW[ls is DOT])
    lo, hi = loglim(d[d.cell.isin([s[0] for s in LAYER])].err_fro_q25.min(), d[d.cell.isin([s[0] for s in LAYER])].err_fro_q75.max(), head=0.16)
    ax.set_ylim(lo, hi); ax.set_ylabel(r'$\|\rho_\theta-\rho\|_F$'); tag(ax, 'a', r'$\alpha=0.6$')
    cells = c[c.alpha.isin([0.3, 0.6]) & c.cell.isin([s[0] for s in LAYER])]
    lo, hi = loglim(cells.min_eig_median.min(), max(cells.min_eig_median.max(), cells.min_eig_exact.max()), head=0.16, foot=0.02)
    for ax, a, s in ((axs[1], 0.6, 'b'), (axs[2], 0.3, 'c')):
        d = c[c.alpha == a]; g = d[d.cell == LAYER[1][0]]
        ax.loglog(g.tau, g.min_eig_exact, color=BLACK, lw=2.0, zorder=2)
        for cell, col, ls, _ in LAYER:
            g = d[d.cell == cell]; ax.loglog(g.tau, g.min_eig_median, color=col, ls=ls, lw=LW[ls is DOT], zorder=3)
        t = np.array([1e-6, 1e-2]); y0 = lo * 30                                          # guide lines of slope alpha and 2 alpha in the free corner
        for p, lab in ((a, r'$\tau^{\alpha}$'), (2 * a, r'$\tau^{2\alpha}$')):
            ax.loglog(t, y0 * (t / t[0]) ** p, color=GUIDE, lw=0.6, zorder=1)
            ax.text(t[1] * 1.5, y0 * (t[1] / t[0]) ** p, lab, color=GREY, fontsize=7, va='center')
        ax.set_ylim(lo, hi); tag(ax, s, rf'$\alpha={a:g}$')
    axs[1].set_ylabel(r'$\lambda_{\min}$'); axs[2].tick_params(labelleft=False)
    for ax in axs:
        ax.set_xlim(TAU_MIN, 1.0); decades(ax, 'x', -8, 0, 2); ax.set_xlabel(r'$\tau$')
    decades(axs[1], 'y', math.ceil(math.log10(lo)), 0, 2); decades(axs[2], 'y', math.ceil(math.log10(lo)), 0, 2)
    h = [Line2D([], [], color=col, ls=ls, lw=LW[ls is DOT], label=lab) for _, col, ls, lab in LAYER] + [Line2D([], [], color=BLACK, lw=2.0, label='exact')]
    legend_top(fig, h); save(fig, 2)


# ------------------------------------------------------------------------------------- Figure 3: positivity for d = 3 and d = 4 (B6)
CASES = [('qutrit', 0.5, 'o', r'$d{=}3$, $\alpha{=}0.5$'), ('qutrit', 0.8, 's', r'$d{=}3$, $\alpha{=}0.8$'),
         ('two_qubit', 0.5, '^', r'$d{=}4$, $\alpha{=}0.5$'), ('two_qubit', 0.8, 'D', r'$d{=}4$, $\alpha{=}0.8$')]
LINTHRESH = 1e-6


def figure_3():
    """Error and smallest eigenvalue of every run by output map; smallest eigenvalue along the trajectory."""
    runs = read('Neural_B6_runs.csv'); st = read('Neural_B6_stepper.csv'); st = st[st.N == 128]; c = read('Neural_B6_curves.csv')
    runs = runs[(runs.stage == 'main') & (runs.status == 'ok') & (runs.gate == 'pow') & runs['map'].isin(MAPS)]
    fig = new_fig(); wa = 35.5; wc = (W_MM - 10.5 - 2 * wa - 2 * 13.0 - 6.5 - 3.4) / 2
    xa = 10.5; xb = xa + wa + 13.0; xc = xb + wa + 13.0; xd = xc + wc + 6.5
    axa, axb, axc, axd = axes(fig, xa, wa), axes(fig, xb, wa), axes(fig, xc, wc), axes(fig, xd, wc)
    for k, m in enumerate(MAPS):
        for q, (system, a, mk, _) in enumerate(CASES):
            d = runs[(runs.system == system) & (runs.alpha == a) & (runs['map'] == m)]; x = k + 0.2 * (q - 1.5)
            kw = dict(ls='none', marker=mk, ms=2.3 if mk != 'D' else 2.0, mew=0.35, alpha=0.9, zorder=3, **face(MAP_COLOR[m], 'full' if a == 0.5 else 'none'))
            axa.plot(np.full(len(d), x), 100 * d.rel_fro, **kw); axb.plot(np.full(len(d), x), d.min_eig, **kw)
            f = 100 * st[(st.system == system) & (st.alpha == a)].rel_fro.iloc[0]
            axa.plot([x - 0.085, x + 0.085], [f, f], color=BLACK, lw=1.0, solid_capstyle='butt', zorder=4)
    for ax in (axa, axb):
        for k in range(1, len(MAPS)):
            ax.axvline(k - 0.5, color=RULE, lw=0.5, zorder=1)
        ax.set_xlim(-0.5, len(MAPS) - 0.5); ax.set_xticks(range(len(MAPS))); ax.set_xticklabels([MAP_TICK[m] for m in MAPS])
        ax.tick_params(axis='x', length=0); ax.set_xlabel('output map')
    axa.set_yscale('log'); axa.set_ylim(*loglim(100 * min(runs.rel_fro.min(), st.rel_fro.min()), 100 * runs.rel_fro.max(), head=0.15))
    axa.yaxis.set_major_formatter(PLAIN); axa.set_ylabel(r'$e_F$ (%)'); tag(axa, 'a')
    axb.axhline(0.0, color=BLACK, lw=0.5, zorder=2); axb.set_yscale('symlog', linthresh=LINTHRESH, linscale=0.6)
    kb = math.floor(math.log10(-runs.min_eig.min())) + 1
    axb.set_ylim(-10.0 ** kb * 2, LINTHRESH * 8); axb.yaxis.set_major_locator(FixedLocator([-10.0 ** k for k in range(kb, -6, -1)] + [0.0]))
    axb.yaxis.set_minor_locator(NullLocator()); axb.set_ylabel(r'$\mathrm{min}_{\tau}\,\lambda_{\min}(\tau)$', labelpad=0.5); tag(axb, 'b')
    for ax, system, s in ((axc, 'qutrit', 'c'), (axd, 'two_qubit', 'd')):
        g = c[(c.system == system) & (c.alpha == 0.5) & (c.tau >= TAU_MIN * (1 - 1e-9))]
        lin = g[g.cell == 'lin/hard-pow']; fac = g[g.cell == 'factor/hard-pow']
        ax.plot(lin.tau, lin.min_eig_exact, color=BLACK, lw=2.0, zorder=2)
        ax.plot(lin.tau, lin.min_eig_min, color=BLUE, ls=DOT, lw=0.9, zorder=3); ax.plot(lin.tau, lin.min_eig_median, color=BLUE, zorder=4)
        ax.plot(fac.tau, fac.min_eig_median, color=VERMILION, ls=DASH, zorder=5)
        ax.axhline(0.0, color=BLACK, lw=0.4, zorder=1); ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=0.6)
        ax.set_xlim(TAU_MIN, 1.0); decades(ax, 'x', -8, 0, 4); ax.set_xlabel(r'$\tau$'); ax.set_ylim(-3e-2, 3.0)
        ax.yaxis.set_major_locator(FixedLocator([-1e-2, -1e-4, 0.0, 1e-4, 1e-2])); ax.yaxis.set_minor_locator(NullLocator())
        ax.annotate('exact', xy=(TAU_MIN * 1.3, 0.0), xytext=(0, 2.0), textcoords='offset points', ha='left', va='bottom', fontsize=7)
        tag(ax, s, rf'$d={3 if system == "qutrit" else 4}$, $\alpha=0.5$')
    axc.set_ylabel(r'$\lambda_{\min}(\tau)$', labelpad=0.5); axd.tick_params(labelleft=False)
    axa.legend(handles=[Line2D([], [], color=BLACK, lw=1.0, label='L1 time stepper')], loc='lower center', bbox_to_anchor=(0.5, 0.115), handlelength=1.0)
    cases = [Line2D([], [], ls='none', marker=mk, ms=3.2, mew=0.5, label=lab, **face(BLACK, 'full' if a == 0.5 else 'none')) for _, a, mk, lab in CASES]
    lines = [Line2D([], [], color=BLUE, label='linear head'), Line2D([], [], color=BLUE, ls=DOT, lw=0.9, label='its minimum'),
             Line2D([], [], color=VERMILION, ls=DASH, label='factor')]
    legend_row(fig, [(None, cases, dict(handlelength=1.0, columnspacing=1.2, handletextpad=0.3)), (None, lines, dict(handlelength=2.0, columnspacing=1.1))],
               centers=[(xa + xb + wa) / 2, (xc + xd + wc) / 2])
    save(fig, 3)


# ------------------------------------------------------------------------------------- Figure 4: FENE-P (B5)
FENEP = [('sym-rel', GREY, '-', 0.6, 'unconstrained, relative'), ('sym-comp', LIGHT, DASH, 0.6, 'unconstrained, component-scaled'),
         ('blogm-nat', ORANGE, '-', 0.8, 'bounded-trace exponential map')]


def figure_4():
    """Stretch and smallest eigenvalue in start-up of uniaxial extension with the FENE-P term, all seeds."""
    c = read('B5_fenep_curves.csv'); fig = new_fig(); w = (W_MM - 10.0 - 13.5 - 3.0) / 2
    axa, axb = axes(fig, 10.0, w), axes(fig, 10.0 + w + 13.5, w); ref = c[c.variant == 'reference']
    for ax, col in ((axa, 'C11'), (axb, 'min_eig')):
        ax.plot(ref.t_over_lambda, ref[col], color=BLACK, lw=2.0, zorder=4)
        for v, colr, ls, lw, _ in FENEP:
            for _, g in c[c.variant == v].groupby('seed'):
                ax.plot(g.t_over_lambda, g[col], color=colr, ls=ls, lw=lw, zorder=5 if v == 'blogm-nat' else 3)
        ax.set_xlim(0.0, round(c.t_over_lambda.max() / c.tau.max(), 6)); ax.set_xticks(np.arange(0.0, ax.get_xlim()[1] + 1e-9, 0.5)); ax.set_xlabel(r'$t/\lambda$')
    axa.set_ylabel(r'$C_{11}$'); axa.set_ylim(c.C11.min() - 3, c.C11.max() * 1.22); tag(axa, 'a')
    axb.axhline(0.0, color=BLACK, lw=0.4, zorder=1); axb.set_yscale('symlog', linthresh=1e-3, linscale=0.6)
    kb = math.floor(math.log10(-c.min_eig.min())) + 1
    axb.set_ylim(-2.5 * 10.0 ** kb, 60.0); axb.yaxis.set_major_locator(FixedLocator([-10.0 ** k for k in range(kb, -3, -2)] + [0.0, 1e-2, 1.0]))
    axb.yaxis.set_minor_locator(NullLocator()); axb.set_ylabel(r'$\lambda_{\min}$', labelpad=0.5); tag(axb, 'b')
    h = [Line2D([], [], color=BLACK, lw=2.0, label='reference')] + [Line2D([], [], color=colr, ls=ls, lw=lw + 0.3, label=lab) for _, colr, ls, lw, lab in FENEP]
    legend_top(fig, h, handlelength=2.0, columnspacing=1.3); save(fig, 4)


# ------------------------------------------------------------------------------------- Figure 5: parametric surrogate
def figure_5():
    """Error over the parameter rectangle for two surrogates and the time stepper; total time against the number of queries."""
    g = read('Neural_surrogate_grid.csv'); st = read('Neural_surrogate_stepper.csv'); T = read('Table_surrogate.csv', TD); B = read('Table_surrogate_breakeven.csv', TD)
    med = g.groupby(['method', 'alpha', 'Gdown']).rel_fro.median().reset_index(); s128 = st[st.N == 128]
    grids = [('additive map', med[med.method == 'surrogate add']), ('linear head', med[med.method == 'surrogate lin']), (r'L1, $N=128$', s128)]
    vmin = 100 * min(d.rel_fro.min() for _, d in grids); vmax = 100 * max(d.rel_fro.max() for _, d in grids)
    fig = new_fig(); x0, wh, gap = 10.0, 25.0, 3.0; top = TOP - 3.2
    for k, (lab, d) in enumerate(grids):
        ax = axes(fig, x0 + k * (wh + gap), wh, top=top); pv = d.pivot_table(index='Gdown', columns='alpha', values='rel_fro')
        xs, ys = pv.columns.values, pv.index.values; dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        im = ax.pcolormesh(xs, ys, 100 * pv.values, cmap=SEQ, norm=LogNorm(vmin, vmax), shading='nearest', edgecolors='white', linewidth=0.4)
        ax.set_xlim(xs[0] - dx / 2, xs[-1] + dx / 2); ax.set_ylim(ys[0] - dy / 2, ys[-1] + dy / 2 + 1.25 * dy)       # free band for the panel label
        ax.set_xticks(np.round(np.linspace(xs[0], xs[-1], 3), 6)); ax.set_yticks(np.round(np.linspace(ys[0], ys[-1], 4), 6)); ax.set_xlabel(r'$\alpha$')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=1.5)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(r'$\gamma_\downarrow$')
        tag(ax, 'abc'[k], lab)
    xc = x0 + 3 * wh + 2 * gap + 1.2; cax = fig.add_axes([xc / W_MM, BOT / H_MM, 1.8 / W_MM, (H_MM - BOT - top) * (8 / 9.25) / H_MM])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.ax.yaxis.set_major_formatter(PLAIN); cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.set_label(r'$e_F$ (%)', labelpad=1.0); cb.ax.tick_params(length=1.8, which='major'); cb.ax.tick_params(length=1.0, which='minor')
    xd = xc + 1.8 + 21.0; ax = axes(fig, xd, W_MM - xd - 3.4, top=top); Q = np.logspace(2, 8, 241)
    sur = T[T.method.str.startswith('surrogate') & (T.status == 'ok')]; stp = T[T.method.str.startswith('l1_stepper')]
    rep = B[B.training_time == 'idle repeat'].iloc[0]
    lines = [(sur[sur.method == 'surrogate add'].train_wall_s.median(), sur[sur.method == 'surrogate add'].inference_ms_per_query.median(), GREEN, '-', 1.8, 'additive map'),
             (sur[sur.method == 'surrogate lin'].train_wall_s.median(), sur[sur.method == 'surrogate lin'].inference_ms_per_query.median(), BLUE, '-', 0.8, 'linear head'),
             (rep.train_wall_s, rep.inference_ms_per_query, BLUE, DASH, 0.8, f'repeated alone, {rep.train_wall_s:.0f} s')]
    steppers = [(int(m.split('=')[1]), stp[stp.method == m].stepper_ms_per_query.iloc[0], ls) for m, ls in (('l1_stepper N=32', DOT), ('l1_stepper N=64', DASH), ('l1_stepper N=128', '-'))]
    for t0, ms, col, ls, lw, _ in lines:
        ax.loglog(Q, t0 + Q * ms * 1e-3, color=col, ls=ls, lw=lw, zorder=3)
        for _, sm, _ in steppers:
            if sm > ms and Q[0] < t0 / ((sm - ms) * 1e-3) < Q[-1]:
                q = t0 / ((sm - ms) * 1e-3); ax.plot([q], [q * sm * 1e-3], ls='none', marker='o', ms=2.6, mfc='white', mec=BLACK, mew=0.5, zorder=6)
    for N, sm, ls in steppers:
        ax.loglog(Q, Q * sm * 1e-3, color=BLACK, ls=ls, lw=0.8, zorder=4)
    ax.set_xlim(Q[0], Q[-1]); decades(ax, 'x', 2, 8, 2); decades(ax, 'y', -1, 5, 2); ax.set_ylim(Q[0] * steppers[0][1] * 1e-3 / 2, 10 ** 6.9)
    ax.set_xlabel('trajectory queries'); ax.set_ylabel('total time (s)', labelpad=0.5); tag(ax, 'd')
    first = ax.legend(handles=[Line2D([], [], color=col, ls=ls, lw=lw, label=lab) for _, _, col, ls, lw, lab in lines], loc='upper left',
                      bbox_to_anchor=(0.03, 0.865), handlelength=2.0)
    ax.add_artist(first)
    ax.legend(handles=[Line2D([], [], color=BLACK, ls=ls, lw=0.8, label=rf'L1, $N={N}$') for N, _, ls in steppers[::-1]], loc='lower right',
              bbox_to_anchor=(0.985, 0.03), handlelength=2.0)
    save(fig, 5)


# ------------------------------------------------------------------------------------- Figure 6: tail of the excitation (Dirac model)
def figure_6():
    """Upper-band population on a ray, its coefficient against the direction, and excitation density against the cutoff."""
    R = read('FigDet_tail_rays.csv'); cut = read('FigDet_B4_cutoff.csv'); T = read('Table_Dirac_tail.csv', TD)
    R = R[R.study.isin(['rays', 'alpha1'])].copy(); R['p'] = R.quantity.str.extract(r'K\^(\d) n_c').astype(int); R['nc'] = R.value / R.K ** R.p
    fig = new_fig(); w = (W_MM - 12.0 - 2 * 12.5 - 1.6) / 3; axa, axb, axc = (axes(fig, 12.0 + k * (w + 12.5), w) for k in range(3))
    ray = R[(R.zeta == 0.0) & (R.phi_deg == 90.0) & (R.xi == 1.0)]                              # perpendicular to the polarisation axis
    for a, (col, mk, _) in ORDER.items():
        g = ray[ray.alpha == a].sort_values('K'); axa.loglog(g.K, g.nc, color=col, marker=mk, ms=3.2, lw=0.8, zorder=3)
    for a, p, lab in ((0.8, 2, r'$\kappa^{-2}$'), (1.0, 4, r'$\kappa^{-4}$')):
        g = ray[ray.alpha == a].sort_values('K'); k = np.array([8.0, 64.0]); y0 = g.nc.iloc[-1] * (4.0 if p == 2 else 0.2)
        axa.loglog(k, y0 * (k / 64.0) ** -p, color=GUIDE, lw=0.6, zorder=1); axa.text(11.0, y0 * (11.0 / 64.0) ** -p * (2.0 if p == 2 else 0.35), lab, color=GREY, fontsize=7, ha='center', va='center')
    axa.set_xlim(1.7, 76); axa.xaxis.set_major_locator(FixedLocator([2, 4, 8, 16, 32, 64])); axa.xaxis.set_major_formatter(PLAIN); axa.xaxis.set_minor_locator(NullLocator())
    axa.set_ylim(*loglim(ray.nc.min() * 0.2, ray.nc.max(), head=0.16)); decades(axa, 'y', -10, 0, 2)
    axa.set_xlabel(r'$\kappa$'); axa.set_ylabel(r'$n_c(t_f)$', labelpad=0.5); tag(axa, 'a', r'linear, $\varphi=%g^{\circ}$' % ray.phi_deg.iloc[0])
    K = R[(R.K == R.K.max()) & (R.alpha < 1)].copy(); K['k2nc'] = K.nc * K.K ** 2; phi = np.linspace(0.0, 90.0, 181)
    for a in (0.9, 0.8):
        col, mk, _ = ORDER[a]
        lin = K[(K.zeta == 0.0) & (K.alpha == a) & (K.xi == 1.0)].sort_values('phi_deg'); cir = K[(K.zeta == 1.0) & (K.alpha == a)].sort_values('phi_deg')
        axb.plot(phi, lin[lin.phi_deg == 90.0].analytic.iloc[0] * np.sin(np.radians(phi)) ** 2, color=col, lw=0.8, zorder=2)
        axb.plot(phi, np.full(len(phi), cir.analytic.iloc[0]), color=col, lw=0.8, ls=DASH, zorder=2)
        axb.plot(lin.phi_deg, lin.k2nc, ls='none', marker=mk, ms=3.6, zorder=4, **face(col)); axb.plot(cir.phi_deg, cir.k2nc, ls='none', marker=mk, ms=3.6, mew=0.7, zorder=5, **face(col, 'none'))
    axb.set_xlim(-6, 96); axb.set_xticks([0, 45, 90]); axb.set_ylim(-0.012, K.k2nc.max() * 1.3); axb.set_xlabel(r'$\varphi$ (deg)')
    axb.set_ylabel(r'$\kappa^{2}n_c(t_f)$', labelpad=1.0); tag(axb, 'b', r'$\kappa=%g$' % R.K.max())
    axb.legend(handles=[Line2D([], [], ls='none', marker='o', ms=3.4, mfc=BLACK, mec=BLACK, label='computed'),
                        Line2D([], [], color=BLACK, lw=0.8, label=r'$\mathcal{C}_\alpha(t_f;\varphi)$')], loc='upper left', bbox_to_anchor=(0.03, 0.855), handlelength=1.6)
    for a, (col, mk, _) in ORDER.items():
        for z, fill, ls in ((0.0, 'full', '-'), (1.0, 'none', DASH)):
            g = cut[(cut.zeta == z) & (cut.alpha == a)].sort_values('kappa_c')
            s = T[(T.study == 'log_slope') & (T.zeta == z) & (T.alpha == a) & T.quantity.str.contains('between 16 and 32')]
            if len(s):                                                                           # slope of the logarithmic law through the last point
                kc = np.array([g.kappa_c.min(), g.kappa_c.max()]); axc.plot(kc, g.N_tot.iloc[-1] + s.analytic.iloc[0] * np.log(kc / kc[1]), color=GUIDE, lw=0.6, zorder=1)
            axc.plot(g.kappa_c, g.N_tot, color=col, ls=ls, lw=0.7, marker=mk, ms=2.6, mew=0.5, zorder=3, **face(col, fill))
    axc.set_xscale('log'); axc.set_xlim(0.88, 36); axc.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32])); axc.xaxis.set_major_formatter(PLAIN)
    axc.xaxis.set_minor_locator(NullLocator()); axc.set_ylim(0.0, cut.N_tot.max() * 1.25); axc.set_xlabel(r'$\kappa_c$')
    axc.set_ylabel(r'$\mathcal{N}_{\mathrm{tot}}(t_f;\kappa_c)$', labelpad=1.0); tag(axc, 'c')
    h = [Line2D([], [], color=col, marker=mk, ms=3.2, lw=0.8, label=rf'$\alpha={a:g}$') for a, (col, mk, _) in ORDER.items()]
    h += [Line2D([], [], color=BLACK, marker='o', ms=3.2, lw=0.8, label='linear polarization'),
          Line2D([], [], color=BLACK, marker='o', ms=3.2, lw=0.8, ls=DASH, mfc='white', label='circular polarization'),
          Line2D([], [], color=GUIDE, lw=0.6, label='slopes')]
    legend_top(fig, h); save(fig, 6)


# ------------------------------------------------------------------------------------- Figure 7: sheet current (Dirac model)
T_UNRESOLVED = 15.0          # for alpha = 1 the trace is not resolved by the momentum grid after this time and is cut there


def figure_7():
    """Band-referenced current for linear polarisation and the peak current against the cutoff for both references."""
    tr = read('FigDet_B4_current_traces.csv'); cut = read('FigDet_B4_cutoff.csv'); F = read('Table_Dirac_B4_fits.csv', TD)
    tr = tr[tr.zeta == 0.0]; cut = cut[cut.zeta == 0.0]; F = F[(F.zeta == 0.0) & (F.quantity == 'Jx_band')]
    fig = new_fig(); w = (W_MM - 12.0 - 2 * 12.5 - 1.6) / 3; axa, axb, axc = (axes(fig, 12.0 + k * (w + 12.5), w) for k in range(3))
    for a, (col, mk, _), ls in zip(ORDER, ORDER.values(), ('-', DASH, DASHDOT)):
        g = tr[(tr.alpha == a) & np.isinf(tr.kappa_c) & (tr.t >= -20.0) & (tr.t <= (T_UNRESOLVED if a == 1.0 else 20.0))]
        axa.plot(g.t, g.Jx_band, color=col, ls=ls, lw=0.9, zorder=3 if a < 1 else 4)
    axa.set_xlim(-20, 20); axa.set_xticks([-20, -10, 0, 10, 20]); axa.set_xlabel(r'$t$'); axa.set_ylabel(r'$J_x$', labelpad=0.5)
    y = tr[np.isinf(tr.kappa_c) & (tr.t.abs() <= 20)].Jx_band; axa.set_ylim(y.min() * 1.08, y.max() * 1.45); tag(axa, 'a')
    for a, (col, mk, _) in ORDER.items():
        g = cut[(cut.alpha == a) & (cut.kappa_c >= 3)].sort_values('kappa_c'); f = F[F.alpha == a].iloc[0]; x = np.array([0.0, 1.0 / 3.0])
        axb.plot(x, f.limit + f.coefficient * x, color=col, lw=0.7, zorder=2); axb.plot(1.0 / g.kappa_c, g.peak_Jx_band, ls='none', marker=mk, ms=3.0, zorder=3, **face(col))
        axb.plot([0.0], [f.limit], ls='none', marker=mk, ms=4.6, mfc='white', mec=col, mew=0.9, clip_on=False, zorder=5)
        g = cut[cut.alpha == a].sort_values('kappa_c'); axc.loglog(g.kappa_c, g.peak_Jx_initref, ls='none', marker=mk, ms=3.0, mew=0.6, zorder=3, **face(col, 'none'))
    axb.set_xlim(-0.012, 0.35); axb.set_xticks([0, 0.1, 0.2, 0.3]); axb.set_xlabel(r'$1/\kappa_c$'); axb.set_ylabel(r'$\mathrm{max}_t\,|J_x|$', labelpad=1.0)
    yb = cut[cut.kappa_c >= 3].peak_Jx_band; axb.set_ylim(yb.min() - 0.02, yb.max() + 0.06); tag(axb, 'b', r'referred to $P_-(t)$')
    slope = 4.0 * math.asinh(1.0) / math.pi ** 2          # response of the filled band displaced within the square of half-width kappa_c
    kc = np.array([cut.kappa_c.min(), cut.kappa_c.max()]); amax = tr.a_x.abs().max(); axc.loglog(kc, slope * kc * amax, color=BLACK, lw=0.7, zorder=2)
    axc.text(3.2, slope * 3.2 * amax * 0.42, rf'${slope:.3f}\,\kappa_c\,\max|a|$', fontsize=7, ha='left', va='center')
    axc.set_xlim(0.85, 40); axc.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32])); axc.xaxis.set_major_formatter(PLAIN); axc.xaxis.set_minor_locator(NullLocator())
    axc.set_ylim(*loglim(cut.peak_Jx_initref.min(), cut.peak_Jx_initref.max(), head=0.16)); axc.yaxis.set_major_formatter(PLAIN)
    axc.set_xlabel(r'$\kappa_c$'); axc.set_ylabel(r'$\mathrm{max}_t\,|J_x|$', labelpad=1.0); tag(axc, 'c', r'referred to $\rho(t_0)$')
    h = [Line2D([], [], color=col, ls=ls, marker=mk, ms=3.0, lw=0.9, label=rf'$\alpha={a:g}$') for (a, (col, mk, _)), ls in zip(ORDER.items(), ('-', DASH, DASHDOT))]
    h.append(Line2D([], [], color=BLACK, lw=0.7, marker='o', ms=4.6, mfc='white', mec=BLACK, mew=0.9, label=r'linear fit and limit $\kappa_c\to\infty$'))
    legend_top(fig, h, handlelength=3.4); save(fig, 7)


FIGURES = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5, 6: figure_6, 7: figure_7}

if __name__ == '__main__':
    todo = [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(FIGURES)
    for n in todo:
        FIGURES[n]()
