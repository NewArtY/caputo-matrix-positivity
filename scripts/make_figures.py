"""Entry point of the figure stage: every figure from the tables and figure data written by scripts/aggregate.py.

Runs the plotting scripts,
    final           scripts/plot_final.py           figures/final/Figure_01.pdf ... Figure_07.pdf   the figures of the article
    deterministic   scripts/plot_deterministic.py   figures/candidates/FigDet_*.pdf
    neural          scripts/plot_neural.py          figures/candidates/Neural_*.pdf
    b5              scripts/plot_b5.py              figures/candidates/B5_*.pdf
and writes figures/figure-manifest.csv: one row per figure with the plotting script, the data files it is drawn
from, the page size and whether the fonts are embedded. All figures are vector PDF with embedded TrueType fonts.
The figures of the article are 165 mm x 44 mm (DejaVu Sans); the other figures show further views of the same data
and are laid out for a text width of 190 mm (DejaVu Serif).

Usage: python scripts/make_figures.py [--only final|deterministic|neural|b5 ...] [--preview <dir for PNG copies>]
"""
import argparse, csv, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T, F = 'tables/data/', 'figures/data/'
FINAL_SIZE = (165.0, 44.0)          # mm
# folder under figures/, file stem, group, data files
FIGURES = [
    ('final', 'Figure_01', 'final', [F + 'Neural_B2_runs.csv', F + 'Neural_B2_stepper.csv']),
    ('final', 'Figure_02', 'final', [F + 'Neural_B2_error_curves.csv']),
    ('final', 'Figure_03', 'final', [F + 'Neural_B6_runs.csv', F + 'Neural_B6_stepper.csv', F + 'Neural_B6_curves.csv']),
    ('final', 'Figure_04', 'final', [F + 'B5_fenep_curves.csv']),
    ('final', 'Figure_05', 'final', [F + 'Neural_surrogate_grid.csv', F + 'Neural_surrogate_stepper.csv', T + 'Table_surrogate.csv', T + 'Table_surrogate_breakeven.csv']),
    ('final', 'Figure_06', 'final', [F + 'FigDet_tail_rays.csv', F + 'FigDet_B4_cutoff.csv', T + 'Table_Dirac_tail.csv']),
    ('final', 'Figure_07', 'final', [F + 'FigDet_B4_current_traces.csv', F + 'FigDet_B4_cutoff.csv', T + 'Table_Dirac_B4_fits.csv']),
    ('candidates', 'FigDet_E_verification', 'deterministic', [T + 'Table_E1_E3.csv']),
    ('candidates', 'FigDet_SOE', 'deterministic', [T + 'Table_SOE.csv', T + 'Table_SOE_kernel.csv', T + 'Table_SOE_residual.csv', T + 'Table_SOE_dirac_chunk.csv']),
    ('candidates', 'FigDet_Dirac_B3', 'deterministic', [F + 'FigDet_B3_trajectories.csv', F + 'FigDet_B3_convergence.csv']),
    ('candidates', 'FigDet_Dirac_current', 'deterministic', [F + 'FigDet_B4_current_traces.csv', F + 'FigDet_B4_cutoff.csv', T + 'Table_Dirac_B4.csv']),
    ('candidates', 'FigDet_Dirac_tail', 'deterministic', [T + 'Table_Dirac_tail.csv', F + 'FigDet_B4_cutoff.csv', F + 'FigDet_tail_constant_vs_tf.csv']),
    ('candidates', 'FigDet_Dirac_maps', 'deterministic', [F + 'FigDet_B4_maps_zeta1.npz']),
    ('candidates', 'Neural_B2_factorial', 'neural', [F + 'Neural_B2_runs.csv', F + 'Neural_B2_stepper.csv']),
    ('candidates', 'Neural_B2_effects', 'neural', [T + 'Table_B2_paired.csv']),
    ('candidates', 'Neural_B2_effects_distance_to_L1', 'neural', [T + 'Table_B2_paired.csv']),
    ('candidates', 'Neural_B2_initial_layer', 'neural', [F + 'Neural_B2_error_curves.csv']),
    ('candidates', 'Neural_B6_positivity', 'neural', [F + 'Neural_B6_runs.csv', F + 'Neural_B6_stepper.csv', F + 'Neural_B6_curves.csv']),
    ('candidates', 'Neural_B7_estimates', 'neural', [F + 'Neural_B7_estimates.csv', T + 'Table_B7_estimates.csv']),
    ('candidates', 'Neural_B1_design', 'neural', [F + 'Neural_B1_inverse_runs.csv', T + 'Table_B1_design.csv']),
    ('candidates', 'Neural_surrogate', 'neural', [F + 'Neural_surrogate_grid.csv', F + 'Neural_surrogate_stepper.csv', T + 'Table_surrogate.csv']),
    ('candidates', 'B5_model_eigenvalues', 'b5', [F + 'B5_model_eigenvalues.csv']),
    ('candidates', 'B5_forward_errors_a0.6', 'b5', [F + 'B5_forward_runs.csv', F + 'B5_forward_stepper.csv']),
    ('candidates', 'B5_forward_errors_a0.9', 'b5', [F + 'B5_forward_runs.csv', F + 'B5_forward_stepper.csv']),
    ('candidates', 'B5_forward_min_eig', 'b5', [F + 'B5_forward_min_eig_curves.csv']),
    ('candidates', 'B5_inverse_estimates', 'b5', [F + 'B5_inverse_estimates.csv', F + 'B5_inverse_crb.csv']),
    ('candidates', 'B5_fenep_trajectories', 'b5', [F + 'B5_fenep_curves.csv']),
]
SCRIPT = {'final': 'plot_final.py', 'deterministic': 'plot_deterministic.py', 'neural': 'plot_neural.py', 'b5': 'plot_b5.py'}


def pdf_info(path):
    """(width in mm, height in mm, fonts embedded) of a one-page PDF written by matplotlib."""
    b = open(path, 'rb').read()
    m = re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]', b)
    w, h = (float(m.group(1)) / 72 * 25.4, float(m.group(2)) / 72 * 25.4) if m else (float('nan'), float('nan'))
    nfont = len(re.findall(rb'/Type\s*/FontDescriptor', b)); nfile = len(re.findall(rb'/FontFile[23]?\s', b))
    return round(w, 1), round(h, 1), 'yes' if nfont and nfile >= nfont and b'/Type3' not in b else 'no'


def main():
    ap = argparse.ArgumentParser(description='All figures from the tables and figure data.')
    ap.add_argument('--only', action='append', choices=list(SCRIPT)); ap.add_argument('--preview')
    a = ap.parse_args(); groups = a.only or list(SCRIPT)
    missing = sorted({d for _, _, g, data in FIGURES if g in groups for d in data if not os.path.exists(os.path.join(ROOT, d))})
    if missing:
        sys.exit('missing input (run scripts/aggregate.py first): ' + ', '.join(missing))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1'); t_start = time.time()
    for g in groups:
        t0 = time.perf_counter(); cmd = [sys.executable, os.path.join(ROOT, 'scripts', SCRIPT[g])] + (['--preview', a.preview] if a.preview else [])
        print(f'--- {g}: python scripts/{SCRIPT[g]}', flush=True)
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if rc:
            sys.exit(f'figures of the group {g} FAILED (exit status {rc})')
        print(f'--- {g}: done in {time.perf_counter() - t0:.0f} s', flush=True)
    rows = []; stale = []; bad = []
    for folder, name, g, data in FIGURES:
        rel = f'figures/{folder}/{name}.pdf'; p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            if g in groups:
                sys.exit(f'figure {rel} was not written')
            continue
        if g in groups and os.path.getmtime(p) < t_start - 1:
            stale.append(rel)
        w, h, emb = pdf_info(p); rows.append([rel, 'scripts/' + SCRIPT[g], '; '.join(data), w, h, emb])
        if emb != 'yes' or (folder == 'final' and (w, h) != FINAL_SIZE) or (folder != 'final' and not w <= 190.5):
            bad.append(rel)
    if stale:
        sys.exit('not rewritten by this run: ' + ', '.join(stale))
    with open(os.path.join(ROOT, 'figures', 'figure-manifest.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['figure', 'script', 'data', 'width_mm', 'height_mm', 'fonts_embedded']); w.writerows(rows)
    print(f'wrote figures/figure-manifest.csv ({len(rows)} figures)')
    known = {f'{folder}/{name}.pdf' for folder, name, _, _ in FIGURES}
    other = sorted(f'{d}/{f}' for d in ('final', 'candidates') if os.path.isdir(os.path.join(ROOT, 'figures', d))
                   for f in os.listdir(os.path.join(ROOT, 'figures', d)) if f'{d}/{f}' not in known)
    if other:
        print(f'note: {len(other)} files in figures/final and figures/candidates are not outputs of this script: ' + ', '.join(other))
    if bad:
        sys.exit('fonts not embedded or page size not as specified: ' + ', '.join(bad))
    return 0


if __name__ == '__main__':
    sys.exit(main())
