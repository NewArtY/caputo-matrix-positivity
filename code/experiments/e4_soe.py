"""E4: sum-of-exponentials (SOE) history -- kernel accuracy, cost, and work-precision comparison.

  python e4_soe.py kernel     (a) number of modes and REALISED relative kernel error (new relative criterion and
                                  the legacy absolute criterion), growth of M with log(1/tol) and log(T/delta)
  python e4_soe.py exact      closed-form (matrix Mittag-Leffler) solution at ALL nodes of the finest uniform and
                                  graded meshes (cached; the coarser meshes are nested in them)
  python e4_soe.py solver     (b) work-precision study of the implicit L1 time stepper:
                                  full history / SOE history  x  uniform / graded mesh
  python e4_soe.py memory     peak Python-heap memory (tracemalloc) in SEPARATE, untimed runs
  python e4_soe.py residual   (c) torch residual operator, full convolution vs SOE recurrence (forward + backward)
  python e4_soe.py dirac      (d) full vs SOE for a Dirac momentum chunk (smooth solution, uniform mesh)
  python e4_soe.py all        everything in this order
  python e4_soe.py job <solver> <N>     one timed configuration (called in a fresh process by `solver`)

Timing protocol (see configs/soe.json: solver.timing_protocol): fresh process per configuration, one BLAS thread,
an untimed warm-up solve, several timed repeats, median and range reported as such; tracemalloc is never active
while a time is measured. Errors are maximum-norm errors against the closed-form solution over ALL nodes (the
initial layer included), over the nodes with t >= T/64, and at the final time.
Outputs: results/raw/E4_soe/*.csv|json|npz (with provenance records)."""
import det_common as dc
import json, os, subprocess, sys, time
import numpy as np
from fracsp import operators as op, reference as ref, exact

HERE = dc.HERE
OUT = os.path.join(dc.RAW, 'E4_soe'); os.makedirs(OUT, exist_ok=True)
CFGP = os.path.join(HERE, '..', 'configs', 'soe.json')
cfg = json.load(open(CFGP))
SOLVERS = ['full_uniform_row', 'full_uniform_tab', 'soe_uniform', 'full_graded_row', 'full_graded_table', 'soe_graded']


def meta(extra=None):
    return dc.provenance(os.path.abspath(__file__), CFGP, extra)


def bloch_A(phys):
    Om, Gd, Gp = phys['Omega'], phys['Gdown'], phys['Gphi']
    G2 = Gd / 2 + Gp
    return np.array([[-G2, 0, 0], [0, -G2, -Om], [0, Om, -Gd]]), np.array([0, 0, -Gd])


def initial_states(c):
    rng = np.random.default_rng(c['initial_state_seed'])
    v = rng.normal(size=(c['K'], 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0, 1, size=(c['K'], 1)) ** (1 / 3)


def mesh_of(solver, N):
    c = cfg['solver']
    return ref.graded_mesh(c['T'], N, c['grading_r']) if 'graded' in solver else np.arange(N + 1) * (c['T'] / N)


def make_runner(solver, N, store_every=1):
    """Returns (run, info): run() performs one complete solve and returns (t, X)."""
    c = cfg['solver']; a = c['alpha']; A, b = bloch_A(c['phys']); X0 = initial_states(c); K = c['K']
    Ab = np.broadcast_to(A, (K, 3, 3)); bb = np.broadcast_to(b, (K, 3))
    gen = lambda tn: (Ab, bb)
    t = mesh_of(solver, N); info = dict(M=0, soe_rel_err=0.0, soe_abs_err=0.0, setup_s=0.0, T_over_delta=0.0)
    if solver.startswith('soe'):
        delta = ref.soe_delta_for_mesh(t); t0 = time.perf_counter()
        om, la, ea, er = op.soe_kernel(a, delta, c['T'], c['soe_tol'])
        info.update(M=len(om), soe_rel_err=er, soe_abs_err=ea, setup_s=time.perf_counter() - t0, T_over_delta=c['T'] / delta)
    if solver == 'full_uniform_row':
        run = lambda: ref.solve_l1(gen, None, X0, t, a, store_every=store_every, weights='row')
    elif solver == 'full_uniform_tab':
        run = lambda: ref.solve_l1(gen, None, X0, t, a, store_every=store_every, weights='uniform')
    elif solver == 'soe_uniform':
        run = lambda: ref.solve_l1_soe(gen, None, X0, c['T'] / N, N, a, om, la, store_every=store_every)
    elif solver == 'full_graded_row':
        run = lambda: ref.solve_l1(gen, None, X0, t, a, store_every=store_every, weights='row')
    elif solver == 'full_graded_table':
        run = lambda: ref.solve_l1(gen, None, X0, t, a, store_every=store_every, weights='table')
    elif solver == 'soe_graded':
        run = lambda: ref.solve_l1_soe_nonuniform(gen, None, X0, t, a, om, la, store_every=store_every)
    else:
        raise ValueError(solver)
    return run, info


def job(solver, N):
    """One timed configuration in this (fresh) process. Prints one JSON line."""
    c = cfg['solver']; thr = dc.disable_power_throttling()
    warm, _ = make_runner(solver, min(N, c['warmup_N'])); warm()                 # untimed warm-up
    run, info = make_runner(solver, N)
    t0 = time.perf_counter(); tt, X = run(); first = time.perf_counter() - t0
    walls = [first]
    reps = c['repeats'] if first < c['slow_threshold_s'] else c['repeats_slow']
    for _ in range(reps - 1):
        t0 = time.perf_counter(); tt, X = run(); walls.append(time.perf_counter() - t0)
    np.save(os.path.join(OUT, f'sol_{solver}_N{N}.npy'), X)
    print(json.dumps(dict(solver=solver, N=N, walls=walls, power_throttling_optout=thr, **info)))


# ------------------------------------------------------------------------------------------------------------------
def exp_kernel(log):
    k = cfg['kernel']; rows = []
    hdr = 'alpha,T,T_over_delta,tol,criterion,M,M_estimate,dy,max_abs_err,max_rel_err,rel_err_over_tol'
    for a in k['alphas']:
        for T in k['T_list']:
            for tod in k['T_over_delta']:
                for tol in k['tols']:
                    for crit in ('relative', 'absolute'):
                        om, la, ea, er = op.soe_kernel(a, T / tod, T, tol, criterion=crit, n_check=k['n_check'])
                        dy = op.soe_design(a, T / tod, T, tol, crit)[0]
                        est = op.soe_mode_estimate(a, T / tod, T, tol) if crit == 'relative' else float('nan')
                        rows.append(f'{a},{T:g},{tod:g},{tol:g},{crit},{len(om)},{est:.2f},{dy:.6f},{ea:.4e},{er:.4e},{er / tol:.4f}')
    dc.write_csv(os.path.join(OUT, 'E4a_kernel.csv'), hdr, rows,
                 meta(dict(note='criterion=relative: construction of this version (relative error budget on [delta,T]); '
                                'criterion=absolute: legacy construction of the first version (lower truncation independent of T). '
                                'max_rel_err is the REALISED sup of |K-K_SOE|/K on n_check log-spaced points of [delta,T] incl. end points')))
    log('E4a kernel table:', len(rows), 'rows')
    # windows actually used in the Dirac and E4b computations
    rows = []
    for a, T, N in k['dirac_windows']:
        for crit in ('relative', 'absolute'):
            om, la, ea, er = op.soe_kernel(a, T / N, T, 1e-10, criterion=crit)
            rows.append(f'{a},{T:g},{N},1e-10,{crit},{len(om)},{ea:.4e},{er:.4e},{er / 1e-10:.3f}')
            log('  window', rows[-1])
    dc.write_csv(os.path.join(OUT, 'E4a_kernel_windows.csv'), 'alpha,T,N,tol,criterion,M,max_abs_err,max_rel_err,rel_err_over_tol', rows, meta())


def _exact_chunk(args):
    tchunk = args
    c = cfg['solver']; A, b = bloch_A(c['phys']); X0 = initial_states(c)
    return exact.bloch_exact_many(A, b, X0, tchunk, c['alpha'])


def exact_on(name, t, log):
    """Closed-form solution at the nodes t (cached in results/raw/E4_soe/exact_<name>.npz)."""
    path = os.path.join(OUT, f'exact_{name}.npz')
    if os.path.exists(path):
        z = np.load(path)
        if len(z['t']) == len(t) and np.array_equal(z['t'], t):
            return z['X']
    t1 = time.perf_counter()
    nch = max(1, min(48, len(t) // 500))
    parts = [t[i::nch] for i in range(nch)]
    res = dc.pool_map(_exact_chunk, parts, log=None)
    X = np.empty((len(t), cfg['solver']['K'], 3))
    for i in range(nch):
        X[i::nch] = res[i]
    dc.write_npz(path, meta(dict(note='closed-form matrix Mittag-Leffler solution (mpmath series, 30 digits)')), t=t, X=X)
    log(f'exact solution on {name}: {len(t)} nodes, {time.perf_counter() - t1:.1f} s')
    return X


def exp_exact(log):
    c = cfg['solver']
    Nu = max(c['uniform_N']); Ng = max(c['graded_N'])
    Xu = exact_on(f'uniform_N{Nu}', np.arange(Nu + 1) * (c['T'] / Nu), log)
    Xg = exact_on(f'graded_N{Ng}', ref.graded_mesh(c['T'], Ng, c['grading_r']), log)
    return Xu, Xg


def exact_for(solver, N, Xu, Xg):
    c = cfg['solver']
    if 'graded' in solver:
        Ng = max(c['graded_N']); return Xg[::Ng // N]          # t_n(N) = t_{n Ng/N}(Ng): nested graded meshes
    Nu = max(c['uniform_N']); return Xu[::Nu // N]


def exp_solver(log):
    c = cfg['solver']; Xu, Xg = exp_exact(log)
    load0 = dc.system_load_percent()
    plan = []
    for N in c['uniform_N']:
        for s in ('full_uniform_row', 'full_uniform_tab', 'soe_uniform'):
            if s == 'full_uniform_row' and N > c['uniform_row_N_max']:
                continue
            plan.append((s, N))
    for N in c['graded_N']:
        for s in ('full_graded_row', 'full_graded_table', 'soe_graded'):
            if s == 'full_graded_table' and N > c['graded_table_N_max']:
                continue
            plan.append((s, N))
    hdr = ('solver,mesh,history,N,M,T_over_delta,soe_rel_kernel_err,n_repeats,wall_median_s,wall_min_s,wall_max_s,setup_s,'
           'err_max_all_nodes,err_max_t_ge_T64,err_final_time,absdiff_to_full_same_mesh,reldiff_to_full_same_mesh,walls_s')
    rows = []; sols = {}; res_all = []
    for s, N in plan:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), 'job', s, str(N)], capture_output=True, text=True)
        if r.returncode != 0:
            log('JOB FAILED', s, N, r.stderr[-500:]); raise RuntimeError(f'job {s} {N} failed')
        q = json.loads(r.stdout.strip().splitlines()[-1]); res_all.append(q)
        X = np.load(os.path.join(OUT, f'sol_{s}_N{N}.npy')); os.remove(os.path.join(OUT, f'sol_{s}_N{N}.npy'))
        ex = exact_for(s, N, Xu, Xg); t = mesh_of(s, N)
        err = np.abs(X - ex).max(axis=(1, 2))
        e_all = err[1:].max(); e_out = err[t >= c['T'] / 64 - 1e-12].max(); e_fin = err[-1]
        mesh = 'graded' if 'graded' in s else 'uniform'; hist = 'soe' if s.startswith('soe') else 'full'
        key = (mesh, N)
        if hist == 'full' and key not in sols:
            sols[key] = X
        if hist == 'soe' and key in sols:
            ad = float(np.abs(X - sols[key]).max()); rd = ad / float(np.abs(sols[key]).max())
        else:
            ad = rd = float('nan')
        w = np.array(q['walls'])
        rows.append(f"{s},{mesh},{hist},{N},{q['M']},{q['T_over_delta']:.4g},{q['soe_rel_err']:.3e},{len(w)},{np.median(w):.5f},{w.min():.5f},{w.max():.5f},"
                    f"{q['setup_s']:.5f},{e_all:.6e},{e_out:.6e},{e_fin:.6e},{ad:.3e},{rd:.3e},{'|'.join(f'{x:.5f}' for x in w)}")
        log(rows[-1])
        if hist == 'soe':
            sols.pop(key, None)
    load1 = dc.system_load_percent()
    dc.write_csv(os.path.join(OUT, 'E4b_solver.csv'), hdr, rows,
                 meta(dict(timing_protocol=c['timing_protocol'], system_load_percent_before=load0, system_load_percent_after=load1,
                           power_throttling_optout=all(q.get('power_throttling_optout') for q in res_all) if sys.platform == 'win32' else 'n/a',
                           note='errors: max over the K states and 3 components of |x - x_exact| (absolute). err_max_all_nodes: all nodes n>=1; '
                                'err_max_t_ge_T64: nodes with t_n >= T/64; err_final_time: t = T. absdiff_to_full_same_mesh = max|x_SOE - x_full| over all '
                                'nodes (ABSOLUTE); reldiff = that divided by max|x_full|. wall_*: wall-clock seconds of the solver call '
                                '(SOE kernel construction = setup_s, not included). Machine shared with other jobs: see system_load_percent_*')))


def exp_memory(log):
    import tracemalloc
    c = cfg['solver']; rows = []
    for N in c['memory_N']:
        for s in SOLVERS:
            if s == 'full_graded_table' and N > c['graded_table_N_max']:
                continue
            run, info = make_runner(s, N, store_every=N)          # only the final state is stored
            tracemalloc.start(); run(); peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            rows.append(f"{s},{N},{info['M']},{peak / 2 ** 20:.4f}"); log('memory', rows[-1])
    dc.write_csv(os.path.join(OUT, 'E4b_memory.csv'), 'solver,N,M,peak_heap_MB', rows,
                 meta(dict(note='peak Python-heap memory (tracemalloc) of one solve in a separate UNTIMED run; only the final state is stored '
                                '(store_every = N), so the figure is the working memory of the solver')))


def exp_residual(log):
    import torch
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    c = cfg['residual']; rows = []
    for N in c['N_list']:
        h = 1.0 / N; t = torch.arange(N + 1) * h
        F0 = torch.stack([torch.sin((j + 1) * 3 * t) + t ** 0.6 for j in range(c['channels'])], 1)
        om, la, ea, er = op.soe_kernel(c['alpha'], h, 1.0, c['soe_tol'])
        Df = None
        for mode in ['full', 'soe']:
            ws = []
            for rep in range(c['repeats'] + 1):                  # the first evaluation is an untimed warm-up
                F = F0.clone().requires_grad_(True); t0 = time.perf_counter()
                D = op.l1_uniform(F, c['alpha'], h) if mode == 'full' else op.l1_soe_uniform(F, c['alpha'], h, om, la)
                D.pow(2).sum().backward(); w = time.perf_counter() - t0
                if rep > 0:
                    ws.append(w)
            if mode == 'full':
                Df = D.detach()
            ad = float((D.detach() - Df).abs().max()); rd = ad / float(Df.abs().max())
            ws = np.array(ws)
            rows.append(f'{N},{mode},{len(om) if mode == "soe" else 0},{er if mode == "soe" else 0:.3e},{len(ws)},{np.median(ws):.5f},{ws.min():.5f},{ws.max():.5f},{ad:.3e},{rd:.3e}')
            log('residual', rows[-1])
    dc.write_csv(os.path.join(OUT, 'E4c_residual.csv'),
                 'N,mode,M,soe_rel_kernel_err,n_repeats,wall_median_s,wall_min_s,wall_max_s,absdiff_to_full,reldiff_to_full', rows,
                 meta(dict(torch_threads=1, note='forward + backward pass of sum(D^2) for an 8-channel trial function; one untimed warm-up, then n_repeats '
                                                 'timed evaluations in the same process; reldiff = max|D_SOE - D_full| / max|D_full|',
                           system_load_percent=dc.system_load_percent())))


def exp_dirac(log):
    """Cost of full versus SOE history for a chunk of K Dirac wave vectors on the uniform mesh of B4."""
    from fracsp import dirac
    c = cfg['dirac_step_cost']; dcfg = json.load(open(os.path.join(HERE, '..', 'configs', 'dirac.json')))
    p = dict(dcfg['pulse']); p.update(delta=dcfg['delta'], Gdown=dcfg['Gdown'], Gup=dcfg['Gup'], Gphi=dcfg['Gphi'], zeta=1.0)
    t0w, tfw = dcfg['window']; T = tfw - t0w; a = c['alpha']; K = c['K']
    rng = np.random.default_rng(7); kap = rng.uniform(-8, 8, size=(K, 2))
    r0 = dirac.ground_state(kap, 1, p, t0w)
    gen = lambda tn: dirac.bloch_generator(kap, 1, t0w + tn, p)
    rows = []
    for N in c['N_list']:
        h = T / N
        om, la, ea, er = op.soe_kernel(a, h, T, c['soe_tol'])
        runs = dict(full_uniform_tab=lambda: ref.solve_l1(gen, None, r0, np.arange(N + 1) * h, a, store_every=N, weights='uniform'),
                    soe_uniform=lambda: ref.solve_l1_soe(gen, None, r0, h, N, a, om, la, store_every=N))
        fin = {}
        for name, run in runs.items():
            ws = []
            for rep in range(c['repeats']):
                t1 = time.perf_counter(); tt, R = run(); ws.append(time.perf_counter() - t1)
            fin[name] = R[-1]; ws = np.array(ws)
            ad = float(np.abs(fin[name] - fin['full_uniform_tab']).max())
            rows.append(f"{name},{N},{K},{len(om) if name.startswith('soe') else 0},{er if name.startswith('soe') else 0:.3e},{len(ws)},{np.median(ws):.4f},{ws.min():.4f},{ws.max():.4f},"
                        f"{np.median(ws) / N / K * 1e6:.4f},{ad:.3e}")
            log('dirac', rows[-1])
    dc.write_csv(os.path.join(OUT, 'E4d_dirac_chunk.csv'),
                 'solver,N,K,M,soe_rel_kernel_err,n_repeats,wall_median_s,wall_min_s,wall_max_s,microseconds_per_point_step,absdiff_final_bloch_to_full', rows,
                 meta(dict(note='complete solves of the Dirac Bloch equations for K random wave vectors in |kappa_x|,|kappa_y|<=8, zeta=1, valley K, window [-30,30]; '
                                'first repeat doubles as warm-up (median reported)', system_load_percent=dc.system_load_percent())))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what == 'job':
        job(sys.argv[2], int(sys.argv[3])); sys.exit(0)
    dc.disable_power_throttling()
    log = dc.Log(os.path.join(OUT, f'E4_{what}.log'))
    todo = dict(kernel=[exp_kernel], exact=[exp_exact], solver=[exp_solver], memory=[exp_memory], residual=[exp_residual], dirac=[exp_dirac],
                all=[exp_kernel, exp_solver, exp_memory, exp_residual, exp_dirac])[what]
    for f in todo:
        f(log)
    log('done'); log.close()
