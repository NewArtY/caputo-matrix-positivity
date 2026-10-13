# Code and data for: Positivity preservation in matrix-valued Caputo evolution equations: continuous problem, L1 time stepping, and residual-based neural networks

Nikolai S. Akintsov (School of Artificial Intelligence and Computer Science, Nantong University, Nantong 226019, China; ORCID 0000-0002-1040-1292),
Artem P. Nevecheria (Department of Mathematical and Computer Methods, Kuban State University, Krasnodar 350040, Russia; ORCID 0000-0001-6736-4691),
Stepan N. Andreev (Joint-Stock Company "Center for Research and Development", Moscow 101000, Russia; ORCID 0000-0003-3588-2894),
Qing-Hua Qin (Institute of Advanced Interdisciplinary Technology, Shenzhen MSU-BIT University, Shenzhen 518172, China; ORCID 0000-0003-0948-784X).

DOI of this archive: `10.5281/zenodo.23346554`. DOI of the article: `<DOI OF THE ARTICLE TO BE INSERTED>`.

This archive contains the source code, the configurations, every raw numerical result, the tables and the
figures of the article. Subject: Caputo time-fractional evolution equations whose state is a positive-semidefinite
matrix (a density matrix, a conformation tensor); implicit L1 time stepping on uniform and graded meshes;
sum-of-exponentials (SOE) compression of the history; physics-informed networks whose output maps keep the state
admissible.

All tables and figures are regenerated from the stored raw results in about two minutes (Section 4). Every campaign
can be rerun from scratch (Section 5).

## 1. Layout

```
README.md  LICENSE  CITATION.cff  .zenodo.json  requirements.txt
results-manifest.csv        principal quantities of the tables: id, value, table, raw source, scripts
code/
  fracsp/                   Python package (Section 2)
  experiments/              experiment drivers, one group per benchmark (Section 2); det_common.py: shared helpers
  configs/                  one JSON file per benchmark with all parameters
  configs/quick/            reduced configurations used only by scripts/smoke_test.py
  tests/                    pytest suites (455 tests)
scripts/
  aggregate.py              entry point: all tables and figure data from the raw results
  aggregate_deterministic.py, aggregate_neural.py, aggregate_b5.py     called by aggregate.py
  make_tables.py            the tables of the article (LaTeX) from tables/data; called by aggregate.py
  make_figures.py           entry point: all figures
  plot_final.py             the figures of the article; called by make_figures.py
  plot_deterministic.py, plot_neural.py, plot_b5.py                    further figures; called by make_figures.py
  rerun_timings.py          reruns the timing measurements only (Section 6)
  smoke_test.py             runs every driver in a reduced configuration (Section 4)
  check_provenance.py       checks the provenance records of the raw results (Section 8)
results/raw/                raw results, one folder per benchmark (178 MB, 5117 files)
tables/data/                tables (CSV); side-car <name>.meta.json with note, raw sources and provenance where present
tables/final/               the tables of the article, Table_01.tex ... Table_08.tex, and two tables quoted in its text
figures/data/               the data drawn in the figures (CSV, one NPZ)
figures/final/              the figures of the article, Figure_01.pdf ... Figure_07.pdf (vector PDF, fonts embedded)
figures/candidates/         further figures of the same data (vector PDF, fonts embedded)
figures/figure-manifest.csv figure, plotting script, data files, page size
```

Table and figure numbers follow the article (Section 10).

## 2. Package and benchmarks

Package `code/fracsp` (version string 0.1.0, recorded in the result files):

| module | content |
|---|---|
| `operators.py` | L1 Caputo operators on uniform and nonuniform meshes (torch, differentiable in the order), SOE kernel and fast L1 operator |
| `reference.py` | implicit L1 solvers (numpy): full history, SOE history on uniform and nonuniform meshes; product-trapezoidal rule; tools for the positivity of the L1 step; Richardson extrapolation |
| `mittag.py`, `exact.py` | Mittag-Leffler functions in multiprecision arithmetic and exact solutions of linear problems |
| `layers.py` | output maps for density matrices, gates of the exact initial condition, projection on the density matrices |
| `pinn.py` | networks, GKSL generators, training loop, implicit L1 time stepper in the metric of the networks, error and admissibility metrics, provenance record, campaign launcher |
| `dirac.py` | laser-driven massive-Dirac two-band model in Bloch form; analytical constants of the high-momentum tail |
| `dirac_cutoff.py` | coefficient of 1/kappa_c in the cutoff dependence of the band-referenced sheet current, from the terms of order kappa^-3 of the large-momentum expansion |
| `spd.py` | Caputo conformation-tensor models (upper-convected Maxwell, Giesekus, FENE-P): exact solutions, implicit L1 stepper with solvers for the admissible solution of the nonlinear step, output maps for symmetric positive definite tensors, residual norms |

Nonlinear step in `spd.py`. With the FENE-P or the Giesekus term, the equation of one implicit L1 step can have
several real solutions; exactly one of them is admissible (positive definite and, for FENE-P, with a trace below
`b`), for every step size. `solve_l1(..., nonlinear='safeguarded')`, the default, runs Newton's iteration from the
previous iterate and accepts its result only if it is admissible and its residual is small; otherwise the step is
solved by `fenep_step_root` (a scalar equation for the Peterlin factor, solved by bracketing) or
`giesekus_step_root` (an algebraic Riccati equation). `nonlinear='root'` solves every step by the root solver.
Two variants return the result of Newton's iteration without a test: `nonlinear='newton-full'` takes full steps,
and `nonlinear='newton'` (also `solve_l1_newton`) halves a step of the FENE-P law as long as the new iterate would
have a trace of at least `b`; for the Giesekus law the two coincide. Both start from the previous iterate and stop
when the Frobenius norm of the update is at most `tol (1 + |C|_F)` with `tol = 1e-12`, after at most 60 iterations.
Either can converge to an inadmissible solution without any error. With full steps this also happens on meshes
within the step bound of the linear law, where an iterate passes the pole `tr C = b`
(`Table_B5_fenep_step.csv`, `Table_B5_fenep_step_case2.csv`); with halved steps it was observed only above that
bound. Where the iterates of `'newton'` are admissible, the default returns the same arrays bit for bit, and the
root solver agrees to round-off.

Benchmarks. Drivers are in `code/experiments`, configurations in `code/configs`, raw results in `results/raw`.

| id | subject | drivers | configuration | raw results |
|---|---|---|---|---|
| E1-E3 | L1 operator on smooth and weakly singular functions (E1, E1T), derivative with respect to the order (E2), implicit L1 solver against the Mittag-Leffler solution (E3) | `e1_e3_verification.py` | `verification.json` | `E1_E3/` |
| E4 | SOE history: kernel error and number of modes, work-precision of the L1 steppers, residual operator, Dirac momentum chunk | `e4_soe.py` | `soe.json` | `E4_soe/` |
| E5 | positivity of the implicit L1 step for random time-dependent GKSL generators | `l1_positivity.py` | `verification.json` | `E5_positivity/` |
| B1 | scalar relaxation, forward and inverse (unknown order), design input x gate | `b1_floor.py`, `b1_ls_mesh.py`, `b1_scalar.py`, `run_b1_campaign.py` | `b1_scalar.json` | `B1_design/` |
| B2 | driven two-level system (d = 2), factorial design output map x initial condition x gate | `b2_prepare.py`, `b2_density.py`, `run_b2_campaign.py`, `b2_timing.py` | `b2_density.json` | `B2_factorial/` |
| surrogate | parametric surrogate of the two-level system over (order, relaxation rate) | `b2_surrogate.py`, `b2_timing.py` | `b2_surrogate.json` | `B2_surrogate/` |
| B3, B4 | laser-driven massive-Dirac model: single wave vector (B3), momentum-resolved densities and sheet current with their checks (B4), high-momentum tail, closed form of the 1/kappa_c coefficient of the current | `dirac_reference.py`, `dirac_tail_check.py`, `dirac_cutoff_coefficient.py` | `dirac.json` | `DIRAC_v2/` |
| B5 | conformation tensor of a viscoelastic fluid: forward problem, identification of order and relaxation time, FENE-P, positivity of time stepping, solutions of the FENE-P step | `b5_conformation.py`, `b5_inverse.py`, `b5_fenep.py`, `b5_positivity.py`, `b5_case_selection.py`, `b5_fenep_step.py` | `b5_forward.json`, `b5_inverse.json`, `b5_fenep.json` | `B5_conformation/` |
| B6 | forward problems for d = 3 and d = 4 with the state on or near the boundary of the state space | `b6_prepare.py`, `b6_positivity.py`, `b6_campaign.py` | `b6_positivity.json` | `B6_positivity/` |
| B7 | identification of the order and of two rates of the two-level system; networks against least squares through the L1 stepper | `b7_inverse.py`, `b7_campaign.py` | `b7_inverse.json` | `B7_inverse/` |

Each neural benchmark folder has the sub-folders `reference/` (exact solutions, L1 stepper, checks) and one
sub-folder per campaign stage (`pilot`, `main`, `extras`, `long`, `mesh`, `timing`) with one JSON file per run and
the console log `campaign.log`. `B5_conformation/` has the sub-folders `forward`, `inverse`, `fenep`, `positivity`
and `logs`; the files `logs/*pilot*.log` are pilot runs that were used to fix the design and enter no table.

## 3. Installation

Python 3.14 with the packages of `requirements.txt` (NumPy, SciPy, PyTorch (CPU), mpmath; pandas and matplotlib for the
tables and figures; pytest for the tests). No GPU is used.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)        source .venv/bin/activate      (Linux, macOS)
python -m pip install -r requirements.txt
```

The package is used in place: the drivers and the tests add `code/` to the module search path. Nothing is installed.

## 4. Checks that take minutes

All commands are given from the root of the archive.

| purpose | command | time |
|---|---|---|
| tests (455) | `cd code` then `python -m pytest -q tests` | 25 to 85 s |
| all tables (CSV and the LaTeX tables of the article) and figure data from the stored raw results, compared with the stored files | `python scripts/aggregate.py --check` | 30 to 40 s |
| the same after deleting the stored tables and figure data | `python scripts/aggregate.py --clean --check` | 80 to 100 s |
| the LaTeX tables of the article alone (`--out <dir>`: another output folder) | `python scripts/make_tables.py --extra` | 1 s |
| all figures (`--only final`: the seven figures of the article, 4 s) | `python scripts/make_figures.py` | 20 s |
| provenance records of the raw results | `python scripts/check_provenance.py` | 3 s |
| every driver in a reduced configuration | `python scripts/smoke_test.py --workers 4` | 6 min |

`aggregate.py --check` reports for every table and figure-data file whether it was regenerated identically; in the
environment of Section 7 all files are identical byte for byte (side-car records up to their time stamp), and
`make_figures.py` rewrites identical PDF files. On another platform the numbers can differ in the last digits;
`--rtol 1e-9` then accepts differences up to that relative size. The aggregation runs with one BLAS thread, so that
the result does not depend on the number of processor cores. One file, `figures/data/FigDet_tail_constant_vs_tf.csv`
(analytical tail constants, about 45 s), is computed by the aggregation itself and not from raw results; it is
recomputed only when it is missing or with `--clean`. `tables/data/Table_Examples_continuous.csv` (numbers of the
article that follow from closed forms, 5 s) is also computed by the aggregation, at every run.

`smoke_test.py` copies `code/` and `scripts/` to a temporary directory, replaces the configurations by
`code/configs/quick/*.json` (tiny meshes, 150 + 20 optimiser iterations) and runs every stage of every benchmark
there, the timing passes and the provenance check included. It does not touch `results/raw`, and its numbers are
not meaningful.

## 5. Rerunning the computations from scratch

Commands from `code/experiments`. `6` is the number of worker processes; every worker uses one thread. The times are
those recorded in the logs and result files of this archive on a laptop (Intel Core i7-13620H, 16 logical CPUs, 16 GB)
that was shared with other jobs during most stages, with 6 workers unless another number is given. The training runs
that were repeated alone on the idle machine (Section 6) took 2.4 to 4.9 times less wall time than in the campaigns.
Campaign launchers skip runs whose result file exists, so the stored result files of a stage have to be moved away
before the stage is rerun.

```
# deterministic part                                                      recorded time
python e1_e3_verification.py all                                          1 min
python l1_positivity.py                                                   0.5 min
python e4_soe.py all                                                      10 min   (timings: Section 6)
python dirac_reference.py B3                                              2 min
python dirac_reference.py B4                                              70 min   (4 workers)
python dirac_reference.py B4check                                         40 min   (4 workers)
python dirac_tail_check.py all                                            5 min
python dirac_cutoff_coefficient.py                                        1 min

# B2: references, then the stages pilot, main, extras, long
python b2_prepare.py 6                                                    3 min
python run_b2_campaign.py all 6                                           17 + 69 + 13 + 8 min   (2227 training runs)
python b2_timing.py b2 0.6 pure 0                                         2 min    (18 runs, one after another)

# B6
python b6_prepare.py 6                                                    0.5 min
python b6_campaign.py all 6                                               17 + 5 min             (408 training runs)

# B7
python b7_inverse.py prepare 6                                            1 min
python b7_campaign.py all 6                                               4 + 15 min             (156 training runs, 40 least-squares jobs)

# B1: references, then the stages main, mesh
python b1_floor.py                                                        1 min
python b1_ls_mesh.py                                                      6 min (least squares through the L1 time stepper on finer meshes)
python run_b1_campaign.py all 6                                           16 min (main, 360 runs) + 64 min (mesh, 180 runs, recorded with 1 worker)

# surrogate (about 1 GB of memory per worker)
python b2_surrogate.py prepare 6                                          0.5 min
python b2_surrogate.py campaign 1                                         57 min   (6 trainings one after another; 4 min each on the idle machine)
python b2_timing.py surrogate                                             1 min
python b2_surrogate.py train --map lin --seed 0 --stage timing            4 min

# B5
python b5_positivity.py                                                   0.5 min
python b5_case_selection.py                                               0.5 min
python b5_conformation.py prepare                                         0.5 min
python b5_conformation.py campaign --workers 3 --seeds 0-4                122 min  (840 training runs, 3 workers)
python b5_conformation.py campaign --workers 3 --seeds 0-4 --variants sym-rel      11 min   (120 runs, 2 workers)
python b5_conformation.py run shear 100 0.6 rotated logm nat 0 --budget 3 --save   1 min each; also "sym rel" and "sym fro", seeds 0, 1, 2
python b5_inverse.py nls                                                  13 min   (120 least-squares fits)
python b5_inverse.py campaign --workers 3 --seeds 0-9 --variants factor-fro,sym-fro   76 min   (120 runs)
python b5_inverse.py campaign --workers 3 --seeds 0-4 --variants logm-nat,sym-comp    42 min   (60 runs, 1 worker)
python b5_fenep.py prepare                                                0.5 min
python b5_fenep_step.py --workers 6                                       1 min    (after "b5_fenep.py prepare", whose references it checks)
python b5_fenep.py campaign --workers 3 --seeds 0-4                       22 min   (140 runs, 2 workers)

# tables and figures (from the root of the archive)
python scripts/aggregate.py
python scripts/make_figures.py
```

The sum of the training times stored in the result files is 25 hours of single-thread wall time for the 4626
training runs of the campaigns (measured on the shared machine); the deterministic part takes about 2 hours with
4 to 6 workers.

Single runs can be repeated from the command line; each driver documents its options in its first lines, for example
`python b2_density.py --alpha 0.6 --state pure --map factor --ic hard --gate pow --seed 0`. Such runs are written to
the sub-folder `manual` of the benchmark and do not enter the tables.

## 6. Timing measurements

Wall times depend on the load of the machine. Three groups of numbers come from dedicated timing measurements and
can be repeated alone, on an idle machine, with

```
python scripts/rerun_timings.py            (about 30 min; options: --only soe|b2|surrogate, --max-load, --no-aggregate)
```

The script keeps the previous measurement in a folder `previous_timing_<UTC time>` next to the files it replaces,
reruns the measurements, and then runs `aggregate.py` and `make_figures.py`. Table entries that depend on it:

| stage | tables (columns) | status of the stored values |
|---|---|---|
| `soe` | `Table_SOE.csv` (`wall_*`, `setup_s`, `time_ratio_*`), `Table_SOE_matched.csv` (`wall_*`, `speedup_*`), `Table_SOE_residual.csv` and `Table_SOE_dirac_chunk.csv` (`wall_*`, `time_ratio_*`, `microseconds_per_point_step`); figure `FigDet_SOE` (panels c to f); of the article: `Table_S1_soe.tex` (times and ratios) | two complete runs, both taken while other jobs were running (system load in the side-car record of `Table_SOE.csv`); to be repeated on an idle machine |
| `b2` | `Table_B2_timing.csv` | 18 runs repeated one after another on the idle machine; each reproduced the campaign result bit for bit |
| `surrogate` | `Table_surrogate.csv` (`inference_ms_*`, `stepper_ms_per_query`, `break_even_*`), `Table_surrogate_breakeven.csv` (rows `idle repeat`, `break_even_*`); figure `Neural_surrogate`; of the article: panel (d) of `Figure_05.pdf` | measured on the idle machine |

The errors, numbers of modes and trained networks in these tables do not depend on the timing run. All other wall
and CPU times in the tables (`wall_median_s`, `cpu_median_s`, `train_wall_s`, `wall_ms`, `median wall, s`, ...) were
recorded during the campaigns on the shared machine; in the article these are the last column of `Table_07.tex` and
the training times of the campaign in `Figure_05.pdf`. They are comparable within a campaign stage and are
indicative otherwise; they change only if the campaign is rerun.

## 7. Environment of the stored results and reproducibility

Every result file of the archive that records its environment (3500 files) reports the same one:

| | |
|---|---|
| Python | 3.14.3 |
| NumPy / SciPy / mpmath | 2.4.2 / 1.18.1 / 1.3.0 (BLAS: scipy-openblas, one thread) |
| PyTorch | 2.10.0+cpu, one thread per process, double precision |
| pandas / matplotlib / pytest | 3.0.5 / 3.11.1 / 9.1.1 (used for the tables, figures and tests; not recorded in the result files) |
| platform | Windows-11-10.0.26200-SP0, Intel Core i7-13620H, 16 logical CPUs, 15.7 GB |
| time of the runs | 11 October 2026, 18:43 to 23:30 UTC; the three result files named in Section 8 (exception 1) and `DIRAC_v2/current_cutoff_coefficient.csv`: 12 October 2026 |

- Deterministic computations (E1-E5, B3, B4, tail, references, L1 steppers, least-squares fits, aggregation) are
  expected to be reproduced to round-off on any platform. On the machine above, every stage that was recomputed
  for this release was reproduced bit for bit (Section 9).
- Trained networks are reproduced bit for bit on the same machine with the same software: seeds are fixed, every
  process uses one thread, and the result does not depend on the number or order of the worker processes (Section 9;
  `Table_B2_timing.csv`). Across platforms, processor types or library versions the individual runs differ, and only
  the statistics over seeds (medians, counts, paired ratios) are expected to be reproduced.
- Seeds: networks are initialised with `torch.manual_seed(seed)`; data noise, initial guesses and observation times
  come from `numpy.random.default_rng` with the seeds stated in the configuration files.

## 8. Provenance

Every raw result is tied to a provenance record: UTC time stamp, versions of Python and of the numerical packages,
platform, processor, seed, the resolved configuration of the run or the fingerprint of its configuration file, and
SHA-256 fingerprints of the source files that produced it. No host name, user name or absolute path is recorded.
`python scripts/check_provenance.py` lists, per folder, where the record is found and compares every recorded
fingerprint with the files of the archive.

| where the record is | files | folders |
|---|---|---|
| in the result file (keys of a JSON file, `# key: value` header lines of a CSV file, array `meta` of an NPZ file) | 3500 | all; in B5 only the files named below |
| in a JSON record of the same folder (`prepare.json`, `timing.json`, `<name>.provenance.json`, the result file of the same run): reference arrays, stepper tables, stored trajectories | 112 | B1, B2, B5, B6, B7, surrogate |
| in the folder-level record `results/raw/B5_conformation/provenance.json` | 1459 | B5 |

Two exceptions to "recorded at run time":

1. B5. 1459 result files of `B5_conformation/` were written before these drivers recorded provenance. Each file
   contains its seed and parameters; versions, platform and time stamps are in the folder-level record, which was
   compiled afterwards (file modification times; the environment of the machine). A subset of these files, covering
   every stage and every output map, was recomputed with the code of this archive and agrees with the stored files
   in every stored value (Section 9). Three result files of `positivity/` were written on 12 October 2026 by the drivers of
   this archive, after the solver of the nonlinear step in `spd.py` had been revised (Section 2), and carry their own
   record: `positivity.json` and `fenep_step.json` (key `provenance`), `subordination_monte_carlo.json` (record
   `subordination_monte_carlo.provenance.json`). `positivity.json` contains all values of the file it replaces,
   unchanged, and new keys `safeguarded_*` and `step_condition_lhs`. The three files were written again on the same
   day, after `spd.py` had received the variant `newton-full` (Section 2): every value of the earlier files is
   reproduced, and `fenep_step.json` has the new keys `newton_full_*`, `newton_iteration` and `case2`. The two
   entries of `revisions` in the folder-level record list the sources that changed and the check of the files that
   were not rewritten. `DIRAC_v2/current_cutoff_coefficient.csv`, whose record includes the fingerprint of `spd.py`
   although its driver does not use this module, was recomputed after that change, with identical data.
2. Earlier state of a source file. 25 deterministic result files (`E1_E3/*.csv`, `E5_positivity/*.csv`,
   `DIRAC_v2/B3_*`, `E4_soe/E4a_*`, `E4b_*`, `E4c_residual.csv`, `E4d_dirac_chunk.csv`, `E4_soe/exact_*.npz`) record a
   fingerprint of `det_common.py` (and, for seven files of `E4_soe/`, of `operators.py`, `layers.py` or `pinn.py`)
   that differs from the file in the archive: these files were produced before the last edits of those sources.
   `check_provenance.py` lists them. The 22 files that are not timing measurements were recomputed with the code of
   this archive and are identical to the stored ones (Section 9). For the three timing files (`E4b_solver.csv`,
   `E4c_residual.csv`, `E4d_dirac_chunk.csv`) the sources that differ are `det_common.py` (helpers without numerical
   content) and `pinn.py` (not used by `e4_soe.py`). For all other result files the recorded fingerprints equal the
   files of the archive; result files written from now on also record the fingerprint of `spd.py`, which became a
   module of the package after the runs of the other benchmarks and is used only by B5. The docstring of
   `fracsp/__init__.py`, a file that contains nothing but the version string, was corrected after the runs;
   `check_provenance.py` knows both fingerprints and counts the result files that record the earlier one in the
   column `docstring changed`.

Tables carry a side-car `<name>.meta.json` (deterministic part and B5) with the note that defines the columns, the
raw sources and the provenance of the aggregation. `results-manifest.csv` lists 11207 principal quantities of the
summary tables with their table, raw source and scripts; it is rewritten by `aggregate.py` and repeats table cells
verbatim.

## 9. Verification of this release

The archive was checked on the machine of Section 7 on 12 October 2026, in two passes. The first pass used a fresh
copy of the tree with a new virtual environment built from `requirements.txt`. After it, the solver of the
nonlinear step in `spd.py` was revised (Section 2), `dirac_cutoff.py` and two drivers were added, and the checks of
the first table were repeated in a copy of the tree, with the Python installation of the machine and without a
new virtual environment:

| check, second pass | result |
|---|---|
| tests | 450 passed, 20 s |
| `scripts/aggregate.py --clean --check` | all 136 output files (56 CSV tables, 38 side-car records, 31 figure-data files, the manifest, 10 LaTeX tables) regenerated identically, 84 s |
| `scripts/make_figures.py` | all 27 figures rewritten byte for byte, 21 s; the seven figures of the article are 165 mm x 44 mm with all fonts embedded |
| `scripts/check_provenance.py` | 5071 result files with a record, none without; the 25 files of Section 8 are listed |
| `scripts/smoke_test.py --workers 6` | 38 commands, none failed, 363 s |
| deterministic files of B5 recomputed with the revised solver: 12 exact references and 24 stepper baselines of the linear law, 4 references and 4 stepper baselines of FENE-P, the four result files of `positivity/` | 48 of 48 files identical to the stored ones in every value (wall times and provenance records excluded), 2.5 min |
| three training runs of B5 repeated with the stored seeds (one forward run, two FENE-P runs) | every value of the printed run summaries (31 to 33 keys per run) identical to the stored files (timings excluded) |
| `dirac_cutoff_coefficient.py` recomputed | data identical to the stored file, 50 s |

| check, first pass (not repeated) | result |
|---|---|
| deterministic stages recomputed: `e1_e3_verification.py all`, `l1_positivity.py`, `e4_soe.py kernel`, `exact`, `memory`, `dirac_reference.py B3` | 22 result files identical to the stored ones in every value (wall times excluded), 2.5 min |
| single runs repeated with the stored seeds: 6 of B2, 2 of B6, 3 of B1, one network and one least-squares fit of B7 | 13 of 13 identical in every stored value (timings excluded) |
| subset of B5 recomputed: 24 stepper baselines and 12 exact references, 13 forward, 7 FENE-P and 4 inverse training runs, 6 least-squares fits, Cramer-Rao bounds, FENE-P references, the files of `positivity/` | 82 of 82 files identical in every stored value (timings excluded), 77 s |

After the second pass, `spd.py` received the variant `newton-full` (Section 2), `b5_fenep_step.py` and the
aggregation of its tables were extended, `Table_Examples_continuous.csv` was added to the deterministic aggregation,
and four tests were added. The following checks were then repeated on the same day; the aggregation, the figures
and the smoke test were run in a copy of the tree:

| check, third pass | result |
|---|---|
| tests | 454 passed, 25 s |
| `scripts/aggregate.py --clean --check` | all 140 output files (58 CSV tables, 40 side-car records, 31 figure-data files, the manifest, 10 LaTeX tables) regenerated identically, 88 s |
| CSV tables against the second pass | 54 of 56 identical byte for byte; `Table_B5_fenep_step.csv` and `Table_B5_fenep_step_case.csv` have new columns and rows for the iteration with full steps and new labels for the iteration with halved steps, and their earlier values are unchanged; `Table_B5_fenep_step_case2.csv` and `Table_Examples_continuous.csv` are new |
| `scripts/make_figures.py` | all 27 figures rewritten byte for byte, 21 s |
| `b5_positivity.py`, `b5_fenep_step.py` and `dirac_cutoff_coefficient.py` rerun in the archive | every value of the four earlier result files reproduced (wall times and provenance records excluded) |
| `scripts/check_provenance.py` | 5071 result files with a record, none without; the 25 files of Section 8 are listed |
| `scripts/smoke_test.py --workers 6` | 38 commands, none failed, 345 s |

After the third pass the captions of five LaTeX tables of the article (`Table_01`, `02`, `04`, `05`, `07`) and one
row heading of `Table_01` were reworded in `make_tables.py`; no table entry changed. The tests (454 passed) and
`scripts/aggregate.py --check` (all 140 output files regenerated identically) were repeated afterwards.

On 13 October 2026 the aggregation was extended so that every number quoted in the article can be read from a
table (Section 10). `Table_B7_start_dependence.csv` is new, and columns were added to `Table_B2_budget.csv`,
`Table_B2_factorial.csv`, `Table_B2_soft_ic_pilot.csv`, `Table_B6_accuracy.csv`, `Table_B7_data_weight_pilot.csv`,
`Table_surrogate.csv`, `Table_B5_inverse_summary.csv`, `Table_B5_fenep_summary.csv` and `Neural_B7_estimates.csv`;
no earlier column changed. `Table_S2_paired.tex` has a new row with the gate effect for the block-gate map. Every
LaTeX table has a space of 3 pt between its caption and the top rule, which is taken back below the table, so that
the height of the float is unchanged. `Figure_02.pdf` also shows the additive map, and the legend entry of the
dotted curve of `Figure_03.pdf` and of `Neural_B6_positivity.pdf` now names what is drawn, the minimum over the
seeds at each time. No module of `fracsp`, no experiment driver, no configuration file and no raw result changed.

| check, fourth pass | result |
|---|---|
| tests | 454 passed |
| `scripts/aggregate.py --check` | all 141 output files (59 CSV tables, 40 side-car records, 31 figure-data files, the manifest, 10 LaTeX tables) regenerated identically |
| output files against the third pass | 118 of 140 identical (side-car records compared without time stamp and environment, as by `--check`); eight CSV tables and one figure-data file have new columns and unchanged earlier columns, two side-car records list the new columns, the manifest has the new quantities, and the ten LaTeX tables have the two spacing lines (`Table_S2_paired.tex` also the new row and a longer caption) |
| `scripts/make_figures.py` | 24 of 27 figures rewritten byte for byte; `Figure_02.pdf`, `Figure_03.pdf` and `Neural_B6_positivity.pdf` differ as described |
| `scripts/check_provenance.py` | 5071 result files with a record, none without; the 25 files of Section 8 are listed |

The smoke test was not repeated, because it runs the experiment drivers only.

On 13 October 2026, in a fifth pass, one deterministic computation and three aggregated items were added.
`code/experiments/b1_ls_mesh.py` repeats the least-squares estimate of the order through the implicit L1 time
stepper, the reference of benchmark B1, on the common mesh (N = 128, r = 3) and on meshes with 4 and 16 times as
many steps (N = 512 and 2048), for the noise-free data and, up to N = 512, for the ten noisy data sets. It writes
`results/raw/B1_design/reference/ls_mesh.csv` (69 rows, 6 min) with its provenance record; its 33 rows for the
common mesh repeat the column `alpha_ls` of `discrete_optimum.csv` digit for digit. The aggregation writes
`Table_B1_ls_mesh.csv`, and `Table_06.tex` has a row for N = 512 (0.3000, 0.4999 and 0.7996 for the orders 0.3, 0.5 and 0.8 from noise-free
data). `Table_B1_design.csv` has four new columns: `final_objective_median`, `final_objective_min` and
`final_objective_max` (training objective at the end of the runs, the residual plus 100 times the data misfit) and
`objective_joint_optimum` (the same objective at its minimum over the order and all nodal values).
`Table_Examples_continuous.csv` has seven new rows for a dephasing rate that is switched off at t = 1
(Remark 2.4(d) of the article). `make_tables.py --extra` also writes `Table_S3_training.tex`, the training
settings of all networks, from `code/configs/*.json` and the result tables. No module of `fracsp`, no earlier
driver, no configuration file and no earlier raw result changed; one test was added.

| check, fifth pass | result |
|---|---|
| tests | 455 passed, 85 s |
| `scripts/aggregate.py --check` | all 143 output files (60 CSV tables, 40 side-car records, 31 figure-data files, the manifest, 11 LaTeX tables) regenerated identically |
| output files against the fourth pass | CSV tables: 57 of 59 identical byte for byte, `Table_B1_design.csv` with four new columns and unchanged earlier columns, `Table_Examples_continuous.csv` with seven new rows and unchanged earlier rows, `Table_B1_ls_mesh.csv` new. Figure data: all 30 CSV files identical. LaTeX tables: 9 of 10 identical, `Table_06.tex` with the new row and a longer caption, `Table_S3_training.tex` new. Manifest: the 11029 earlier rows unchanged, 178 added. Side-car records: unchanged apart from time stamps, the hash of `aggregate_deterministic.py` in the records of the deterministic tables, and the record of the examples table |
| `scripts/make_figures.py` | not run: no figure-data file changed |
| `scripts/check_provenance.py` | 5073 result files with a record, none without; the 25 files of Section 8 are listed |
| `scripts/smoke_test.py --workers 4 --only B1 provenance` | 5 commands, among them the new driver, none failed, 27 s |

The sources used by the first two rows of the table of the first pass did not change after it, apart from the
docstring of `fracsp/__init__.py`. Of the B5 subset in its third row, the deterministic files and three
training runs were repeated with the revised `spd.py` (first table); the other training runs and the least-squares
fits were not. They do not call the solver of the nonlinear step: the forward and inverse problems use the linear
law, and the FENE-P training runs use it only through their references, stepper baselines and a-priori scales,
which the revised solver reproduces bit for bit (`Table_B5_fenep_stored_check.csv`).

Not recomputed for this check: B4 with its checks and the tail study (about 2 hours), the timing measurements of E4
(Section 6), and the campaigns as a whole. `scripts/rerun_timings.py` was run in the reduced configuration only.
The archive was not tested on Linux or macOS.

## 10. Map of tables and figures

Table and figure numbers follow the article. Raw results are in `results/raw/`, tables in `tables/data/`, figure
data in `figures/data/`; drivers are in `code/experiments/`. Aggregation scripts: D =
`scripts/aggregate_deterministic.py`, N = `scripts/aggregate_neural.py`, M = `scripts/aggregate_b5.py`.

### Tables of the article

`tables/final/Table_NN.tex` (LaTeX) are written by `scripts/make_tables.py` from the CSV tables named below; the
rounding rules are stated in the first lines of the script, and no number is typed in. `Table_S1_soe.tex`,
`Table_S2_paired.tex` and `Table_S3_training.tex` (option `--extra`) are the Tables S1 to S3 of the supplementary
material of the article (`supplementary.tex` in the manuscript folder): they hold the numbers that the article
quotes in the text of its Sections 5.2 and 5.3 and the training settings referred to in Section 5.1.

| article | file | content | data (`tables/data/`) | drivers |
|---|---|---|---|---|
| Table 1 | `Table_01.tex` | verification of the L1 formula and of the implicit L1 time stepper | `Table_E1_E3_summary.csv` | `e1_e3_verification.py` |
| Table 2 | `Table_02.tex` | factorial study of the two-level system: error by output map, initial condition and gate | `Table_B2_factorial.csv` | `run_b2_campaign.py`, `b2_prepare.py` |
| Table 3 | `Table_03.tex` | positivity for d = 3 and d = 4: error and smallest eigenvalue | `Table_B6_accuracy.csv`, `Table_B6_positivity.csv` | `b6_campaign.py`, `b6_prepare.py` |
| Table 4 | `Table_04.tex` | conformation tensor with the linear law: admissibility and error | `Table_B5_forward_admissibility.csv`, `Table_B5_forward_summary_long.csv`, `Table_B5_forward_stepper.csv` | `b5_conformation.py` |
| Table 5 | `Table_05.tex` | FENE-P in uniaxial extension | `Table_B5_fenep_summary.csv` | `b5_fenep.py` |
| Table 6 | `Table_06.tex` | identification of the order in the scalar problem | `Table_B1_design.csv`, `Table_B1_ls_mesh.csv` | `run_b1_campaign.py`, `b1_floor.py`, `b1_ls_mesh.py` |
| Table 7 | `Table_07.tex` | identification of the order and of two rates of the two-level system, networks against least squares | `Table_B7_estimates.csv` | `b7_campaign.py`, `b7_inverse.py` |
| Table 8 | `Table_08.tex` | laser-driven Dirac model: excitation density, logarithmic slope, valley contrast, peak current | `Table_Dirac_B4.csv`, `Table_Dirac_B4_fits.csv`, `Table_Dirac_tail.csv` | `dirac_reference.py`, `dirac_tail_check.py` |
| Table S1 | `Table_S1_soe.tex` | SOE history against full history: errors, times, time ratios | `Table_SOE.csv` | `e4_soe.py` |
| Table S2 | `Table_S2_paired.tex` | paired effects of the factors in the factorial study | `Table_B2_paired.csv` | `run_b2_campaign.py` |
| Table S3 | `Table_S3_training.tex` | training settings of all networks | `code/configs/*.json`, `Table_B2_factorial.csv`, `Table_B6_accuracy.csv`, `Table_surrogate.csv` | none (configuration files) |

Runs of the campaigns that the article counts. B2 (`Table_B2_factorial.csv`, column `cell`): of the 1440 runs of
the stage `main` the article uses 1050, namely `unc`, `lin` and `factor` with `soft-tuned`, `hard-exp` and
`hard-pow`, `add` with `hard-exp` and `hard-pow` (three orders, three states and 10 seeds each) and `gate2b`, the
block-gate map of the article, with both gates for the pure state; of the 270 runs of `extras` it uses the 180 runs
`*/hard-pow/singular` with the second input tau^alpha at the orders 0.3 and 0.6. Of these 1230 runs, 960 have a
hard initial condition. The archive also holds `herm/*` (270 runs), `gate2/*` (60), `unc/soft-w1` and
`factor/soft-w1` (60) and `bloch/hard-pow` of `extras` (90), which the article does not use. B6
(`Table_B6_positivity.csv`): the article counts the 160 runs of `unc`, `herm` and `lin` of the stage `main`, the
196 runs of `factor`, `add` and `gate2b` (160 of `main`, 36 of `long`) and the 12 runs of `lin` of `long`; the 40
runs of `gate2/hard-pow` are not used. B5, forward problem (`Table_B5_forward_summary_long.csv`): the article uses
the 840 runs of the seven combinations of output map and residual norm of its Table 4; the 120 runs of the variant
`anchor-fro` (output map C0 + s_C g A A^T, which enforces C >= C0, a property that the exact solutions do not
have) are not used. B7 (`Table_B7_start_dependence.csv`, rows `PINN lin and PINN add`): the 80 runs of the
linear head and the additive map fall into 32 runs started from an order of at most 0.40 and 48 started from at
least 0.58, the same starts in all four designs; `alpha_init` in `Neural_B7_estimates.csv` is the start of each run.

Newton sweep of B5 (`Table_B5_fenep_step.csv`). The iteration with full steps fails in 418 of the 648 runs of
uniaxial extension. With the stopping rule of the sweep (update below 1e-12 (1 + |C|_F), at most 60 iterations per
step) it converges in every time step in 358 of them and stops converging in 60. The article writes "about 360"
for this split, because it depends on the tolerance in a few runs whose iterates, once inadmissible, are sensitive
to rounding: `spd.solve_l1(..., nonlinear='newton-full', tol=...)` on the meshes of the sweep gives 358 and 60 for
the tolerances 1e-12 and 1e-11, 354 and 64 for 1e-13 and 357 and 61 for 1e-10, and with the tolerance 1e-12 the
same numbers for 40 to 1000 iterations per step. The 230 admissible runs, the total of 418, the 391 and 27 failures above and within the step
bound and the 15 convergent runs among the 27 are the same for all these settings. These repetitions are not part
of the stored results.

### Figures of the article

`figures/final/Figure_NN.pdf` are drawn by `scripts/plot_final.py`: one row of panels, 165 mm x 44 mm, vector PDF with
embedded TrueType fonts (DejaVu Sans, axis labels 8 pt, tick labels and legends 7 pt), panel labels (a), (b), ...
inside the panels. Colours (Okabe-Ito palette), the same in all figures: L1 time stepper, exact and reference
solutions black; unconstrained output grey; linear head blue; factor map vermilion; additive map green; block-gate
map purple; exponential map orange; in the figures of the Dirac model the orders 1, 0.9 and 0.8 are black, blue and
vermilion. Every series also differs by position, marker shape, fill or line style, so that no figure depends on
colour alone. Tick labels of the output maps: unc. = unconstrained, lin. = linear head, fac. = factor, add. =
additive, block = block gate.

| article | file | content | data | drivers |
|---|---|---|---|---|
| Figure 1 | `Figure_01.pdf` | factorial study of the two-level system: error of every run and median per initial state, by output map and initial condition, for the orders 0.3 (a), 0.6 (b), 0.9 (c); lines: L1 time stepper | `Neural_B2_runs.csv`, `Neural_B2_stepper.csv` | `run_b2_campaign.py`, `b2_prepare.py` |
| Figure 2 | `Figure_02.pdf` | initial layer for the pure initial state: error for the order 0.6 (a), smallest eigenvalue for the orders 0.6 (b) and 0.3 (c); factor map with both gates, linear head, block-gate and additive map | `Neural_B2_error_curves.csv` | `run_b2_campaign.py`, `b2_prepare.py` |
| Figure 3 | `Figure_03.pdf` | positivity for d = 3 and d = 4: error (a) and smallest eigenvalue (b) of every run by output map; smallest eigenvalue along the trajectory at the order 0.5 for d = 3 (c) and d = 4 (d): exact, median and minimum over the seeds for the linear head, median for the factor map | `Neural_B6_runs.csv`, `Neural_B6_stepper.csv`, `Neural_B6_curves.csv` | `b6_campaign.py`, `b6_prepare.py` |
| Figure 4 | `Figure_04.pdf` | FENE-P, uniaxial extension, Wi = 5, order 0.6, all seeds: stretch (a) and smallest eigenvalue (b) | `B5_fenep_curves.csv` | `b5_fenep.py` |
| Figure 5 | `Figure_05.pdf` | parametric surrogate: error over the parameter rectangle for two surrogates (a, b) and for the L1 time stepper (c); total time against the number of queries (d) | `Neural_surrogate_grid.csv`, `Neural_surrogate_stepper.csv`, `Table_surrogate.csv`, `Table_surrogate_breakeven.csv` | `b2_surrogate.py`, `b2_timing.py` |
| Figure 6 | `Figure_06.pdf` | Dirac model, tail of the upper-band population: on a ray (a), coefficient against the direction (b), excitation density against the cutoff (c) | `FigDet_tail_rays.csv`, `FigDet_B4_cutoff.csv`, `Table_Dirac_tail.csv` | `dirac_tail_check.py`, `dirac_reference.py` |
| Figure 7 | `Figure_07.pdf` | Dirac model, sheet current for linear polarization: current in time (a), peak against the cutoff for the band-referenced current (b) and for the current referred to the initial state (c) | `FigDet_B4_current_traces.csv`, `FigDet_B4_cutoff.csv`, `Table_Dirac_B4_fits.csv` | `dirac_reference.py` |

### All tables in `tables/data`

The first column names the table or figure of the article that uses the file; it is empty for tables that are not
used in the article.

| article | table file | content | script | raw results |
|---|---|---|---|---|
| | `Table_E1_E3.csv` | E1, E1T, E3: errors and observed orders for every mesh and refinement pair | D | `E1_E3/` |
| Table 1 | `Table_E1_E3_summary.csv` | one row per convergence series: order of the last pair and its range | D | `E1_E3/` |
| | `Table_E2_alpha_derivative.csv` | E2: derivative with respect to the order, automatic differentiation, closed form, finite differences, 60-digit reference | D | `E1_E3/E2_alpha_gradient.csv` |
| | `Table_L1_positivity.csv` | E5 and Dirac trajectories: smallest eigenvalue, trace and Hermiticity defects, Choi matrices | D | `E5_positivity/`, `DIRAC_v2/` |
| text (Remark 2.4, Section 4.1) | `Table_Examples_continuous.csv` | numbers that follow from closed forms: extrema of the modulus of E_0.8(-ix), coherence factor of the dephasing example with a temporarily negative rate and with a rate that is switched off, number of nodes of the graded mesh in the initial layer | D | none |
| text (S1) | `Table_SOE.csv` | E4: work-precision of six L1 solvers (errors, wall times, memory, time ratios) | D | `E4_soe/E4b_*` |
| | `Table_SOE_matched.csv` | E4: steps and time needed to reach the accuracy of the uniform SOE solver | D | `E4_soe/E4b_*` |
| text (Section 3.4) | `Table_SOE_kernel.csv` | E4: number of modes and realised kernel error for both truncation criteria; ratio of the number of modes to its leading-order estimate | D | `E4_soe/E4a_*` |
| | `Table_SOE_modes.csv` | E4: growth of the number of modes with the tolerance and with the window | D | `E4_soe/E4a_kernel.csv` |
| | `Table_SOE_residual.csv` | E4: forward and backward pass of the residual operator, convolution against SOE recurrence | D | `E4_soe/E4c_residual.csv` |
| | `Table_SOE_dirac_chunk.csv` | E4: full against SOE history for 256 Dirac wave vectors | D | `E4_soe/E4d_dirac_chunk.csv` |
| | `Table_Dirac_B3.csv` | B3: final conduction population, Richardson extrapolated, with the independent second-order scheme | D | `DIRAC_v2/B3_summary.csv` |
| | `Table_Dirac_B3_convergence.csv` | B3: convergence histories of both schemes | D | `DIRAC_v2/B3_summary.csv` |
| Table 8 | `Table_Dirac_B4.csv` | B4: densities, valley contrast and peak sheet current for 16 cutoffs and the cutoff limit, with error estimates | D | `DIRAC_v2/B4_*.npz`, `B4check_*.npz` |
| | `Table_Dirac_B4_cutoff.csv` | B4: all quantities against the cutoff; logarithmic slope of the total density | D | `DIRAC_v2/B4_*.npz` |
| Table 8, Figure 7 | `Table_Dirac_B4_fits.csv` | B4: cutoff extrapolations of the current and of the valley contrast; fitted 1/kappa_c coefficient of the current against its closed form | D | `DIRAC_v2/B4_*.npz`, `current_cutoff_coefficient.csv` |
| | `Table_Dirac_B4_runs.csv` | B4: resolution, realised SOE error, positivity, cost of the runs | D | `DIRAC_v2/B4_*.npz` |
| | `Table_Dirac_B4_gridcheck.csv` | B4: momentum-quadrature check with halved spacings | D | `DIRAC_v2/B4check_grid_*.npz` |
| | `Table_Dirac_B4_timecheck.csv` | B4: L1 with Richardson extrapolation against the product-trapezoidal rule | D | `DIRAC_v2/B4check_time_*.npz` |
| | `Table_Dirac_B4_odecheck.csv` | B4, order 1: tolerance check of the Runge-Kutta reference | D | `DIRAC_v2/B4check_ode_*.npz` |
| Table 8, Figure 6 | `Table_Dirac_tail.csv` | high-momentum tail: rays, dissipation switched off, final time, order 1, analytical constants, logarithmic slopes, valley contrast, shell exponents | D | `DIRAC_v2/tail_*.csv`, `B4_*.npz` |
| Table 2 | `Table_B2_factorial.csv` | B2: error and admissibility statistics of every cell (stage, order, state), L1 stepper as reference rows | N | `B2_factorial/main`, `extras`, `long`, `reference` |
| | `Table_B2_marginal.csv` | B2: geometric means over states and seeds | N | `B2_factorial/main` |
| text (S2) | `Table_B2_paired.csv` | B2: paired comparisons of cells (ratios, wins, Wilcoxon test) | N | `B2_factorial/main`, `extras`, `long` |
| text (Section 5.3) | `Table_B2_budget.csv` | B2: four-fold optimiser budget against the default, with the iteration numbers of both budgets | N | `B2_factorial/main`, `long` |
| text (Section 5.3) | `Table_B2_soft_ic_pilot.csv` | B2: selection of the weight of the soft initial condition, with the seeds of the pilot runs | N | `B2_factorial/pilot` |
| | `Table_B2_timing.csv` | B2: sequential timing pass and bit-for-bit comparison | N | `B2_factorial/timing` |
| Table 3 | `Table_B6_accuracy.csv` | B6: error statistics of every cell with its optimiser budget, L1 stepper as reference rows | N | `B6_positivity/main`, `long`, `reference` |
| Table 3 | `Table_B6_positivity.csv` | B6: frequency, size and duration of negative eigenvalues | N | `B6_positivity/main`, `long` |
| | `Table_B6_paired.csv` | B6: paired comparisons of maps and gates | N | `B6_positivity/main`, `long` |
| Table 7 | `Table_B7_estimates.csv` | B7: bias, standard deviation and root-mean-square error of the estimates, Cramer-Rao bounds | N | `B7_inverse/main`, `reference/crb.csv` |
| | `Table_B7_vs_least_squares.csv` | B7: distance of every method to the least-squares estimate | N | `B7_inverse/main` |
| text (Section 5.6) | `Table_B7_start_dependence.csv` | B7: network fits by the initial guess of the order: runs that end at the lower bound of the order, and the other runs against least squares on the same data sets | N | `B7_inverse/main` |
| | `Table_B7_data_weight_pilot.csv` | B7: selection of the data weight, with the realizations of the pilot runs | N | `B7_inverse/pilot` |
| Table 6 | `Table_B1_design.csv` | B1: estimates of the order and forward errors for the design input x gate, discrete optima, mesh gradings, training objective at the end of the runs | N | `B1_design/main`, `mesh`, `reference` |
| Table 6 | `Table_B1_ls_mesh.csv` | B1: least squares through the L1 time stepper on the common mesh and on finer meshes | N | `B1_design/reference/ls_mesh.csv` |
| | `Table_B1_discrete_optimum.csv` | B1: discrete optima of the inverse problem | N | `B1_design/reference/discrete_optimum.csv` |
| | `Table_B1_l1_floor.csv` | B1: implicit L1 stepper on the meshes of the networks | N | `B1_design/reference/l1_floor.csv` |
| | `Table_B1_paired.csv` | B1: paired effects of input and gate | N | `B1_design/main` |
| Figure 5 | `Table_surrogate.csv` | surrogate: accuracy, admissibility, cost per query, break-even number of queries, sizes of the training pool and of the test sets; L1 stepper rows | N | `B2_surrogate/main`, `reference`, `timing.json` |
| Figure 5 | `Table_surrogate_breakeven.csv` | surrogate: break-even against every stepper mesh and both training times | N | `B2_surrogate/main`, `timing` |
| Table 4 | `Table_B5_forward_stepper.csv` | B5: implicit L1 stepper on the collocation mesh | M | `B5_conformation/forward/stepper_*.json` |
| | `Table_B5_forward_a0.6_aligned.csv`, `_a0.6_rotated`, `_a0.9_aligned`, `_a0.9_rotated` | B5 forward: median error; log-Euclidean error; runs that lose positive definiteness, per output map and residual norm | M | `B5_conformation/forward` |
| Table 4 | `Table_B5_forward_summary_long.csv` | B5 forward: every configuration | M | `B5_conformation/forward` |
| | `Table_B5_forward_best_vs_stepper.csv` | B5 forward: most accurate network variant against the stepper | M | `B5_conformation/forward` |
| Table 4 | `Table_B5_forward_admissibility.csv` | B5 forward: all runs pooled | M | `B5_conformation/forward` |
| | `Table_B5_forward_budget.csv` | B5 forward: three-fold optimiser budget | M | `B5_conformation/forward/*_budget3.json` |
| text (Section 5.6) | `Table_B5_inverse_summary.csv`, `Table_B5_inverse_paired.csv` | B5 inverse: networks against least squares through the stepper, Cramer-Rao bounds; starts of the least-squares fits | M | `B5_conformation/inverse` |
| Table 5 | `Table_B5_fenep_summary.csv` | B5 FENE-P: error and admissibility per variant; difference of the two solutions behind the reference | M | `B5_conformation/fenep` |
| | `Table_B5_positivity_steppers.csv`, `_nonlinear`, `_exact` | B5: positive definiteness of explicit and implicit L1 stepping and of the exact solution; nonlinear laws with Newton's iteration (steps halved at the pole of the FENE-P term, result not tested) and with the safeguarded solver | M | `B5_conformation/positivity/positivity.json` |
| text (Sections 5.5, 7) | `Table_B5_fenep_step.csv` | B5: FENE-P step on 648 + 504 meshes: Newton's iteration with full steps and with steps halved at the pole, against the root solver and the safeguarded solver; failures above and within the step bound of the linear law | M | `B5_conformation/positivity/fenep_step.json` |
| text (Section 5.5) | `Table_B5_fenep_step_case.csv` | B5: Wi = 20, order 0.6, b = 50, 256 uniform steps: the three real solutions of the first step, Newton and admissible iterates, reference | M | `B5_conformation/positivity/fenep_step.json` |
| text (Section 5.5) | `Table_B5_fenep_step_case2.csv` | B5: Wi = 20, order 0.9, b = 10, 512 uniform steps, a mesh within the step bound of the linear law: Newton's iteration with full steps passes the pole tr C = b at step 9; iterates before and at that step, admissible solution | M | `B5_conformation/positivity/fenep_step.json` |
| text (Section 5.5) | `Table_B5_fenep_stored_check.csv` | B5: references, stepper baselines and scales of the FENE-P benchmark recomputed with the root solver | M | `B5_conformation/positivity/fenep_step.json`, `fenep/reference_*.npz` |

### Further figures in `figures/candidates`

These figures are not in the article; they show further views of the same data (convergence histories, paired
effects, momentum maps, the benchmarks B1, B3 and B7, the forward and inverse problems of B5). All are vector PDF
with embedded TrueType fonts (DejaVu Serif, 8 pt labels), at most 190 mm wide, in a palette of three hues (blue,
orange, aqua); series are also distinguished by marker or line style.

| figure file | content | script | data |
|---|---|---|---|
| `FigDet_E_verification.pdf` | E3 and E1T: convergence on uniform and graded meshes | `plot_deterministic.py` | `Table_E1_E3.csv` |
| `FigDet_SOE.pdf` | E4: number of modes, realised kernel error, wall time, work-precision, time ratio | `plot_deterministic.py` | `Table_SOE*.csv` |
| `FigDet_Dirac_B3.pdf` | B3: conduction population in time, convergence of both schemes | `plot_deterministic.py` | `FigDet_B3_*.csv` |
| `FigDet_Dirac_current.pdf` | B4: sheet current, both definitions, cutoff dependence, intraband and interband parts | `plot_deterministic.py` | `FigDet_B4_current_traces.csv`, `FigDet_B4_cutoff.csv`, `Table_Dirac_B4.csv` |
| `FigDet_Dirac_tail.pdf` | tail: rays against the analytical constants, exponents, densities against the cutoff, final time | `plot_deterministic.py` | `Table_Dirac_tail.csv`, `FigDet_B4_cutoff.csv`, `FigDet_tail_constant_vs_tf.csv` |
| `FigDet_Dirac_maps.pdf` | B4: momentum maps of the final conduction population | `plot_deterministic.py` | `FigDet_B4_maps_zeta1.npz` |
| `Neural_B2_factorial.pdf` | B2: error of every cell, seeds and medians, L1 stepper | `plot_neural.py` | `Neural_B2_runs.csv`, `Neural_B2_stepper.csv` |
| `Neural_B2_effects.pdf`, `Neural_B2_effects_distance_to_L1.pdf` | B2: paired ratios for gate, initial condition and output map | `plot_neural.py` | `Table_B2_paired.csv` |
| `Neural_B2_initial_layer.pdf` | B2: error and smallest eigenvalue against time on a logarithmic axis | `plot_neural.py` | `Neural_B2_error_curves.csv` |
| `Neural_B6_positivity.pdf` | B6: error, smallest eigenvalue per run and in time | `plot_neural.py` | `Neural_B6_*.csv` |
| `Neural_B7_estimates.pdf` | B7: estimates against the noise level with Cramer-Rao bands | `plot_neural.py` | `Neural_B7_estimates.csv`, `Table_B7_estimates.csv` |
| `Neural_B1_design.pdf` | B1: error of the identified order for the design input x gate; mesh grading | `plot_neural.py` | `Neural_B1_inverse_runs.csv`, `Table_B1_design.csv` |
| `Neural_surrogate.pdf` | surrogate: error over the parameter rectangle, total time against the number of queries | `plot_neural.py` | `Neural_surrogate_*.csv`, `Table_surrogate.csv` |
| `B5_model_eigenvalues.pdf` | B5: extreme eigenvalues of the exact conformation tensor | `plot_b5.py` | `B5_model_eigenvalues.csv` |
| `B5_forward_errors_a0.6.pdf`, `B5_forward_errors_a0.9.pdf` | B5 forward: error of every run by output map and residual norm | `plot_b5.py` | `B5_forward_runs.csv`, `B5_forward_stepper.csv` |
| `B5_forward_min_eig.pdf` | B5 forward: smallest eigenvalue along the trajectory | `plot_b5.py` | `B5_forward_min_eig_curves.csv` |
| `B5_inverse_estimates.pdf` | B5 inverse: estimates of order and relaxation time | `plot_b5.py` | `B5_inverse_estimates.csv`, `B5_inverse_crb.csv` |
| `B5_fenep_trajectories.pdf` | B5 FENE-P: stretch and smallest eigenvalue in time | `plot_b5.py` | `B5_fenep_curves.csv` |

Figure data that no figure of this archive draws: `FigDet_E1_truncation.csv`, `FigDet_E3_convergence.csv`,
`FigDet_SOE_kernel.csv`, `FigDet_SOE_work_precision.csv`, `FigDet_tail_shell_profile.csv`, `Neural_B1_forward_runs.csv`. Stored trajectories of seed 0 (`*_test_rho.npy` in `B2_factorial/main` and
`B6_positivity/main`) are raw results that no table or figure uses.

## 11. Names that refer to an earlier version

Some column names, notes and docstrings mention "the first version", "the first manuscript version" or "the first
campaign". They refer to an earlier, unpublished version of the manuscript and of this code, whose results are not
part of this archive:

- Key `init` of `fracsp.dirac.current_integrands`, columns `peak_J*_initref` of `Table_Dirac_B4_cutoff.csv` and
  `peak_J*_sea` of `Table_Dirac_B4.csv`: the sheet current referred to the initial state, the definition of that
  version, and its difference to the current of the article; it grows linearly with the momentum cutoff and is
  kept to document this. The quantity of the article is the current referred to the instantaneous valence band
  (`peak_J*_band`).
- `criterion='absolute'` of `fracsp.operators.soe_kernel` (rows `absolute` of `Table_SOE_kernel.csv`): the
  truncation rule of that version; all results use `criterion='relative'`.
- Direct, SP-soft and SP-fPINN were the names of the B2 cells `unc/soft-w1`, `factor/soft-w1` and
  `factor/hard-exp`; `plain` and `singular` those of the B1 cells `plain/exp` and `singular/pow`.
  `fracsp.pinn.Model` implements the former and is covered by a test of its equivalence with `DensityNet`.
- `DIRAC_v2` is the name of the folder of the Dirac results; there is no other version in the archive.
- The equation numbers (4.25), (4.27) and (5.30) in a docstring of `fracsp/operators.py` and in two comments of
  `tests/test_all.py` are those of that version; the docstring was left as it is because the fingerprint of
  `operators.py` is recorded in the result files.

## 12. Licence and citation

Source code (`code/`, `scripts/`, `requirements.txt`): MIT License, see `LICENSE`. Data and figures (`results/`,
`tables/`, `figures/`, `results-manifest.csv`): Creative Commons Attribution 4.0 International (CC BY 4.0).
Copyright: the authors named above.

Please cite the article and this archive; the metadata are in `CITATION.cff`.
