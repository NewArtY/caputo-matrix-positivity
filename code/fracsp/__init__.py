"""fracsp: Caputo time-fractional evolution equations whose state is a positive-semidefinite matrix (a density
matrix, a conformation tensor): implicit L1 time stepping, sum-of-exponentials history, and residual-based
networks with admissible output maps.

Modules
-------
operators     L1 Caputo operators on uniform and nonuniform meshes (torch, differentiable in the order), SOE kernel
              and fast L1 operator
reference     implicit L1 solvers (numpy): full history, SOE history on uniform and nonuniform meshes;
              product-trapezoidal rule; positivity of the L1 step; Richardson extrapolation
mittag        Mittag-Leffler functions in multiprecision arithmetic (mpmath)
exact         exact solutions of linear problems (matrix Mittag-Leffler functions)
layers        output maps for density matrices, gates of the exact initial condition, projection onto the density
              matrices
pinn          networks, GKSL generators, training loop, implicit L1 time stepper in the metric of the networks,
              error and admissibility metrics, provenance record, campaign launcher
dirac         laser-driven massive-Dirac two-band model in Bloch form; constants of the high-momentum tail
dirac_cutoff  next order of the high-momentum expansion; dependence of the sheet current on the momentum cutoff
spd           conformation-tensor models (upper-convected Maxwell, Giesekus, FENE-P): exact solutions, implicit L1
              time stepper with the admissible solution of the nonlinear step, output maps for symmetric positive
              definite tensors, residual norms
"""
__version__ = "0.1.0"
