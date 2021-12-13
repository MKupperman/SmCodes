# SMCODES: Spectral Multigrid Chebyshev Ordinary Differential Equation Solver

_Cade Ballew, Michael Kupperman_  Amath 570

A python implementation of SMCODES, a semi-implicit time-stepping method with spectral flair.


A minimal example:

```python
from SMCODES import SmcSolver

testfun = lambda x: - x
smc = SmcSolver(f, u0=1, hstep=1e-1, stages=4, do_implicit_solve=True)
times, soln = smc.solve()

```
