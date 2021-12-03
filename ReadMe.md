# SMCODES: Spectral Multigrid Chebyshev Ordinary Differential Equation Solver

_Cade Ballew, Michael Kupperman_  Amath 570

A python implementation of SMCODES, a diagonally-implicit time-stepping method with spectral flair.



```python
from SMCODES import SmcSolver

testfun = lambda x: - x
tmax = 1
h = 0.1
y0 = 0

smc = SmcSolver(fun=testfun, u0=1, hstep=hstep, stages=stage)
t, y = smc.solve(tmax=tmax)
```
