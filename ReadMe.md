# SMCODES: Spectral Multigrid Chebyshev Ordinary Differential Equation Solver

_Cade Ballew, Michael Kupperman_  Amath 570

A python implementation of SMCODES, a diagonally-implicit time-stepping method with spectral flair.

```python
from SMCODES import smcodes
testfun = lambda t,x: - x
tspan = [0, 1]
h = 0.1
y0 = 0
[t,y] = smcodes(testfun, tspan=tspan, h=h, y0=y0)
```
