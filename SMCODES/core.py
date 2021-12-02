"""
Core algorithm for 
Spectral Multigrid Chebyshev Ordinary Differental Equation Solver
"SMCODES"

Only handles autonomous ODEs
"""

import typing

import numpy as np
import numpy.typing as npt

from chebutils import chebdiff


def smcodes_1step(fun:typing.Callable, y0:np.typing.ArrayLike, nstages:int) -> np.typing.ArrayLike:
    # First we preallocate the general matrices and nodes that we'll need a lot
    Dmats = {}
    xnodes = {}
    for idx in range(nstages):
        D, x = chebdiff(n=idx, h=1, use_numba=True)
        Dmats[idx], xnodes[idx]  = D, x 
    
    topnode_values = np.zeros(nstages+1)
    topnode_values[0] = fun(y0)

    for subpdeg in range(np.degrees):
        for subprob in None:

            pass
    
    return True

def solve_cheb_subproblem(D:npt.ArrayLike, u:npt.ArrayLike, fun:typing.Callable) -> npt.ArrayLike:
    """
    Solve the Du = f(u) subproblem in space. 
    """

    assert len(u.shape) == 1, 'Did not get a vector of time values'
    n = u.shape[0]
    return True