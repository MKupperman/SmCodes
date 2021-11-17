from typing import Union
import numpy as np
import numpy.typing as npt  # Seems to resolve an issue.... not sure what's going on with this import
import typing

from numba import njit


def chebdiff(n:int=2, h:float=1, usenumba=False) -> typing.Union[npt.ArrayLike, npt.ArrayLike]: 
    """
    Obtains the chebyshev differentiation matrix and chebyshev points
    on the rescaled interval $[-h,h]$. By default, $h=1$ and we obtain the
    "canonical" differentiation matrix.

    Handles type checking before dispatching a call to a numba compiled function

    Code translated from "cheb.m" availible at:
    https://people.maths.ox.ac.uk/trefethen/spectral.html

    """

    n,h = checknh(n,h)
    if usenumba:  # Use the numba version
        D, x = _chebdiff_n(n,h)
    else:  # Nopython
        D, x = _chebdiff(n,h)

    return D, x

def _chebdiff(n:int=2,h:float=1) -> typing.Union[npt.ArrayLike, npt.ArrayLike]:
    """
    The implementation 
    """

    x =_chebpoints(n=n,h=h)  # bypass safety checks for numba
    # x = chebpoints(n=n, h=h)
    c = np.ones(n + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1) ** (np.arange(n + 1))
    X = np.stack([x for _ in range(n + 1)]).T
    dX = X - X.T
    D = (c.reshape(n+1, 1) * (1 / c)) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, x

@njit
def _chebdiff_n(n:int=2,h:float=1) -> typing.Union[npt.ArrayLike, npt.ArrayLike]:
    """
    The implementation of the spectral differentiation matrix
    """

    x =_chebpoints(n=n,h=h)  # bypass safety checks for numba
    # x = chebpoints(n=n, h=h)
    c = np.ones(n + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1) ** (np.arange(n + 1))
    X = np.zeros((n+1, n+1))
    for rdx in range(n+1):
        X[rdx, :] = x
    #X = np.stack([x for _ in range(n + 1)]).T
    dX = X - X.T
    D = (c.reshape(n+1, 1) * (1 / c)) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, x

def chebpoints(n:int=2,h:float=1) -> npt.ArrayLike:
    n,h = checknh(n,h)
    return _chebpoints(n=n, h=h)


@njit
def _chebpoints(n:int=2,h:float=1) -> npt.ArrayLike:
    """ 
    Compute the n+1 chebyshev points rescaled on $[-h,h]$.
    """
    data = np.pi * np.arange(n + 1) / n
    x = np.cos(data) * h
    return x


def checknh(n,h):
    """
    Utility function to check that n and h are "good" values
    """
    assert type(n) is int  # For shape
    assert np.can_cast(type(h), float)  # For size
    return n, np.asarray(h, dtype=float)
