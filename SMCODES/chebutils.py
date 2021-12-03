import typing

import numpy as np
from numba import njit
from numpy import typing as npt  # Seems to resolve an issue.... not sure what's going on with this import


def chebdiff(n: int = 2, h: float = 1, use_numba: bool = False, flip: bool = False) \
        -> typing.Union[npt.ArrayLike, npt.ArrayLike]:
    """
    Obtains the chebyshev differentiation matrix and chebyshev points
    on the rescaled interval $[-h,h]$. By default, $h=1$ and we obtain the
    "canonical" differentiation matrix.cl

    Handles type checking before dispatching a call to a numba compiled function

    Code translated from "cheb.m" available at:
    https://people.maths.ox.ac.uk/trefethen/spectral.html

    """

    n, h = check_nh(n, h)
    if use_numba:  # Use the numba version
        D, x = _chebdiff_n(n, h)
    else:  # No Python mode
        D, x = _chebdiff(n, h)
    if flip:  # Flip it so we get nodes in increasing order
        # D = np.flipud(D)  # TODO - check the symmetries here
        x = np.flip(x)
    return D, x


# noinspection DuplicatedCode
def _chebdiff(n: int = 2, h: float = 1) -> typing.Union[npt.ArrayLike, npt.ArrayLike]:
    """
    The implementation 
    """

    x = _chebpoints(n=n, h=h)  # bypass safety checks for numba
    c = np.ones(n + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1) ** (np.arange(n + 1))
    X = np.stack([x for _ in range(n + 1)]).T
    dX = X - X.T
    D = (c.reshape(n + 1, 1) * (1 / c)) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, x


# noinspection DuplicatedCode
@njit
def _chebdiff_n(n: int = 2, h: float = 1) -> typing.Union[npt.ArrayLike, npt.ArrayLike]:
    """
    The implementation of the spectral differentiation matrix
    """

    x = _chebpoints(n=n, h=h)  # bypass safety checks for numba
    # x = chebpoints(n=n, h=h)
    c = np.ones(n + 1)
    c[0] = 2
    c[-1] = 2
    c *= (-1) ** (np.arange(n + 1))
    X = np.zeros((n + 1, n + 1))
    for rdx in range(n + 1):
        X[rdx, :] = x
    # X = np.stack([x for _ in range(n + 1)]).T
    dX = X - X.T
    D = (c.reshape(n + 1, 1) * (1 / c)) / (dX + np.eye(n + 1))
    D = D - np.diag(np.sum(D, axis=1))

    return D, x


def chebpoints(n: int = 2, h: float = 1) -> npt.ArrayLike:
    n, h = check_nh(n, h)
    return _chebpoints(n=n, h=h)


@njit
def _chebpoints(n: int = 2, h: float = 1) -> npt.ArrayLike:
    """ 
    Compute the n+1 chebyshev points rescaled on $[-h,h]$.
    """
    data = np.pi * np.arange(n + 1) / n
    x = np.cos(data) * h
    return x


def check_nh(n: int, h: float):
    """
    Utility function to check that n and h are "good" values.
    Ensures that the values returned are an integer and a float

    Args:
        n (int): candidate integer
        h (float):

    Returns:
        n (int): cleaned integer
        h:
    """
    assert type(n) is int  # For shape
    assert np.can_cast(type(h), float)  # For size
    return n, float(h)  # do the cast here
