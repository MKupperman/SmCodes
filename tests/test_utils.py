import pytest
from SMCODES.chebutils import chebdiff
import numpy as np

def test_chebdiff():
    for use_numba in [False, True]:
        for hdx in range(6):  # h=0,1,2,...6 gives a good range of discretizations
            for n in range(1, 10):  # Test first 10 cases
                h = 2**(-n)  # determine the actual step size
                D, x = chebdiff(n, h=h, use_numba=use_numba)
                assert np.allclose(np.sum(D, axis=1), 0, atol=1e-7) == True, f'Case n={n}, h={h} failed to satisfy tolerance'
                assert len(x.shape) == 1, 'Did not get a vector back, got something with depth'
                assert x.shape[0] == n+1, f'Did not get n+1 points back, got {np.length(x)} points instead' 
                # Now let's verify that we got shifted and scaled Chebyshev points back
                inverted_cheb_pts = np.arccos(x/h)/np.pi*2*(n)
                print(inverted_cheb_pts)
                assert np.allclose(inverted_cheb_pts - 2*np.arange(n+1), 0), f'Did not get Chebyshev points within tolerance'
                # First we check chebyshev points

    return True
