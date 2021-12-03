"""
Core algorithm for 
Spectral Multigrid Chebyshev Ordinary Differental Equation Solver
"SMCODES"

Only handles autonomous ODEs
"""

import typing

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .chebutils import chebdiff


class ChebProblem(object):
    def __init__(self, f: typing.Callable, degree: int,
                 u0: float, h: float, diffmatDict: dict, xnodes: dict):
        """ Construct a Chebyshev update step problem to solve.

            A problem class for handling the problems contained
            in each step.
        """
        self.f = f
        self.diffmatDict = diffmatDict
        self.xnodes = xnodes
        self.h = h
        self.degree = degree
        self.subproblems = []  # Recursive referencing
        self.u = np.zeros(self.degree)
        self.u0 = u0
        self.u[0] = u0
        self.solution = None  # will store the full u vector
        self.can_solve = False  # bool_flag

    def __len__(self):
        """ Return the length of the step we are trying to take


        """

    def solve(self):
        """ Solve the problem using the values we have

        Do a Linear least squares for an initial guess
        of the solution value. Then run out a Nelder-mead simplex
        search for the fixed point via a rootfinding.
        """

        D = self.diffmatDict[self.degree].copy()
        D *= 1 / self.h
        b_known = self.f(self.u)  # u is the known values
        # This can probably be optimized
        D1 = D[:-1, :-1]  # the upper block
        c = D1.dot(self.u)
        c2 = D[-1, :-1].dot(self.u)
        # c + ax = b => x =(b-c)/a
        bright = b_known - c
        # Now do a linear-least-squares solve
        A = D[-1, :-1]  # basically a column vector
        # dot here handles the degenerate case of A = [#]
        sol_upper = (1 /A.T.dot(A)) * A.T.dot(bright)
        # might be a faster with more direct method here instead
        # But this will give us a decent start for the adaptive last step

        # Now we handle the nonlinear equation
        np.set_printoptions(precision=16)
        soln = minimize(lambda x: np.linalg.norm((self.f(x) - c2 - x * D[-1, -1])),
                        x0=sol_upper, method='Nelder-Mead',
                        options={"disp": False, "xatol": 1e-14, "fatol": 1e-14, "maxiter": 500})
        # c + ax = f(x) => x = (f(x) - a)/c
        # These tolerances are aggressive at 1e-12, but we're going for broke!
        # Golden might be more efficient if we could work out the bounds,

        self.solution = soln.x.copy()
        return self.solution.copy()  # copy it over

    def recursive_solve_subproblems(self):
        """ Instruct the subproblems to solve.

        The smallest subproblem (2 points) should be solved first,
        then we attempt to solve the 3-point problem (which solves the 2
        point sub-problem first, etc).

        For each possible subproblem,
        """
        for newproblem_index in range(1, self.degree):
            # new step size is current length * new node position
            # dict[number of nodes][which node you want] access pattern
            newh = self.xnodes[self.degree][newproblem_index] * self.h
            newproblem = ChebProblem(f=self.f, degree=newproblem_index, u0=self.u0,
                                     h=newh, diffmatDict=self.diffmatDict, xnodes=self.xnodes)
            self.subproblems.append(newproblem)
            # recursive solve subproblem finishes with a solve and returns the new value
            self.u[newproblem_index] = newproblem.recursive_solve_subproblems()

        self.can_solve = True
        return self.solve()


class SmcSolver(object):
    def __init__(self, fun: typing.Callable, u0: float, hstep: float, t0:float=0, stages: int = 2):
        """ Get usual suspects for an ODE solver routine. """
        self.stages = stages
        self.Dmats = {}
        self.xnodes = {}
        self.u0 = np.asarray(u0)  # convert it now
        self.hstep = hstep
        self.f = fun  # the function to solve u' = f(u)
        for idx in range(1, self.stages + 1):
            # build spectral diff mat + points on [0,1] - rescale on the fly
            D, x = chebdiff(n=idx, h=0.5, use_numba=True, flip=True)
            self.Dmats[idx] = D
            self.xnodes[idx] = x + 0.5
        self.main_problem = None
        self.t = t0
        self.times = None

    def _solve(self):
        """ Run the solver for 1 step. Does not update the problem For debug purposes primarily. """
        self.main_problem = ChebProblem(f=self.f, degree=self.stages,
                                        u0=self.u0, xnodes=self.xnodes,
                                        diffmatDict=self.Dmats, h=self.hstep)
        return self.main_problem.recursive_solve_subproblems()

    def solve(self, tmax=1):
        uvals = [self.u0]
        self.times = [self.t]
        while self.t < tmax:
            # setup a new problem
            self.main_problem = ChebProblem(f=self.f, degree=self.stages,
                                            u0=uvals[-1], xnodes=self.xnodes,
                                            diffmatDict=self.Dmats, h=self.hstep)
            self.t += self.hstep  # move forward time
            unew = self.main_problem.recursive_solve_subproblems()
            uvals.append(unew)
            self.times.append(self.t)

        return np.array(self.times), np.asarray(uvals)
