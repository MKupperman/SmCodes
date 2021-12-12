import numpy as np


class TestFunction(object):
    """ A utility harness for counting function evaluations"""

    def __init__(self, function):
        self.fun = function
        self.numcalls = 0

    def __call__(self, t, x):
        self.numcalls += 1
        return self.fun(t, x)

    def num_calls(self):
        return self.numcalls
