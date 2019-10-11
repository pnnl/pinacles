import numpy as np 
import numba 

class TDMA_solver: 
    def __init__(self, n): 

        self._n = n 
        self._scratch = np.zeros(self._n, dtype=np.double)

        return

    @property
    def n(self): 
        return self._n 

    def solve(self): 

        return