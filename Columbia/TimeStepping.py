import numpy as np 
from Columbia import TimeStepping_impl as TS_impl 
class RungeKuttaBase: 

    def __init__(self, namelist, Grid, PrognosticState): 
        self._Grid = Grid 
        self._PrognosticState = PrognosticState
        self._t = 0 
        self.n_rk_step = 0
        self._rk_step = 0 
        self.cfl_target = 0.7
        self._dt = 1.0

        return

    def initialize(self): 

        return 

    def update(self):

        return 

class RungeKutta2ndSSP(RungeKuttaBase): 
    def __init__(self, namelist, Grid, PrognosticState):
        RungeKuttaBase.__init__(self, namelist, Grid, PrognosticState)
        self.Tn = None 
        self.n_rk_step = 2 
        self._rk_step = 0.0 
        return 

    def initialize(self): 
        self._Tn = np.empty_like(self._PrognosticState.state_array, order='F')
        self._Tn[:,:,:,:]=0.0
        return 

    def update(self): 
        present_state = self._PrognosticState.state_array
        present_tend = self._PrognosticState.tend_array

        if self._rk_step == 0: 
            self._Tn = np.copy(present_state, order='F')
            TS_impl.rk2ssp_s0(self._Tn, present_tend, self._dt )
            self._rk_step = 1 

        else:
            TS_impl.rk2ssp_s1(self._Tn, present_state, 
                present_tend, self._dt)
            self._rk_step = 0 
            self._t += self._dt

        return 

def factory(namelist, Grid, PrognosticState): 
    
    return RungeKutta2ndSSP(namelist, Grid, PrognosticState)
