def TimeSteppingFactory(namelist): 
    
    if namelist['timestepping'] == 'RungeKutta2ndSSP': 
        return RungeKutta2ndSSP(namelist)

    return 


class RungeKuttaBase: 

    def __init__(self, namelist, Grid, PrognosticState, DiagnosticState): 
        self._Grid = Grid 
        self._PrognosticState = PrognosticState
        self._DiagnosticState = DiagnosticState
        self._t = 0 
        self.n_rk_step = 0
        self.cfl_target = namelist['timestepping']['cfl_target']

        return

    def update(self):

        return 


def RungeKutta2ndSSP(RungeKuttaBase): 
    def __init__(self, namelist, Grid, PrognosticStae, DiagnosticState): 
        RungeKuttaBase.__init__(self, namelist, Grid, PrognosticState, DiagnosticState)
        return 