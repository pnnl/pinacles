def TimeSteppingFactory(namelist): 
    
    if namelist['timestepping'] == 'RungeKutta2ndSSP': 
        return RungeKutta2ndSSP(namelist)

    return 


class RungeKuttaBase: 

    def __init__(self, namelist): 

        self.T = 0 
        self.n_rk_step = 0
        self.cfl_target = namelist['timestepping']['cfl_target']

        return

    def update(self):

        return 


def RungeKutta2ndSSP(RungeKuttaBase): 
    def __init__(self, namelist): 
        RungeKuttaBase.__init__(self, namelist)
        return 