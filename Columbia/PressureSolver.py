import numpy as np
from Columbia.PressureSolver_impl import divergence

class PressureSolver: 
    def __init__(self, Grid, Ref, VelocityState): 
        
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState

        return 

    def update(self): 
    

        #First get views in to the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        div = np.empty_like(u)

        rho0  = self._Ref.rho0 
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx 

        #First compute diverge of wind field
        divergence(dxs, rho0, rho0_edge, u, v, w, div)

        return 



def factory(namelist, Grid, Ref, VelocityState): 
    return PressureSolver(Grid, Ref, VelocityState)