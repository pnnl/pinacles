import numpy as numpy
from Columbia.PressureSolver_impl import divergence

class PressureSolver: 
    def __init__(self, Grid, Ref, VelocityState): 
        
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState

        return 

    def update(self): 
    
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        return 



def factory(namelist, Grid, Ref, VelocityState): 
    return PressureSolver(Grid, Ref, VelocityState)