import numpy as np
import numba 


class BoundaryBrinkman:

    @staticmethod 
    @numba.njit()
    def add_boundary(terrain_fraction, u, v, w, ut, vt, wt):

        shape = terrain_fraction.shape
        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                for k in range(shape[2]-1):

                    if terrain_fraction[i,j,k] > 0.0:
                        for surr in [0, 1]:

                            ut[i+surr,j,k] = 0.0 #0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i+1,j,k])  * u[i,j,k]
                            vt[i,j+surr,k] = 0.0 #0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i,j+1,k])  * v[i,j,k]
                            wt[i,j,k+surr] = 0.0 #0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i,j,k+1])  * w[i,j,k]
                            
                            u[i+surr,j,k] = 0.0
                            v[i,j+surr,k] = 0.0
                            w[i,j,k+surr] = 0.0
        return


    def __init__(self, Grid, DiagnosticState, VelocityState, TimeSteppingController):

        self._Grid = Grid
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._DiagnosticState.add_variable('terrain_fraction')

        return

    def update(self):


        terrain_fraction = self._DiagnosticState.get_field('terrain_fraction')
        terrain_fraction[14:22,60:68,:14] =  1.0 #/self._TimeSteppingController.dt
        terrain_fraction[14+12:22+12,60+12:68+12,:14] =  1.0 
        terrain_fraction[14+12:22+12,60:68,:20] =  1.0 
        #terrain_fraction[15:21,61:67,:13] =  0.7
        #terrain_fraction[16:20,62:66,:13] =  10.4

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        wt = self._VelocityState.get_tend('w')

        #ut[14:22,60:68,:14] -= 1.0/ * u[14:22,60:68,:14] 
        #vt[14:22,60:68,:14] -= 1.0/ * v[14:22,60:68,:14]
        #wt[14:22,60:68,:14] -= 1.0/ * w[14:22,60:68,:14] 

        self.add_boundary(terrain_fraction, u, v, w, ut, vt, wt)

        #Add boundary 


        return