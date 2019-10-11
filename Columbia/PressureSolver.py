import numpy as np
from Columbia.PressureSolver_impl import divergence
import mpi4py_fft as fft
from mpi4py import MPI 

class PressureSolver:
    def __init__(self, Grid, Ref, VelocityState):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState

        #Set up the diagonals for the solve
        self._a = None
        self._b = None
        self._c = None

        self._set_center_diagional()
        self._set_upperlower_diagonals()

        #Setup the Fourier Transform
        self._div =  fft.DistArray(self._Grid.n , self._Grid.subcomms)
        self._div = self._div.redistribute(0)
        self._fft =  fft.PFFT(self._Grid.subcomms, darray=self._div, axes=(1,0))
        self._div_hat = fft.newDistArray(self._fft, forward_output=True)


        self._fft.forward(self._div, self._div_hat)
        self._div_hat = self._div_hat.redistribute(2)
        print(self._div_hat.shape)
        #z3 = fft.newDistArray(self._fft, forward_output=True)
        
        
        #import time
        #t1 = time.time() 
        #self._fft.forward(self._div, z3)
        #t2 = time.time() 
        #print(t2-t1)

        print(self._div.shape)

        return

    def _set_center_diagional(self): 

        self._b = np.zeros(self._Grid.n[2], dtype=np.double)

        return 


    def _set_upperlower_diagonals(self): 
        self._a = np.zeros(self._Grid.n[2]-1, dtype=np.double)
        self._b = np.zeros(self._Grid.n[2]-1, dtype=np.double)
        return 

    def update(self): 
    
        #First get views in to the velocity components
        #u = self._VelocityState.get_field('u')
        #v = self._VelocityState.get_field('v')
        #w = self._VelocityState.get_field('w')

        #div = np.empty_like(u)

        #rho0  = self._Ref.rho0 
        #rho0_edge = self._Ref.rho0_edge

        #dxs = self._Grid.dx 
        #n_halo = self._Grid.n_halo

        #First compute divergence of wind field
        #divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)

        div = fft.DistArray(self._Grid.n, self._Grid.subcomms)
        div[:,:,:] = 22.0
        div_cpy = div.copy()
        div_0 = div.redistribute(0)

        div_hat =  fft.newDistArray(self._fft, forward_output=True)
        self._fft.forward(div_0, div_hat)


        div_hat_2 = div_hat.redistribute(2)


        print(div_hat_2[0,0,0])

        div_hat = div_hat_2.redistribute(1)


        self._fft.backward(div_hat, div_0)

        div = div_0.redistribute(2)


        return 



def factory(namelist, Grid, Ref, VelocityState): 
    return PressureSolver(Grid, Ref, VelocityState)