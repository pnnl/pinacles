import numpy as np
from Columbia.PressureSolver_impl import divergence
from Columbia.TDMA import Thomas, PressureTDMA
import mpi4py_fft as fft
from mpi4py import MPI

class PressureSolver:
    def __init__(self, Grid, Ref, VelocityState):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState


        #Setup the Fourier Transform
        div =  fft.DistArray(self._Grid.n , self._Grid.subcomms, dtype=np.complex)
        div = div.redistribute(0)
        self._fft =  fft.PFFT(self._Grid.subcomms, darray=div, axes=(1,0), transforms={})

        self._TMDA_solve = PressureTDMA(self._Grid)

        return


    def update(self):

        #First get views in to the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        rho0  = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo

        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        #First compute divergence of wind field
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)

        div_0 = div.redistribute(0)

        div_hat =  fft.newDistArray(self._fft, forward_output=True)
        self._fft.forward(div_0, div_hat)


        div_hat_2 = div_hat.redistribute(2)

        #The TDM solver goes here
        divh2_real = div_hat_2.real
        divh2_img = div_hat_2.imag

        Thomas(divh2_real, self._a, self._b, self._c)
        Thomas(divh2_img, self._a, self._b, self._c)


        div_hat_2 = divh2_real + divh2_img * 0j

        div_hat = div_hat_2.redistribute(1)


        self._fft.backward(div_hat, div_0)

        div = div_0.redistribute(2)


        return



def factory(namelist, Grid, Ref, VelocityState):
    return PressureSolver(Grid, Ref, VelocityState)