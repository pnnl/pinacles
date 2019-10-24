import numpy as np
from Columbia.PressureSolver_impl import divergence, fill_pressure, apply_pressure
from Columbia.TDMA import Thomas, PressureTDMA
import mpi4py_fft as fft
from mpi4py import MPI

class PressureSolver:
    def __init__(self, Grid, Ref, VelocityState, DiagnosticState):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        #Add dynamic pressure as a diagnsotic state
        self._DiagnosticState.add_variable('dynamic pressure')

        #Setup the Fourier Transform
        div =  fft.DistArray(self._Grid.n , self._Grid.subcomms, dtype=np.complex)
        div = div.redistribute(0)
        self._fft =  fft.PFFT(self._Grid.subcomms, darray=div, axes=(1,0), transforms={})

        self._TMDA_solve = PressureTDMA(self._Grid, self._Ref)

        return

    def update(self):

        #First get views in to the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        #u[25:-25,25:-25,25:-25] = 2.0
        #if MPI.COMM_WORLD.rank == 0: 
        #    u[5:10,5:10,5:10] = 1.0


        dynp = self._DiagnosticState.get_field('dynamic pressure')

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

        #Place the pressure solve here 
        self._TMDA_solve.solve(divh2_real)
        self._TMDA_solve.solve(divh2_img)

        div_hat_2 = divh2_real + divh2_img * 1j

        div_hat = div_hat_2.redistribute(1)

        self._fft.backward(div_hat, div_0)

        div = div_0.redistribute(2)

        fill_pressure(n_halo, div, dynp)

        #TODO add single vairable exchange
        self._DiagnosticState.boundary_exchange()

        apply_pressure(dxs, dynp, u, v, w)
        self._VelocityState.boundary_exchange()
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)
        print('Divergence', np.amax(div))

        import pylab as plt
        plt.contourf(u[:,:,25],200)
        plt.colorbar()
        plt.show()



        #print(np.amin(dynp), np.amax(dynp))
        #import sys; sys.exit()



        return



def factory(namelist, Grid, Ref, VelocityState, DiagnosticState):
    return PressureSolver(Grid, Ref, VelocityState, DiagnosticState)