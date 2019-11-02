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

        self._wavenumber_substarts = None
        self._wavenumber_n = None

        #Add dynamic pressure as a diagnsotic state
        self._DiagnosticState.add_variable('dynamic pressure')

        #Setup the Fourier Transform
        div =  fft.DistArray(self._Grid.n , self._Grid.subcomms, dtype=np.complex)
        div = div.redistribute(0)
        self._fft =  fft.PFFT(self._Grid.subcomms, darray=div, axes=(1,0), transforms={})


        self.fft_local_starts()
        self._TMDA_solve = PressureTDMA(self._Grid, self._Ref, self._wavenumber_substarts, self._wavenumber_n)


        return


    def fft_local_starts(self):
        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        div_hat =  fft.newDistArray(self._fft, forward_output=True)
        div_hat2 = div_hat.redistribute(2)

        self._wavenumber_n = div_hat2.shape
        print(self._wavenumber_n)
        self._wavenumber_substarts = div_hat2.substart

        #import sys; sys.exit()

        return

    def update(self):

        self._VelocityState.remove_mean('w')

        #First get views in to the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        dynp = self._DiagnosticState.get_field('dynamic pressure')

        rho0  = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo
        local_start = self._Grid.local_start

        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        #First compute divergence of wind field
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)


        #print('u', u.shape)

        div_0 = div.redistribute(0)

        div_hat =  fft.newDistArray(self._fft, forward_output=True)
        self._fft.forward(div_0, div_hat)

        div_hat_2 = div_hat.redistribute(2)
        #print(div_hat_2[1,1,1], self._wavenumber_substarts)
        #import sys; sys.exit()
        #print('div_hat_2 ', div_hat_2.shape)
        #import sys; sys.exit() 


        #The TDM solver goes here
        divh2_real = div_hat_2.real
        divh2_img = div_hat_2.imag

        #Place the pressure solve here
        self._TMDA_solve.solve(divh2_real)
        self._TMDA_solve.solve(divh2_img)

        div_hat_2 = divh2_real + divh2_img * 1j
        if self._wavenumber_substarts[0] == 0 and self._wavenumber_substarts[1] == 0:
            div_hat_2[0,0,:] = 0.0 + 0j #This only works for serial solver

        div_hat = div_hat_2.redistribute(1)

        self._fft.backward(div_hat, div_0)

        div = div_0.redistribute(2)

        fill_pressure(n_halo, div, dynp)

 

        #TODO add single vairable exchange
        self._DiagnosticState.boundary_exchange()
        self._DiagnosticState._gradient_zero_bc('dynamic pressure')

        apply_pressure(dxs, dynp, u, v, w)
        #usl = self._DiagnosticState.get_field_slice_z('dynamic pressure', 5)
        usl = self._VelocityState.get_field_slice_z('u', 5)
        #if MPI.COMM_WORLD.Get_rank() == 0: 
        #    print(np.mean(usl), np.max(usl), np.min(usl), usl[7,10])
        #    import pylab as plt
        #    plt.contourf(usl,46)
        #    plt.show()
        #import sys; sys.exit()


        self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()

        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)

        #print('Divergence 2', np.amax(np.abs(div)))
 
        return



def factory(namelist, Grid, Ref, VelocityState, DiagnosticState):
    return PressureSolver(Grid, Ref, VelocityState, DiagnosticState)