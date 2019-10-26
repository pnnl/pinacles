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


        self._VelocityState.remove_mean('w')

        #u[25:-25,25:-25,25:-25] = 2.0
        #if MPI.COMM_WORLD.rank == 0: 
        #    u[5:10,5:10,5:10] = 1.0


        dynp = self._DiagnosticState.get_field('dynamic pressure')

        rho0  = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo
        local_start = self._Grid.local_start

        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        #First compute divergence of wind field
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)
        #print('Divergence 1', np.amax(np.abs(div)))
        #import pylab as plt
        #plt.figure(11)
        #plt.contourf(div[:,:,50],200)
        #plt.colorbar()

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
        if local_start[0] == 0 and local_start[1] == 0: 
            div_hat_2[0,0,:] = 0.0 + 0j #This only works for serial solver 

        div_hat = div_hat_2.redistribute(1)

        self._fft.backward(div_hat, div_0)

        div = div_0.redistribute(2)

        fill_pressure(n_halo, div, dynp)

        #TODO add single vairable exchange
        self._DiagnosticState.boundary_exchange()
        self._DiagnosticState._gradient_zero_bc('dynamic pressure')

        apply_pressure(dxs, dynp, u, v, w)
        #print('W mean: ', np.mean(np.mean(w[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1],:], axis=0),axis=0))
        self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)

        print('Divergence 2', np.amax(np.abs(div)))


        #plt.figure(13)
        #plt.contourf(v[:,:,50],20)
        #plt.colorbar()
        #plt.figure(14)
        #plt.contourf(w[:,:,50],20)
        #plt.colorbar()
        #plt.figure(15)
        #plt.contourf(div[:,50,:],20)
        #plt.colorbar()
        #plt.show()



        #print(np.amin(dynp), np.amax(dynp))
        #import sys; sys.exit()



        return



def factory(namelist, Grid, Ref, VelocityState, DiagnosticState):
    return PressureSolver(Grid, Ref, VelocityState, DiagnosticState)