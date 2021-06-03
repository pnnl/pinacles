import numpy as np
import mpi4py_fft 
from mpi4py import MPI 
from pinacles.PressureSolver_impl import divergence, fill_pressure, apply_pressure
from pinacles.TDMA import Thomas, PressureTDMA, PressureNonPeriodicTDMA
import functools
from scipy.fft import dctn, idctn

class dct_mpi4py:
    
    def __init__(self,n, subcomms):
        self._subcomms = subcomms
        self._n = n

        self.p2 = mpi4py_fft.pencil.Pencil(self._subcomms, self._n, axis=2)
        self.p1 = self.p2.pencil(1)
        self.p0 = self.p1.pencil(0)

        self.transfer21 = self.p2.transfer(self.p1, np.double)
        self.transfer10 = self.p1.transfer(self.p0, np.double)

        self.a2 = np.zeros(self.p2.subshape, dtype=np.double)
        self.a1 = np.zeros(self.p1.subshape, dtype=np.double)
        self.a0 = np.zeros(self.p0.subshape, dtype=np.double)


        return


    def forward(self, data):

        np.copyto(self.a2, data) 
        self.transfer21.forward(self.a2, self.a1)
        self.a1[:,:,:] = dctn(self.a1, axes=1, type=2)
    
        self.transfer10.forward(self.a1, self.a0)
        self.a0[:,:,:] = dctn(self.a0, axes=0, type=2)

        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)



        return self.a2
    
    
    def backward(self, data):

        np.copyto(self.a2, data) 
        self.transfer21.forward(self.a2, self.a1)
        self.a1[:,:,:] = idctn(self.a1, axes=1, type=2)
    
        self.transfer10.forward(self.a1, self.a0)
        self.a0[:,:,:] = idctn(self.a0, axes=0, type=2)

        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)

        return self.a2
    

class PressureSolverNonPeriodic:
    def __init__(self, Grid, Ref, VelocityState, DiagnosticState):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        self._wavenumber_substarts = None
        self._wavenumber_n = None

        #Add dynamic pressure as a diagnsotic state
        self._DiagnosticState.add_variable('dynamic pressure', long_name='Dynamic Pressure', units='Pa', latex_name='p^*')

        self.dct = dct_mpi4py(self._Grid.n, self._Grid.subcomms)

        self.fft_local_starts()

        return

    def fft_local_starts(self):
        div = mpi4py_fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        #div_hat =  mpi4py_fft.newDistArray(self._fft, forward_output=True)

        self._wavenumber_n = div.shape #div_hat2.shape
        self._wavenumber_substarts = div.substart

        return

    def initialize(self):
        self._TMDA_solve = PressureNonPeriodicTDMA(self._Grid, self._Ref, self._wavenumber_substarts, self._wavenumber_n)

        return

    def update(self):

        #First get views in to the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        v.fill(0.0)
        u.fill(0.0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            u[32:48, 32:48, 10:20] = 1.0

        dynp = self._DiagnosticState.get_field('dynamic pressure')

        rho0  = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo
      
        div = np.empty(self._Grid.local_shape, dtype=np.double) #mpi4py_fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.double)
  
        #First compute divergence of wind field
        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)
        import time; t0 = time.time()


        div_hat = self.dct.forward(div)
        self._TMDA_solve.solve(div_hat)
        p = self.dct.backward(div_hat)

        fill_pressure(n_halo, p, dynp)
        self._DiagnosticState.boundary_exchange('dynamic pressure')
        self._DiagnosticState._gradient_zero_bc('dynamic pressure')

        apply_pressure(dxs, dynp, u, v, w)

        self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()

        t1 = time.time()
        print(t1 -t0)

        divergence(n_halo,dxs, rho0, rho0_edge, u, v, w, div)
        import pylab as plt
        plt.figure()
        plt.contourf(div[4:-4,4:-4,16].T)
        plt.title(str(MPI.COMM_WORLD.Get_rank()))
        plt.colorbar()
        plt.show()
    

        
        
        import sys; sys.exit()
        return