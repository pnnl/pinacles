import numpy as np
from Columbia.PressureSolver_impl import divergence
from Columbia.TDMA import Thomas
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
        div =  fft.DistArray(self._Grid.n , self._Grid.subcomms, dtype=np.complex)
        div = div.redistribute(0)
        self._fft =  fft.PFFT(self._Grid.subcomms, darray=div, axes=(1,0), transforms={})

        self._compute_modified_wavenumbers()
        self._set_center_diagional()
        self._set_upperlower_diagonals()

        return

    def _set_center_diagional(self):

        self._b = np.ones(self._Grid.n[2], dtype=np.double)


        return

    def _compute_modified_wavenumbers(self):
        nl = self._Grid.nl
        local_start = self._Grid.local_start
        n_h = self._Grid.n_halo
        dx = self._Grid.dx
        n = self._Grid.n 
        xl = self._Grid.x_local[n_h[0]:-n_h[0]]
        yl = self._Grid.y_local[n_h[1]:-n_h[1]]

        self._kx2 = np.zeros(nl[0], dtype=np.double)
        self._ky2 = np.zeros(nl[1], dtype=np.double)
        #TODO the code below feels a bit like boilerplate
        for ii in range(nl[0]): 
            i = local_start[0] + ii 
            if i <= n[0]/2: 
                xi = np.double(i)
            else: 
                xi = np.double(i - n[0])
            self._kx2[ii] = (2.0 * np.cos((2.0 * np.pi/n[0]) * xi)-2.0)/dx[0]/dx[0]        

        for jj in range(nl[1]): 
            j = local_start[1] + jj
            if j <= n[1]/2: 
                yi = np.double(jj)
            else: 
                yi = np.double(j - n[1])
            self._ky2[jj] = (2.0 * np.cos((2.0 * np.pi/n[1]) * yi)-2.0)/dx[1]/dx[1]     

        return

    def _set_upperlower_diagonals(self):
        self._a = np.zeros(self._Grid.n[2], dtype=np.double)
        self._c = np.zeros(self._Grid.n[2], dtype=np.double)
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