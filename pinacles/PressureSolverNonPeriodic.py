import numpy as np
import mpi4py_fft
from mpi4py import MPI
from pinacles.PressureSolver_impl import divergence, fill_pressure, apply_pressure
from pinacles.TDMA import Thomas, PressureTDMA, PressureNonPeriodicTDMA
from pinacles.ParallelFFTs import dct_mpi4py

import functools
from scipy.fft import dctn, idctn


class PressureSolverNonPeriodic:
    def __init__(self, Grid, Ref, VelocityState, DiagnosticState):
        """ Contructor for non-periodic pressure solver

        Args:
            Grid (class): PINACLES Grid class
            Ref (class): PINACLES  Reference class
            VelocityState (class): PINACLES Container class containing velocity state
            DiagnosticState (class): PINACLES Container class containing diagnostic model state
        """

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        self._wavenumber_substarts = None
        self._wavenumber_n = None

        # Add dynamic pressure as a diagnsotic state
        self._DiagnosticState.add_variable(
            "dynamic pressure",
            long_name="Dynamic Pressure",
            units="Pa",
            latex_name="p^*",
        )

        self.dct = dct_mpi4py(self._Grid.n, self._Grid.subcomms)

        self._fft_local_starts()

        return

    def _fft_local_starts(self):
        """ compute the number of global number of modified wave numbers and the starting point
        in the global array
        """
        div = mpi4py_fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)

        # Dimensions (global) of modified wave number array
        self._wavenumber_n = div.shape 

        # Starting points for this rank in the globa modified wave number array
        self._wavenumber_substarts = div.substart

        return

    def initialize(self):
        self._TMDA_solve = PressureNonPeriodicTDMA(
            self._Grid, self._Ref, self._wavenumber_substarts, self._wavenumber_n
        )

        return

    def _radiation_davies(self, u, v, w):
        """[summary]

        Args:
            u (array): u component
            v (array): v component
            w (array): w component
        """

        nh = self._Grid.n_halo
        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        # Set boundary conditions on u
        if low_rank[0]:
            u[nh[0] - 1, :, :] = 0.1

        if high_rank[0]:
            u[-nh[0] - 1, :, :] = 0.0

        # Set boundary conditions on v
        if low_rank[1]:
            v[:, nh[1] - 1, :] = 0.0

        if high_rank[1]:
            v[:, -nh[1] - 1, :] = 0.0

        return

    def _make_homogeneous(self, div, div_copy):
        # Set boundary conditions on u

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank
        nh = self._Grid.n_halo

        if low_rank[0]:
            div_copy[nh[0], :, :] += div[nh[0]-1, :, :]

        if high_rank[0]:
            div_copy[-1, :, :] = 0.0

        # Set boundary conditions on v
        if low_rank[1]:
            div_copy[:, 1, :] = 0.0

        if high_rank[1]:
            div_copy[:, -1, :] = 0.0

        return


    def _make_non_homogeneous(self, p):

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        if low_rank[0]:
            div[0, :, :] = 0.0

        if high_rank[0]:
            div[-1, :, :] = 0.0

        # Set boundary conditions on v
        if low_rank[1]:
            div[:, 1, :] = 0.0

        if high_rank[1]:
            div[:, -1, :] = 0.0



        return


    def update(self):

        # First get views in to the velocity components
        u = self._VelocityState.get_tend("u")
        v = self._VelocityState.get_tend("v")
        w = self._VelocityState.get_tend("w")

        dynp = self._DiagnosticState.get_field("dynamic pressure")

        v.fill(0.0)
        u.fill(0.0)
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            u[32:48, 32:48, 10:20] = 1.0

        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo

        # Set boundary conditions
        self._radiation_davies(u, v, w)

        div = np.empty(self._Grid.local_shape, dtype=np.double)



        # First compute divergence of wind field
        divergence(n_halo, dxs, rho0, rho0_edge, u, v, w, div)
        div_copy = np.copy(div)
        self._make_homogeneous(div, div_copy)
        import time

        t0 = time.time()

        div_hat = self.dct.forward(div_copy)
        self._TMDA_solve.solve(div_hat)
        p = self.dct.backward(div_hat)

        fill_pressure(n_halo, p, dynp)
        self._DiagnosticState.boundary_exchange("dynamic pressure")
        self._DiagnosticState._gradient_zero_bc("dynamic pressure")

        apply_pressure(dxs, dynp, u, v, w)

        self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()

        t1 = time.time()
        print(t1 - t0)

        divergence(n_halo, dxs, rho0, rho0_edge, u, v, w, div)
        import pylab as plt

        plt.figure()
        plt.contourf(div[3:-3, 3:-3, 16].T)
        plt.title(str(MPI.COMM_WORLD.Get_rank()))
        plt.colorbar()
        plt.show()

        import sys

        sys.exit()
        return
