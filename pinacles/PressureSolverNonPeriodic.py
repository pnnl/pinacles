import numpy as np
import mpi4py_fft
from mpi4py import MPI
from pinacles.PressureSolverNonPeriodic_impl import (
    divergence,
    divergence_ghost,
    fill_pressure,
    apply_pressure,
    apply_pressure_open,
    apply_pressure_open_new,
)
from pinacles.TDMA_NonPeriodic import Thomas, PressureNonPeriodicTDMA
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

        # Compute vel_bounds vel_starts
        self._vel_starts = []
        self._vel_ends = []
        n_halo = self._Grid.n_halo
        ngrid_local = self._Grid.ngrid_local
        for i in range(3):
            if self._Grid.low_rank[i]:
                self._vel_starts.append(n_halo[i])
            else:
                self._vel_starts.append(0)

            if self._Grid.high_rank[i]:
                self._vel_ends.append(ngrid_local[i] - n_halo[i] -1 )
            else:
                self._vel_ends.append(ngrid_local[i] - 1)
        self._vel_ends = tuple(self._vel_ends)
        self._vel_starts = tuple(self._vel_starts)

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

        ibl = self._Grid.ibl
        ibu = self._Grid.ibu

        ibl_edge = self._Grid.ibl_edge
        ibu_edge = self._Grid.ibu_edge

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        # Set boundary conditions on u
        if low_rank[0]:
            u[ibl_edge[0], :, :] = 0.1

        if high_rank[0]:
            u[ibu_edge[0], :, :] = 0.1

        # Set boundary conditions on v
        if low_rank[1]:
            v[:, ibl_edge[1], :] = 0.0

        if high_rank[1]:
            v[:, ibu_edge[1], :] = 0.0

        return

    def _make_homogeneous(self, div, div_copy):
        # Set boundary conditions on u

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        ibl_edge = self._Grid.ibl_edge
        ibu_edge = self._Grid.ibu_edge

        ibl = self._Grid.ibl
        ibu = self._Grid.ibu

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank
        if low_rank[0]:
            div_copy[ibl[0], :, :] += div[ibl[0] - 1, :, :]
        if high_rank[0]:
            div_copy[ibu[0], :, :] -= div[ibu[0] + 1, :, :]

        # Set boundary conditions on v
        if low_rank[1]:
            div_copy[:, ibl[1], :] += div[:, ibl[1] - 1, :]

        if high_rank[1]:
            div_copy[:, ibu[1], :] -= div[:, ibu[1] + 1, :]

        return

    def _correct_mass_leak(self, u, v):

        nh = self._Grid.n_halo
        n = self._Grid.n

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        ibl_edge = self._Grid.ibl_edge
        ibu_edge = self._Grid.ibu_edge

        ibl = self._Grid.ibl
        ibu = self._Grid.ibu

        rho0 = self._Ref.rho0
        dx = self._Grid.dx

        # rho0[:] = 1.0

        leak = 0

        print(ibl_edge, nh)

        # Compute the amount of mass entering the system
        if low_rank[0]:
            leak -= (
                np.sum(
                    u[ibl_edge[0], nh[1] : -nh[1], nh[2] : -nh[2]]
                    #* rho0[np.newaxis, nh[2] : -nh[2]]
                )
                *dx[1]*dx[2]
            )

        if high_rank[0]:
            leak += (
                np.sum(
                    u[ibu_edge[0]+1, nh[1] : -nh[1], nh[2] : -nh[2]]
                    #* rho0[np.newaxis, nh[2] : -nh[2]]
                )
                *dx[1]*dx[2]
            )

        if low_rank[1]:
            leak -= (
                np.sum(
                    v[nh[0] : -nh[0], ibl_edge[1], nh[2] : -nh[2]]
                    #* rho0[np.newaxis, nh[2] : -nh[2]]
                )
                *dx[0]*dx[2]
            )

        if high_rank[1]:
            leak += (
                np.sum(
                    v[nh[0] : -nh[0], ibu_edge[1]+1, nh[2] : -nh[2]]
                    #* rho0[np.newaxis, nh[2] : -nh[2]]
                )
                *dx[0]*dx[2]
            )

        linear_mass = (2 * self._Grid.l[0] * self._Grid.l[2] + 2 * self._Grid.l[1] * self._Grid.l[2])

        u_fix_leak_local = np.array([leak / linear_mass])
        u_fix_leak_global = np.empty_like(u_fix_leak_local)
        MPI.COMM_WORLD.Allreduce(u_fix_leak_local, u_fix_leak_global, MPI.SUM),
        u_fix_leak = u_fix_leak_global[0]

        print('fix_leak', u_fix_leak, leak)


        # print(leak,  linear_mass, u_fix_leak)

        # import sys; sys.exit()

        if low_rank[0]:
            u[
                : ibl_edge[0] + 1, :, :
            ] += u_fix_leak #* rho0[np.newaxis, np.newaxis, :] # / rho0[np.newaxis, np.newaxis, nh[2]:-nh[2]]
        if high_rank[0]:
            u[
                ibu_edge[0] :, :, :
            ] -= u_fix_leak #* rho0[np.newaxis, np.newaxis, :] # / rho0[np.newaxis, np.newaxis, nh[2]:-nh[2]]

        if low_rank[1]:
            v[
                :, : ibl_edge[1] + 1, :
            ] += u_fix_leak #* rho0[np.newaxis, np.newaxis, :] # /  rho0[np.newaxis, np.newaxis, nh[2]:-nh[2]]
        if high_rank[1]:
            v[
                :, ibu_edge[1] :, :
            ] -= u_fix_leak #* rho0[np.newaxis, np.newaxis, :] # /  rho0[np.newaxis, np.newaxis, nh[2]:-nh[2]]

        # if low_rank[0]:
        #   np.add(u[:ibl_edge[0]+1, ibl[1] : ibu[1], nh[2]:-nh[2]],  u_fix_leak,out=u[:ibl_edge[0]+1, ibl[1] : ibu[1], nh[2]:-nh[2]])

        # if high_rank[0]:
        #    np.add(u[ibu_edge[0]:, ibl[1] : ibu[1], nh[2]:-nh[2]], -u_fix_leak, out = u[ibu_edge[0]:, ibl[1] : ibu[1], nh[2]:-nh[2]])

        # if low_rank[1]:
        #    np.add(v[ibl[0]:ibu[0], :ibl_edge[1]+1, nh[2]:-nh[2]], u_fix_leak, out=  v[ibl[0]:ibu[0], :ibl_edge[1]+1, nh[2]:-nh[2]])

        # if high_rank[1]:
        #   np.add(v[ibl[0]:ibu[0], ibu_edge[1]:, nh[2]:-nh[2]],-u_fix_leak, v[ibl[0]:ibu[0], ibu_edge[1]:, nh[2]:-nh[2]])

        return

    def _make_non_homogeneous(self, rho0, div, p, dynp):

        low_rank = self._Grid.low_rank
        high_rank = self._Grid.high_rank

        ibl = self._Grid.ibl
        ibu = self._Grid.ibu
        dx = self._Grid.dx
        nh = self._Grid.n_halo

        if low_rank[0]:
            dynp[ibl[0] - 1, :, :] = (
                dynp[ibl[0], :, :]
                - dx[0] * dx[0] * div[ibl[0] - 1, :, :] / rho0[np.newaxis, :]
            )

        if high_rank[0]:
            dynp[ibu[0] + 1, :, :] = (
                dynp[ibu[0], :, :]
                + dx[0] * dx[0] * div[ibu[0] + 1, :, :] / rho0[np.newaxis, :]
            )

        if low_rank[1]:
            dynp[:, ibl[1] - 1, :] = (
                dynp[:, ibl[1], :]
                - dx[1] * dx[1] * div[:, ibl[1] - 1, :] / rho0[np.newaxis, :]
            )

        if high_rank[1]:
            dynp[:, ibu[1] + 1, :] = (
                dynp[:, ibu[1], :]
                + dx[1] * dx[1] * div[:, ibu[1] + 1, :] / rho0[np.newaxis, :]
            )

        return

    def update(self):

        ibl_edge = self._Grid.ibl_edge
        ibu_edge = self._Grid.ibu_edge

        ibl = self._Grid.ibl
        ibu = self._Grid.ibu
        n_halo = self._Grid.n_halo

        # First get views in to the velocity components
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        # self._correct_mass_leak(u, v)
        # print(np.max(u0 - u))
        self._correct_mass_leak(u, v)
       #self._correct_mass_leak(u, v)


        w[:, :, n_halo[2] - 1] = 0.0
        w[:, :, ibu_edge[2]] = 0.0

        # if MPI.COMM_WORLD.Get_rank() == 0:
        #    u[32:48, 32:48, 10:20] = 1.0

        #self._VelocityState.remove_mean("w")

        dynp = self._DiagnosticState.get_field("dynamic pressure")

        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx

        # Set boundary conditions
        # self._radiation_davies(u, v, w)

        div = np.zeros(self._Grid.ngrid_local, dtype=np.double)

        # First compute divergence of wind field
        divergence_ghost(n_halo, dxs, rho0, rho0_edge, u, v, w, div)

        #div = div - np.mean(np.mean(div[n_halo[0] : -n_halo[0],n_halo[1] : -n_halo[1],:],axis=0),axis=0)[np.newaxis, np.newaxis, :]

        # Make the BCS Homogeneous
        if self._Grid.low_rank[0]:
            div[ibl[0] - 1, :, :] = 0.0
        if self._Grid.high_rank[0]:
            div[ibu[0] + 1, :, :] = 0.0

        if self._Grid.low_rank[1]:
            div[:, ibl[1] - 1, :] = 0.0
        if self._Grid.high_rank[1]:
            div[:, ibu[1] + 1, :] = 0.0

        div_copy = np.copy(div)
        self._make_homogeneous(div, div_copy)

        # import pylab as plt

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.contourf(div_copy[3:-3, 3:-3, 16].T)
        # plt.colorbar()
        # plt.title('div')
        # plt.show()
        import time

        # time.sleep(2.0)
        t0 = time.time()

        div_hat = self.dct.forward(
            np.copy(
                div_copy[
                    n_halo[0] : -n_halo[0],
                    n_halo[1] : -n_halo[1],
                    n_halo[2] : -n_halo[2],
                ]
            )
        )
        self._TMDA_solve.solve(div_hat)
        p = self.dct.backward(div_hat)

        fill_pressure(n_halo, p, dynp)
        self._DiagnosticState.boundary_exchange("dynamic pressure")
        # self._DiagnosticState._gradient_zero_bc("dynamic pressure")

        # import pylab as plt
        # plt.figure(1)
        # plt.contourf(u[:,:,10].T)

        self._make_non_homogeneous(self._Ref.rho0, div, p, dynp)
        # apply_pressure_open(n_halo, dxs, dynp, u, v, w)
        ##apply_pressure(dxs, dynp, u, v, w)
        apply_pressure_open_new(
            n_halo, self._vel_starts, self._vel_ends, dxs, dynp, u, v, w
        )

        self._VelocityState.remove_mean('w')

        #self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()


        #w[:,:,n_halo[2]-1]=0.0
        #w[:,:,ibu_edge[2]] = 0.0
        # divergence_ghost(n_halo, dxs, rho0, rho0_edge, u, v, w, div)
        # print('Divergence', np.amax(np.abs(div[3:-3, 3:-3, 16])))

        return
