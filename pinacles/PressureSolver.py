import numpy as np
from pinacles.PressureSolver_impl import divergence, fill_pressure, apply_pressure
from pinacles.PressureSolverNonPeriodic import PressureSolverNonPeriodic
from pinacles.TDMA import Thomas, PressureTDMA
import mpi4py_fft as fft
from mpi4py import MPI
from pinacles import ParallelFFTs


class PressureSolver:
    def __init__(self, Timers, Grid, Ref, VelocityState, DiagnosticState):

        self._Timers = Timers
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

        # Setup the Fourier Transform
        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        div = div.redistribute(0)

        self.FFT = ParallelFFTs.fft_mpi4py(self._Grid.n, self._Grid.subcomms)

        self.fft_local_starts()

        self._div_work = np.empty(self.FFT.p2.subshape, dtype=np.complex)

        self._Timers.add_timer("PressureSolver_update")
        return

    def initialize(self):
        self._TMDA_solve = PressureTDMA(
            self._Grid, self._Ref, self._wavenumber_substarts, self._wavenumber_n
        )
        return

    def fft_local_starts(self):
        div = fft.DistArray(self._Grid.n, self._Grid.subcomms, dtype=np.complex)
        # div_hat =  fft.newDistArray(self._fft, forward_output=True)
        # div_hat2 = div_hat.redistribute(2)

        self._wavenumber_n = div.shape
        self._wavenumber_substarts = div.substart

        return

    def update(self):

        self._Timers.start_timer("PressureSolver_update")

        self._VelocityState.remove_mean("w")

        # First get views in to the velocity components
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        dynp = self._DiagnosticState.get_field("dynamic pressure")

        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        dxs = self._Grid.dx
        n_halo = self._Grid.n_halo

        # First compute divergence of wind field
        divergence(n_halo, dxs, rho0, rho0_edge, u, v, w, self._div_work)

        div_hat_2 = self.FFT.forward(self._div_work)

        # The TDM solver goes here
        # divh2_real = div_hat_2.real
        # divh2_img = div_hat_2.imag

        # Place the pressure solve here
        # self._TMDA_solve.solve(divh2_real)
        self._TMDA_solve.solve(div_hat_2)

        # div_hat_2 = divh2_real + divh2_img * 1j
        if self._wavenumber_substarts[0] == 0 and self._wavenumber_substarts[1] == 0:
            div_hat_2[0, 0, :] = 0.0 + 0j

        fill_pressure(n_halo, self.FFT.backward(div_hat_2), dynp)

        # TODO add single vairable exchange
        self._DiagnosticState.boundary_exchange("dynamic pressure")
        self._DiagnosticState._gradient_zero_bc("dynamic pressure")

        apply_pressure(dxs, dynp, u, v, w)

        self._VelocityState.boundary_exchange()
        self._VelocityState.update_all_bcs()

        self._Timers.end_timer("PressureSolver_update")

        return


def factory(namelist, Timer, Grid, Ref, VelocityState, DiagnosticState):
    return PressureSolverNonPeriodic(Grid, Ref, VelocityState, DiagnosticState)
