import numpy as np
import time
from pinacles import MomentumAdvection_impl
from pinacles import UtilitiesParallel


class MomentumAdvectionBase:
    def __init__(self, namelist, Timers, Grid, Ref, ScalarState, VelocityState):
        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState

        self._fu = None  # Function for computing u-fluxes
        self._fv = None  # Function for computing v-fluxes
        self._fz = None  # Function for computing z-fluxes
        self.flux_function_factory(namelist)

        self._fluxx = np.zeros(self._Grid.ngrid_local, dtype=np.double)
        self._fluxy = np.zeros_like(self._fluxx)
        self._fluxz = np.zeros_like(self._fluxx)

        return

    def update(self):

        return

    def flux_function_factory(self, namelist):

        # Get the number of halo point
        n_halo = self._Grid.n_halo

        scheme = namelist["momentum_advection"]["type"].upper()
        if scheme == "WENO5":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 3)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno5
            self._fv = MomentumAdvection_impl.v_advection_weno5
            self._fw = MomentumAdvection_impl.w_advection_weno5
        if scheme == "WENO5_Z":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 3)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno5_z
            self._fv = MomentumAdvection_impl.v_advection_weno5_z
            self._fw = MomentumAdvection_impl.w_advection_weno5_z
        elif scheme == "WENO5_BASE":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 3)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno5_base
            self._fv = MomentumAdvection_impl.v_advection_weno5_base
            self._fw = MomentumAdvection_impl.w_advection_weno5_base
        elif scheme == "WENO7":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno7
            self._fv = MomentumAdvection_impl.v_advection_weno7
            self._fw = MomentumAdvection_impl.w_advection_weno7
        elif scheme == "WENO7_Z":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno7_z
            self._fv = MomentumAdvection_impl.v_advection_weno7_z
            self._fw = MomentumAdvection_impl.w_advection_weno7_z
        elif scheme == "WENO7_BASE":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_weno7_base
            self._fv = MomentumAdvection_impl.v_advection_weno7_base
            self._fw = MomentumAdvection_impl.w_advection_weno7_base
        elif scheme == "SECOND":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 3)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_2nd
            self._fv = MomentumAdvection_impl.v_advection_2nd
            self._fw = MomentumAdvection_impl.w_advection_2nd
        elif scheme == "FOURTH":
            UtilitiesParallel.print_root(
                "\t \t Using " + scheme + " momentum advection"
            )
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._fu = MomentumAdvection_impl.u_advection_4th
            self._fv = MomentumAdvection_impl.v_advection_4th
            self._fw = MomentumAdvection_impl.w_advection_4th

        # Make sure functions have for each velocity component
        assert self._fu is not None
        assert self._fv is not None
        assert self._fw is not None

        return


class MomentumWENO(MomentumAdvectionBase):
    def __init__(self, namelist, Timers, Grid, Ref, ScalarState, VelocityState):
        MomentumAdvectionBase.__init__(
            self, namelist, Timers, Grid, Ref, ScalarState, VelocityState
        )

        self._Timers.add_timer("MomentumWENO_update")
        return

    def update(self):
        self._Timers.start_timer("MomentumWENO_update")

        # Get values from thermodynamic reference state
        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge

        dxi = self._Grid.dxi

        # Retrieve velocities from container
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        u_t = self._VelocityState.get_tend("u")
        v_t = self._VelocityState.get_tend("v")
        w_t = self._VelocityState.get_tend("w")

        # TODO Move the allocation somewhere else.
        fluxx = self._fluxx
        fluxy = self._fluxy
        fluxz = self._fluxz

        # Here we return the fluxes. We could capture these for output
        # U Component
        self._fu(rho0, rho0_edge, u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.uv_flux_div(
            dxi[0], dxi[1], dxi[2], alpha0, fluxx, fluxy, fluxz, u_t
        )

        # V Component
        self._fv(rho0, rho0_edge, u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.uv_flux_div(
            dxi[0], dxi[1], dxi[2], alpha0, fluxx, fluxy, fluxz, v_t
        )

        # W Component
        self._fw(rho0, rho0_edge, u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.w_flux_div(
            dxi[0], dxi[1], dxi[2], alpha0_edge, fluxx, fluxy, fluxz, w_t
        )

        self._Timers.end_timer("MomentumWENO_update")
        return


def factory(namelist, Timers, Grid, Ref, ScalarState, VelocityState):
    return MomentumWENO(namelist, Timers, Grid, Ref, ScalarState, VelocityState)
