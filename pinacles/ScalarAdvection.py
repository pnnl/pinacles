import numba
import numpy as np
import time

from pinacles import UtilitiesParallel, parameters
from pinacles.scalar_advection.flux_divergence import (
    flux_divergence,
    flux_divergence_bounded,
    flux_divergence_split_monotone,
    flux_divergence_monotone,
)

from pinacles.scalar_advection.first_order import first_order
from pinacles.scalar_advection.second_order import second_order
from pinacles.scalar_advection.wrf5 import wrf5_advection
from pinacles.scalar_advection.weno5_base import weno5_advection_base
from pinacles.scalar_advection.weno5_z import weno5_advection_z
from pinacles.scalar_advection.weno5 import weno5_advection
from pinacles.scalar_advection.weno7_base import weno7_advection_base
from pinacles.scalar_advection.weno7_z import weno7_advection_z
from pinacles.scalar_advection.weno7 import weno7_advection

from mpi4py import MPI


class ScalarAdvectionBase:
    def __init__(self, Timers, Grid, Ref, ScalarState, VelocityState, TimeStepping):

        self._name = "ScalarAdvection"
        self._Timers = Timers
        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._Ref = Ref
        self._TimeStepping = TimeStepping

    def update(self):
        pass

    @property
    def name(self):
        return self._name


@numba.njit(fastmath=True)
def compute_phi_range(phi):
    phi_max = -1e23
    shape = phi.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                phi_max = max(phi_max, abs(phi[i, j, k]))

    return max(phi_max, 1.0)


@numba.njit(fastmath=True)
def rescale_scalar(phi, phi_range, phi_norm):
    shape = phi.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                phi_norm[i, j, k] = phi[i, j, k] / phi_range


@numba.njit(fastmath=True)
def phi_has_nonzero(phi):
    shape = phi.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if abs(phi[i, j, k]) > 0.0:
                    return True
    return False


class ScalarWENO(ScalarAdvectionBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, ScalarState, VelocityState, TimeStepping
    ):
        ScalarAdvectionBase.__init__(
            self, Timers, Grid, Ref, ScalarState, VelocityState, TimeStepping
        )
        self._flux_profiles = {}

        self._flux_function = None
        self.flux_function_factory(namelist)

        # Allocate work arrays for fluxes
        self._fluxx = np.zeros(self._Grid.ngrid_local, dtype=np.double)
        self._fluxy = np.zeros_like(self._fluxx)
        self._fluxz = np.zeros_like(self._fluxx)

        self._fluxx_low = np.zeros_like(self._fluxx)
        self._fluxy_low = np.zeros_like(self._fluxx)
        self._fluxz_low = np.zeros_like(self._fluxx)

        self._phi_norm = np.zeros_like(self._fluxx)

        self._Timers.add_timer("ScalarWENO_update")

        # Now optionally override the default flux divergence functions. To do this add  
        # lists of variable names to the namelist/input file scalar_advection 
        # dictionary items with keys (default, bounded, split_emono, emono) 
        # corresponding to the flux divergence function you want to be used 
        # for a particular variable. 
        for fd in ["default", "bounded", "split_emono", "emono"]:
            nml_s = namelist["scalar_advection"]
            if fd in nml_s:
                for scalar in nml_s[fd]:
                    UtilitiesParallel.print_root(
                        "\t Setting scalar flux divergence of "
                        + scalar
                        + " to "
                        + fd
                        + "."
                    )
                    self._ScalarState.override_flux_divergence(scalar, fd)

        UtilitiesParallel.print_root(self._ScalarState._flux_divergence)

    def flux_function_factory(self, namelist):
        n_halo = self._Grid.n_halo
        scheme = namelist["scalar_advection"]["type"].upper()
        if scheme == "WENO5":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._flux_function = weno5_advection
        elif scheme == "WENO5_Z":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._flux_function = weno5_advection_z
        elif scheme == "WRF5":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._flux_function = wrf5_advection
        elif scheme == "WENO5_BASE":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._flux_function = weno5_advection_base
        elif scheme == "WENO7":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 5)  # Check that we have enough halo points
            self._flux_function = weno7_advection
        elif scheme == "WENO7_Z":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 5)  # Check that we have enough halo points
            self._flux_function = weno7_advection_z
        elif scheme == "WENO7_BASE":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 5)  # Check that we have enough halo points
            self._flux_function = weno7_advection_base

        assert self._flux_function is not None

    def io_initialize(self, this_grp):
        profiles_grp = this_grp["profiles"]
        for var in self._ScalarState.names:
            if "ff" in var:
                continue
            v = profiles_grp.createVariable(
                "w" + var + "_resolved",
                np.double,
                dimensions=(
                    "time",
                    "z",
                ),
            )
            v.long_name = "Resolved flux of " + var
            v.units = "m s^{-1} " + self._ScalarState.get_units(var)
            v.standard_name = "w " + self._ScalarState._latex_names[var]

        # Add the thetali flux
        v = profiles_grp.createVariable(
            "w" + "T" + "_resolved",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Resolved flux of temperature"
        v.units = "m s^{-1} K"
        v.standard_name = "wT"

        v = profiles_grp.createVariable(
            "w" + "thetali" + "_resolved",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Resolved flux of liquid-ice potential temperature"
        v.units = "m s^{-1} K"
        v.standard_name = "w \theta_{li}"

    def io_update(self, this_grp):
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        for var in self._ScalarState.names:

            if "ff" in var:
                continue

            flux_mean = UtilitiesParallel.ScalarAllReduce(
                self._flux_profiles[var] / npts
            )

            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp["profiles"]
                profiles_grp["w" + var + "_resolved"][-1, :] = flux_mean[
                    n_halo[2] : -n_halo[2]
                ]

        # Compute the thetali flux
        if my_rank == 0:
            profiles_grp = this_grp["profiles"]
            if "qc" in self._ScalarState.names and "qr" in self._ScalarState.names:
                wql = (
                    profiles_grp["wqc_resolved"][-1, :]
                    + profiles_grp["wqr_resolved"][-1, :]
                )
                # Liquid water flux
                wT = (
                    profiles_grp["ws_resolved"][-1, :]
                    + (wql * parameters.LV) * parameters.ICPD
                )
                # Temperature Flux
                wthetali_sgs = (
                    wT[:] - (wql * parameters.LV / parameters.CPD)
                ) / self._Ref.exner[n_halo[2] : -n_halo[2]]
            elif "qc" in self._ScalarState.names:
                wql = profiles_grp["wqc_resolved"][-1, :]
                wT = (
                    profiles_grp["ws_resolved"][-1, :]
                    + (wql * parameters.LV) * parameters.ICPD
                )
                wthetali_sgs = (
                    wT[:] - (wql * parameters.LV / parameters.CPD)
                ) / self._Ref.exner[n_halo[2] : -n_halo[2]]
            else:
                wT = profiles_grp["ws_resolved"][-1, :]
                wthetali_sgs = wT[:] / self._Ref.exner[n_halo[2] : -n_halo[2]]

                # Write to the netcdf file
                profiles_grp["wT_resolved"][-1, :] = wT
                profiles_grp["wthetali_resolved"][-1] = wthetali_sgs

    def initialize_io_arrays(self):
        for var in self._ScalarState.names:
            self._flux_profiles[var] = np.zeros(self._Grid.ngrid[2], dtype=np.double)

    def update(self):

        # For now we assume that all scalars are advected with this scheme. This doesn't have to
        # remain true.

        self._Timers.start_timer("ScalarWENO_update")

        # Get the velocities (No copy done here)
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        # Get the relevant reference variables
        # TODO there is acopy hiding here
        rho0 = self._Ref.rho0
        alpha0 = self._Ref.alpha0
        rho0_edge = self._Ref.rho0_edge

        dt = self._TimeStepping.dt

        # Allocate arrays for storing fluxes
        # TODO define these as class data
        fluxx = self._fluxx
        fluxy = self._fluxy
        fluxz = self._fluxz

        fluxx_low = self._fluxx_low
        fluxy_low = self._fluxy_low
        fluxz_low = self._fluxz_low

        phi_norm = self._phi_norm

        nhalo = self._Grid.n_halo
        # Now iterate over the scalar variables
        for var in self._ScalarState.names:
            phi_range = 1.0

            # Get a scalar field (No copy done here)
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)
            io_flux = self._flux_profiles[var]
            io_flux.fill(0)
            # Now compute the WENO fluxes
            flux_divergence_type = self._ScalarState.flux_divergence_type(var)
            if flux_divergence_type != "DEFAULT":

                if phi_has_nonzero(
                    phi
                ):  # If fields are zero everywhere no need to do any advection so skip-it!
                    phi_range = compute_phi_range(phi)
                    rescale_scalar(phi, phi_range, phi_norm)

                    # First compute the higher order fluxes, for now we do it with WENO
                    self._flux_function(
                        rho0,
                        rho0_edge,
                        u,
                        v,
                        w,
                        phi_norm,
                        fluxx,
                        fluxy,
                        fluxz,
                    )

                    # Now compute the lower order upwind fluxes these are used if high-order fluxes
                    # break boundness.
                    first_order(
                        rho0,
                        rho0_edge,
                        u,
                        v,
                        w,
                        phi_norm,
                        fluxx_low,
                        fluxy_low,
                        fluxz_low,
                    )

                    if flux_divergence_type == "EMONO":
                        # Essentially monotone advection scheme
                        flux_divergence_monotone(
                            nhalo,
                            self._Grid.dxi[0],
                            self._Grid.dxi[1],
                            self._Grid.dxi[2],
                            alpha0,
                            fluxx,
                            fluxy,
                            fluxz,
                            fluxx_low,
                            fluxy_low,
                            fluxz_low,
                            dt,
                            phi,
                            io_flux,
                            phi_range,
                            phi_t,
                        )
                    elif flux_divergence_type == "SPLIT_EMONO":
                        flux_divergence_split_monotone(
                            nhalo,
                            self._Grid.dxi[0],
                            self._Grid.dxi[1],
                            self._Grid.dxi[2],
                            alpha0,
                            fluxx,
                            fluxy,
                            fluxz,
                            fluxx_low,
                            fluxy_low,
                            fluxz_low,
                            dt,
                            phi,
                            io_flux,
                            phi_range,
                            phi_t,
                        )

                    else:
                        # Bounded advection scheme between 0 and 1
                        flux_divergence_bounded(
                            nhalo,
                            self._Grid.dxi[0],
                            self._Grid.dxi[1],
                            self._Grid.dxi[2],
                            alpha0,
                            fluxx,
                            fluxy,
                            fluxz,
                            fluxx_low,
                            fluxy_low,
                            fluxz_low,
                            0.0,
                            1.0,
                            dt,
                            phi,
                            io_flux,
                            phi_range,
                            phi_t,
                        )

            else:
                phi_range = compute_phi_range(phi)
                rescale_scalar(phi, phi_range, phi_norm)

                self._flux_function(
                    rho0,
                    rho0_edge,
                    u,
                    v,
                    w,
                    phi_norm,
                    fluxx,
                    fluxy,
                    fluxz,
                )
                # second_order(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)
                # Now compute the flux divergences
                flux_divergence(
                    nhalo,
                    self._Grid.dxi[0],
                    self._Grid.dxi[1],
                    self._Grid.dxi[2],
                    alpha0,
                    fluxx,
                    fluxy,
                    fluxz,
                    io_flux,
                    phi_range,
                    phi_t,
                )

        self._Timers.end_timer("ScalarWENO_update")
