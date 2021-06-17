import numpy as np
from mpi4py import MPI
import numba
from pinacles import UtilitiesParallel, parameters


@numba.njit(fastmath=True)
def compute_fluxes(
    n_halo,
    dx,
    dxi,
    rho0,
    rho0_edge,
    phi,
    eddy_diffusivity,
    fluxx,
    fluxy,
    fluxz,
    io_flux,
    phi_t,
):

    shape = phi.shape
    # kz_fact = dx[2]*dx[2]/((dx[0] * dx[1]))
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):

                fluxx[i, j, k] = (
                    -0.5
                    * (eddy_diffusivity[i, j, k] + eddy_diffusivity[i + 1, j, k])
                    * (phi[i + 1, j, k] - phi[i, j, k])
                    * dxi[0]
                    * rho0[k]
                )
                fluxy[i, j, k] = (
                    -0.5
                    * (eddy_diffusivity[i, j, k] + eddy_diffusivity[i, j + 1, k])
                    * (phi[i, j + 1, k] - phi[i, j, k])
                    * dxi[1]
                    * rho0[k]
                )
                fluxz[i, j, k] = (
                    -0.5
                    * (eddy_diffusivity[i, j, k] + eddy_diffusivity[i, j, k + 1])
                    * (phi[i, j, k + 1] - phi[i, j, k])
                    * dxi[2]
                    * rho0_edge[k]
                )  # * kz_fact

    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):

                io_flux[k] += fluxz[i, j, k] / rho0_edge[k]

                phi_t[i, j, k] -= (
                    (fluxx[i, j, k] - fluxx[i - 1, j, k]) * dxi[0] / rho0[k]
                )
                phi_t[i, j, k] -= (
                    (fluxy[i, j, k] - fluxy[i, j - 1, k]) * dxi[1] / rho0[k]
                )
                phi_t[i, j, k] -= (
                    (fluxz[i, j, k] - fluxz[i, j, k - 1]) * dxi[2] / rho0[k]
                )
    return


class ScalarDiffusion:
    def __init__(self, namelist, Timers, Grid, Ref, DiagnosticState, ScalarState):

        self._name = "ScalarDiffusion"

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._DiagnosticState = DiagnosticState
        self._ScalarState = ScalarState
        self._flux_profiles = {}

        self._fluxx = np.zeros(self._Grid.ngrid_local, dtype=np.single)
        self._fluxy = np.zeros_like(self._fluxx)
        self._fluxz = np.zeros_like(self._fluxx)

        self._Timers.add_timer("ScalarDiffusion_update")
        return

    def io_initialize(self, this_grp):
        profiles_grp = this_grp["profiles"]
        for var in self._ScalarState.names:
            if "ff" in var:
                continue

            v = profiles_grp.createVariable(
                "w" + var + "_sgs", np.single, dimensions=("time", "z",)
            )
            v.long_name = "SGS flux of " + var
            v.units = "m s^{-1} " + self._ScalarState.get_units(var)
            v.standard_name = "w " + self._ScalarState._latex_names[var] + " sgs"
        # Add the thetali flux
        v = profiles_grp.createVariable(
            "w" + "T" + "_sgs", np.single, dimensions=("time", "z",)
        )
        v.long_name = "SGS flux of temperature"
        v.units = "m s^{-1} K"
        v.standard_name = "wT sgs"

        v = profiles_grp.createVariable(
            "w" + "thetali" + "_sgs", np.single, dimensions=("time", "z",)
        )
        v.long_name = "SGS flux of liquid-ice potential temperature"
        v.units = "m s^{-1} K"
        v.standard_name = "w \theta_{li} sgs"

        return

    def io_update(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        for var in self._ScalarState.names:
            if "ff" in var:
                return

            flux_mean = UtilitiesParallel.ScalarAllReduce(
                self._flux_profiles[var] / npts
            )

            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp["profiles"]
                profiles_grp["w" + var + "_sgs"][-1, :] = flux_mean[
                    n_halo[2] : -n_halo[2]
                ]

        # Compute the thetali flux
        if my_rank == 0:
            profiles_grp = this_grp["profiles"]
            if "qc" in self._ScalarState.names and "qr" in self._ScalarState.names:
                wql = profiles_grp["wqc_sgs"][-1, :] + profiles_grp["wqr_sgs"][-1, :]
                # Liquid water flux
                wT = (
                    profiles_grp["ws_sgs"][-1, :]
                    + (wql * parameters.LV) * parameters.ICPD
                )
                # Temperature Flux
                wthetali_sgs = (
                    wT[:] - (wql * parameters.LV / parameters.CPD)
                ) / self._Ref.exner[n_halo[2] : -n_halo[2]]
            elif "qc" in self._ScalarState.names:
                wql = profiles_grp["wqc_sgs"][-1, :]
                wT = (
                    profiles_grp["ws_sgs"][-1, :]
                    + (wql * parameters.LV) * parameters.ICPD
                )
                wthetali_sgs = (
                    wT[:] - (wql * parameters.LV / parameters.CPD)
                ) / self._Ref.exner[n_halo[2] : -n_halo[2]]
            else:
                wT = profiles_grp["ws_sgs"][-1, :]
                wthetali_sgs = wT[:] / self._Ref.exner[n_halo[2] : -n_halo[2]]

            profiles_grp["wT_sgs"][-1, :] = wT
            profiles_grp["wthetali_sgs"][-1] = wthetali_sgs

        return

    def initialize_io_arrays(self):

        for var in self._ScalarState.names:
            self._flux_profiles[var] = np.zeros(self._Grid.ngrid[2], dtype=np.single)

        return

    def update(self):

        self._Timers.start_timer("ScalarDiffusion_update")

        n_halo = self._Grid.n_halo
        dxi = self._Grid.dxi
        dx = self._Grid.dx

        rho0 = self._Ref.rho0
        alpha0 = self._Ref.alpha0
        rho0_edge = self._Ref.rho0_edge

        eddy_diffusivity = self._DiagnosticState.get_field("eddy_diffusivity")

        fluxx = self._fluxx
        fluxy = self._fluxy
        fluxz = self._fluxz

        for var in self._ScalarState.names:
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)
            io_flux = self._flux_profiles[var]
            io_flux.fill(0.0)
            compute_fluxes(
                n_halo,
                dx,
                dxi,
                rho0,
                rho0_edge,
                phi,
                eddy_diffusivity,
                fluxx,
                fluxy,
                fluxz,
                io_flux,
                phi_t,
            )

        self._Timers.end_timer("ScalarDiffusion_update")

        return

    @property
    def name(self):
        return self._name
