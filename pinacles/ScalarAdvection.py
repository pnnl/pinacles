import numba
import numpy as np
import time
from pinacles.interpolation_impl import interp_weno7, interp_weno5, centered_second
from pinacles import UtilitiesParallel, parameters
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

        return

    def update(self):
        return

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

    return


@numba.njit(fastmath=True)
def phi_has_nonzero(phi):
    shape = phi.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if abs(phi[i, j, k]) > 0.0:
                    return True
    return False


@numba.njit(fastmath=True)
def weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):
        
    phi_shape = phi.shape

    u_loc = np.empty((u.shape[0]), dtype=np.double)
    phi_loc = np.empty((phi.shape[0]), dtype=np.double)
    fluxx_loc = np.empty((fluxx.shape[0]), dtype=np.double)
    
    for j in range(2, phi_shape[1] - 3):
        for k in range(2, phi_shape[2] - 3):
            phi_loc[:] = phi[:,j,k]

            if np.count_nonzero(phi_loc) != 0:
                u_loc[:] = u[:,j,k]
                fluxx_loc[:] = fluxx[:,j,k]
                rho_loc = rho0[k]
                for i in range(2, phi_shape[0] - 3):
                    # First compute x-advection
                    if u_loc[i] >= 0:
                        fluxx_loc[i] = (
                            rho_loc
                            * u_loc[i] 
                            * interp_weno5(
                                phi_loc[i - 2],
                                phi_loc[i - 1],
                                phi_loc[i],
                                phi_loc[i + 1],
                                phi_loc[i + 2],
                            )
                        )
                    else:
                        fluxx_loc[i] = (
                            rho_loc
                            * u_loc[i]
                            * interp_weno5(
                                phi_loc[i],
                                phi_loc[i + 2],
                                phi_loc[i + 1],
                                phi_loc[i],
                                phi_loc[i - 1],
                            )
                        )
                fluxx[:,j,k] = fluxx_loc


    v_loc = np.empty((v.shape[1]), dtype=np.double)
    phi_loc = np.empty((phi.shape[1]), dtype=np.double)
    fluxy_loc = np.empty((fluxy.shape[1]), dtype=np.double)

    for i in range(2, phi_shape[0] - 3):
        for k in range(2, phi_shape[2] - 3):
            phi_loc[:] = phi[i,:,k]
            if np.count_nonzero(phi_loc) != 0:
                v_loc[:] = v[i,:,k]
                fluxy_loc[:] = fluxy[i,:,k]
                rho_loc = rho0[k]

                for j in range(2, phi_shape[1] - 3):
                    # First compute y-advection
                
                    if v_loc[j] >= 0:
                        fluxy_loc[j] = (
                            rho_loc
                            * v_loc[j]
                            * interp_weno5(
                                phi_loc[j - 2],
                                phi_loc[j - 1],
                                phi_loc[j],
                                phi_loc[j + 1],
                                phi_loc[j + 2],
                            )
                        )
                    else:
                        fluxy_loc[j] = (
                            rho_loc
                            * v_loc[j]
                            * interp_weno5(
                                phi_loc[j + 3],
                                phi_loc[j + 2],
                                phi_loc[j + 1],
                                phi_loc[j],
                                phi_loc[j - 1],
                            )
                        )
                fluxy[i,:,k] = fluxy_loc

    w_loc = np.empty((w.shape[2]), dtype=np.double)
    phi_loc = np.empty((phi.shape[2]), dtype=np.double)
    fluxz_loc = np.empty((fluxy.shape[2]), dtype=np.double)

    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):

            phi_loc[:] = phi[i,j,:]
            if np.count_nonzero(phi_loc) != 0:
                w_loc[:] = w[i,j,:]
                fluxz_loc[:] = fluxz[i,j,:] 

                for k in range(2, phi_shape[2] - 3):
                    # First compute y-advection
                    if w_loc[k] >= 0:
                        fluxz_loc[k] = (
                            rho0_edge[k]
                            * w_loc[k]
                            * interp_weno5(
                                phi_loc[k - 2],
                                phi_loc[k - 1],
                                phi_loc[k],
                                phi_loc[k + 1],
                                phi_loc[k + 2],
                            )
                        )
                    else:
                        fluxz_loc[k] = (
                            rho0_edge[k]
                            * w_loc[k]
                            * interp_weno5(
                                phi_loc[k + 3],
                                phi_loc[k + 2],
                                phi_loc[k + 1],
                                phi_loc[k],
                                phi_loc[k - 1],
                            )
                        )

                fluxz[i,j,:] = fluxz_loc
    return


@numba.njit(fastmath=True)
def weno7_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi.shape
    for i in range(3, phi_shape[0] - 4):
        for j in range(3, phi_shape[1] - 4):
            for k in range(3, phi_shape[2] - 4):
                # First compute x-advection
                if u[i, j, k] >= 0:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno7(
                            phi[i - 3, j, k],
                            phi[i - 2, j, k],
                            phi[i - 1, j, k],
                            phi[i, j, k],
                            phi[i + 1, j, k],
                            phi[i + 2, j, k],
                            phi[i + 3, j, k],
                        )
                    )
                else:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno7(
                            phi[i + 4, j, k],
                            phi[i + 3, j, k],
                            phi[i + 2, j, k],
                            phi[i + 1, j, k],
                            phi[i, j, k],
                            phi[i - 1, j, k],
                            phi[i - 2, j, k],
                        )
                    )

                # First compute y-advection
                if v[i, j, k] >= 0:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno7(
                            phi[i, j - 3, k],
                            phi[i, j - 2, k],
                            phi[i, j - 1, k],
                            phi[i, j, k],
                            phi[i, j + 1, k],
                            phi[i, j + 2, k],
                            phi[i, j + 3, k],
                        )
                    )
                else:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno7(
                            phi[i, j + 4, k],
                            phi[i, j + 3, k],
                            phi[i, j + 2, k],
                            phi[i, j + 1, k],
                            phi[i, j, k],
                            phi[i, j - 1, k],
                            phi[i, j - 2, k],
                        )
                    )

                # First compute y-advection
                if w[i, j, k] >= 0:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno7(
                            phi[i, j, k - 3],
                            phi[i, j, k - 2],
                            phi[i, j, k - 1],
                            phi[i, j, k],
                            phi[i, j, k + 1],
                            phi[i, j, k + 2],
                            phi[i, j, k + 3],
                        )
                    )
                else:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno7(
                            phi[i, j, k + 4],
                            phi[i, j, k + 3],
                            phi[i, j, k + 2],
                            phi[i, j, k + 1],
                            phi[i, j, k],
                            phi[i, j, k - 1],
                            phi[i, j, k - 2],
                        )
                    )
    return


theta = 1.0


@numba.njit(fastmath=True)
def weno5_advection_flux_limit(
    nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t
):
    phi_shape = phi.shape
    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):
            for k in range(2, phi_shape[2] - 3):
                # First compute x-advection
                if u[i, j, k] >= 0:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno5(
                            phi[i - 2, j, k],
                            phi[i - 1, j, k],
                            phi[i, j, k],
                            phi[i + 1, j, k],
                            phi[i + 2, j, k],
                        )
                    )

                    fluxlow = rho0[k] * u[i, j, k] * phi[i, j, k]
                else:
                    fluxx[i, j, k] = (
                        rho0[k]
                        * u[i, j, k]
                        * interp_weno5(
                            phi[i + 3, j, k],
                            phi[i + 2, j, k],
                            phi[i + 1, j, k],
                            phi[i, j, k],
                            phi[i - 1, j, k],
                        )
                    )
                    fluxlow = rho0[k] * u[i, j, k] * phi[i + 1, j, k]

                denom = phi[i + 1, j, k] - phi[i, j, k]
                if denom != 0.0:
                    r = (phi[i, j, k] - phi[i - 1, j, k]) / denom
                    # flim = np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(
                        0.0, np.minimum(theta * r, np.minimum((1 + r) / 2.0, theta))
                    )
                    fluxx[i, j, k] = fluxlow - flim * (fluxlow - fluxx[i, j, k])

                # First compute y-advection
                if v[i, j, k] >= 0:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno5(
                            phi[i, j - 2, k],
                            phi[i, j - 1, k],
                            phi[i, j, k],
                            phi[i, j + 1, k],
                            phi[i, j + 2, k],
                        )
                    )
                    fluxlow = rho0[k] * v[i, j, k] * phi[i, j, k]
                else:
                    fluxy[i, j, k] = (
                        rho0[k]
                        * v[i, j, k]
                        * interp_weno5(
                            phi[i, j + 3, k],
                            phi[i, j + 2, k],
                            phi[i, j + 1, k],
                            phi[i, j, k],
                            phi[i, j - 1, k],
                        )
                    )
                    fluxlow = rho0[k] * v[i, j, k] * phi[i, j + 1, k]
                denom = phi[i, j + 1, k] - phi[i, j, k]
                if denom != 0.0:
                    r = (phi[i, j, k] - phi[i, j - 1, k]) / denom
                    # flim =  np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(
                        0.0, np.minimum(theta * r, np.minimum((1 + r) / 2.0, theta))
                    )
                    fluxy[i, j, k] = fluxlow - flim * (fluxlow - fluxy[i, j, k])

                # First compute y-advection
                if w[i, j, k] >= 0:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno5(
                            phi[i, j, k - 2],
                            phi[i, j, k - 1],
                            phi[i, j, k],
                            phi[i, j, k + 1],
                            phi[i, j, k + 2],
                        )
                    )
                    fluxlow = rho0_edge[k] * w[i, j, k] * phi[i, j, k]
                else:
                    fluxz[i, j, k] = (
                        rho0_edge[k]
                        * w[i, j, k]
                        * interp_weno5(
                            phi[i, j, k + 3],
                            phi[i, j, k + 2],
                            phi[i, j, k + 1],
                            phi[i, j, k],
                            phi[i, j, k - 1],
                        )
                    )
                    fluxlow = rho0_edge[k] * w[i, j, k] * phi[i, j, k + 1]

                denom = phi[i, j, k + 1] - phi[i, j, k]
                if denom != 0.0:
                    r = (phi[i, j, k] - phi[i, j, k - 1]) / denom
                    # flim =  np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(
                        0.0, np.minimum(theta * r, np.minimum((1 + r) / 2.0, theta))
                    )
                    fluxz[i, j, k] = fluxlow - flim * (fluxlow - fluxz[i, j, k])

    return


@numba.njit(fastmath=True)
def first_order(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):

    phi_shape = phi.shape

    u_loc = np.empty((u.shape[0]), dtype=np.double)
    phi_loc = np.empty((phi.shape[0]), dtype=np.double)
    fluxx_loc = np.empty((fluxx.shape[0]), dtype=np.double)



    for j in range(2, phi_shape[1] - 3):
        for k in range(2, phi_shape[2] - 3):
            phi_loc[:] = phi[:,j,k]

            if np.count_nonzero(phi_loc) != 0:
                u_loc[:] = u[:,j,k]
                fluxx_loc[:] = fluxx[:,j,k]
                rho_loc = rho0[k]

                for i in range(2, phi_shape[0] - 3):
                    # First compute x-advection
                    fluxx_loc[i] = 0.5 * (abs(u_loc[i] + u_loc[i]) *  rho_loc  * phi_loc[i] + abs(u_loc[i] - u_loc[i])* rho_loc * u_loc[i] * phi_loc[i + 1])

                fluxx[:,j,k] = fluxx_loc

    v_loc = np.empty((v.shape[1]), dtype=np.double)
    phi_loc = np.empty((phi.shape[1]), dtype=np.double)
    fluxy_loc = np.empty((fluxx.shape[1]), dtype=np.double)

    for i in range(2, phi_shape[0] - 3):
        for k in range(2, phi_shape[2] - 3):
            phi_loc[:] = phi[i,:,k]

            if np.count_nonzero(phi_loc) != 0:
                v_loc[:] = v[i,:,k]
                fluxy_loc[:] = fluxy[i,:,k]
                rho_loc = rho0[k]

                for j in range(2, phi_shape[1] - 3):
                    # First compute y-advection
                    fluxy_loc[j] = 0.5 * (abs(v_loc[j] + v_loc[j]) *  rho_loc  * phi_loc[j] + abs(v_loc[j] - v_loc[j])* rho_loc * v_loc[j] * phi_loc[j  + 1])


                fluxy[i,:,k] = fluxy_loc


    w_loc = np.empty((w.shape[2]), dtype=np.double)
    phi_loc = np.empty((phi.shape[2]), dtype=np.double)
    fluxz_loc = np.empty((fluxz.shape[2]), dtype=np.double)

    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):
            phi_loc[:] = phi[i,j,:]
            
            if np.count_nonzero(phi_loc) != 0:
                w_loc[:] = w[i,j,:]
                fluxz_loc[:] = fluxz[i,j,:]
                

                for k in range(2, phi_shape[2] - 3):
                    rho_loc = rho0_edge[k]
                    # First compute y-advection
                    fluxz_loc[k] = 0.5 * (abs(w_loc[k] + w_loc[k]) *  rho_loc  * phi_loc[k] + abs(w_loc[k] - w_loc[k])* rho_loc * w_loc[k] * phi_loc[k  + 1])

                fluxz[i,j,:] = fluxz_loc
    return


@numba.njit(fastmath=True)
def second_order(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):

    phi_shape = phi.shape
    for i in range(2, phi_shape[0] - 3):
        for j in range(2, phi_shape[1] - 3):
            for k in range(2, phi_shape[2] - 3):
                # First compute x-advection
                fluxx[i, j, k] = (
                    rho0[k]
                    * u[i, j, k]
                    * centered_second(phi[i, j, k], phi[i + 1, j, k])
                )

                # First compute y-advection
                fluxy[i, j, k] = (
                    rho0[k]
                    * v[i, j, k]
                    * centered_second(phi[i, j, k], phi[i, j + 1, k])
                )

                # First compute y-advection
                fluxz[i, j, k] = (
                    rho0_edge[k]
                    * w[i, j, k]
                    * centered_second(phi[i, j, k], phi[i, j, k + 1])
                )

    return


@numba.njit(fastmath=True)
def flux_divergence(
    nhalo, idx, idy, idzi, alpha0, fluxx, fluxy, fluxz, io_flux, phi_range, phi_t
):
    phi_shape = phi_t.shape
    # TODO Tighten range of loops
    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )
                io_flux[k] += (
                    (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * alpha0[k] * phi_range
                )
    return


@numba.njit(fastmath=True)
def flux_divergence_bounded(
    nhalo,
    idx,
    idy,
    idzi,
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
):
    phi_shape = phi_t.shape
    # TODO Tighten range of loops
    for i in range(1, phi_shape[0] - 1):
        for j in range(1, phi_shape[1] - 1):
            for k in range(1, phi_shape[2] - 1):

                tend_tmp = -alpha0[k] * (
                    (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                    + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                    + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                )

                if phi[i, j, k] + tend_tmp * dt < 0.0:

                    fluxx[i - 1, j, k] = fluxx_low[i - 1, j, k]
                    fluxx[i, j, k] = fluxx_low[i, j, k]

                    fluxy[i, j - 1, k] = fluxy_low[i, j - 1, k]
                    fluxy[i, j, k] = fluxy_low[i, j, k]

                    fluxz[i, j, k - 1] = fluxz_low[i, j, k - 1]
                    fluxz[i, j, k] = fluxz_low[i, j, k]

    for i in range(nhalo[0], phi_shape[0] - nhalo[0]):
        for j in range(nhalo[1], phi_shape[1] - nhalo[1]):
            for k in range(nhalo[2], phi_shape[2] - nhalo[2]):
                io_flux[k] += (fluxz[i, j, k] + fluxz[i, j, k - 1]) * 0.5 * phi_range
                phi_t[i, j, k] -= (
                    alpha0[k]
                    * (
                        (fluxx[i, j, k] - fluxx[i - 1, j, k]) * idx
                        + (fluxy[i, j, k] - fluxy[i, j - 1, k]) * idy
                        + (fluxz[i, j, k] - fluxz[i, j, k - 1]) * idzi
                    )
                    * phi_range
                )

    return


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

        return

    def flux_function_factory(self, namelist):
        n_halo = self._Grid.n_halo
        scheme = namelist["scalar_advection"]["type"].upper()
        if scheme == "WENO5":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 3)  # Check that we have enough halo points
            self._flux_function = weno5_advection
        elif scheme == "WENO7":
            UtilitiesParallel.print_root("\t \t Using " + scheme + " scalar advection")
            assert np.all(n_halo >= 4)  # Check that we have enough halo points
            self._flux_function = weno7_advection

        assert self._flux_function is not None

        return

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

        return

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

        return

    def initialize_io_arrays(self):
        for var in self._ScalarState.names:
            self._flux_profiles[var] = np.zeros(self._Grid.ngrid[2], dtype=np.double)

        return

    def update(self):

        # For now we assume that all scalars are advected with this scheme. This doesn't have to
        # remain true.

        self._Timers.start_timer("ScalarWENO_update")

        # Get the velocities (No copy done here)
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        # Get the releveant reference variables
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
            if "ff" in var or "plume" in var or var in ["qc", "qr"]:
                if phi_has_nonzero(
                    phi
                ):  # If fields are zero everywhere no need to do any advection so skip-it!
                    
                    fluxx.fill(0.0)
                    fluxy.fill(0.0)
                    fluxz.fill(0.0)

                    fluxx_low.fill(0.0)
                    fluxy_low.fill(0.0)
                    fluxz_low.fill(0.0)
                    phi_range = compute_phi_range(phi)
                    rescale_scalar(phi, phi_range, phi_norm)
                    # phi_norm = phi

                    # Divide by range
                    # np.divide(phi, phi_range, out=phi_norm)

                    # First compute the higher order fluxes, for now we do it with WENO
                    self._flux_function(
                        nhalo,
                        rho0,
                        rho0_edge,
                        u,
                        v,
                        w,
                        phi_norm,
                        fluxx,
                        fluxy,
                        fluxz,
                        phi_t,
                    )

                    # Now compute the lower order upwind fluxes these are used if high-order fluxes
                    # break boundness.
                    first_order(
                        nhalo,
                        rho0,
                        rho0_edge,
                        u,
                        v,
                        w,
                        phi_norm,
                        fluxx_low,
                        fluxy_low,
                        fluxz_low,
                        phi_t,
                    )

                    # Now insure the that the advection does not violate boundeness of scalars.
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
                        dt,
                        phi,
                        io_flux,
                        phi_range,
                        phi_t,
                    )
                #   weno5_advection_flux_limit(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)
            else:

                # phi_range = max(np.max(np.abs(phi)), 1.0)

                # Divide by range
                # np.divide(phi, phi_range, out=phi_norm)
                phi_range = compute_phi_range(phi)
                rescale_scalar(phi, phi_range, phi_norm)
                # phi_norm = phi

                self._flux_function(
                    nhalo,
                    rho0,
                    rho0_edge,
                    u,
                    v,
                    w,
                    phi_norm,
                    fluxx,
                    fluxy,
                    fluxz,
                    phi_t,
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

        return
