from pinacles.SGS import SGSBase
import numpy as np
import numba


@numba.njit(fastmath=True)
def tke_ell(cn, e, buoyancy_frequency, delta):
    if buoyancy_frequency > 1e-10:
        ell = max(min(cn * np.sqrt(max(e, 0.0) / buoyancy_frequency), delta), 1e-10)
    else:
        ell = delta
    return ell


@numba.njit(fastmath=True)
def tke_viscosity_diffusivity(dx, e, buoyancy_frequency, visc, diff, cn, ck):
    shape = diff.shape

    delta = (dx[0] * dx[1] * dx[2]) ** (1.0 / 3.0)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ell = tke_ell(cn, e[i, j, k], buoyancy_frequency[i, j, k], delta)
                visc[i, j, k] = ck * ell * np.sqrt(max(e[i, j, k], 0.0))
                prt = delta / (delta + 2.0 * ell)
                diff[i, j, k] = visc[i, j, k] / prt

    return


@numba.njit(fastmath=True)
def tke_dissipation(dx, e, buoyancy_frequency, et, cn, ck):

    shape = et.shape

    delta = (dx[0] * dx[1] * dx[2]) ** (1.0 / 3.0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                ell = tke_ell(cn, e[i, j, k], buoyancy_frequency[i, j, k], delta)
                ceps = 1.9 * ck + (0.93 - 1.9 * ck) * ell / delta
                et[i, j, k] += -ceps * (max(e[i, j, k], 0.0) ** 1.5) / ell

    return


@numba.njit(fastmath=True)
def tke_shear_production(visc, strain_rate_mag, et):

    shape = et.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                et[i, j, k] += (
                    visc[i, j, k] * strain_rate_mag[i, j, k] * strain_rate_mag[i, j, k]
                )

    return


@numba.njit(fastmath=True)
def tke_buoyancy_production(diff, buoyancy_frequency, et):

    shape = et.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                et[i, j, k] += -diff[i, j, k] * buoyancy_frequency[i, j, k]

    return


class SGSTKE(SGSBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState
    ):

        self._ScalarState = ScalarState

        # Initialize the SGS baseclass
        SGSBase.__init__(
            self, namelist, Timers, Grid, Ref, VelocityState, DiagnosticState
        )

        # Add diagnostic fields
        self._DiagnosticState.add_variable(
            "eddy_diffusivity",
            long_name="Eddy Diffusivity",
            units="m^2s^-1",
            latex_name="\D_t",
        )
        self._ScalarState.add_variable(
            "tke_sgs",
            long_name="Subgrid-scale turbulence kinetic energy",
            units="m^2s^-2",
            latex_name=r"e_{sgs}",
            flux_divergence="EMONO",
        )
        self._DiagnosticState.add_variable(
            "eddy_viscosity",
            long_name="Eddy Viscosity",
            units="m^2s^-1",
            latex_name="\nu_t",
        )

        self._cn = 0.76
        try:
            self._cn = namelist["sgs"]["tke"]["cn"]
        except BaseException:
            pass

        self._ck = 0.1
        try:
            self._ck = namelist["sgs"]["tke"]["ck"]
        except BaseException:
            self._ck = 0.1

        self.init = False

        return

    def update(self):

        dx = self._Grid.dx

        e = self._ScalarState.get_field("tke_sgs")
        et = self._ScalarState.get_tend("tke_sgs")

        if np.amax(e) == 0.0:
            e.fill(1e-8)

        buoyancy_frequency = self._DiagnosticState.get_field("bvf")
        eddy_viscosity = self._DiagnosticState.get_field("eddy_viscosity")
        eddy_diffusivity = self._DiagnosticState.get_field("eddy_diffusivity")
        strain_rate_mag = self._DiagnosticState.get_field("strain_rate_mag")

        tke_viscosity_diffusivity(
            dx,
            e,
            buoyancy_frequency,
            eddy_viscosity,
            eddy_diffusivity,
            self._cn,
            self._ck,
        )

        tke_dissipation(dx, e, buoyancy_frequency, et, self._cn, self._ck)

        tke_shear_production(eddy_viscosity, strain_rate_mag, et)

        tke_buoyancy_production(eddy_diffusivity, buoyancy_frequency, et)

        return
