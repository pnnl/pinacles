from pinacles.SGS import SGSBase
import numpy as np
import numba


@numba.njit(fastmath=True)
def compute_visc(
    n_halo,
    dx,
    z,
    strain_rate_mag,
    bvf,
    cs,
    pr,
    eddy_viscosity,
    eddy_diffusivity,
    tke_sgs,
):

    shape = eddy_viscosity.shape

    filt_scale = (dx[0] * dx[1] * dx[2]) ** (1.0 / 3.0)
    pri = 1.0 / pr

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # filt_scale  = np.sqrt(1.0/(1.0/((dx[0] * dx[1] * dx[2] )**(1.0/3.0))**2.0 + 1.0/(0.4 * z[k])**2.0))
                # Compute the stratification correction
                fb = 1
                if bvf[i, j, k] > 0 and strain_rate_mag[i, j, k] > 1e-10:
                    fb = (
                        max(
                            0.0,
                            1.0
                            - bvf[i, j, k]
                            / (
                                pr * strain_rate_mag[i, j, k] * strain_rate_mag[i, j, k]
                            ),
                        )
                        ** (1.0 / 2.0)
                    )
                # Compute the eddy viscosity with a correction for
                # stratification
                eddy_viscosity[i, j, k] = (cs * filt_scale) ** 2.0 * (
                    fb * strain_rate_mag[i, j, k]
                )

                tke_sgs[i, j, k] = (eddy_viscosity[i, j, k] / (filt_scale * 0.1)) ** 2.0
                # Compute the eddy diffusivty from the  eddy viscosity using an assumed
                # inverse SGS Prandtl number tune this using
                eddy_diffusivity[i, j, k] = eddy_viscosity[i, j, k] * pri

    eddy_viscosity[:, :, n_halo[2] - 1] = eddy_viscosity[:, :, n_halo[2]]
    eddy_diffusivity[:, :, n_halo[2] - 1] = eddy_diffusivity[:, :, n_halo[2]]
    return


class Smagorinsky(SGSBase):
    def __init__(self, namelist, Timers, Grid, Ref, VelocityState, DiagnosticState):

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
        self._DiagnosticState.add_variable(
            "tke_sgs",
            long_name="Subgrid-scale turbulence kinetic energy",
            units="m^2s^-2",
            latex_name=r"e_{sgs}",
        )
        self._DiagnosticState.add_variable(
            "eddy_viscosity",
            long_name="Eddy Viscosity",
            units="m^2s^-1",
            latex_name="\nu_t",
        )

        # Read values in from namelist if not there set defaults
        try:
            self._cs = namelist["sgs"]["smagorinsky"]["cs"]
        except BaseException:
            self._cs = 0.17

        try:
            self._prt = namelist["sgs"]["smagorinsky"]["Prt"]
        except BaseException:
            self._prt = 1.0 / 3.0

        self._Timers.add_timer("SGSSmagorinsky_update")

        return

    def update(self):

        self._Timers.start_timer("SGSSmagorinsky_update")

        # Get the grid spacing from the Grid class
        dx = self._Grid.dx
        z = self._Grid.z_local
        n_halo = self._Grid.n_halo

        # Get the necessary 3D fields from the field containers
        strain_rate_mag = self._DiagnosticState.get_field("strain_rate_mag")
        eddy_viscosity = self._DiagnosticState.get_field("eddy_viscosity")
        eddy_diffusivity = self._DiagnosticState.get_field("eddy_diffusivity")
        tke_sgs = self._DiagnosticState.get_field("tke_sgs")
        bvf = self._DiagnosticState.get_field("bvf")

        # Compute the viscosity
        compute_visc(
            n_halo,
            dx,
            z,
            strain_rate_mag,
            bvf,
            self._cs,
            self._prt,
            eddy_viscosity,
            eddy_diffusivity,
            tke_sgs,
        )

        self._Timers.end_timer("SGSSmagorinsky_update")

        return
