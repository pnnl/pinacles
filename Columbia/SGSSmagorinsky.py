from Columbia.SGS import SGSBase
import numpy as np
import numba


@numba.njit(fastmath=True)
def compute_visc(dx, strain_rate_mag, bvf, cs, pr,
                 eddy_viscosity, eddy_diffusivity):

    shape = eddy_viscosity.shape

    filt_scale  = (dx[0] * dx[1])**(1.0/2.0)
    pri = 1.0/pr

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # Compute the stratification correction
                fb = 1
                if bvf[i, j, k] > 0 and strain_rate_mag[i, j, k] > 0.0:
                    fb = max(0.0, 1.0 -
                             bvf[i, j, k] /
                             (pr *
                              strain_rate_mag[i, j, k] *
                                 strain_rate_mag[i, j, k]))**(1.0 /
                                                              2.0)
                # Compute the eddy viscosity with a correction for
                # stratification
                eddy_viscosity[i, j, k] = (
                    (cs * filt_scale) ** 2.0 * (fb * strain_rate_mag[i, j, k]))

                # Compute the eddy diffusivty from the  eddy viscosity using an assumed
                # inverse SGS Prandtl number tune this using
                eddy_diffusivity[i, j, k] = eddy_viscosity[i, j, k] * pri

    return


class Smagorinsky(SGSBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, DiagnosticState):

        # Initialize the SGS baseclass
        SGSBase.__init__(
            self,
            namelist,
            Grid,
            Ref,
            VelocityState,
            DiagnosticState)

        # Add diagnostic fields
        self._DiagnosticState.add_variable('eddy_diffusivity')
        self._DiagnosticState.add_variable('eddy_viscosity')

        # Read values in from namelist if not there set defaults
        try:
            self._cs = namelist['sgs']['smagorinsky']['cs']
        except BaseException:
            self._cs = 0.17

        try:
            self._prt = namelsit['sgs']['smagorinsky']['Prt']
        except BaseException:
            self._prt = 1.0 / 3.0

        return

    def update(self):

        # Get the grid spacing from the Grid class
        dx = self._Grid.dx

        # Get the necessary 3D fields from the field containers
        strain_rate_mag = self._DiagnosticState.get_field('strain_rate_mag')
        eddy_viscosity = self._DiagnosticState.get_field('eddy_viscosity')
        eddy_diffusivity = self._DiagnosticState.get_field('eddy_diffusivity')
        bvf = self._DiagnosticState.get_field('bvf')

        # Compute the viscosity
        compute_visc(
            dx,
            strain_rate_mag,
            bvf,
            self._cs,
            self._prt,
            eddy_viscosity,
            eddy_diffusivity)

        return
