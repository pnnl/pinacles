from Columbia.SGS import SGSBase
import numpy as np
import numba


@numba.njit(fastmath=True)
def compute_visc(dx, strain_rate_mag, bvf, cs, pr, eddy_viscosity, eddy_diffusivity):


    filt_scale_squared = (dx[0] * dx[1] * dx[2])**(1.0/3.0)
    pri = 1.0/pr

    shape = eddy_viscosity.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                fb = 1
                if bvf[i,j,k] > 0 and strain_rate_mag[i,j,k] > 0.0:
                    fb = max(0.0, 1.0 - bvf[i,j,k]/(pr *  strain_rate_mag[i,j,k] * strain_rate_mag[i,j,k]))**(1.0/2.0)


                eddy_viscosity[i,j,k] = min( (cs * filt_scale_squared) **2.0  * fb  * strain_rate_mag[i,j,k], 100.0)

                eddy_diffusivity[i,j,k] = eddy_viscosity[i,j,k] * pri 


    return


class Smagorinsky(SGSBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, DiagnosticState):

        SGSBase.__init__(self, namelist, Grid, Ref, VelocityState, DiagnosticState)


        self._DiagnosticState.add_variable('eddy_diffusivity')
        self._DiagnosticState.add_variable('eddy_viscosity')


        #Read values in from namelist
        try:
            self._cs = namelist['SGS']['Smagorinsky']['cs']
        except:
            self._cs = 0.17

        try:
            self._prt = namelsit['SGS']['Smagorinsky']['Prt']
        except:
            self._prt = 1.0/3.0


        return

    def update(self):

        dx = self._Grid.dx
        
        #Call the SGS model
        strain_rate_mag = self._DiagnosticState.get_field('strain_rate_mag')
        eddy_viscosity = self._DiagnosticState.get_field('eddy_viscosity')
        eddy_diffusivity = self._DiagnosticState.get_field('eddy_diffusivity')
        bvf = self._DiagnosticState.get_field('bvf')

        compute_visc(dx, strain_rate_mag, bvf, self._cs, self._prt, eddy_viscosity, eddy_diffusivity)



        return