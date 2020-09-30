import numpy as np
import numba
from Columbia import Surface, Surface_impl
from Columbia import parameters


class SurfaceStableBubble(Surface.SurfaceBase):

    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
        Surface.SurfaceBase.__init__(self, namelist, Grid, Ref, VelocityState,
            ScalarState, DiagnosticState)

        self._theta_flux = 0.0
        self._z0 = 0.0
        self.bflux_from_thflux()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc) + self._buoyancy_flux
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)

        return

    def update(self):
	#No surface implementation for this case

        return
