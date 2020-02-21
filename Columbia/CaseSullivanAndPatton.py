import numpy as np
import numba
from Columbia import Surface, Surface_impl, Forcing_impl, Forcing
from Columbia import parameters

class SurfaceSullivanAndPatton(Surface.SurfaceBase):

    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
        Surface.SurfaceBase.__init__(self, namelist, Grid, Ref, VelocityState,
            ScalarState, DiagnosticState)

        self._theta_flux = 0.24
        self._z0 = 0.1
        self.bflux_from_thflux()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc) + self._buoyancy_flux
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)

        return

    def update(self):

        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge

        #Get Fields
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')

        u0 = self._Ref.u0
        v0 = self._Ref.v0 

        #Get Tendnecies
        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')

        #Get surface slices
        usfc = u[:,:,nh[2]]
        vsfc = v[:,:,nh[2]]
        utsfc = ut[:,:,nh[2]]
        vtsfc = vt[:,:,nh[2]]
        stsfc = st[:,:,nh[2]]

        #Compute the windspeed, friction velocity, and surface stresses
        Surface_impl.compute_windspeed_sfc(usfc, vsfc, u0, v0, self.gustiness, self._windspeed_sfc)
        Surface_impl.compute_ustar_sfc(self._windspeed_sfc, self._bflx_sfc, self._z0, self._Grid.dx[2]/2.0, self._ustar_sfc)
        Surface_impl.tau_given_ustar(self._ustar_sfc, usfc, vsfc, u0, v0, self._windspeed_sfc, self._taux_sfc, self._tauy_sfc)

        #Add the tendencies
        #utsfc += self._taux_sfc * dxi2 * alpha0[nh[2]]/alpha0_edge[nh[2]-1]
        #vtsfc += self._tauy_sfc * dxi2 * alpha0[nh[2]]/alpha0_edge[nh[2]-1]
        #stsfc +=  self._theta_flux * parameters.CPD*exner_edge[nh[2]-1] * dxi2 * alpha0[nh[2]]/alpha0_edge[nh[2]-1]

        shf = np.zeros_like(self._taux_sfc) + self._theta_flux * exner_edge[nh[2]-1]

        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._taux_sfc, ut)
        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tauy_sfc, vt)
        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, shf, st)

        return

class ForcingSullivanAndPatton(Forcing.ForcingBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
        Forcing.ForcingBase.__init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)

        self._f = 1.0e-4

        self._ug = np.zeros_like(self._Grid.z_global) + 1.0
        self._vg = np.zeros_like(self._ug)
        
        return


    def update(self):

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')

        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')

        u0 = self._Ref.u0
        v0 = self._Ref.v0 

        Forcing_impl.large_scale_pgf(self._ug, self._vg, self._f, u, v, u0, v0, vt, ut)
        return