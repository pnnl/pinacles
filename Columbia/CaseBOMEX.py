import numpy as np
from Columbia import parameters
from Columbia import Surface, Surface_impl, Forcing_impl, Forcing

class SurfaceBOMEX(Surface.SurfaceBase): 
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState): 

        Surface.SurfaceBase.__init__(self, namelist, Grid, Ref, VelocityState,
            ScalarState, DiagnosticState)


        self._theta_flux = 8.0e-3 # K m/s
        self._qv_flux = 5.2e-5
        self._ustar_ = 0.28 #m/s
        self._theta_surface = 299.1 #K
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


        #Get Tendnecies
        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')
        qvt = self._ScalarState.get_tend('qv')

        #Get surface slices
        usfc = u[:,:,nh[2]]
        vsfc = v[:,:,nh[2]]
        utsfc = ut[:,:,nh[2]]
        vtsfc = vt[:,:,nh[2]]
        stsfc = st[:,:,nh[2]]
        qvtsfc = qvt[:,:,nh[2]+1]



        Surface_impl.compute_windspeed_sfc(usfc, vsfc, self.gustiness, self._windspeed_sfc)
        Surface_impl.tau_given_ustar(self._ustar_sfc, usfc, vsfc, self._windspeed_sfc, self._taux_sfc, self._tauy_sfc)

        shf = np.zeros_like(self._taux_sfc) + self._theta_flux * exner_edge[nh[2]-1]
        qv_flx_sf = np.zeros_like(self._taux_sfc) + self._qv_flux
        #Surface_impl.iles_surface_flux_application(10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._taux_sfc, ut)
        #Surface_impl.iles_surface_flux_application(10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tauy_sfc, vt)
        Surface_impl.iles_surface_flux_application(10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, shf, st)
        Surface_impl.iles_surface_flux_application(10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, qv_flx_sf , qvt)
    
        return

class ForcingBOMEX(Forcing.ForcingBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
        Forcing.ForcingBase.__init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)

        self._f = 0.376e-4

        zl = self._Grid.z_local
        exner = self._Ref.exner

        #Set Geostrophic wind
        self._ug = np.zeros_like(self._Grid.z_global)
        for k in range(zl.shape[0]): 
            self._ug[k] = self._ug[k] = -10.0 + (1.8e-3)*zl[k]
        self._vg = np.zeros_like(self._ug)


        #Set heating rate
        self._heating_rate = np.zeros_like(self._Grid.z_global)

        #Set large scale cooling
        # Convert given form of tendencies (theta) to temperature tendency
        for k in range(zl.shape[0]): 
            if zl[k] <= 1500.0:
                self._heating_rate[k] = (-2.0/(3600 * 24.0))  * exner[k]     #K/s
            if zl[k] > 1500.0:
                self._heating_rate[k] = (-2.0/(3600 * 24.0) + (zl[k] - 1500.0)
                                 * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner[k]


        return 

    def update(self): 

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')

        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')

        #Forcing_impl.large_scale_pgf(self._ug, self._vg, self._f, u, v, vt, ut)
        st += self._heating_rate[np.newaxis, np.newaxis, :]

        #Forcing_impl.large_scale_pgf(self._ug, self._vg, self._f, u, v, vt, ut)

        return 