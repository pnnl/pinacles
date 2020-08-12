import numpy as np 
import netCDF4 as nc
from Columbia import Surface, Forcing
from Columbia import parameters

'''
CK: Here I am starting with the simplest case and assuming the start time of the forcing
files is the same as the simulation start time. This can easily be revisited, and made 
more sophisticated when/if needed

Assume heat fluxes are given
'''
class SurfaceTestbed(Surface.SurfaceBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController): 

        Surface.SurfaceBase.__init__(self, namelist, Grid, Ref, VelocityState,
            ScalarState, DiagnosticState)
        
        self._TimeSteppingController = TimeSteppingController
        
        surface_file = namelist['surface']['filepath']
        surface_data = nc.Dataset(surface_file, 'r')
        self._forcing_times = surface_data.variables['time'][:]
        self._forcing_shf = surface_data.variables['sensible_heat_flux'][:]
        self._forcing_lhf = surface_data.variables['latent_heat_flux'][:]
        self._forcing_skintemp = surface_data.variables['skin_temperature'][:]
        self._forcing_ustar = surface_data.variables['friction_velocity'][:]
        # Read off other variables needed for radiation..?

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
       


        # Open and read the file
        return
    
    
    def update(self):
        current_time = self._TimeSteppingController.time()
 
        # Interpolate to the current time
        shf_interp = np.interp(current_time, self._forcing_times, self._forcing_shf)
        lhf_interp = np.interp(current_time, self._forcing_times, self._forcing_lhf)
        ustar_interp = np.interp(current_time, self._forcing_times, self._forcing_ustar)

        # Get grid & reference profile info
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        exner_edge = self._Ref.exner_edge

        # Get fields
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')

        # Get tendencies
        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')
        qvt = self._ScalarState.get_tend('qv')

        # Get surface slices
        usfc = u[:,:,nh[2]]
        vsfc = v[:,:,nh[2]]
        utsfc = ut[:,:,nh[2]]
        vtsfc = vt[:,:,nh[2]]
        stsfc = st[:,:,nh[2]]
        qvtsfc = qvt[:,:,nh[2]]

        # Compute the surface stress & apply it
        ustar_sfc = np.zeros_like(self._windspeed_sfc) + ustar_interp
        Surface_impl.compute_windspeed_sfc(usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc)
        Surface_impl.tau_given_ustar(ustar_sfc, usfc, vsfc, self._Ref.u0, self._Ref.v0, self._windspeed_sfc, self._taux_sfc, self._tauy_sfc)
        Surface_impl.surface_flux_application(dxi2, nh, alpha0, alpha0_edge, self._taux_sfc, ut)
        Surface_impl.surface_flux_application(dxi2, nh, alpha0, alpha0_edge, self._tauy_sfc, vt)



       # Apply the heat fluxes
        s_flx_sf = np.zeros_like(self._taux_sfc) + shf_interp * alpha0_edge[nh[2]-1]/parameters.CPD
        qv_flx_sf = np.zeros_like(self._taux_sfc) + lhf_interp * alpha0_edge[nh[2]-1]/parameters.LV
        Surface_impl.surface_flux_application(dxi2, nh, alpha0, alpha0_edge, s_flx_sf, st)
        Surface_impl.surface_flux_application(dxi2, nh, alpha0, alpha0_edge, qv_flx_sf , qvt)


        return

        



