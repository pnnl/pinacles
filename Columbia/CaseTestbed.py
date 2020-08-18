import numpy as np 
import netCDF4 as nc
from Columbia import Surface, Surface_impl, Forcing, Forcing_impl
from Columbia import parameters
from scipy import interpolate

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
        
        file = namelist['testbed']['input_filepath']
        data = nc.Dataset(file, 'r')
        surface_data = data.groups['surface']
        self._forcing_times = surface_data.variables['time'][:]
        self._forcing_shf = surface_data.variables['sensible_heat_flux'][:]
        self._forcing_lhf = surface_data.variables['latent_heat_flux'][:]
        self._forcing_skintemp = surface_data.variables['skin_temperature'][:]
        self._forcing_ustar = surface_data.variables['friction_velocity'][:]
        # Read off other variables needed for radiation..?

        data.close()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        
        return
    
    
    def update(self):
        current_time = self._TimeSteppingController.time()
 
        # Interpolate to the current time
     
        shf_interp = interpolate.interp1d(self._forcing_times, self._forcing_shf,fill_value='extrapolate', assume_sorted=True )(current_time)
        lhf_interp = interpolate.interp1d(self._forcing_times, self._forcing_lhf,fill_value='extrapolate', assume_sorted=True )(current_time)
        ustar_interp = interpolate.interp1d(self._forcing_times, self._forcing_ustar,fill_value='extrapolate', assume_sorted=True )(current_time)
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

class ForcingTestbed(Forcing.ForcingBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):

        Forcing.ForcingBase.__init__(self, namelist, Grid, 
        Ref, VelocityState, ScalarState, DiagnosticState)
        self._TimeSteppingController = TimeSteppingController
        
        file = namelist['testbed']['input_filepath']
        data = nc.Dataset(file, 'r')
        forcing_data = data.groups['forcing']
        lat = forcing_data.variables['latitude']
        self._f = 2.0 * parameters.OMEGA* np.sin(latitude * np.pi / 180.0 )
        zl = self._Grid.z_local
      

        # Read in the data, we want to 
        forcing_z = forcing_data.variables['z'][:]
        self._forcing_times =forcing_data.variables['time'][:]
        raw_ug = forcing_data.variables['u_geostrophic'][:,:]
        raw_vg = forcing_data.variables['v_geostrophic'][:,:]
        raw_subsidence = forcing_data.variables['subsidence'][:,:]
        raw_adv_qt = forcing_data.variables['qt_advection'][:,:]
        raw_adv_theta = forcing_data.variables['theta_advection'][:,:]

        data.close()

        self._ug = interpolate.interp1d(forcing_z, raw_ug, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        self._vg = interpolate.interp1d(forcing_z, raw_vg, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        self._subsidence = interpolate.interp1d(forcing_z, raw_subsidence, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        self._adv_qt = interpolate.interp1d(forcing_z, raw_adv_qt, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        self._adv_theta = interpolate.interp1d(forcing_z, raw_adv_theta, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)

        return
        
    def update(self):
        current_time = self._TimeSteppingController.time()

        # interpolate in time
        ug = interpolate.interp1d(self._forcing_times,self._ug, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        vg = interpolate.interp1d(self._forcing_times,self._vg, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        subsidence = interpolate.interp1d(self._forcing_times,self._subsidence, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        adv_qt = interpolate.interp1d(self._forcing_times,self._adv_qt, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        adv_theta = interpolate.interp1d(self._forcing_times,self._adv_theta, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
       
        exner = self._Ref.exner

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')


        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')
        qvt = self._ScalarState.get_tend('qv')

        st += (self._adv_theta * exner)[np.newaxis, np.newaxis, :]

        Forcing_impl.large_scale_pgf(ug, vg, self._f ,u, v, self._Ref.u0, self._Ref.v0, ut, vt)

        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2],s, st)
        Forcing_impl.apply_subsidence(subsidence, self._Grid.dxi[2],qv, qvt)
        
        return








        



