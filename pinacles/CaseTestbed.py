import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import Surface, Surface_impl, Forcing, Forcing_impl
from pinacles import parameters
from scipy import interpolate
from pinacles import UtilitiesParallel


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
        self._forcing_times = surface_data.variables['times'][:]
        self._forcing_shf = surface_data.variables['sensible_heat_flux'][:]
        self._forcing_lhf = surface_data.variables['latent_heat_flux'][:]
        try:
            self._forcing_skintemp = surface_data.variables['skin_temperature'][:]
        except:
            self._forcing_skintemp = surface_data.variables['surface_temperature'][:]
            UtilitiesParallel.print_root('\t \t surface skin temp inferred from LW radiative fluxes')
        self._forcing_ustar = surface_data.variables['friction_velocity'][:]
        # Read off other variables needed for radiation..?

        data.close()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)

        # Use these to store the fluxes for output
        self._shf = self._forcing_shf[0]
        self._lhf = self._forcing_lhf[0]
        self._ustar = self._forcing_ustar[0]
        return

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp['timeseries']

        #Add thermodynamic fluxes
        timeseries_grp.createVariable('shf', np.double, dimensions=('time',))
        timeseries_grp.createVariable('lhf', np.double, dimensions=('time',))
        timeseries_grp.createVariable('ustar', np.double, dimensions=('time',))
        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp['timeseries']

            timeseries_grp['shf'][-1] = self._shf
            timeseries_grp['lhf'][-1] = self._lhf

            timeseries_grp['ustar'][-1] = self._ustar

        return

    def update(self):
        current_time = self._TimeSteppingController.time

        # Interpolate to the current time
        shf_interp = interpolate.interp1d(self._forcing_times, self._forcing_shf,fill_value='extrapolate', assume_sorted=True )(current_time)
        lhf_interp = interpolate.interp1d(self._forcing_times, self._forcing_lhf,fill_value='extrapolate', assume_sorted=True )(current_time)
        ustar_interp = interpolate.interp1d(self._forcing_times, self._forcing_ustar,fill_value='extrapolate', assume_sorted=True )(current_time)
        self.T_surface = interpolate.interp1d(self._forcing_times, self._forcing_skintemp, fill_value='extrapolate', assume_sorted=True )(current_time)
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

        # Store the surface fluxes for output
        self._shf = shf_interp
        self._lhf = lhf_interp
        self._ustar = ustar_interp

        return



'''
Note regarding set-up of original LASSO cases (e.g. HISCALE tested case)
These simulations use the initial sounding of horizontal winds as the geostrophic winds.
Here, I require the proper specification of geostrophic winds to be made in the input file 
so that the source code remains tidy.
'''

class ForcingTestbed(Forcing.ForcingBase):
    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):

        Forcing.ForcingBase.__init__(self, namelist, Grid,
        Ref, VelocityState, ScalarState, DiagnosticState)
        self._TimeSteppingController = TimeSteppingController

        file = namelist['testbed']['input_filepath']
        self._momentum_forcing_method = namelist['testbed']['momentum_forcing']
        # Options: relaxation, geostrophic, none


        data = nc.Dataset(file, 'r')
        forcing_data = data.groups['forcing']
        lat = forcing_data.variables['latitude'][0]
        self._f = 2.0 * parameters.OMEGA* np.sin(lat * np.pi / 180.0 )
        zl = self._Grid.z_local

        # Read in the data, we want to
        forcing_z = forcing_data.variables['z'][:]
        self._forcing_times =forcing_data.variables['times'][:]
        if self._momentum_forcing_method == 'geostrophic':
            raw_ug = forcing_data.variables['u_geostrophic'][:,:]
            raw_vg = forcing_data.variables['v_geostrophic'][:,:]
        if self._momentum_forcing_method == 'relaxation':
            raw_ur = forcing_data.variables['u_relaxation'][:,:]
            raw_vr = forcing_data.variables['v_relaxation'][:,:]

        
        raw_adv_qt = forcing_data.variables['qt_advection'][:,:]
        # take temperature over theta if we have both
        try:
            raw_adv_temperature= forcing_data.variables['temperature_advection'][:,:]
            self._use_temperature_advection = True
        except:
            raw_adv_theta = forcing_data.variables['theta_advection'][:,:]
            self._use_temperature_advection = False
        
        # take given vertical advection tendencies over subsidence if we have both
        # But we still need to get the subsidence 
        raw_subsidence = forcing_data.variables['subsidence'][:,:]
        try:
            raw_vtend_qt = forcing_data.variables['qt_vertical_tendency'][:,:]
            # I am omitting the logic here to get vertical advection of theta
            raw_vtend_temperature =  forcing_data.variables['temperature_vertical_tendency'][:,:]
            self._use_vertical_tendency = True
        
        except:
            
            self._use_vertical_tendency = False

        data.close()

        if self._momentum_forcing_method == 'geostrophic':
            self._ug = interpolate.interp1d(forcing_z, raw_ug, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
            self._vg = interpolate.interp1d(forcing_z, raw_vg, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        if self._momentum_forcing_method == 'relaxation':
            self._ur = interpolate.interp1d(forcing_z, raw_ur, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
            self._vr = interpolate.interp1d(forcing_z, raw_vr, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)

        self._subsidence = interpolate.interp1d(forcing_z, raw_subsidence, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        self._adv_qt = interpolate.interp1d(forcing_z, raw_adv_qt, axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        if self._use_temperature_advection:
            self._adv_temperature = interpolate.interp1d(forcing_z, raw_adv_temperature, 
                                                            axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        else:
            self._adv_theta = interpolate.interp1d(forcing_z, raw_adv_theta, 
                                                    axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        if self._use_vertical_tendency:
            self._vtend_temperature = interpolate.interp1d(forcing_z, raw_vtend_temperature, 
                                                            axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
            
            self._vtend_qt = interpolate.interp1d(forcing_z, raw_vtend_qt, 
                                                            axis=1,fill_value='extrapolate',assume_sorted=True)(zl)
        
        z_top = self._Grid.l[2]
        _depth = namelist['damping']['depth']
        znudge = z_top -_depth
        self._compute_relaxation_coefficient(znudge,3600.0)

        


        return

    def update(self):
        current_time = self._TimeSteppingController.time

        # interpolate in time
        if self._momentum_forcing_method == 'geostrophic':
            current_ug = interpolate.interp1d(self._forcing_times,self._ug, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
            current_vg = interpolate.interp1d(self._forcing_times,self._vg, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        if self._momentum_forcing_method == 'relaxation':
            current_ur = interpolate.interp1d(self._forcing_times,self._ur, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
            current_vr = interpolate.interp1d(self._forcing_times,self._vr, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        current_subsidence = interpolate.interp1d(self._forcing_times,self._subsidence, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        current_adv_qt = interpolate.interp1d(self._forcing_times,self._adv_qt, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        if self._use_temperature_advection:
            current_adv_temperature = interpolate.interp1d(self._forcing_times,self._adv_temperature, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        else:
            current_adv_theta = interpolate.interp1d(self._forcing_times,self._adv_theta, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
        if self._use_vertical_tendency:
            current_vtend_temperature = interpolate.interp1d(self._forcing_times,self._vtend_temperature, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)
            current_vtend_qt = interpolate.interp1d(self._forcing_times,self._vtend_qt, axis=0,fill_value='extrapolate',assume_sorted=True)(current_time)


        exner = self._Ref.exner

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')


        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        st = self._ScalarState.get_tend('s')
        qvt = self._ScalarState.get_tend('qv')

        if self._use_temperature_advection:
            st += (current_adv_temperature)[np.newaxis, np.newaxis, :]
        else:
            st += (current_adv_theta * exner)[np.newaxis, np.newaxis, :]
        qvt += (current_adv_qt)[np.newaxis, np.newaxis, :]
        if self._momentum_forcing_method == 'geostrophic':
            Forcing_impl.large_scale_pgf(current_ug, current_vg, self._f ,u, v, self._Ref.u0, self._Ref.v0, ut, vt)
        if self._momentum_forcing_method == 'relaxation':
            Forcing_impl.relax_velocities(current_ur, current_vr,  u, v, self._Ref.u0, self._Ref.v0, ut, vt, self._relaxation_coefficient)
        
        if self._use_vertical_tendency:
             st += (current_vtend_temperature + current_subsidence * parameters.G * parameters.ICPD)[np.newaxis, np.newaxis, :]
             qvt += (current_vtend_qt)[np.newaxis, np.newaxis, :]
        else:
            Forcing_impl.apply_subsidence(current_subsidence, self._Grid.dxi[2],s, st)
            Forcing_impl.apply_subsidence(current_subsidence, self._Grid.dxi[2],qv, qvt)

        return


    def _compute_relaxation_coefficient(self,znudge,timescale):

        self._relaxation_coefficient = np.zeros(self._Grid.ngrid[2], dtype=np.double)
       
        z = self._Grid.z_global
        
       
        for k in range(self._Grid.ngrid[2]):
            self._relaxation_coefficient[k] = (1.0/timescale)
            if z[k] < znudge:
                self._relaxation_coefficient[k] *=  np.sin((np.pi / 2.0) * (1.0 - (znudge - z[k]) / znudge))**2.0
               

        return