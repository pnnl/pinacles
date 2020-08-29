import numpy as np
import numba
from mpi4py import MPI
from Columbia import Surface, Surface_impl, Forcing_impl, Forcing
from Columbia import UtilitiesParallel
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
        self._tflx = np.zeros_like(self._windspeed_sfc)

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

        #Compute the surface temperature flux
        self._tflx[:,:] =  self._theta_flux * exner_edge[nh[2]-1]

        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._taux_sfc, ut)
        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tauy_sfc, vt)
        Surface_impl.iles_surface_flux_application(25.0, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tflx, st)


        return


    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp['timeseries']

        #Add surface windspeed
        timeseries_grp.createVariable('wind_horizontal', np.double, dimensions=('time',))

        # Add surface stresses
        timeseries_grp.createVariable('ustar', np.double, dimensions=('time',))
        timeseries_grp.createVariable('taux', np.double, dimensions=('time',))
        timeseries_grp.createVariable('tauy', np.double, dimensions=('time',))

        #Add thermodynamic fluxes
        timeseries_grp.createVariable('tflx', np.double, dimensions=('time',))
        timeseries_grp.createVariable('shf', np.double, dimensions=('time',))


        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        windspeed = np.sum(self._windspeed_sfc[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1]])/npts
        windspeed = UtilitiesParallel.ScalarAllReduce(windspeed)

        ustar = np.sum(self._ustar_sfc[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1]])/npts
        ustar = UtilitiesParallel.ScalarAllReduce(ustar)

        taux = np.sum(self._taux_sfc[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1]])/npts
        taux = UtilitiesParallel.ScalarAllReduce(taux)

        tauy = np.sum(self._taux_sfc[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1]])/npts
        tauy = UtilitiesParallel.ScalarAllReduce(tauy)

        tflx = np.sum(self._tflx[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1]])/npts
        tflx = UtilitiesParallel.ScalarAllReduce(tflx)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp['timeseries']

            timeseries_grp['wind_horizontal'][-1] = windspeed
            timeseries_grp['ustar'][-1] = ustar
            timeseries_grp['taux'][-1] = taux
            timeseries_grp['tauy'][-1] = tauy
            timeseries_grp['tflx'][-1] = tflx
            timeseries_grp['shf'][-1] = tflx * parameters.CPD
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