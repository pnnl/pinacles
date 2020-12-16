from pinacles.Microphysics import MicrophysicsBase, water_path, water_fraction, water_fraction_profile
from pinacles.wrf_physics import kessler
from pinacles import UtilitiesParallel
from pinacles.WRFUtil import to_wrf_order, wrf_tend_to_our_tend, wrf_theta_tend_to_our_tend, to_our_order
from pinacles import parameters
from mpi4py import MPI
import numpy as np
import numba

@numba.njit
def compute_qvs(temp, pressure):
    ep2 = 287./461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65 
    svpt0 = 273.15
    es        = 1000.*svp1*np.exp(svp2*(temp-svpt0)/(temp-svp3))
    qvs       = ep2*es/(pressure-es)

    return qvs

@numba.njit
def compute_rh(qv, temp, pressure):
    return qv/compute_qvs(temp, pressure)


class MicroKessler(MicrophysicsBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

        MicrophysicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)

        self._ScalarState.add_variable('qv', long_name = 'water vapor mixing ratio', units='kg kg^{-1}', latex_name = 'q_v')
        self._ScalarState.add_variable('qc', long_name = 'cloud water mixing ratio', units='kg kg^{-1}', latex_name = 'q_c')
        self._ScalarState.add_variable('qr', long_name = 'rain water mixing ratio', units='kg kg^{-1}', latex_name = 'q_{r}')

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        self._wrf_dims = (self._our_dims[0] -2*nhalo[0], self._our_dims[2]-2*nhalo[2], self._our_dims[1]-2*nhalo[1])


        self._RAINNC = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        self._RAINNCV =  np.zeros_like(self._RAINNC)

        self._rain_rate = 0.0
        return

    def update(self):

        #Get variables from the model state
        T = self._DiagnosticState.get_field('T')
        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')
        qc = self._ScalarState.get_field('qc')
        qr = self._ScalarState.get_field('qr')

        exner = self._Ref.exner

        #Build arrays from reference state make sure these are properly Fortran/WRF
        #ordered
        nhalo = self._Grid.n_halo

        #Some of the memory allocation could be done at init (TODO)
        rho_wrf = np.empty(self._wrf_dims, dtype=np.double, order='F')
        exner_wrf = np.empty_like(rho_wrf)
        T_wrf = np.empty_like(rho_wrf)
        qv_wrf = np.empty_like(rho_wrf)
        qc_wrf = np.empty_like(rho_wrf)
        qr_wrf = np.empty_like(rho_wrf)

        dz_wrf = np.empty_like(rho_wrf)
        z = np.empty_like(rho_wrf)

        dz_wrf.fill(self._Grid.dx[2])
        z[:,:,:] = self._Grid.z_global[np.newaxis,nhalo[2]:-nhalo[2],np.newaxis]
        rho_wrf[:,:,:] = self._Ref.rho0[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        exner_wrf[:,:,:] = exner[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]


        #TODO Need to fill these
        dt = self._TimeSteppingController.dt
        xlv = 2.5E6
        cp = 1004.0
        ep2 = 287./461.6
        svp1 = 0.6112
        svp2 = 17.67
        svp3 = 29.65
        svpT0 = 273.15
        rhow = 1000.0

        ids = 1; jds = 1; kds = 1
        iide = 1; jde = 1; kde = 1
        ims=1; jms = 1; kms = 1
        ime=self._wrf_dims[0]; jme=self._wrf_dims[2]; kme=self._wrf_dims[1]
        its=1; jts=1; kts=1
        ite=ime; jte=jme; kte=kme

        to_wrf_order(nhalo, T/self._Ref.exner[np.newaxis,np.newaxis,:], T_wrf)

        to_wrf_order(nhalo, qv, qv_wrf)
        to_wrf_order(nhalo, qc, qc_wrf)
        to_wrf_order(nhalo, qr, qr_wrf)

        rain_accum_old = np.sum(self._RAINNC)
        kessler.module_mp_kessler.kessler(T_wrf, qv_wrf, qc_wrf, qr_wrf, rho_wrf, exner_wrf,
            dt, z, xlv, cp,
            ep2, svp1, svp2, svp3, svpT0, rhow,
            dz_wrf,
            self._RAINNC, self._RAINNCV,
            ids,iide, jds,jde, kds,kde,
            ims,ime, jms,jme, kms,kme,
            its,ite, jts,jte, kts,kte)

        to_our_order(nhalo, qv_wrf, qv)
        to_our_order(nhalo, qc_wrf, qc)
        to_our_order(nhalo, qr_wrf, qr)

        #Update the energy (TODO Move this to numba)
        T_wrf *= self._Ref.exner[np.newaxis,nhalo[2]:-nhalo[2],np.newaxis]
        s_wrf = T_wrf + (parameters.G*z- parameters.LV*(qc_wrf + qr_wrf))*parameters.ICPD
        to_our_order(nhalo, s_wrf, s)

        self._rain_rate = (np.sum(self._RAINNC) - rain_accum_old)/dt

        return

    def io_initialize(self, nc_grp):
        timeseries_grp = nc_grp['timeseries']
        profiles_grp = nc_grp['profiles']
        
        #Cloud fractions 
        v = timeseries_grp.createVariable('CF', np.double, dimensions=('time',))
        v.long_name = 'Cloud Fraction'
        v.standard_name = 'CF'
        v.units = ''

        v = timeseries_grp.createVariable('RF', np.double, dimensions=('time',))
        v.long_name = 'Rain Fraction'
        v.standard_name = 'RF'
        v.units = ''

        v = timeseries_grp.createVariable('LWP', np.double, dimensions=('time',))
        v.long_name = 'Liquid Water Path'
        v.standard_name = 'LWP'
        v.units = 'kg/m^2'

        v = timeseries_grp.createVariable('RWP', np.double, dimensions=('time',))
        v.long_name = 'Rain Water Path'
        v.standard_name = 'RWP'
        v.units = 'kg/m^2'

        v = timeseries_grp.createVariable('VWP', np.double, dimensions=('time',))
        v.long_name = 'Water Vapor Path'
        v.standard_name = 'VWP'
        v.units = 'kg/m^2'

        #Precipitation
        v = timeseries_grp.createVariable('RAINNC', np.double, dimensions=('time',))
        v.long_name = 'accumulated surface precip'
        v.units = 'mm'
        v.latex_name = 'rainnc'

        timeseries_grp.createVariable('RAINNCV', np.double, dimensions=('time',))
        v.long_name = 'one time step accumulated surface precip'
        v.units = 'mm'
        v.latex_name = 'rainncv'

        timeseries_grp.createVariable('rain_rate', np.double, dimensions=('time',))

        #Now add cloud fraction and rain fraction profiles
        v = profiles_grp.createVariable('CF', np.double, dimensions=('time', 'z',))
        v.long_name = 'Cloud Fraction'
        v.standard_name = 'CF'
        v.units = ''

        profiles_grp.createVariable('RF', np.double, dimensions=('time', 'z',))
        v.long_name = 'Rain Fraction'
        v.standard_name = 'RF'
        v.units = ''

        return

    def io_update(self, nc_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()

        n_halo = self._Grid.n_halo
        dz = self._Grid.dx[2]
        rho = self._Ref.rho0
        npts = self._Grid.n[0] * self._Grid.n[1]

        qc = self._ScalarState.get_field('qc')
        qv = self._ScalarState.get_field('qv')
        qr = self._ScalarState.get_field('qr')

        #First compute liqud water path
        lwp = water_path(n_halo, dz, npts, rho, qc)
        lwp = UtilitiesParallel.ScalarAllReduce(lwp)

        rwp = water_path(n_halo, dz, npts, rho, qr)
        rwp = UtilitiesParallel.ScalarAllReduce(rwp)

        vwp = water_path(n_halo, dz, npts, rho, qv)
        vwp = UtilitiesParallel.ScalarAllReduce(vwp)

        #Compute cloud and rain fraction
        cf = water_fraction(n_halo, npts, qc, threshold=1e-5)
        cf = UtilitiesParallel.ScalarAllReduce(cf)

        cf_prof = water_fraction_profile(n_halo, npts, qc, threshold=1e-5)
        cf_prof = UtilitiesParallel.ScalarAllReduce(cf_prof)

        rf = water_fraction(n_halo, npts, qr)
        rf = UtilitiesParallel.ScalarAllReduce(rf)

        rf_prof = water_fraction_profile(n_halo, npts, qr, threshold=1e-5)
        rf_prof = UtilitiesParallel.ScalarAllReduce(rf_prof)

        rainnc = np.sum(self._RAINNC)/npts
        rainnc = UtilitiesParallel.ScalarAllReduce(rainnc)
        rainncv = np.sum(self._RAINNCV)/npts
        rainncv = UtilitiesParallel.ScalarAllReduce(rainncv)

        rr = UtilitiesParallel.ScalarAllReduce(self._rain_rate/npts)

        if my_rank == 0:
            timeseries_grp = nc_grp['timeseries']
            profiles_grp = nc_grp['profiles']

            timeseries_grp['CF'][-1] = cf
            timeseries_grp['RF'][-1] = rf
            timeseries_grp['LWP'][-1] = lwp
            timeseries_grp['RWP'][-1] = rwp
            timeseries_grp['VWP'][-1] = vwp

            timeseries_grp['RAINNC'][-1] = rainnc
            timeseries_grp['RAINNCV'][-1] = rainncv
            timeseries_grp['rain_rate'][-1] = rr

            profiles_grp['CF'][-1,:] = cf_prof[n_halo[2]:-n_halo[2]]
            profiles_grp['RF'][-1,:] = rf_prof[n_halo[2]:-n_halo[2]]

        return

    def get_qc(self):
        return self._ScalarState.get_field('qc') + self._ScalarState.get_field('qr')