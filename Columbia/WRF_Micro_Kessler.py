from Columbia.Microphysics import MicrophysicsBase, water_path, water_fraction
from Columbia.wrf_physics import kessler
from Columbia import UtilitiesParallel
from Columbia.WRFUtil import to_wrf_order, wrf_tend_to_our_tend, wrf_theta_tend_to_our_tend
from Columbia import parameters
from mpi4py import MPI
import numpy as np
import numba

@numba.njit
def compute_rh(qv, temp, pressure):
    ep2 = 287./461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15

    es        = 1000.*svp1*np.exp(svp2*(temp-svpt0)/(temp-svp3))
    qvs       = ep2*es/(pressure-es)
    return qv/qvs


class MicroKessler(MicrophysicsBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

        MicrophysicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)

        self._ScalarState.add_variable('qv')
        self._ScalarState.add_variable('qc')
        self._ScalarState.add_variable('qr')

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
        qv = self._ScalarState.get_field('qv')
        qc = self._ScalarState.get_field('qc')
        qr = self._ScalarState.get_field('qr')

        s_tend = self._ScalarState.get_tend('s')
        qv_tend = self._ScalarState.get_tend('qv')
        qc_tend = self._ScalarState.get_tend('qc')
        qr_tend = self._ScalarState.get_tend('qr')

        exner = self._Ref.exner

        #Build arrays from reference state make sure these are properly Fortran/WRF
        #ordered 
       # our_dims= self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
       # wrf_dims = (self._our_dims[0] -2*nhalo[0], self_our_dims[2]-2*nhalo[2], self._our_dims[1]-2*nhalo[1])
    
        #Some of the memory allocation could be done at init

        rho_wrf = np.empty(self._wrf_dims, dtype=np.double, order='F')
        exner_wrf = np.empty_like(rho_wrf) 
        T_wrf = np.empty_like(rho_wrf)
        qv_wrf = np.empty_like(rho_wrf)
        qc_wrf = np.empty_like(rho_wrf)
        qr_wrf = np.empty_like(rho_wrf)

        #RAINNC = np.empty((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        #RAINNCV = np.empty_like(RAINNC)

        dz_wrf = np.empty_like(rho_wrf)
        z = np.empty_like(rho_wrf)

        dz_wrf.fill(self._Grid.dxi[2])
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


        #print('Print 2!')
        to_wrf_order(nhalo, qv, qv_wrf)
        #print('Print 3!')
        to_wrf_order(nhalo, qc, qc_wrf)
        #print('Print 4!')
        to_wrf_order(nhalo, qr, qr_wrf)
        #print('Print 5!')
        #import pylab as plt
       # plt.figure(1)
       # plt.plot(qv_wrf[5,:,5])
        #plt.show()

        #Convert T to potential temperature (Flip order here)
        #T_wrf = T_wrf / exner[np.newaxis,::-1,np.newaxis]

        #print('Here 1')
        rain_accum_old = np.sum(self._RAINNC)
        kessler.module_mp_kessler.kessler(T_wrf, qv_wrf, qc_wrf, qr_wrf, rho_wrf, exner_wrf,
            dt, z, xlv, cp,
            ep2, svp1, svp2, svp3, svpT0, rhow,
            dz_wrf,
            self._RAINNC, self._RAINNCV,
            ids,iide, jds,jde, kds,kde,
            ims,ime, jms,jme, kms,kme,
            its,ite, jts,jte, kts,kte)

        self._rain_rate = (np.sum(self._RAINNC) - rain_accum_old)/dt
        #print('Here 2')

        wrf_theta_tend_to_our_tend(nhalo, dt, exner, T_wrf, T, s_tend)
        #print(np.amax(s_tend - s_b4))
        wrf_tend_to_our_tend(nhalo, dt, qv_wrf, qv, qv_tend)
        wrf_tend_to_our_tend(nhalo, dt, qc_wrf, qc, qc_tend)
        #print('qc', np.amax(qc), np.amax(qc_tend))
        wrf_tend_to_our_tend(nhalo,dt, qr_wrf, qr, qr_tend)
        #print('qr', np.amax(qr), np.amax(qr_tend))

        #Add in tendencies
        s_tend -= parameters.LV*parameters.ICPD * np.add(qc_tend, qr_tend)

        #pressure = np.zeros_like(qv) + self._Ref.p0[np.newaxis, np.newaxis,:]
        #rh = compute_rh(qv, T, pressure)
        #print('RH: ', np.amax(rh))

        #plt.plot(T[8,8,nhalo[0]:-nhalo[0]]/exner[nhalo[0]:-nhalo[0]])
        #plt.plot(T_wrf[5,:,5])
        #plt.show()
        #import sys; sys.exit()

        return

    def io_initialize(self, nc_grp):
        timeseries_grp = nc_grp['timeseries']

        timeseries_grp.createVariable('CF', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RF', np.double, dimensions=('time',))
        timeseries_grp.createVariable('LWP', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RWP', np.double, dimensions=('time',))
        timeseries_grp.createVariable('VWP', np.double, dimensions=('time',))


        timeseries_grp.createVariable('RAINNC', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RAINNCV', np.double, dimensions=('time',))
        timeseries_grp.createVariable('rain_rate', np.double, dimensions=('time',))
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

        rf = water_fraction(n_halo, npts, qr)
        rf = UtilitiesParallel.ScalarAllReduce(rf)


        rainnc = np.sum(self._RAINNC)/npts
        rainnc = UtilitiesParallel.ScalarAllReduce(rainnc)
        rainncv = np.sum(self._RAINNCV)/npts
        rainncv = UtilitiesParallel.ScalarAllReduce(rainncv)

        rr = UtilitiesParallel.ScalarAllReduce(self._rain_rate/npts)

        if my_rank == 0:
            timeseries_grp = nc_grp['timeseries']

            timeseries_grp['CF'][-1] = cf
            timeseries_grp['RF'][-1] = rf
            timeseries_grp['LWP'][-1] = lwp
            timeseries_grp['RWP'][-1] = rwp
            timeseries_grp['VWP'][-1] = vwp

            timeseries_grp['RAINNC'][-1] = rainnc
            timeseries_grp['RAINNCV'][-1] = rainncv
            timeseries_grp['rain_rate'][-1] = rr

        return