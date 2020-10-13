from pinacles.Microphysics import MicrophysicsBase, water_path, water_fraction, water_fraction_profile
from pinacles.wrf_physics import kessler
from pinacles.wrf_physics import p3
from pinacles import UtilitiesParallel
from pinacles.WRFUtil import to_wrf_order, wrf_tend_to_our_tend, wrf_theta_tend_to_our_tend, to_our_order
from pinacles import parameters
from mpi4py import MPI
import numba
import numpy as np

class MicroP3(MicrophysicsBase):
    def __init__(self, namelist, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

            MicrophysicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)

            try:
                lookup_file_dir = namelist['microphysics']['lookup_file_dir']
            except:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('Need to provide location of P3 lookup files in namelist')
                
            nCat = 1
            stat = 1
            abort_on_err = False
            model = 'PINACLES'


            p3.module_mp_p3.p3_init(lookup_file_dir, nCat, model, stat, abort_on_err)

            #Allocate microphysical/thermodyamic variables
            self._ScalarState.add_variable('qv')
            self._ScalarState.add_variable('qc')
            self._ScalarState.add_variable('qr')
            self._ScalarState.add_variable('qnr')
            self._ScalarState.add_variable('qi1')
            self._ScalarState.add_variable('qni1')
            self._ScalarState.add_variable('qir1')
            self._ScalarState.add_variable('qib1')

            self._DiagnosticState.add_variable('reflectivity')

            nhalo = self._Grid.n_halo
            self._our_dims = self._Grid.ngrid_local
            nhalo = self._Grid.n_halo
            self._wrf_dims = (self._our_dims[0] -2*nhalo[0], self._our_dims[2]-2*nhalo[2], self._our_dims[1]-2*nhalo[1])
            
         

            self._itimestep = 0
            self._RAINNC = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), order='F', dtype=np.float32)
            self._SR = np.zeros_like(self._RAINNC)
            self._RAINNCV =  np.zeros_like(self._RAINNC)
            self._SNOWNC = np.zeros_like(self._RAINNC)
            self._SNOWNCV =  np.zeros_like(self._RAINNC)



            return

    def update(self):

        #Get variables from the model state
        T = self._DiagnosticState.get_field('T')
        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')
        qc = self._ScalarState.get_field('qc')
        qr = self._ScalarState.get_field('qr')
        qnr = self._ScalarState.get_field('qnr')
        qi1 = self._ScalarState.get_field('qi1')
        qni1 = self._ScalarState.get_field('qni1')
        qir1 = self._ScalarState.get_field('qir1')
        qib1 = self._ScalarState.get_field('qib1')

        w = self._VelocityState.get_field('w')

        reflectivity = self._DiagnosticState.get_field('reflectivity')

        exner = self._Ref.exner
        p0 = self._Ref.p0
        nhalo = self._Grid.n_halo

        rho_wrf = np.empty(self._wrf_dims, order='F',dtype=np.float32)
        exner_wrf = np.empty_like(rho_wrf)
        p0_wrf = np.empty_like(rho_wrf)
        T_wrf = np.empty_like(rho_wrf)
        qv_wrf = np.empty_like(rho_wrf)
        qc_wrf = np.empty_like(rho_wrf)
        qr_wrf = np.empty_like(rho_wrf)
        qnr_wrf = np.empty_like(rho_wrf)
        nc_wrf = np.empty_like(rho_wrf)
        qi1_wrf = np.empty_like(rho_wrf)
        qni1_wrf = np.empty_like(rho_wrf)
        qir1_wrf = np.empty_like(rho_wrf)
        qib1_wrf = np.empty_like(rho_wrf)
        w_wrf = np.empty_like(rho_wrf)
        th_old = np.empty_like(rho_wrf)
        qv_old = np.empty_like(rho_wrf)


        reflectivity_wrf = np.empty_like(rho_wrf)
        diag_effc = np.empty_like(rho_wrf)
        diag_effi = np.empty_like(rho_wrf)
        diag_vmi = np.empty_like(rho_wrf)
        diag_di = np.empty_like(rho_wrf)
        diag_rhopo = np.empty_like(rho_wrf)

        dz_wrf = np.empty_like(rho_wrf)
        z = np.empty_like(rho_wrf)


        dz_wrf.fill(self._Grid.dx[2])
        z[:,:,:] = self._Grid.z_global[np.newaxis,nhalo[2]:-nhalo[2],np.newaxis]
        rho_wrf[:,:,:] = self._Ref.rho0[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        exner_wrf[:,:,:] = exner[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        p0_wrf[:,:,:] = p0[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]

        dt = self._TimeSteppingController.dt

        ids = 1; jds = 1; kds = 1
        ide = 1; jde = 1; kde = 1
        ims=1; jms = 1; kms = 1
        ime=self._wrf_dims[0]; jme=self._wrf_dims[2]; kme=self._wrf_dims[1]
        its=1; jts=1; kts=1
        ite=ime; jte=jme; kte=kme

        #Reorder arrays
        to_wrf_order(nhalo, T/self._Ref.exner[np.newaxis,np.newaxis,:], T_wrf)
        to_wrf_order(nhalo, qv, qv_wrf)
        to_wrf_order(nhalo, qc, qc_wrf)
        to_wrf_order(nhalo, qr, qr_wrf)
        to_wrf_order(nhalo, w, w_wrf)
        to_wrf_order(nhalo, qnr, qnr_wrf)
        to_wrf_order(nhalo, qi1, qi1_wrf)
        to_wrf_order(nhalo, qni1, qni1_wrf)
        to_wrf_order(nhalo, qir1, qir1_wrf)
        to_wrf_order(nhalo, qib1, qib1_wrf)

        for i in range(self._wrf_dims[0]):
            for j in range(self._wrf_dims[1]):
                for k in range(self._wrf_dims[2]):
                    th_old[i,j,k] = T_wrf[i,j,k]
                    qv_old[i,j,k] = qv_wrf[i,j,k]




        n_iceCat = 1
        nc_wrf[:,:,:] = 100.0e6 # 70e6
        #T_wrf,qv_wrf,qc_wrf,qr_wrf,qnr_wrf)
        p3.module_mp_p3.mp_p3_wrapper_wrf(T_wrf,qv_wrf,qc_wrf,qr_wrf,qnr_wrf,
                                th_old, qv_old,
                                exner_wrf, p0_wrf, dz_wrf, w_wrf, dt, self._itimestep,
                                self._RAINNC,self._RAINNCV,self._SR,self._SNOWNC,self._SNOWNCV,n_iceCat,
                                ids, ide, jds, jde, kds, kde ,
                                ims, ime, jms, jme, kms, kme ,
                                its, ite, jts, jte, kts, kte ,
                                reflectivity_wrf,diag_effc,diag_effi,
                                diag_vmi,diag_di,diag_rhopo,
                                qi1_wrf,qni1_wrf,qir1_wrf,qib1_wrf,nc_wrf)


        #Update prognosed fields
        to_our_order(nhalo, qv_wrf, qv)
        to_our_order(nhalo, qc_wrf, qc)
        to_our_order(nhalo, qr_wrf, qr)
        to_our_order(nhalo, qnr_wrf, qnr)
        to_our_order(nhalo, qi1_wrf, qi1)
        to_our_order(nhalo, qni1_wrf, qni1)
        #to_our_order(nhalo, nc_wrf, nc)


        #Update the energys (TODO Move this to numba)
        T_wrf *= self._Ref.exner[np.newaxis,nhalo[2]:-nhalo[2],np.newaxis]
        s_wrf = T_wrf + (parameters.G*z- parameters.LV*(qc_wrf + qr_wrf) - parameters.LS*(qi1_wrf))*parameters.ICPD
        to_our_order(nhalo, s_wrf, s)


        #wrf_theta_tend_to_our_tend(nhalo, dt, exner, T_wrf, T, s_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qv_wrf, qv, qv_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qc_wrf, qc, qc_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qr_wrf, qr, qr_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qnr_wrf, qnr, qnr_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qi1_wrf, qi1, qi1_tend)
        #wrf_tend_to_our_tend(nhalo, dt, qni1_wrf, qni1, qni1_tend)
        #wrf_tend_to_our_tend(nhalo, dt, nc_wrf, nc, nc_tend)
        to_our_order(nhalo, reflectivity_wrf, reflectivity)

        #Add in tendencies
        #s_tend -= parameters.LV*parameters.ICPD * np.add(qc_tend, qr_tend)

        self._itimestep += 1
        return

    def io_initialize(self, nc_grp):
        timeseries_grp = nc_grp['timeseries']
        profiles_grp = nc_grp['profiles']

        timeseries_grp.createVariable('CF', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RF', np.double, dimensions=('time',))
        timeseries_grp.createVariable('LWP', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RWP', np.double, dimensions=('time',))
        timeseries_grp.createVariable('VWP', np.double, dimensions=('time',))

        timeseries_grp.createVariable('RAINNC', np.double, dimensions=('time',))
        timeseries_grp.createVariable('RAINNCV', np.double, dimensions=('time',))

        #Now add cloud fraction and rain fraction profiles
        profiles_grp.createVariable('CF', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('RF', np.double, dimensions=('time', 'z',))

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

        rainnc = np.sum(self._RAINNC)/npts
        rainnc = UtilitiesParallel.ScalarAllReduce(rainnc)

        rf_prof = water_fraction_profile(n_halo, npts, qr, threshold=1e-5)
        rf_prof = UtilitiesParallel.ScalarAllReduce(rf_prof)

        rainncv = np.sum(self._RAINNCV)/npts
        rainncv = UtilitiesParallel.ScalarAllReduce(rainncv)

        if my_rank == 0:
            timeseries_grp = nc_grp['timeseries']

            timeseries_grp['CF'][-1] = cf
            timeseries_grp['RF'][-1] = rf
            timeseries_grp['LWP'][-1] = lwp
            timeseries_grp['RWP'][-1] = rwp
            timeseries_grp['VWP'][-1] = vwp

            timeseries_grp['RAINNC'][-1] = rainnc
            timeseries_grp['RAINNCV'][-1] = rainncv

        return
    

    def get_qc(self):
        return self._ScalarState.get_field('qc') + self._ScalarState.get_field('qr')
