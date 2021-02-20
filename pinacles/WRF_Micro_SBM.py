import numpy as np
from pinacles import parameters
from mpi4py import MPI
from pinacles import UtilitiesParallel
import time
from pinacles.Microphysics import MicrophysicsBase, water_path, water_fraction, water_fraction_profile
from pinacles.wrf_physics import module_mp_fast_sbm_warm
from pinacles.WRFUtil import to_wrf_order, to_wrf_order_4d, to_our_order_4d, to_our_order
from pinacles.WRFUtil import to_wrf_order_halo, to_wrf_order_4d_halo, to_our_order_4d_halo, to_our_order_halo
module_mp_fast_sbm = module_mp_fast_sbm_warm
class MicroSBM(MicrophysicsBase):

    def __init__(self,namelist, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):
        MicrophysicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)


        self._ScalarState.add_variable('qv')
        #TODO for now adding these as prognostic variables but probably unnecessary
        self._list_of_ScalarStatevars = ['qc', 'qr',
            'qnc', 'qnr', 'qna','qna_nucl']

        long_names = {'qc': 'cloud-water mixing ratio',
                      'qr': 'rain-water mixing ratio',
                      'qnc': 'cloud droplet number concentraiton',
                      'qnr': 'rain drop number concentration',
                      'qna': 'aerosol number concentration',
                      'qna_nucl': 'regeneration aerosol number concentration'}

        units = {'qc':'kg kg^-1',
                 'qr':  'kg kg^-1',
                 'qnc': 'kg^-1',
                 'qnr': 'kg^-1',
                 'qna': 'kg^-1',
                 'qna_nucl': 'kg^-1'}


        for var in self._list_of_ScalarStatevars:
            self._ScalarState.add_variable(var, units='kg kg^-1', latex_name=var, long_name=long_names[var])

        #Add new diag fields
        self._io_fields = ['MA', 'LH_rate', 'CE_rate', 'CldNucl_rate', 'difful_tend', 'diffur_tend',
            'tempdiffl', 'saturation', 'n_reg_ccn']

        long_names = {'MA': '', 
                      'LH_rate':'Latent heat rate',
                      'CE_rate':'Condensation / evaporation rate',
                      'CldNucl_rate':'Cloud nucleation rate',
                      'difful_tend':'liquid mass change rate due to droplet diffusional growth',
                      'diffur_tend':'rain mass change rate due to droplet diffusional growth',
                      'tempdiffl':'latent heat rate due to droplet diffusional growth',
                      'saturation':'Saturaiton Ratio',
                      'n_reg_ccn':'Aerosol Regeneration Rate'}

        units = {'MA': '',
                      'LH_rate':'K s^{-1}',
                      'CE_rate':'kg kg^{-1} s^{-1}',
                      'CldNucl_rate':'kg kg^{-1} s^{-1}',
                      'difful_tend':'kg kg^{-1} s^{-1}',
                      'diffur_tend':'kg kg^{-1} s^{-1}',
                      'tempdiffl':'K s^{-1}',
                      'saturation':'',
                      'n_reg_ccn':''}

        for var in self._io_fields:
            self._DiagnosticState.add_variable(var, latex_name=var, long_name=long_names[var])

        self._sbm_in_iofield = ['T_sbm_in', 'qv_sbm_in', 'qc_sbm_in', 'qr_sbm_in',
            'qnc_sbm_in', 'qnr_sbm_in', 'qna_sbm_in', 'qna_nucl_sbm_in']

        long_names = {'T_sbm_in': 'temperature going into sbm',
                      'qv_sbm_in': 'water vapor mixing ratio going into sbm',
                      'qc_sbm_in': 'cloud-water mixing ratio going into sbm',
                      'qr_sbm_in': 'rain-water mixing ratio going into sbm',
                      'qnc_sbm_in': 'cloud droplet number concentraiton going into sbm',
                      'qnr_sbm_in': 'rain drop number concentration going into sbm',
                      'qna_sbm_in': 'aerosol number concentration going into sbm',
                      'qna_nucl_sbm_in': 'regeneration aerosol number concentration going into sbm'}

        units = {'T_sbm_in': 'K',
                      'qv_sbm_in': 'kg/kg',
                      'qc_sbm_in': 'kg/kg',
                      'qr_sbm_in': 'kg/kg',
                      'qnc_sbm_in': 'kg/kg',
                      'qnr_sbm_in': 'kg/kg',
                      'qna_sbm_in': 'kg/kg',
                      'qna_nucl_sbm_in': 'kg/kg'}

        for var in self._sbm_in_iofield :
            self._DiagnosticState.add_variable(var, latex_name=var, long_name=long_names[var])

        self._bin_start = self._ScalarState.nvars
        self._qc_start = self._ScalarState.nvars 
        for i in range(1,34):
            name = "ff1i" + str(i)
            self._ScalarState.add_variable(name, units='kg kg^{-1}', long_name='liquid bin mass ' + str(i) )
        self._qc_end = self._ScalarState.nvars
        #Add aersol bins
        for i in range(1,34):
            name = 'ff8i' + str(i)
            self._ScalarState.add_variable(name, units='kg kg^{-1}', long_name='aerosol bin mass ' + str(i))

        for i in range(1,34):
            name = 'ff8in' + str(i)
            self._ScalarState.add_variable(name,units='kg kg^{-1}', long_name='(regeneration) aerosol bin mass ' + str(i))
        self._bin_end = self._ScalarState.nvars

        # Call add output container diagnostics to the container
        self.add_output_container_diags()

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        self._wrf_dims = (self._our_dims[0] -2*nhalo[0], self._our_dims[2]-2*nhalo[2], self._our_dims[1]-2*nhalo[1])
        #self._wrf_dims = (self._our_dims[0],self._our_dims[2], self._our_dims[1])

        self._th_old = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        self._qv_old = np.zeros(self._wrf_dims, order='F', dtype=np.double)

        self._RAINNC = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        self._RAINNCV =  np.zeros_like(self._RAINNC)

        self._rain_rate = 0.0

        self._itimestep = 1
        self._call_count = 0

        # Unless we find a valid aerosol data file, assume we want to use lognormal distributions
       
        ccncon1 = 90.0
        radius_mean1 = 0.03e-4
        sig1 = 1.28

        ccncon2 = 15
        radius_mean2 =  0.14e-4
        sig2 = 1.75

        ccncon3 = 0.0
        radius_mean3 = 0.31000e-04
        sig3 = 2.70000
        
        # Determine if an analytical aerosol size distribution is to be used, or a data file
        try:
            aero_in = namelist['microphysics']['aero_sbm']
            ccncon1 = aero_in['ccncon'][0]
            sig1 = aero_in['sig'][0]
            radius_mean1 = aero_in['rm'][0]

            ccncon2 = aero_in['ccncon'][1]
            sig2 = aero_in['sig'][1]
            radius_mean2 = aero_in['rm'][1]

            ccncon3 = aero_in['ccncon'][2]
            sig3 = aero_in['sig'][2]
            radius_mean3 = aero_in['rm'][2]
        except:
            UtilitiesParallel.print_root(' Analytical aerosol distribution parameters are not defined in namelist')
            UtilitiesParallel.print_root(' Default values will be used unless an aerosol file has been defined')

        try:
            aerosol_species = namelist['microphysics']['aerosol_species']
            #options are sea_salt, ammonium_bisulfate
            if not aerosol_species in ['sea_salt', 'ammonium_bisulfate']:
                aerosol_species = 'sea_salt'
                UtilitiesParallel.print_root('Warning: Unknown aerosol species, defaulting to sea_salt')
        except:
            aerosol_species = 'sea_salt'
            UtilitiesParallel.print_root(' No aerosol species specified, defaulting to sea_salt.')
        
        if aerosol_species == 'sea_salt':
            mwaero_in = 22.9 + 35.5
            ions_in = 2
            ro_solute_in = 2.16
        elif aerosol_species == 'ammonium_bisulfate':
            mwaero_in = 115.0
            ions_in = 2
            ro_solute_in = 1.79
             

        # Read in the aerosol file here which previously was read in the fortran code
        # and assumed to be named 'CCN_size_33bin.dat'
        ccn_size_bin_dat = np.ones((33,3),dtype=np.double,order='F') * -9999
        try:
            aerosol_file = namelist['microphysics']['aerosol_file']
            UtilitiesParallel.print_root('Trying to read a specified aerosol file')

            if MPI.COMM_WORLD.Get_rank() == 0:
                ccn_size_bin_dat = np.asfortranarray(np.loadtxt(aerosol_file))
                print(np.shape(ccn_size_bin_dat))
        
            ccn_size_bin_dat =  MPI.COMM_WORLD.bcast(ccn_size_bin_dat)
            
        except:
            UtilitiesParallel.print_root(' Did not read in aerosol data, will use lognormal distributions')


        module_mp_fast_sbm.module_mp_warm_sbm.warm_hucminit(5.0,
        ccncon1, radius_mean1, sig1,
        ccncon2, radius_mean2, sig2,
        ccncon3, radius_mean3, sig3,
        ccn_size_bin_dat,
        mwaero_in, ions_in, ro_solute_in)

        return

    def add_output_container_diags(self):

        # Add the drop/droplet mass bins
        self.sbm_output_container_index_map = {}
        meta = []
        for i in range(33):
            self.sbm_output_container_index_map['ff1i' + str(i) + '_bfcc'] = i
            meta.append(('kg kg^{-1}', 'liquid bin mass ' + str(i) +' at coal'))

       # Set the indicies for the process rates
        names = ['nc_autoconv',
                'qc_autoconv',
                'qr_autoconv',
                'nr_autoconv',
                'qv_autoconv',
                't_autoconv',
                'w_autoconv',
                'auto_cldmsink_b',
                'auto_cldnsink_b',
                'accr_cldmsink_b',
                'accr_cldnsink_b',
                'selfc_rainnchng_b']

        meta.append(('kg^-1', 'cloud water autoconversion number'))
        meta.append(('kg kg^-1', 'cloud water autoconversion mass'))

        meta.append(('kg kg^-1', 'rain water autoconversion mass'))
        meta.append(('kg^-1', 'rain water autoconversion number'))
        meta.append(('kg kg^-1', 'water vapor autoconversion mass'))
        meta.append(('K', 'temperature'))
        meta.append(('m s^-1', 'autoconversion vertical velocity'))
        meta.append(('kg kg^-1 s^-1', 'autoconversion cloud mass sink'))
        meta.append(('kg^-1 s^-1', 'autoconversion cloud number sink'))
        meta.append(('kg kg^-1 s^-1', 'accretion cloud mass sink'))
        meta.append(('kg^-1 s^-1', 'accretion cloud number sink'))
        meta.append(('kg^-1 s^-1', 'self-collection cloud number change'))

        # Compute the inidicies
        for name in names:
            self.sbm_output_container_index_map[name] = len(self.sbm_output_container_index_map) 

        #Add variables to the diagnostic state
        count = 0
        for name in self.sbm_output_container_index_map:
            self._DiagnosticState.add_variable(name, units=meta[count][0], long_name=meta[count][1])
            count += 1
        return


    def update(self):

        #Get grid information
        nhalo = self._Grid.n_halo

        #Let's build a dictionary wrf_ordered variables
        wrf_vars = {}

        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')
        wrf_vars['qv'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        to_wrf_order(nhalo, qv, wrf_vars['qv'])

        #First re-order velocity variables
        for v in ['w', 'u', 'v']:
            wrf_vars[v] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
            var = self._VelocityState.get_field(v)
            to_wrf_order(nhalo, var, wrf_vars[v])

        #Now re-order scalar variables
        for v in self._list_of_ScalarStatevars:
            wrf_vars[v] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
            var = self._ScalarState.get_field(v)
            to_wrf_order(nhalo, var, wrf_vars[v])

        #Need to compute potential temperature
        wrf_vars['th_old'] = self._th_old
        wrf_vars['qv_old'] = self._qv_old
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field('T')
        if self._itimestep == 1:
            to_wrf_order(nhalo, T/exner[np.newaxis, np.newaxis, :], wrf_vars['th_old'])
            self._qv_old[:,:,:] = wrf_vars['qv'][:,:,:]
        wrf_vars['th_phy']=np.zeros(self._wrf_dims, order='F', dtype=np.double)
        to_wrf_order(nhalo, T/exner[np.newaxis, np.newaxis, :], wrf_vars['th_phy'])

        #Now reorder the bin array
        wrf_vars['chem_new'] = np.zeros((self._wrf_dims[0], self._wrf_dims[1], self._wrf_dims[2], 99),
            order='F', dtype=np.double)

        #This is an expensive transpose
        chem_new = self._ScalarState._state_array.array[self._bin_start:self._bin_end,:,:,:]
        to_wrf_order_4d(nhalo, chem_new, wrf_vars['chem_new'])

        # #Setup reference state profiles
        wrf_vars['dz8w'] =  np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['dz8w'].fill(self._Grid.dx[2])
        wrf_vars['rho_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['rho_phy'][:,:,:] = self._Ref.rho0[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        wrf_vars['p_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['p_phy'][:,:,:] =  self._Ref.p0[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        wrf_vars['pi_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['pi_phy'][:,:,:] = exner[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]
        #wrf_vars['th_phy'] = np.copy(wrf_vars['th_old'])
        wrf_vars['diagflag'] = True
        wrf_vars['num_sbmradar'] = 1
        wrf_vars['sbmradar'] =  np.zeros((self._wrf_dims[0], self._wrf_dims[1], self._wrf_dims[2], wrf_vars['num_sbmradar']),
             order='f', dtype=np.double)

        wrf_vars['RAINNC'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['RAINNCV'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['SNOWNC'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['SNOWNCV'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['GRAUPELNC'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['GRAUPELNCV'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['SR'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['xland'] = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), dtype=np.double, order='F')
        wrf_vars['domain_id'] = 1
        wrf_vars['MA'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['LH_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['CE_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['DS_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['Melt_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['Frz_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['CldNucl_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['n_reg_ccn'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['IceNucl_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['difful_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['diffur_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['tempdiffl'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['automass_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['autonum_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['nprc_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['saturation'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)

        #Get grid dimensions
        ids = 1; jds = 1; kds = 1
        ide = self._wrf_dims[0]; jde = self._wrf_dims[2]; kde = self._wrf_dims[1]
        ims=1; jms = 1; kms = 1
        ime=self._wrf_dims[0]; jme=self._wrf_dims[2]; kme=self._wrf_dims[1]
        its=1; jts=1; kts=1
        ite=ime; jte=jme; kte=kme

        dt = self._TimeSteppingController.dt,

        z = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        z[:,:,:] = self._Grid.z_global[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]

        sbm_output_container = np.zeros((self._wrf_dims[0],self._wrf_dims[1],self._wrf_dims[2],45),dtype=np.double,order='F')
        KRDROP=14

        #Call sbm!

        #Store input fields
        sbm_in_vars = {'T_sbm_in':'T' , 'qv_sbm_in':'qv', 'qc_sbm_in':'qc', 'qr_sbm_in':'qr',
            'qnc_sbm_in':'qnc', 'qnr_sbm_in':'qnr', 'qna_sbm_in':'qna', 'qna_nucl_sbm_in':'qna_nucl'}
    
        for key, value in sbm_in_vars.items():
            try:
                v = self._ScalarState.get_field(value)
            except:
                v = self._DiagnosticState.get_field(value)
            v_diag = self._DiagnosticState.get_field(key)
            v_diag[:,:,:] = v[:,:,:]


        rain_accum_old = np.sum(self._RAINNC)
        MPI.COMM_WORLD.barrier()
        t0 = time.time()

        module_mp_fast_sbm.module_mp_warm_sbm.warm_sbm(wrf_vars['w'],
                                                      wrf_vars['u'],
                                                      wrf_vars['v'],
                                                      wrf_vars['th_old'],
                                                      wrf_vars['chem_new'],
                                                      self._itimestep,
                                                      dt,
                                                      self._Grid.dx[0],
                                                      self._Grid.dx[1],
                                                      wrf_vars['dz8w'],
                                                      wrf_vars['rho_phy'],
                                                      wrf_vars['p_phy'],
                                                      wrf_vars['pi_phy'],
                                                      wrf_vars['th_phy'],
                                                      wrf_vars['xland'],
                                                      wrf_vars['domain_id'],
                                                      wrf_vars['qv'],
                                                      wrf_vars['qc'],
                                                      wrf_vars['qr'],
                                                      wrf_vars['qv_old'],
                                                      wrf_vars['qnc'],
                                                      wrf_vars['qnr'],
                                                      wrf_vars['qna'],
                                                      wrf_vars['qna_nucl'],
                                                      ids,ide, jds,jde, kds,kde,
                                                      ims,ime, jms,jme, kms,kme,
                                                      its,ite, jts,jte, kts,kte,
                                                      KRDROP,
                                                      wrf_vars['sbmradar'],
                                                      wrf_vars['MA'],
                                                      wrf_vars['LH_rate'],
                                                      wrf_vars['CE_rate'],
                                                      wrf_vars['CldNucl_rate'],
                                                      wrf_vars['n_reg_ccn'],
                                                      sbm_output_container,
                                                      wrf_vars['difful_tend'],
                                                      wrf_vars['diffur_tend'],
                                                      wrf_vars['tempdiffl'],
                                                      diagflag= wrf_vars['diagflag'],
                                                      rainnc=wrf_vars['RAINNC'],
                                                      rainncv=wrf_vars['RAINNCV'],
                                                      sr=wrf_vars['SR'])


        t1 = time.time()
        MPI.COMM_WORLD.barrier()

        self._RAINNC[:,:] = wrf_vars['RAINNC'][:,:]
        self._RAINNCV[:,:] = wrf_vars['RAINNCV'][:,:]
        self._rain_rate = (np.sum(self._RAINNC) - rain_accum_old)/dt
        #Now we need to map back to map back from WRF indexes to PINNACLE indexes
        to_our_order_4d(nhalo,wrf_vars['chem_new'],chem_new)

        #import pylab as plt
        #plt.contourf(wrf_vars['th_old'][:,1,:])

        #plt.show()
        #Now re-order scalar variables
        for v in self._list_of_ScalarStatevars:
            var = self._ScalarState.get_field(v)
            to_our_order(nhalo, wrf_vars[v], var)

        for v in self._io_fields:
            var = self._DiagnosticState.get_field(v)
            to_our_order(nhalo, wrf_vars[v], var)


        for item, key in self.sbm_output_container_index_map.items():
            var = self._DiagnosticState.get_field(item)
            to_our_order(nhalo, sbm_output_container[:,:,:,key], var)

        s_wrf = wrf_vars['th_phy']* self._Ref.exner[np.newaxis, nhalo[2]:-nhalo[2], np.newaxis]  +  (parameters.G*z- parameters.LV*(wrf_vars['qc'] + wrf_vars['qr']))*parameters.ICPD
        to_our_order(nhalo, s_wrf, s)

        to_our_order(nhalo, wrf_vars['qv'], qv)

        self._call_count += 1
        self._itimestep  += 1

        return
    def get_qc(self):
        return self._ScalarState.get_field('qc') +  self._ScalarState.get_field('qr')  #np.sum(self._ScalarState._state_array.array[self._qc_start:self._qc_end,:,:,:], axis=0)

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

        rf = water_fraction(n_halo, npts, qr, threshold=1e-5)
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

            
