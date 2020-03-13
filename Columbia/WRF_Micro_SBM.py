import numpy as np
from Columbia import parameters
import time
from Columbia.Microphysics import MicrophysicsBase
from Columbia.wrf_physics import module_mp_fast_sbm_kobby
from Columbia.WRFUtil import to_wrf_order, to_wrf_order_4d, to_our_order_4d, to_our_order
from Columbia.WRFUtil import to_wrf_order_halo, to_wrf_order_4d_halo, to_our_order_4d_halo, to_our_order_halo
module_mp_fast_sbm = module_mp_fast_sbm_kobby
class MicroSBM(MicrophysicsBase):

    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):
        MicrophysicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)


        self._ScalarState.add_variable('qv')
        #TODO for now adding these as prognostic variables but probably unnecessary
        self._list_of_ScalarStatevars = ['qc', 'qr', 'qi', 'qg', 'qs',
            'qnc', 'qnr', 'qni', 'qns', 'qng', 'qna']

        for var in self._list_of_ScalarStatevars:
            self._DiagnosticState.add_variable(var)

        self._bin_start = self._ScalarState.nvars
        self._qc_start = self._ScalarState.nvars 
        for i in range(1,34):
            name = "ff1i" + str(i)
            self._ScalarState.add_variable(name)
        self._qc_end = self._ScalarState.nvars
        #Add aersol bins
        for i in range(1,34):
            name = 'ff8i' + str(i)
            self._ScalarState.add_variable(name)
        self._bin_end = self._ScalarState.nvars

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        #self._wrf_dims = (self._our_dims[0] -2*nhalo[0], self._our_dims[2]-2*nhalo[2], self._our_dims[1]-2*nhalo[1])
        self._wrf_dims = (self._our_dims[0],self._our_dims[2], self._our_dims[1])

        self._th_old = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        self._qv_old = np.zeros(self._wrf_dims, order='F', dtype=np.double)

        self._itimestep = 1

        module_mp_fast_sbm.module_mp_fast_sbm.fast_hucminit(5.0)

        return

    def update(self):

        t0 = time.time()

        #Get grid information
        nhalo = self._Grid.n_halo

        #Let's build a dictionary wrf_ordered variables
        wrf_vars = {}

        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')
        wrf_vars['qv'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        to_wrf_order_halo(nhalo, qv, wrf_vars['qv'])

        #First re-order velocity variables
        for v in ['w', 'u', 'v']:
            wrf_vars[v] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
            var = self._VelocityState.get_field(v)
            to_wrf_order_halo(nhalo, var, wrf_vars[v])

        #Now re-order scalar variables
        for v in self._list_of_ScalarStatevars:
            wrf_vars[v] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
            var = self._DiagnosticState.get_field(v)
            to_wrf_order_halo(nhalo, var, wrf_vars[v])

        #Need to compute potential temperature
        wrf_vars['th_old'] = self._th_old
        wrf_vars['qv_old'] = self._qv_old
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field('T')
        if self._itimestep == 1:
            to_wrf_order_halo(nhalo, T/exner[np.newaxis, np.newaxis, :], wrf_vars['th_old'])
            self._qv_old = np.copy(wrf_vars['qv'])

        #print(np.amin(T), np.amax(T))
        #import sys; sys.exit()

        #Now reorder the bin array
        wrf_vars['chem_new'] = np.zeros((self._wrf_dims[0], self._wrf_dims[1], self._wrf_dims[2], 66),
            order='F', dtype=np.double)

        #This is an expensive transpose
        chem_new = self._ScalarState._state_array.array[self._bin_start:self._bin_end,:,:,:]

        to_wrf_order_4d(nhalo, chem_new, wrf_vars['chem_new'])

        # #Setup reference state profiles
        wrf_vars['dz8w'] =  np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['dz8w'].fill(self._Grid.dx[2])
        wrf_vars['rho_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['rho_phy'][:,:,:] = self._Ref.rho0[np.newaxis, :, np.newaxis]
        wrf_vars['p_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['p_phy'][:,:,:] =  self._Ref.p0[np.newaxis, :, np.newaxis]
        wrf_vars['pi_phy'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['pi_phy'][:,:,:] = exner[np.newaxis, :, np.newaxis]
        wrf_vars['th_phy'] = np.copy(wrf_vars['th_old'])
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

        wrf_vars['MA'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['LH_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['CE_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['DS_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['Melt_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['Frz_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['CldNucl_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['IceNucl_rate'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['difful_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['diffur_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['tempdiffl'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['automass_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['autonum_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        wrf_vars['nprc_tend'] = np.zeros(self._wrf_dims, order='F', dtype=np.double)

        #Get grid dimensions
        ids = 1; jds = 1; kds = 1
        ide = self._wrf_dims[0]; jde = self._wrf_dims[2]; kde = self._wrf_dims[1]
        ims=1; jms = 1; kms = 1
        ime=self._wrf_dims[0]; jme=self._wrf_dims[2]; kme=self._wrf_dims[1]
        its=1; jts=1; kts=1
        ite=ime; jte=jme; kte=kme

        dt = self._TimeSteppingController.dt,

        z = np.zeros(self._wrf_dims, order='F', dtype=np.double)
        z[:,:,:] = self._Grid.z_global[np.newaxis, :, np.newaxis]

        print('th_old', np.min(wrf_vars['th_old']), np.amax(wrf_vars['th_old']))
        print('th_phy', np.min(wrf_vars['th_old']), np.amax(wrf_vars['th_old']))

        #Call sbm!
        module_mp_fast_sbm.module_mp_fast_sbm.fast_sbm(wrf_vars['w'],
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
                                                      wrf_vars['qv'],
                                                      wrf_vars['qc'],
                                                      wrf_vars['qr'],
                                                      wrf_vars['qi'],
                                                      wrf_vars['qs'],
                                                      wrf_vars['qg'],
                                                      wrf_vars['qv_old'],
                                                      wrf_vars['qnc'],
                                                      wrf_vars['qnr'],
                                                      wrf_vars['qni'],
                                                      wrf_vars['qns'],
                                                      wrf_vars['qng'],
                                                      wrf_vars['qna'],
                                                      ids,ide, jds,jde, kds,kde,
                                                      ims,ime, jms,jme, kms,kme,
                                                      its,ite, jts,jte, kts,kte,
                                                      wrf_vars['sbmradar'],
                                                      wrf_vars['MA'],
                                                      wrf_vars['LH_rate'],
                                                      wrf_vars['CE_rate'],
                                                      wrf_vars['DS_rate'],
                                                      wrf_vars['Melt_rate'],
                                                      wrf_vars['Frz_rate'],
                                                      wrf_vars['CldNucl_rate'],
                                                      wrf_vars['IceNucl_rate'],
                                                      wrf_vars['difful_tend'],
                                                      wrf_vars['diffur_tend'],
                                                      wrf_vars['tempdiffl'],
                                                      wrf_vars['automass_tend'],
                                                      wrf_vars['autonum_tend'],
                                                      wrf_vars['nprc_tend'],
                                                      diagflag= wrf_vars['diagflag'],
                                                      rainnc=wrf_vars['RAINNC'],
                                                      rainncv=wrf_vars['RAINNCV'],
                                                      snownc=wrf_vars['SNOWNC'],
                                                      snowncv=wrf_vars['SNOWNCV'],
                                                      graupelnc=wrf_vars['GRAUPELNC'],
                                                      graupelncv=wrf_vars['GRAUPELNCV'],
                                                      sr=wrf_vars['SR'])
        print('qv', np.min(wrf_vars['qv']), np.amax(wrf_vars['qv']))
        print('qv_old', np.min(wrf_vars['qv_old']), np.amax(wrf_vars['qv_old']))
        print('qc', np.min(wrf_vars['qc']), np.amax(wrf_vars['qc']))
        print('qr', np.min(wrf_vars['qr']), np.amax(wrf_vars['qr']))
        print('th_old', np.min(wrf_vars['th_old']), np.amax(wrf_vars['th_old']))
        print('th_phy', np.min(wrf_vars['th_phy']), np.amax(wrf_vars['th_phy']))
        #Now we need to map back to map back from WRF indexes to PINNACLE indexes
        to_our_order_4d(nhalo,wrf_vars['chem_new'],chem_new)

        #import pylab as plt
        #plt.contourf(wrf_vars['th_old'][:,1,:])
        #plt.show()

        #Now re-order scalar variables
        for v in self._list_of_ScalarStatevars:

            var = self._DiagnosticState.get_field(v)
            to_our_order_halo(nhalo, wrf_vars[v], var)
        print(np.amin(wrf_vars['qc'] ), np.amax(wrf_vars['qc'] ))
        s_wrf = wrf_vars['th_phy']* self._Ref.exner[np.newaxis, :, np.newaxis]  - (parameters.G*z- parameters.LV*(wrf_vars['qc'] + wrf_vars['qr']))*parameters.ICPD
        to_our_order_halo(nhalo, s_wrf, s)
        to_our_order_halo(nhalo, wrf_vars['qv'], qv)
        #print(np.amin(s), np.amax(s))  
        #print('QR MAX!!!', np.amax(wrf_vars['qr']))
        #print('TEMP MAX!!!', np.amax(wrf_vars['th_phy']* self._Ref.exner[np.newaxis,nhalo[2]:-nhalo[2],np.newaxis]  - (parameters.G*z- parameters.LV*(wrf_vars['qc']+wrf_vars['qr']))*parameters.ICPD ))

        #print('Exit')
        #import sys; sys.exit()

        #import pylab as plt
        #plt.contourf(wrf_vars['qv'][3,:,:].T)
        #plt.colorbar()
        ##plt.grid()
        #plt.show()
        #import sys; sys.exit()
        self._itimestep  += 1

        t1 = time.time()
        print('SBM time: ', t1-t0)

        
        return
    def get_qc(self):
        #print('Here', np.sum(self._ScalarState._state_array.array[self._qc_start:self._qc_end,:,:,:], axis=0))
        #import sys; sys.exit()
        return np.sum(self._ScalarState._state_array.array[self._qc_start:self._qc_end,:,:,:], axis=0)/1000.0



