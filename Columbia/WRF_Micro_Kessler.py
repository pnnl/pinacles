from Columbia.wrf_physics import kessler
from Columbia import parameters
import numpy as np
import numba 

@numba.njit
def to_wrf_order(nhalo, our_array, wrf_array): 
    shape = our_array.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]): 
        for j in range(nhalo[1],shape[1]-nhalo[1]): 
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0]
                j_wrf = j - nhalo[1]  
                k_wrf = k - nhalo[2] #shape[2] - 1 - k
                wrf_array[i_wrf,k_wrf,j_wrf] = our_array[i,j,k]
    return 

@numba.njit
def wrf_theta_tend_to_our_tend(nhalo, dt, exner, wrf_out, our_in, stend):
    shape = stend.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]): 
        for j in range(nhalo[1],shape[1]-nhalo[1]): 
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0] 
                j_wrf = j - nhalo[1]  
                k_wrf = k - nhalo[2] #shape[2] - 1 - k
                stend[i,j,k] += (wrf_out[i_wrf,k_wrf,j_wrf]*exner[k]- our_in[i,j,k])/dt

    return

@numba.njit
def wrf_tend_to_our_tend(nhalo, dt, wrf_out, our_in, tend):
    shape = tend.shape
    for i in range(nhalo[0],shape[0]-nhalo[0]): 
        for j in range(nhalo[1],shape[1]-nhalo[1]): 
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                i_wrf = i  -nhalo[0]  
                j_wrf = j - nhalo[1]  
                k_wrf = k - nhalo[2]#shape[2] - 1 - k
                tend[i,j,k] += (wrf_out[i_wrf,k_wrf,j_wrf] - our_in[i,j,k])/dt
    return

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


class MicroKessler():
    def __init__(self, Grid, Ref, ScalarState, DiagnosticState, TimeSteppingController):
       
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._ScalarState.add_variable('qv')
        self._ScalarState.add_variable('qc') 
        self._ScalarState.add_variable('qr')

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
        our_dims= self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        wrf_dims = (our_dims[0] -2*nhalo[0], our_dims[2]-2*nhalo[2], our_dims[1]-2*nhalo[1])
    
        #Some of the memory allocation could be done at init

        rho_wrf = np.empty(wrf_dims, dtype=np.double, order='F')
        exner_wrf = np.empty_like(rho_wrf) 
        T_wrf = np.empty_like(rho_wrf)
        qv_wrf = np.empty_like(rho_wrf)
        qc_wrf = np.empty_like(rho_wrf)
        qr_wrf = np.empty_like(rho_wrf)

        RAINNC = np.empty((wrf_dims[0], wrf_dims[2]), dtype=np.double, order='F')
        RAINNCV = np.empty_like(RAINNC)

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
        ime=wrf_dims[0]; jme=wrf_dims[2]; kme=wrf_dims[1]
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
        kessler.module_mp_kessler.kessler(T_wrf, qv_wrf, qc_wrf, qr_wrf, rho_wrf, exner_wrf,
            dt, z, xlv, cp,
            ep2, svp1, svp2, svp3, svpT0, rhow,
            dz_wrf,
            RAINNC, RAINNCV,
            ids,iide, jds,jde, kds,kde,
            ims,ime, jms,jme, kms,kme,
            its,ite, jts,jte, kts,kte)
        #print('Here 2')

        s_b4 = np.copy(s_tend)
        wrf_theta_tend_to_our_tend(nhalo, dt, exner, T_wrf, T, s_tend)
        #print(np.amax(s_tend - s_b4))
        wrf_tend_to_our_tend(nhalo, dt, qv_wrf, qv, qv_tend)
        wrf_tend_to_our_tend(nhalo, dt, qc_wrf, qc, qc_tend)
        #print('qc', np.amax(qc), np.amax(qc_tend))
        wrf_tend_to_our_tend(nhalo,dt, qr_wrf, qr, qr_tend)
        #print('qr', np.amax(qr), np.amax(qr_tend))


        #pressure = np.zeros_like(qv) + self._Ref.p0[np.newaxis, np.newaxis,:]
        #rh = compute_rh(qv, T, pressure)
        #print('RH: ', np.amax(rh))

        #plt.plot(T[8,8,nhalo[0]:-nhalo[0]]/exner[nhalo[0]:-nhalo[0]])
        #plt.plot(T_wrf[5,:,5])
        #plt.show()
        #import sys; sys.exit()

        return