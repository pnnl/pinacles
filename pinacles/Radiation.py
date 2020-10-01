from Columbia import parameters
import numba
import numpy as np
from mpi4py import MPI
from scipy import interpolate
import pylab as plt
from cffi import FFI
import ctypes
ffi = FFI()


class RRTMG:
    def __init__(self, namelist, Grid, Ref, ScalarState, DiagnosticState, Surf):

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Surf = Surf

        try:
            self._compute_radiation = namelist['radiation']['compute_radiation']
        except:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('Looks like there is no radiation model specified in the namelist!')
            self._compute_radiation = False

        # HACK
        self._compute_radiation = True
        

        ffi.cdef("void c_rrtmg_lw_init(double cpdair);",override=True)
        ffi.cdef("void c_rrtmg_sw_init(double cpdair);", override=True)
        self._lib_lw = ffi.dlopen('/Users/kaul025/PINACLES/Columbia/Columbia/rrtmg/librrtmglw.so')
        self._lib_sw = ffi.dlopen('/Users/kaul025/PINACLES/Columbia/Columbia/rrtmg/librrtmgsw.so')
       

        self._lib_lw.c_rrtmg_lw_init(parameters.CPD)
        self._lib_sw.c_rrtmg_sw_init(parameters.CPD)

        # self._DiagnosticState.add_variable('heating_rate_lw')
        # self._DiagnosticState.add_variable('heating_rate_sw')
        # self._DiagnosticState.add_variable('dTdt_rad')
        # self._DiagnosticState.add_variable('uflux_lw')
        # self._DiagnosticState.add_variable('dflux_lw')
        # self._DiagnosticState.add_variable('uflux_sw')
        # self._DiagnosticState.add_variable('dflux_sw')

        DiagnosticState.add_variable('heating_rate_lw',units='K',  latex_name = 'T', long_name='Temperature')
        DiagnosticState.add_variable('heating_rate_sw',units='K',  latex_name = 'T', long_name='Temperature')
        # DiagnosticState.add_variable('dTdt_rad')
        # DiagnosticState.add_variable('uflux_lw')
        # DiagnosticState.add_variable('dflux_lw')
        # DiagnosticState.add_variable('uflux_sw')
        # DiagnosticState.add_variable('dflux_sw')
        
        # self._DiagnosticState.add_variable('heating_rate_clear')
        # self._DiagnosticState.add_variable('uflux_lw_clear')
        # self._DiagnosticState.add_variable('dflux_lw_clear')
        # self._DiagnosticState.add_variable('uflux_sw_clear')
        # self._DiagnosticState.add_variable('dflux_sw_clear')


        self.p_extension = np.array([600.e2,100.e2,50.e2]) #None
        self.t_extension = np.array([275.0, 275.0,275.0]) #None
        self.qv_extension = np.array([1e-2, 1e-3,1e-4])#None
        self.ql_extension =np.array([0.,0.,0.]) #None
        self.qi_extension = np.array([0.,0.,0.])#None

        # CL WRF values based on 2005 values from 2007 IPCC report
        self._vmr_co2   = 379.0e-6
        self._vmr_ch4   = 1774.0e-9
        self._vmr_n2o   = 319.0e-9
        self._vmr_o2    = 0.209448
        self._vmr_cfc11 = 0.251e-9
        self._vmr_cfc12 = 0.538e-9
        self._vmr_cfc22 = 0.169e-9
        self._vmr_ccl4  = 0.093e-9
        
        self._emis = 0.95
        self._coszen = 0.5
        self._adir = 0.1
        self._adif = 0.1
        self._scon = 1365.0
        self._adjes = 1.0
        self._dyofyr = 0
        
        return

    def update(self):

        if not self._compute_radiation:
            return
        
        ffi.cdef("void c_rrtmg_lw(int ncol, int nlay, int icld, int idrv, \
                double play[], double plev[], double tlay[], double tlev[], \
                double tsfc[], double h2ovmr[], double o3vmr[], double co2vmr[], \
                double ch4vmr[], double n2ovmr[], double o2vmr[], double cfc11vmr[],\
                double cfc12vmr[], double cfc22vmr[], double ccl4vmr[], double emis[],\
                int inflglw, int iceflglw, int liqflglw, double cldfr[], \
                double taucld[], double cicewp[], double cliqwp[], double reice[],\
                double reliq[], double tauaer[], double uflx[], double dflx[], double hr[],\
                double uflxc[], double dflxc[],  double hrc[], double duflx_dt[],\
                double duflxc_dt[]);",override=True)
        ffi.cdef("void c_rrtmg_sw(int ncol, int nlay, int icld, int iaer, double play[], \
                double plev[], double tlay[], double tlev[], double tsfc[], \
                double h2ovmr[], double o3vmr[], double co2vmr[], double ch4vmr[],\
                double n2ovmr[], double o2vmr[], double asdir[], double asdif[] , \
                double aldir[], double aldif[], double coszen[], double adjes, int dyofyr,\
                double scon, int inflgsw, int iceflgsw, int liqflgsw, double cldfr[], \
                double taucld[], double ssacld[], double asmcld[], double fsfcld[],\
                double cicewp[], double cliqwp[], double reice[], double reliq[], \
                double tauaer[], double ssaaer[], double asmaer[], double ecaer[], \
                double swuflx[], double swdflx[], double swhr[], double swuflxc[],\
                 double swdflxc[] , double swhrc[]);", override=True)

        # RRTMG flags. Hardwiring for now
        icld = 1
        idrv = 0
        iaer = 0
        inflglw = 2
        iceflglw = 3
        liqflglw = 1
        inflgsw = 2
        iceflgsw = 3
        liqflgsw = 1

        _nbndlw=16
        _nbndsw=14
        _ngrid_local = self._Grid._ngrid_local
        _nhalo = self._Grid.n_halo
        _nextension = np.shape(self.t_extension)[0]
        _ncol = (_ngrid_local[0]-2*_nhalo[0])* (_ngrid_local[1]-2*_nhalo[1])
        _nlay = _ngrid_local[2] -2*_nhalo[2] + _nextension

        # inputs to RRTMG
        play = np.zeros((_ncol,_nlay), dtype=np.double, order='F') #hPA !!!
        plev = np.zeros((_ncol,_nlay + 1), dtype=np.double, order='F') #hPA !!!
        tlay = np.zeros((_ncol,_nlay), dtype=np.double, order='F')
        tlev = np.zeros((_ncol,_nlay + 1), dtype=np.double, order='F')
        tsfc = np.ones((_ncol),dtype=np.double,order='F') * self._Surf.T_surface
        h2ovmr = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        o3vmr  = np.zeros((_ncol,_nlay),dtype=np.double,order='F') 
        co2vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_co2
        ch4vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_ch4
        n2ovmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_n2o
        o2vmr  = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_o2
        cfc11vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_cfc11
        cfc12vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_cfc12
        cfc22vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_cfc22
        ccl4vmr = np.ones((_ncol,_nlay),dtype=np.double,order='F') * self._vmr_ccl4
        emis = np.ones((_ncol,_nbndlw),dtype=np.double,order='F') * self._emis
        cldfr  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cicewp = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cliqwp = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        reice  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        reliq  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        coszen = np.ones((_ncol),dtype=np.double,order='F') *self._coszen
        asdir = np.ones((_ncol),dtype=np.double,order='F') * self._adir
        asdif = np.ones((_ncol),dtype=np.double,order='F') * self._adif
        aldir = np.ones((_ncol),dtype=np.double,order='F') * self._adir
        aldif = np.ones((_ncol),dtype=np.double,order='F') * self._adif
        taucld_lw  = np.zeros((_nbndlw,_ncol,_nlay),dtype=np.double,order='F')
        tauaer_lw  = np.zeros((_ncol,_nlay,_nbndlw),dtype=np.double,order='F')
        taucld_sw  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        ssacld_sw  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        asmcld_sw  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        fsfcld_sw  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        tauaer_sw  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        ssaaer_sw  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        asmaer_sw  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        ecaer_sw  = np.zeros((_ncol,_nlay,6),dtype=np.double,order='F')

        # Output
        uflx_lw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflx_lw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hr_lw = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        uflxc_lw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflxc_lw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hrc_lw = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        duflx_dt = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        duflxc_dt = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        uflx_sw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflx_sw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hr_sw = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        uflxc_sw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflxc_sw = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hrc_sw = np.zeros((_ncol,_nlay),dtype=np.double,order='F')

        # Set play, plev
        play_col = np.append(self._Ref._P0[_nhalo[2]:-_nhalo[2]], self.p_extension)/100.0
        play =  np.asfortranarray(np.repeat(play_col[np.newaxis,:],_ncol,axis=0))
        plev_col = np.append(self._Ref._P0_edge[_nhalo[2]-1:-_nhalo[2]],  0.5 * (self.p_extension + np.append(self.p_extension[1:],0)))/100.0
        plev = np.asfortranarray(np.repeat(plev_col[np.newaxis,:],_ncol,axis=0))

        # reshape temperature to rrtmg shape (layers)
        to_rrtmg_shape(_nhalo,self._DiagnosticState.get_field('T'),self.t_extension,tlay ) 
       
        # Interpolate temperature to the levels
        # Extrapolate to surface temp between the lowest pressure layer and surface
        t_temp = np.insert(tlay,0, self._Surf.T_surface, axis=1)
        p_temp = np.insert(play,0, self._Ref._Psfc/100.0,axis=1)
        # extrapolate as isothermal between top pressure layer and TOA
        t_temp = np.append(t_temp, np.expand_dims(t_temp[:,-1],axis=1),axis=1)
        p_temp = np.append(p_temp, np.zeros((_ncol,1)),axis=1)
        
        # We use the extrapolate option to handle situations were floating point issues put us out of range
        tlev = interpolate.interp1d(p_temp[0,:],t_temp,axis=1,fill_value='extrapolate')(plev[0,:])
        

        # qv to rrtmg shape; convert to vmr
        to_rrtmg_shape(_nhalo,self._ScalarState.get_field('qv'),self.qv_extension,h2ovmr)
        h2ovmr *= parameters.RV/parameters.RD
        # ql to rrtmg shape; need to convert to path in g/m^2
        if 'ql' in self._ScalarState.names:
            to_rrtmg_shape(_nhalo, self._ScalarState.get_field('ql'), self.ql_extension, cliqwp)
            cliqwp *= 1.e3/parameters.G * (plev[:-1]-plev[1:])

        #  qi to rrtmg shape; need to convert to path in g/m^2
        if 'qi' in self._ScalarState.names:
            to_rrtmg_shape(_nhalo, self._ScalarState.get_field('ql'), self.qi_extension, cicewp)
            cicewp *= 1.e3/parameters.G * (plev[:-1]-plev[1:])
        
    
        print('rad1')
        self._lib_lw.c_rrtmg_lw(_ncol, _nlay,  icld, idrv, 
        as_pointer(play), as_pointer(plev), as_pointer(tlay), as_pointer(tlev), 
        as_pointer(tsfc), as_pointer(h2ovmr), as_pointer(o3vmr), as_pointer(co2vmr), 
        as_pointer(ch4vmr), as_pointer(n2ovmr), as_pointer(o2vmr), as_pointer(cfc11vmr),
        as_pointer(cfc12vmr), as_pointer(cfc22vmr), as_pointer(ccl4vmr), as_pointer(emis),
        inflglw, iceflglw, liqflglw, as_pointer(cldfr), as_pointer(taucld_lw), 
        as_pointer(cicewp), as_pointer(cliqwp), as_pointer(reice), as_pointer(reliq), 
        as_pointer(tauaer_lw), as_pointer(uflx_lw), as_pointer(dflx_lw), as_pointer(hr_lw),
        as_pointer(uflxc_lw), as_pointer(dflxc_lw),  as_pointer(hrc_lw), as_pointer(duflx_dt),as_pointer(duflxc_dt))
        print('rad2')

        self._lib_sw.c_rrtmg_sw(_ncol, _nlay, icld, iaer, as_pointer(play), 
        as_pointer(plev), as_pointer(tlay), as_pointer(tlev), as_pointer(tsfc), 
        as_pointer(h2ovmr), as_pointer(o3vmr), as_pointer(co2vmr), as_pointer(ch4vmr),
        as_pointer(n2ovmr), as_pointer(o2vmr), as_pointer(asdir), as_pointer(asdif), 
        as_pointer(aldir), as_pointer(aldif), as_pointer(coszen), self._adjes, self._dyofyr,
        self._scon, inflgsw, iceflgsw, liqflgsw, as_pointer(cldfr), as_pointer(taucld_sw), 
        as_pointer(ssacld_sw), as_pointer(asmcld_sw), as_pointer(fsfcld_sw), as_pointer(cicewp),
        as_pointer(cliqwp),as_pointer(reice), as_pointer(reliq), 
        as_pointer(tauaer_sw), as_pointer(ssaaer_sw), as_pointer(asmaer_sw), as_pointer(ecaer_sw), 
        as_pointer(uflx_sw), as_pointer(dflx_sw), as_pointer(hr_sw), as_pointer(uflxc_sw),\
        as_pointer(dflxc_sw) , as_pointer(hrc_sw))

        # plt.figure()
        # plt.plot(hr_lw[0,:],play[0,:])
        # plt.plot(hr_sw[0,:],play[0,:])
        # plt.show()

        # ds_hr_lw = self._DiagnosticState.get_field('heating_rate_lw')
        # ds_hr_sw = self._DiagnosticState.get_field('heating_rate_sw')
        # ds_dTdt_rad = self._DiagnosticState.get_field('dTdt_rad')
        # ds_uflux_lw = self._DiagnosticState.get_field('uflux_lw')
        # ds_dflux_lw = self._DiagnosticState.get_field('dflux_lw')
        # ds_uflux_sw = self._DiagnosticState.get_field('uflux_sw')
        # ds_dflux_sw = self._DiagnosticState.get_field('dflux_sw')

        st = self._ScalarState.get_tend('s')
        alpha0 = self._Ref._alpha0

        # to_our_shape(_nhalo, hr_lw, ds_hr_lw)
        # to_our_shape(_nhalo, hr_sw, ds_hr_sw)
        # to_our_shape(_nhalo, uflx_lw, ds_uflux_lw)
        # to_our_shape(_nhalo, dflx_lw, ds_dflux_lw)
        # to_our_shape(_nhalo, uflx_sw, ds_uflux_sw)
        # to_our_shape(_nhalo, dflx_sw, ds_dflux_sw)


        # ds_dTdt_rad = (ds_hr_lw + ds_hr_sw) * alpha0[np.newaxis,np.newaxis,:] /parameters.CPD

        # st += ds_dTdt_rad


        return

# Does this work for plev?
@numba.njit
def to_rrtmg_shape(nhalo, our_array,  extension_array, rrtmg_array):
    shape = our_array.shape
    count = 0
    n_ext = extension_array.shape[0]
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                k_rrtmg = k - nhalo[2] #shape[2] - 1 - k
                rrtmg_array[count,k_rrtmg] = our_array[i,j,k]
            for k in range(n_ext):
                k_rrtmg = shape[2]-2 * nhalo[2] + k
                rrtmg_array[count, k_rrtmg] = extension_array[k]
            count+=1

    return

# does this work for plev?
@numba.njit
def to_our_shape(nhalo, rrtmg_array, our_array):
    shape = our_array.shape
    count = 0
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
        
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                k_rrtmg = k - nhalo[2]
                our_array[i,j,k] = rrtmg_array[count, k_rrtmg]
            count+=1
    return

# function creates cdata variables of a type "double *" from a numpy array             
# additionally checks if the array is contiguous                                       
def as_pointer(numpy_array):
    assert numpy_array.flags['F_CONTIGUOUS'], \
        "array is not contiguous in memory (Fortran order)"
    return ffi.cast("double*", numpy_array.__array_interface__['data'][0])
