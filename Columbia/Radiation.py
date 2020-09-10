from climlab.radiation.rrtm import _rrtmg_lw
from climlab.radiation.rrtm import _rrtmg_sw
from Columbia import parameters
import numba
import numpy as np

class RRTMG:
    def __init__(self, namelist, Grid, Ref, ScalarState, DiagnosticState):

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        try:
            self._compute_radiation = namelist['radiation']['compute_radiation']
        except:
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('Looks like there is no radiation model specified in the namelist!')
            self._compute_radiation = False
        
        _rrtmg_lw.climlab_rrtmg_lw_ini(parameters.CPD)

        self._DiagnosticState.add_variable('heating_rate')
        self._DiagnosticState.add_variable('dTdt_rad')
        self._DiagnosticState.add_variable('uflux_lw')
        self._DiagnosticState.add_variable('dflux_lw')
        self._DiagnosticState.add_variable('uflux_sw')
        self._DiagnosticState.add_variable('dflux_sw')
        
        self._DiagnosticState.add_variable('heating_rate_clear')
        self._DiagnosticState.add_variable('uflux_lw_clear')
        self._DiagnosticState.add_variable('dflux_lw_clear')
        self._DiagnosticState.add_variable('uflux_sw_clear')
        self._DiagnosticState.add_variable('dflux_sw_clear')


        self.p_extension = None
        self.t_extension = None
        self.qv_extension = None
        self.ql_extension = None
        self.qi_extension = None


        return

    def update(self):

        _nbndlw=16
        _nbndsw=14
        _ngrid_local = self._Grid.ngrid_local
        _nhalo = self._Grid.n_halo
        _nextension = np.shape(self.t_extension)[0]
        _ncol = (_ngrid_local[0]-2*_nhalo[0])* (_ngrid_local[1]-2*_nhalo[1])
        _nlay = _ngrid_local[2] -2*_nhalo[2] + _nextension

        # inputs to RRTMG
        play_in = np.zeros((_ncol,_nlay), dtype=np.double, order='F')
        plev_in = np.zeros((_ncol,_nlay + 1), dtype=np.double, order='F')
        tlay_in = np.zeros((_ncol,_nlay), dtype=np.double, order='F')
        tlev_in = np.zeros((_ncol,_nlay + 1), dtype=np.double, order='F')
        tsfc_in = np.ones((_ncol),dtype=np.double,order='F') * Sur.T_surface
        h2ovmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        o3vmr_in  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        co2vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        ch4vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        n2ovmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        o2vmr_in  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cfc11vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cfc12vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cfc22vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        ccl4vmr_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        emis_in = np.ones((_ncol,_nbndlw),dtype=np.double,order='F') * self._emis
        cldfr_in  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cicewp_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        cliqwp_in = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        reice_in  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        reliq_in  = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        coszen_in = np.ones((_ncol),dtype=np.double,order='F') *self._coszen
        asdir_in = np.ones((_ncol),dtype=np.double,order='F') * self._adir
        asdif_in = np.ones((_ncol),dtype=np.double,order='F') * self._adif
        aldir_in = np.ones((_ncol),dtype=np.double,order='F') * self._adir
        aldif_in = np.ones((_ncol),dtype=np.double,order='F') * self._adif
        taucld_lw_in  = np.zeros((nbndlw,_ncol,_nlay),dtype=np.double,order='F')
        tauaer_lw_in  = np.zeros((_ncol,_nlay,_nbndlw),dtype=np.double,order='F')
        taucld_sw_in  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        ssacld_sw_in  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        asmcld_sw_in  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        fsfcld_sw_in  = np.zeros((_nbndsw,_ncol,_nlay),dtype=np.double,order='F')
        tauaer_sw_in  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        ssaaer_sw_in  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        asmaer_sw_in  = np.zeros((_ncol,_nlay,_nbndsw),dtype=np.double,order='F')
        ecaer_sw_in  = np.zeros((_ncol,_nlay,6),dtype=np.double,order='F')

            # Output
        uflx_lw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflx_lw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hr_lw_out = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        uflxc_lw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflxc_lw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hrc_lw_out = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        duflx_dt_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        duflxc_dt_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        uflx_sw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflx_sw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hr_sw_out = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        uflxc_sw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        dflxc_sw_out = np.zeros((_ncol,_nlay +1),dtype=np.double,order='F')
        hrc_sw_out = np.zeros((_ncol,_nlay),dtype=np.double,order='F')
        return

# Does this work for plev?
@numba.njit
def to_rrtmg_shape(nhalo, our_array, rrtmg_array, extension_array):
    shape = our_array.shape
    count = 0
    n_ext = np.shape(extension_array)[0]
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            count += 1
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                k_rrtmg = k - nhalo[2] #shape[2] - 1 - k
                rrtmg_array[count,k_rrtmg]] = our_array[i,j,k]
            for k in range(n_ext):
                k_rrtmg = shape[2]-2 * nhalo[2] + k
                rrtmg_array[count, k_rrtmg] = extension_array[k]

    return

# does this work for plev?
@numba.njit
def to_our_shape(nhalo, rrtmg_array, our_array):
    shape = our_array.shape
    count = 0
    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            count +=1
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                k_rrtmg = k - nhalo[2]
                our_array[i,j,k] = rrtmg_array[count, k_rrtmg]

