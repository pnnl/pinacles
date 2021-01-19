from pinacles import parameters
from pinacles import UtilitiesParallel
import time
import numba
import numpy as np
from mpi4py import MPI
from scipy import interpolate
#import pylab as plt
import netCDF4 as nc
from cffi import FFI
import ctypes
ffi = FFI()


class RRTMG:
    def __init__(self, namelist, Grid, Ref, ScalarState, DiagnosticState, Surf,TimeSteppingController):
        self._name = 'RRTMG'
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Surf = Surf
        self._TimeSteppingController = TimeSteppingController

        try:
            self._compute_radiation = namelist['radiation']['compute_radiation']
        except:
            if namelist['meta']['casename'] == 'testbed':
                self._compute_radiation = True
                UtilitiesParallel.print_root('\t \t Assuming RRTMG should be used for testbed case.')
            else:
                self._compute_radiation = False
                UtilitiesParallel.print_root('\t \t Assuming RRTMG should not be used for this case.')
     
        if not self._compute_radiation:
            return


        try:
            self._radiation_frequency = namelist['radiation']['update_frequency']
        except:
            self._radiation_frequency = 30.0

        #

        ffi.cdef("void c_rrtmg_lw_init(double cpdair);",override=True)
        ffi.cdef("void c_rrtmg_sw_init(double cpdair);", override=True)
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

        self._rrtmg_lib_path = namelist['radiation']['rrtmg_lib_path']
        if self._rrtmg_lib_path[-1] != '/':
            self._rrtmg_lib_path += '/'
        self._lib_lw = ffi.dlopen(self._rrtmg_lib_path + 'librrtmglw.so')
        self._lib_sw = ffi.dlopen(self._rrtmg_lib_path + 'librrtmgsw.so')
       

        self._lib_lw.c_rrtmg_lw_init(parameters.CPD)
        self._lib_sw.c_rrtmg_sw_init(parameters.CPD)

        self._DiagnosticState.add_variable('heating_rate_lw')
        self._DiagnosticState.add_variable('heating_rate_sw')
        self._DiagnosticState.add_variable('dTdt_rad')
        # self._DiagnosticState.add_variable('uflux_lw')
        # self._DiagnosticState.add_variable('dflux_lw')
        # self._DiagnosticState.add_variable('uflux_sw')
        # self._DiagnosticState.add_variable('dflux_sw')

        nl = self._Grid.ngrid_local

       
       
        self._radiation_file_path = namelist['radiation']['input_filepath']
        data = nc.Dataset(self._radiation_file_path, 'r')
        try:
            rad_data = data.groups['radiation_varanal']
            UtilitiesParallel.print_root('\t \t radiation profiles from analysis')
            # rad_data = data.groups['radiation_sonde']
            # print('radiation profiles from sonde')
        except:
            rad_data = data.groups['radiation']
        self._latitude = rad_data.variables['latitude'][0]
        self._longitude = rad_data.variables['longitude'][0]
        day  = rad_data.variables['day_of_year'][0]
        self._hourz_init = rad_data.variables['hour_utc'][0]
        
        self._dyofyr_init = np.floor(day) + self._hourz_init/24.0
        try:
            self._emis =  rad_data.variables['emissivity'][0]
        except:
            self._emis = 0.98
        try:
            albedo = rad_data.variables['albedo'][0]
        except:
            albedo = 0.06
        
        self._adir =  albedo
        self._adif = albedo
        # print(self._emis, self._adir, self._adif)
        data.close()

        # CL WRF values based on 2005 values from 2007 IPCC report
        self._vmr_co2   = 379.0e-6
        self._vmr_ch4   = 1774.0e-9
        self._vmr_n2o   = 319.0e-9
        self._vmr_o2    = 0.209448
        self._vmr_cfc11 = 0.251e-9
        self._vmr_cfc12 = 0.538e-9
        self._vmr_cfc22 = 0.169e-9
        self._vmr_ccl4  = 0.093e-9

        # These needs to be improved
        self._vmr_o3 = 70.0e-9
        
        # self._emis = 1.0
        
        # self._adir = 0.2
        # self._adif = 0.2
        self._scon = 1365.0
        self._adjes = 1.0
        self.dyofyr = 0
        self.hourz = 0


        self.time_elapsed = 10000.0      
        return

    def init_profiles(self):
        if not self._compute_radiation:
            return
        n_halo = self._Grid.n_halo[2]
       
        data = nc.Dataset(self._radiation_file_path, 'r')
        try:
            rad_data = data.groups['radiation_varanal']
            print('radiation profiles from analysis')
            # rad_data = data.groups['radiation_sonde']
            # print('radiation profiles from sonde')
        except:
            rad_data = data.groups['radiation']
        p_data = rad_data.variables['pressure'][:]
        t_data = rad_data.variables['temperature'][:]
        qv_data = rad_data.variables['vapor_mixing_ratio'][:]
        ql_data = rad_data.variables['liquid_mixing_ratio'][:]
        qi_data = rad_data.variables['ice_mixing_ratio'][:]
        data.close()

        # Configure a few buffer points in the pressure profile
        dp_model_top = np.abs(self._Ref._P0_edge[-n_halo] - self._Ref._P0_edge[-n_halo-1])
        p_trial = p_data[p_data<self._Ref._P0[-n_halo]]
        dp_trial = np.abs(p_trial[1]-p_trial[2])
        dp_geom = np.geomspace(dp_model_top*1.5, dp_trial,num=10)
        p_buffer = np.array([self._Ref._P0_edge[-n_halo]])

      
        i=0     
        while p_buffer[i] + 2*dp_trial > p_trial[1] and i < 10:
            p_buffer = np.append(p_buffer,p_buffer[i]-dp_geom[i])
            i+=1
     
        self.p_buffer = p_buffer[1:]
        self.p_extension = p_data[p_data<p_buffer[-1]]
        self.t_extension = t_data[p_data<p_buffer[-1]]
        self.qv_extension = qv_data[p_data<p_buffer[-1]]
        self.ql_extension = ql_data[p_data<p_buffer[-1]]
        self.qi_extension = qi_data[p_data<p_buffer[-1]]

       

        # Set plev
        _nhalo = self._Grid.n_halo
        play_col = np.concatenate((self._Ref._P0[_nhalo[2]:-_nhalo[2]],self.p_buffer, self.p_extension))
        p_ext_full = np.concatenate((self.p_buffer,self.p_extension))
        plev_col = np.append(self._Ref._P0_edge[_nhalo[2]-1:-_nhalo[2]],  0.5 * (p_ext_full + np.append(p_ext_full[1:],0)))
        lw_input_file = self._rrtmg_lib_path + 'rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure']) * 100.0
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        index_o3 = 7

        self._profile_o3 = interpolate_trace_gas(lw_pressure,lw_absorber[:,index_o3],plev_col)


        # plt.figure(1)
        # plt.plot(plev_col[:-1]-plev_col[1:],'o')
        # plt.plot(play_col[:-1]-play_col[1:],'s')

        # plt.figure(2)
        # plt.plot(lw_pressure,lw_absorber[:,index_o3],'o')
        # plt.plot(play_col,self._profile_o3, 's')
        # plt.show()

        self._surf_sw_dn = 0.0
        self._surf_sw_up = 0.0
        self._surf_lw_dn = 0.0
        self._surf_lw_up = 0.0

        self._toa_sw_dn = 0.0
        self._toa_sw_up = 0.0
        self._toa_lw_dn = 0.0
        self._toa_lw_up = 0.0  
        return

    def update(self,  _rk_step):

        if not self._compute_radiation:
            return

        # get the pointers we need in any case  
        ds_dTdt_rad = self._DiagnosticState.get_field('dTdt_rad')
        st = self._ScalarState.get_tend('s')
        if _rk_step == 0:
            self.time_elapsed += self._TimeSteppingController.dt
        
        

        if _rk_step == 0 and self.time_elapsed > self._radiation_frequency:
            self.time_elapsed = 0.0
            # THis should get tested
            self.hourz = self._hourz_init + self._TimeSteppingController.time/3600.0
            self.dyofyr = self._dyofyr_init + np.floor_divide(self._TimeSteppingController.time,86400.0)
            if self.hourz > 24.0:
                self.hourz = np.remainder(self.hourz,24.0)
            self.coszen = cos_sza(self.dyofyr,self.hourz, self._latitude, self._longitude )
           

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
            _ncol = (_ngrid_local[0]-2*_nhalo[0])* (_ngrid_local[1]-2*_nhalo[1])
            _nlay = _ngrid_local[2] -2*_nhalo[2] +np.shape(self.p_extension)[0] + np.shape(self.p_buffer)[0]
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
            coszen = np.ones((_ncol),dtype=np.double,order='F') *self.coszen
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
            play_col = np.concatenate((self._Ref._P0[_nhalo[2]:-_nhalo[2]],self.p_buffer, self.p_extension))
            p_ext_full = np.concatenate((self.p_buffer,self.p_extension))
            play =  np.asfortranarray(np.repeat(play_col[np.newaxis,:],_ncol,axis=0))
            plev_col = np.append(self._Ref._P0_edge[_nhalo[2]-1:-_nhalo[2]],  0.5 * (p_ext_full + np.append(p_ext_full[1:],0)))
            plev = np.asfortranarray(np.repeat(plev_col[np.newaxis,:],_ncol,axis=0))

            # reshape temperature to rrtmg shape (layers)
            to_rrtmg_shape(_nhalo,self._DiagnosticState.get_field('T'),self.t_extension,tlay, self.p_buffer,self._Ref._P0[-_nhalo[2]], self.p_extension[0] ) 
        
            # Interpolate temperature to the levels
            # Extrapolate to surface temp between the lowest pressure layer and surface
            t_temp = np.insert(tlay,0, self._Surf.T_surface, axis=1)
            p_temp = np.insert(play,0, self._Ref._Psfc,axis=1)
            # extrapolate as isothermal between top pressure layer and TOA
            t_temp = np.append(t_temp, np.expand_dims(t_temp[:,-1],axis=1),axis=1)
            p_temp = np.append(p_temp, np.zeros((_ncol,1)),axis=1)
            
            # We use the extrapolate option to handle situations were floating point issues put us out of range
            tlev = interpolate.interp1d(p_temp[0,:],t_temp,axis=1,fill_value='extrapolate')(plev[0,:])

            # plt.figure()
            # plt.plot(play[0,:],tlay[0,:],'-o')
            # plt.plot(self.p_extension[:],self.t_extension[:],'--s')
            
            # fig.show()
            

            # qv to rrtmg shape; convert to vmr
            to_rrtmg_shape(_nhalo,self._ScalarState.get_field('qv'),self.qv_extension,h2ovmr, self.p_buffer, self._Ref._P0[-_nhalo[2]], self.p_extension[0] )
            h2ovmr *= parameters.RV/parameters.RD
            # ql to rrtmg shape; need to convert to path in g/m^2
            if 'ql' in self._ScalarState.names:
                to_rrtmg_shape(_nhalo, self._ScalarState.get_field('ql'), self.ql_extension, cliqwp, self.p_buffer,self._Ref._P0[-_nhalo[2]], self.p_extension[0] )
                cliqwp *= 1.e3/parameters.G * (plev[:-1]-plev[1:])

            #  qi to rrtmg shape; need to convert to path in g/m^2
            if 'qi' in self._ScalarState.names:
                to_rrtmg_shape(_nhalo, self._ScalarState.get_field('ql'), self.qi_extension, cicewp, self.p_buffer,self._Ref._P0[-_nhalo[2]], self.p_extension[0] )
                cicewp *= 1.e3/parameters.G * (plev[:-1]-plev[1:])
            
            play *= 0.01
            plev *= 0.01

            o3vmr = np.asfortranarray(np.repeat(self._profile_o3[np.newaxis,:],_ncol,axis=0))


    
            self._lib_lw.c_rrtmg_lw(_ncol, _nlay,  icld, idrv, 
            as_pointer(play), as_pointer(plev), as_pointer(tlay), as_pointer(tlev), 
            as_pointer(tsfc), as_pointer(h2ovmr), as_pointer(o3vmr), as_pointer(co2vmr), 
            as_pointer(ch4vmr), as_pointer(n2ovmr), as_pointer(o2vmr), as_pointer(cfc11vmr),
            as_pointer(cfc12vmr), as_pointer(cfc22vmr), as_pointer(ccl4vmr), as_pointer(emis),
            inflglw, iceflglw, liqflglw, as_pointer(cldfr), as_pointer(taucld_lw), 
            as_pointer(cicewp), as_pointer(cliqwp), as_pointer(reice), as_pointer(reliq), 
            as_pointer(tauaer_lw), as_pointer(uflx_lw), as_pointer(dflx_lw), as_pointer(hr_lw),
            as_pointer(uflxc_lw), as_pointer(dflxc_lw),  as_pointer(hrc_lw), as_pointer(duflx_dt),as_pointer(duflxc_dt))
            
           

            self._lib_sw.c_rrtmg_sw(_ncol, _nlay, icld, iaer, as_pointer(play), 
            as_pointer(plev), as_pointer(tlay), as_pointer(tlev), as_pointer(tsfc), 
            as_pointer(h2ovmr), as_pointer(o3vmr), as_pointer(co2vmr), as_pointer(ch4vmr),
            as_pointer(n2ovmr), as_pointer(o2vmr), as_pointer(asdir), as_pointer(asdif), 
            as_pointer(aldir), as_pointer(aldif), as_pointer(coszen), self._adjes, np.int(self.dyofyr),
            self._scon, inflgsw, iceflgsw, liqflgsw, as_pointer(cldfr), as_pointer(taucld_sw), 
            as_pointer(ssacld_sw), as_pointer(asmcld_sw), as_pointer(fsfcld_sw), as_pointer(cicewp),
            as_pointer(cliqwp),as_pointer(reice), as_pointer(reliq), 
            as_pointer(tauaer_sw), as_pointer(ssaaer_sw), as_pointer(asmaer_sw), as_pointer(ecaer_sw), 
            as_pointer(uflx_sw), as_pointer(dflx_sw), as_pointer(hr_sw), as_pointer(uflxc_sw),\
            as_pointer(dflxc_sw) , as_pointer(hrc_sw))

            # global number of non-ghost points 
            npts = self._Grid.n[0] * self._Grid.n[1]
            self._surf_sw_dn = np.sum(dflx_sw[:,0])/npts
            self._surf_sw_up = np.sum(uflx_sw[:,0])/npts
            self._surf_lw_dn = np.sum(dflx_lw[:,0])/npts
            self._surf_lw_up = np.sum(uflx_lw[:,0])/npts

            self._toa_sw_dn = np.sum(dflx_sw[:,-1])/npts
            self._toa_sw_up = np.sum(uflx_sw[:,-1])/npts
            self._toa_lw_dn = np.sum(dflx_lw[:,-1])/npts
            self._toa_lw_up = np.sum(uflx_lw[:,-1])/npts           

            # ds_uflux_lw = self._DiagnosticState.get_field('uflux_lw')
            # ds_dflux_lw = self._DiagnosticState.get_field('dflux_lw')
            # ds_uflux_sw = self._DiagnosticState.get_field('uflux_sw')
            # ds_dflux_sw = self._DiagnosticState.get_field('dflux_sw')
            ds_hr_lw = self._DiagnosticState.get_field('heating_rate_lw')
            ds_hr_sw = self._DiagnosticState.get_field('heating_rate_sw')

            rho0 = self._Ref._rho0

            to_our_shape(_nhalo, hr_lw, ds_hr_lw)
            to_our_shape(_nhalo, hr_sw, ds_hr_sw)
            # to_our_shape(_nhalo, uflx_lw, ds_uflux_lw)
            # to_our_shape(_nhalo, dflx_lw, ds_dflux_lw)
            # to_our_shape(_nhalo, uflx_sw, ds_uflux_sw)
            # to_our_shape(_nhalo, dflx_sw, ds_dflux_sw)

            # _ngrid_local = self._Grid._ngrid_local
            # for i in range(_ngrid_local[0]):
            #     for j in range(_ngrid_local[1]):
            #         for k in range(_ngrid_local[2]):
            #             ds_dTdt_rad[i,j,k] =  (ds_hr_lw[i,j,k] + ds_hr_sw[i,j,k])  /86400.0


            ds_dTdt_rad[:,:,:] = (ds_hr_lw + ds_hr_sw)  /86400.0
            ds_hr_lw[:,:,:]  *= rho0[np.newaxis,np.newaxis,:] * parameters.CPD /86400.0
            ds_hr_sw[:,:,:] *= rho0[np.newaxis,np.newaxis,:] * parameters.CPD /86400.0
          

        # FOR ALL RK STEPS
        st[:,:,:] += ds_dTdt_rad[:,:,:]


        return
    
    def io_initialize(self, nc_grp):
        if not self._compute_radiation:
            return
        timeseries_grp = nc_grp['timeseries']
        profiles_grp = nc_grp['profiles']

        v = timeseries_grp.createVariable('surface_sw_down', np.double, dimensions=('time',))
        v.long_name = 'surface shortwave down'
        v.standard_name = 'surface_sw_down'
        v.units = 'W/m^2'

        v = timeseries_grp.createVariable('surface_sw_up', np.double, dimensions=('time',))
        v.long_name = 'surface shortwave up'
        v.standard_name = 'surface_sw_up'
        v.units = 'W/m^2'

    
        v = timeseries_grp.createVariable('surface_lw_down', np.double, dimensions=('time',))
        v.long_name = 'surface longwave down'
        v.standard_name = 'surface_lw_down'
        v.units = 'W/m^2'


        v = timeseries_grp.createVariable('surface_lw_up', np.double, dimensions=('time',))
        v.long_name = 'surface longwave up'
        v.standard_name = 'surface_lw_up'
        v.units = 'W/m^2'

        v = timeseries_grp.createVariable('toa_sw_down', np.double, dimensions=('time',))
        v.long_name = 'TOA shortwave down'
        v.standard_name = 'toa_sw_down'
        v.units = 'W/m^2'

        v = timeseries_grp.createVariable('toa_sw_up', np.double, dimensions=('time',))
        v.long_name = 'TOA shortwave up'
        v.standard_name = 'toa_sw_up'
        v.units = 'W/m^2'

    
        v = timeseries_grp.createVariable('toa_lw_down', np.double, dimensions=('time',))
        v.long_name = 'toa longwave down'
        v.standard_name = 'surface_lw_down'
        v.units = 'W/m^2'


        v = timeseries_grp.createVariable('toa_lw_up', np.double, dimensions=('time',))
        v.long_name = 'toa longwave up'
        v.standard_name = 'toa_lw_up'
        v.units = 'W/m^2'

        return

    def io_update(self, nc_grp):
        if not self._compute_radiation:
            return
        my_rank = MPI.COMM_WORLD.Get_rank()
        if my_rank == 0:
            timeseries_grp = nc_grp['timeseries']
            profiles_grp = nc_grp['profiles']

            timeseries_grp['surface_sw_up'][-1] = UtilitiesParallel.ScalarAllReduce(self._surf_sw_up)
            timeseries_grp['surface_sw_down'][-1] = UtilitiesParallel.ScalarAllReduce(self._surf_sw_dn)
            timeseries_grp['surface_lw_up'][-1] = UtilitiesParallel.ScalarAllReduce(self._surf_lw_up)
            timeseries_grp['surface_lw_down'][-1] = UtilitiesParallel.ScalarAllReduce(self._surf_lw_dn)

            timeseries_grp['toa_sw_up'][-1] = UtilitiesParallel.ScalarAllReduce(self._toa_sw_up)
            timeseries_grp['toa_sw_down'][-1] = UtilitiesParallel.ScalarAllReduce(self._toa_sw_dn)
            timseries_grp['toa_lw_up'][-1] = UtilitiesParallel.ScalarAllReduce(self._toa_lw_up)
            timeseries_grp['toa_lw_down'][-1] = UtilitiesParallel.ScalarAllReduce(self._toa_lw_dn)
        return

    @property
    def name(self):
        return self._name           


# Does this work for plev?
@numba.njit
def to_rrtmg_shape(nhalo, our_array,  extension_array, rrtmg_array, p_buffer, p_mt, p_ext):
    shape = our_array.shape
    count = 0
    
    n_buffer = p_buffer.shape[0]
    n_ext = extension_array.shape[0] 
    mt_index = shape[2]- nhalo[2]-1

    for i in range(nhalo[0],shape[0]-nhalo[0]):
        for j in range(nhalo[1],shape[1]-nhalo[1]):
            slope = (extension_array[0]-our_array[i,j,mt_index])/(p_ext-p_mt)
            for k in range(nhalo[2], shape[2]- nhalo[2]):
                k_rrtmg = k - nhalo[2] #shape[2] - 1 - k
                rrtmg_array[count,k_rrtmg] = our_array[i,j,k]
            for k in range(n_buffer):
                k_rrtmg = shape[2]-2 * nhalo[2] + k
                rrtmg_array[count, k_rrtmg] = slope *(p_buffer[k] - p_mt) + our_array[i,j,mt_index]
            for k in range(n_ext):
                k_rrtmg = shape[2]-2 * nhalo[2] + n_buffer + k
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
    return ffi.cast("double*", numpy_array.ctypes.data)

@numba.njit
def  cos_sza(jday, hourz,  dlat,  dlon):

    epsiln = 0.016733
    sinob = 0.3978
    dpy = 365.242 #degrees per year
    dph = 15.0 #degrees per hour
    day_angle = 2.0*np.pi*(jday-1.)/dpy
    #Hours of Meridian Passage (true solar noon)
    homp =((12.0 + 0.12357*np.sin(day_angle) - 0.004289*np.cos(day_angle) 
            + 0.153809*np.sin(2*day_angle) + 0.060783*np.cos(2*day_angle)))
    hour_angle = dph*(hourz - homp) - dlon
    ang = 279.9348*np.pi/180. + day_angle
    sigma = (ang*180./np.pi + 0.4087*np.sin(ang) + 1.8724*np.cos(ang) - 0.0182*np.sin(2.*ang) + 0.0083*np.cos(2.*ang))*np.pi/180.
    sindlt = sinob*np.sin(sigma)
    cosdlt = np.sqrt(1. - sindlt*sindlt)
    return np.maximum(sindlt*np.sin(np.pi/180.*dlat) +cosdlt*np.cos(np.pi/180.*dlat)*np.cos(np.pi/180.*hour_angle), 0.0)

    
def interpolate_trace_gas(p_trace_pa, trace_vmr, p_pa):
    nlev = np.shape(p_pa)[0]
    trpath = np.zeros(nlev)
    ntrace = np.shape(p_trace_pa)[0]
    interp_vmr = np.zeros(nlev-1)
    for i in range(1,nlev):
        trpath[i] = trpath[i-1]
        if p_pa[i-1]>p_trace_pa[0]:
            trpath[i]+= (p_pa[i-1] - np.maximum(p_pa[i],p_trace_pa[0]))/9.81 * trace_vmr[0]
        for m in range(1,ntrace):
            plow = np.minimum(p_pa[i-1],np.maximum(p_pa[i],p_trace_pa[m-1]))
            pupp = np.minimum(p_pa[i-1],np.maximum(p_pa[i],p_trace_pa[m]))
            if plow>pupp:
                pmid = 0.5 * (plow + pupp)
                wgtlow = (pmid - p_trace_pa[m])/(p_trace_pa[m-1]-p_trace_pa[m])
                wgtupp = (p_trace_pa[m-1]-pmid)/(p_trace_pa[m-1]-p_trace_pa[m])
                trpath[i] += (plow-pupp)/9.81*(wgtlow*trace_vmr[m-1]  + wgtupp*trace_vmr[m])
        if (p_pa[i] < p_trace_pa[-1]):
            trpath[i] += (np.minimum(p_pa[i-1],p_trace_pa[-1]) - p_pa[i])/9.81 * trace_vmr[-1]
    
    for i in range(nlev-1):
        interp_vmr[i] = 9.81 /(p_pa[i]-p_pa[i+1]) * (trpath[i+1] - trpath[i])
    return interp_vmr