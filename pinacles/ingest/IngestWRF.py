import xarray as xr
import numpy as np
import os
from scipy import interpolate
from mpi4py import MPI
import datetime

from pinacles.ingest.IngestBase import IngestBase
from pinacles.ingest.IngestERA5 import IngestERA5

def rbf_interp(lon_in, lat_in, lon_out, lat_out, var, mask, i):
    
    
    lon_lat = (lon_in[mask], lat_in[mask])
    lat_lon_array = np.vstack(lon_lat).T
    
    rbf = interpolate.RBFInterpolator(
            lat_lon_array, var[i, :, :].flatten()[mask], neighbors = 16
    )
    
    
    return rbf(np.vstack((lon_out.flatten(), lat_out.flatten())).T)

def interp_griddata(lon_in, lat_in, lon_out, lat_out, var, mask, i):
    
    lon_lat = (lon_in[mask], lat_in[mask])
    
    return interpolate.griddata(
                lon_lat,var[i, :, :].flatten()[mask], (lon_out, lat_out), method="cubic")
    
rbf_interp = interp_griddata

class IngestWRF(IngestERA5):
    def __init__(self, namelist, Grid, TimeSteppingController):
        
        self._Grid = Grid 
        self._TimesSteppingController = TimeSteppingController
        
        assert "real_data" not in namelist
        self._real_data = namelist["meta"]["real_data"]
        assert os.path.exists(self._real_data)
        
        # Out model time
        self.start_time = datetime.datetime(
            namelist["time"]["year"],
            namelist["time"]["month"],
            namelist["time"]["day"],
            namelist["time"]["hour"],
        )
        
        self.start_time = np.datetime64(self.start_time)
        
        self.end_time = self.start_time + np.timedelta64(
            int(namelist["time"]["time_max"]), "s"
        
        )
        
        #Lat and lon bounds for interpolation
        self.lat_margin = (3000.0 / 1000.0)/110.0 * 5.0
        self.lon_margin = (3000.0 / 1000.0)/110.0 * 5.0
        
        #print(self.lat_margin, self.lon_margin)
        #import sys; sys.exit()
        
        self.sfc_data_file = self._real_data #'./support/test_out_frontal.nc'
        self.atm_data_file = self._real_data #'./support/test_out_frontal.nc'

        #WAVE DATA INGEST
        try:
            self._WW3_data_prefix= namelist["meta"]["WW3_data"]
        except:
            self._WW3_data_prefix = None
        
        if self._WW3_data_prefix is not None:
            self._WW3_lm_file = self._WW3_data_prefix + '_lm.nc'
            self._WW3_lp_file = self._WW3_data_prefix + '_pkwl.nc'
            self._WW3_hs_file = self._WW3_data_prefix + '_hs.nc'

            assert os.path.exists(self._WW3_lm_file)
            assert os.path.exists(self._WW3_lp_file)
            assert os.path.exists(self._WW3_hs_file)
           
        
    def initialize(self):
        self.get_times()
        if self._WW3_data_prefix is not None:
            self.get_times_wave()

        
        
    def get_times(self):
        
        # print("Getting model times.")
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Simulation starts at: ", self.start_time)
            print("Simulation ends at ", self.end_time)
        
            print("Reading in timedata data from input files.")

        
            wrf_data = xr.open_dataset(self.atm_data_file)
        
            
            try:
                self.time_atm = wrf_data.XTIME.values
            except:
                self.time_atm = wrf_data.time.values
            self.time_sfc = self.time_atm
            
            print(self.time_atm, self.time_atm[-1] - self.end_time )
            
            print("Checking input data covers the correct time range.")
            assert self.time_atm[0] - self.start_time <= 0
            assert self.time_atm[-1] - self.end_time >= 0
            print("Great, the data covers the correct range.")
            
            self.atm_timeindx = np.abs(self.time_atm - self.start_time).argmin()
            self.sfc_timeindx = np.abs(self.time_sfc - self.start_time).argmin()

            print("\t Starting index for surface: ", self.sfc_timeindx)
            print("\t Start index for atmosphere: ", self.atm_timeindx)
            
        else:
            self.time_atm = None
            self.time_sfc = None
        
            self.atm_timeindx = None
            self.sfc_timeindx = None
        
            
        # Broadcasting datat to all nodes
        self.time_atm = MPI.COMM_WORLD.bcast(self.time_atm)
        self.time_sfc = MPI.COMM_WORLD.bcast(self.time_sfc)

        self.atm_timeindx = MPI.COMM_WORLD.bcast(self.atm_timeindx)
        self.sfc_timeindx = MPI.COMM_WORLD.bcast(self.sfc_timeindx)  

        if self._WW3_data_prefix is not None:
            self.get_times_wave()
        return
            
    def get_times_wave(self):
        # print("Getting wave data times.")
        if MPI.COMM_WORLD.Get_rank() == 0:
            ww3_data = xr.open_dataset(self._WW3_hs_file)
            # Note: This makes a very mild assumption that times are consistent 
            # among the different wave-watch output files. this should be okay
            # due to requiring them all to have the same "prefix" but just be aware
            #--Colleen
            self.time_wave = ww3_data.time.values
            print("Checking wave input data covers the correct time range.")
            assert self.time_wave[0] - self.start_time <= 0
            assert self.time_wave[-1] - self.end_time >= 0
            print("Great, the wave data covers the correct range.")
            
            self.wave_timeindx = np.abs(self.time_wave - self.start_time).argmin()

            print("\t Starting index for wave: ", self.wave_timeindx)
        else:
            self.time_wave = None
            
        
            self.wave_timeindx = None
            
        
            
        # Broadcasting datat to all nodes
        self.time_wave = MPI.COMM_WORLD.bcast(self.time_wave)
       
        self.wave_timeindx = MPI.COMM_WORLD.bcast(self.wave_timeindx)
   
        return   
    def get_skin_T(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(
                self._real_data
            )
            skin_T = sfc_data.SST.values[self.sfc_timeindx + shift, :, :]
            lon = sfc_data.longitude.values[:,:]
            lat = sfc_data.latitude.values[:,:]
            

        else:
            lon = None
            lat = None
            skin_T = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        skin_T = MPI.COMM_WORLD.bcast(skin_T)

        return lon, lat, skin_T

    


    def get_slp(self, shift=0):

        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(
                self._real_data
            )
            slp = sfc_data.PSFC.values[self.sfc_timeindx + shift, :, :]
            lon = sfc_data.longitude.values[:,:]
            lat = sfc_data.latitude.values[:,:]
        else:
            lon = None
            lat = None
            slp = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        slp = MPI.COMM_WORLD.bcast(slp)

        return lon, lat, slp
    
    
    def interp_height(self, lon, lat, shift=0):

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert lat_shape == lon_shape

        lon_hgt, lat_hgt, hgt = self.get_hgt(shift=shift)

        lon_hgt_grid = lon_hgt.flatten()
        lat_hgt_grid = lat_hgt.flatten()



    
        # Mask data to make interpolation more efficient
        mask = (lon_hgt_grid >= np.amin(lon) - self.lon_margin) & (
            lon_hgt_grid <= np.amax(lon) + self.lon_margin
        )

        mask = (
            mask
            & (lat_hgt_grid >= np.amin(lat) - self.lat_margin)
            & (lat_hgt_grid <= np.amax(lat) + self.lon_margin)
        )
        


        hgt_horizontal = np.zeros(
            (hgt.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        
        for i in range(hgt.shape[0]):
            field = interp_griddata(lon_hgt_grid, lat_hgt_grid, lon, lat, hgt, mask, i)
            hgt_horizontal[i, :, :] = field.reshape(
                hgt_horizontal[i, :, :].shape)

            if np.isnan(hgt_horizontal[i, :, :]).any() == True:
                import pylab as plt 
                plt.pcolor(hgt_horizontal[i, :, :])
                plt.show()


        return hgt_horizontal
    
    def get_hgt(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            hgt = atm_data.Z.values[self.sfc_timeindx + shift, :, :, :]

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            lon = None
            lat = None
            hgt = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        hgt = MPI.COMM_WORLD.bcast(hgt)

        #hgt[0,:,:] = 0.0

        return lon, lat, hgt #0.5 * (hgt[1:,:,:] + hgt[:-1,:,:])

    def get_T(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            t = np.array(atm_data.T[self.sfc_timeindx + shift, :, :, :])
            t2m = np.array(atm_data.T2m[self.sfc_timeindx + shift, :, :])

            t = np.concatenate((t2m.reshape(1, t2m.shape[0], t2m.shape[1]), t), axis=0)
            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            t = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        t = MPI.COMM_WORLD.bcast(t)

        return lon, lat, t
    

    def get_P(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            p = np.array(atm_data.P[self.sfc_timeindx + shift, :, :, :])

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            p = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        p = MPI.COMM_WORLD.bcast(p)

        return lon, lat, p


    def interp_T(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(
            self._Grid.lon_local, self._Grid.lat_local
        )



        lat_shape = lat.shape
        lon_shape = lon.shape


        lon_T, lat_T, T = self.get_T(shift=shift)


        lon_T_grid = lon_T.flatten()
        lat_T_grid = lat_T.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_T_grid >= np.amin(lon) - self.lon_margin) & (lon_T_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_T_grid >= np.amin(lat) - self.lat_margin)
            & (lat_T_grid <= np.amax(lat) + self.lat_margin)
        )

        T_horizontal = np.empty(
            (T.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        for i in range(T.shape[0]):

            field = rbf_interp(lon_T_grid, lat_T_grid, lon, lat, T, mask, i)

            T_horizontal[i, :, :] = field.reshape(
                T_horizontal[i, :, :].shape)


        Ti = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)
        nh = self._Grid.n_halo
        
        for i in range(T_horizontal.shape[1]):
            for j in range(T_horizontal.shape[2]):



                z = hgt_horizontal_interp[:, i, j]
                
                
                z = np.concatenate(([2.0], z))

                ids = np.argsort(z)
                z = z[ids]
                Th =  T_horizontal[ids, i, j]
                interp = interpolate.Akima1DInterpolator(z, Th)
                Ti[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )




        return Ti
    
    

    # def get_T(self, shift=0):
    #     if MPI.COMM_WORLD.Get_rank() == 0:

    #         atm_data = xr.open_dataset(
    #             self._real_data
    #         )

    #         t = np.array(atm_data.T[self.sfc_timeindx + shift, :, :, :])

    #         lat = atm_data.latitude.values[:,:]
    #         lon = atm_data.longitude.values[:,:]

    #     else:
    #         t = None
    #         lat = None
    #         lon = None

    #     lon = MPI.COMM_WORLD.bcast(lon)
    #     lat = MPI.COMM_WORLD.bcast(lat)
    #     t = MPI.COMM_WORLD.bcast(t)

    #     return lon, lat, t
    

    def get_qv(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            qv = np.array(atm_data.QV[self.sfc_timeindx + shift, :, :, :])
            qv2m = np.array(atm_data.QV2m[self.sfc_timeindx + shift, :, :])
            qv = np.concatenate((qv2m.reshape(1, qv2m.shape[0], qv2m.shape[1]), qv), axis=0)

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            qv = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        qv = MPI.COMM_WORLD.bcast(qv)


        return lon, lat, qv

    def interp_qv(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(
            self._Grid.lon_local, self._Grid.lat_local
        )

        lat_shape = lat.shape
        lon_shape = lon.shape


        lon_qv, lat_qv, qv = self.get_qv(shift=shift)

        lon_qv_grid = lon_qv.flatten()
        lat_qv_grid = lat_qv.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_qv_grid >= np.amin(lon) - self.lon_margin) & (lon_qv_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_qv_grid >= np.amin(lat) - self.lat_margin)
            & (lat_qv_grid <= np.amax(lat) + self.lat_margin)
        )

        qv_horizontal = np.empty(
            (qv.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        for i in range(qv.shape[0]):

            field = rbf_interp(lon_qv_grid, lat_qv_grid, lon, lat, qv, mask, i)

            qv_horizontal[i, :, :] = field.reshape(
                qv_horizontal[i, :, :].shape)


        qvi = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)
        nh = self._Grid.n_halo
        for i in range(qv_horizontal.shape[1]):
            for j in range(qv_horizontal.shape[2]):

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([2.0], z))
                
                ids = np.argsort(z)
                z = z[ids]
                qvh = qv_horizontal[ids, i, j]
                interp = interpolate.Akima1DInterpolator(z, qvh)
                
                
                qvi[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )

        # Make sure qv is positive otherwise bad things can happen.

        return qvi
    
    def get_qc(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            qc = np.array(atm_data.QC[self.sfc_timeindx + shift, :, :, :])
            qc = np.concatenate((np.zeros((1, qc.shape[1], qc.shape[2])), qc), axis=0)

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            qc = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        qc = MPI.COMM_WORLD.bcast(qc)

        return lon, lat, qc

    def interp_qc(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(
            self._Grid.lon_local, self._Grid.lat_local
        )

        lat_shape = lat.shape
        lon_shape = lon.shape


        lon_qc, lat_qc, qc = self.get_qc(shift=shift)

        lon_qc_grid = lon_qc.flatten()
        lat_qc_grid = lat_qc.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_qc_grid >= np.amin(lon) - self.lon_margin) & (lon_qc_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_qc_grid >= np.amin(lat) - self.lat_margin)
            & (lat_qc_grid <= np.amax(lat) + self.lat_margin)
        )

        qc_horizontal = np.empty(
            (qc.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        for i in range(qc.shape[0]):

            field = rbf_interp(lon_qc_grid, lat_qc_grid, lon, lat, qc, mask, i)

            qc_horizontal[i, :, :] = field.reshape(
                qc_horizontal[i, :, :].shape)


        qci = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)
        nh = self._Grid.n_halo
        for i in range(qc_horizontal.shape[1]):
            for j in range(qc_horizontal.shape[2]):

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([0.0], z))

                ids = np.argsort(z)
                z = z[ids]
                qch = qc_horizontal[ids, i, j]

                interp = interpolate.Akima1DInterpolator(z, qch)
                qci[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )


        return qci
    
    
    def get_qi(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            qi = np.array(atm_data.QI[self.sfc_timeindx + shift, :, :, :])
            qi = np.concatenate((np.zeros((1, qi.shape[1], qi.shape[2])), qi), axis=0)

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]

        else:
            qi = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        qi = MPI.COMM_WORLD.bcast(qi)

        return lon, lat, qi

    def interp_qi(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(
            self._Grid.lon_local, self._Grid.lat_local
        )

        lat_shape = lat.shape
        lon_shape = lon.shape


        lon_qi, lat_qi, qi = self.get_qi(shift=shift)

        lon_qi_grid = lon_qi.flatten()
        lat_qi_grid = lat_qi.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_qi_grid >= np.amin(lon) - self.lon_margin) & (lon_qi_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_qi_grid >= np.amin(lat) - self.lat_margin)
            & (lat_qi_grid <= np.amax(lat) + self.lat_margin)
        )

        qi_horizontal = np.empty(
            (qi.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        for i in range(qi.shape[0]):

            field = rbf_interp(lon_qi_grid, lat_qi_grid, lon, lat, qi, mask, i)

            qi_horizontal[i, :, :] = field.reshape(
                qi_horizontal[i, :, :].shape)


        qii = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)
        nh = self._Grid.n_halo
        for i in range(qi_horizontal.shape[1]):
            for j in range(qi_horizontal.shape[2]):

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([0.0], z))

                ids = np.argsort(z)
                z = z[ids]
                qih = qi_horizontal[ids, i, j]
                interp = interpolate.Akima1DInterpolator(z, qih)
                qii[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )


        return qii

    def get_u(self, shift=0):

        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            # u10 = np.array(sfc_data.u10[self.sfc_timeindx + shift, :, :])
            u = np.array(atm_data.U[self.atm_timeindx + shift, :, :, :])
            u10m = np.array(atm_data.U10m[self.sfc_timeindx + shift, :, :])
            u = np.concatenate((u10m.reshape(1,u10m.shape[0], u10m.shape[1]), u), axis=0)            
            # u = np.concatenate((u10.reshape(1,u10.shape[0], u10.shape[1]), u), axis=0)

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]
            
            

        else:
            u = None
            lat = None
            lon = None
        u = MPI.COMM_WORLD.bcast(u)
        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)

        return lon, lat, u
    
    def interp_u(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(lon, lat)

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert lat_shape == lon_shape
        assert len(np.shape(height)) == 1

        lon_u, lat_u, u = self.get_u(shift=shift)

        u[:, :, :] = u  # uu[:,:,:]

        lon_u_grid = lon_u.flatten()
        lat_u_grid = lat_u.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_u_grid >= np.amin(lon) - self.lon_margin) & (lon_u_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_u_grid >= np.amin(lat) - self.lat_margin)
            & (lat_u_grid <= np.amax(lat) + self.lon_margin)
        )

        u_horizontal = np.empty(
            (u.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        for i in range(u.shape[0]):


            field = rbf_interp(lon_u_grid, lat_u_grid, lon, lat, u, mask, i)

            u_horizontal[i, :, :] = field.reshape(
                u_horizontal[i, :, :].shape
            ) 
        
        ui = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)

        nh = self._Grid.n_halo

        for i in range(u_horizontal.shape[1]):
            for j in range(u_horizontal.shape[2]):

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([10.0], z))		
                ids = np.argsort(z)
                z = z[ids]
                uh = u_horizontal[ids, i, j]

                interp = interpolate.Akima1DInterpolator(z[:], uh)
                ui[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )


        return ui
    
    def get_v(self, shift=0):

        if MPI.COMM_WORLD.Get_rank() == 0:

            atm_data = xr.open_dataset(
                self._real_data
            )

            # u10 = np.array(sfc_data.u10[self.sfc_timeindx + shift, :, :])
            v = np.array(atm_data.V[self.atm_timeindx + shift, :, :, :])
            v10m = np.array(atm_data.V10m[self.sfc_timeindx + shift, :, :])
            v = np.concatenate((v10m.reshape(1,v10m.shape[0], v10m.shape[1]), v), axis=0)

            lat = atm_data.latitude.values[:,:]
            lon = atm_data.longitude.values[:,:]
            
            

        else:
            v = None
            lat = None
            lon = None
        v = MPI.COMM_WORLD.bcast(v)
        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)

        return lon, lat, v
    
    def interp_v(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(lon, lat)

        lat_shape = lat.shape
        lon_shape = lon.shape

        lon_v, lat_v, v = self.get_v(shift=shift)

        v[:, :, :] = v  # uu[:,:,:]

        lon_v_grid = lon_v.flatten()
        lat_v_grid = lat_v.flatten()

        # Mask data to make interpolation more efficient
        mask = (lon_v_grid >= np.amin(lon) - self.lon_margin) & (lon_v_grid <= np.amax(lon) + self.lon_margin)

        mask = (
            mask
            & (lat_v_grid >= np.amin(lat) - self.lat_margin)
            & (lat_v_grid <= np.amax(lat) + self.lat_margin)
        )

        v_horizontal = np.empty(
            (v.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double
        )
        
        for i in range(v.shape[0]):

            field = rbf_interp(lon_v_grid, lat_v_grid, lon, lat, v, mask, i)

            v_horizontal[i, :, :] = field.reshape(
                v_horizontal[i, :, :].shape
            ) 
        
        vi = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double)

        nh = self._Grid.n_halo

        for i in range(v_horizontal.shape[1]):
            for j in range(v_horizontal.shape[2]):

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([10.0], z))

                z = hgt_horizontal_interp[:, i, j]
                z = np.concatenate(([10.0], z))
                ids = np.argsort(z)
                z = z[ids]

                vh = v_horizontal[ids, i, j]

                interp = interpolate.Akima1DInterpolator(z[:], vh)
                vi[i, j, :] = np.pad(
                    interp.__call__(height[nh[2] : -nh[2]]), nh[2], mode="edge"
                )


        return vi
    
    def get_mean_wavelength(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            lm_data = xr.open_dataset(
                self._WW3_lm_file 
            )
            wave_lm = lm_data.lm.values[self.wave_timeindx + shift, :, :]
            lon = lm_data.longitude.values[:,:]
            lat = lm_data.latitude.values[:,:]
            

        else:
            lon = None
            lat = None
            wave_lm = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        wave_lm = MPI.COMM_WORLD.bcast(wave_lm)

        return lon, lat, wave_lm
    
    def get_peak_wavelength(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            lp_data = xr.open_dataset(
                self._WW3_lp_file 
            )
            wave_lp = lp_data.pkwl.values[self.wave_timeindx + shift, :, :]
            grid = np.meshgrid(lp_data.longitude.values,lp_data.latitude.values)
            lon = (grid[0][:][:]+ 180) % 360 -180
            lat = grid[1][:][:]

        else:
            lon = None
            lat = None
            wave_lp = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        wave_lp = MPI.COMM_WORLD.bcast(wave_lp)

        return lon, lat, wave_lp
    
    def get_peak_wavelength(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            lp_data = xr.open_dataset(
                self._WW3_lp_file 
            )
            wave_lp = lp_data.pkwl.values[self.wave_timeindx + shift, :, :]
            grid = np.meshgrid(lp_data.longitude.values,lp_data.latitude.values)
            lon = (grid[0][:][:]+ 180) % 360 -180
            lat = grid[1][:][:]

        else:
            lon = None
            lat = None
            wave_lp = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        wave_lp = MPI.COMM_WORLD.bcast(wave_lp)

        return lon, lat, wave_lp
    
        
    def get_wave_height(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            hs_data = xr.open_dataset(
                self._WW3_hs_file 
            )
            wave_hs = hs_data.hs.values[self.wave_timeindx + shift, :, :]
            grid = np.meshgrid(hs_data.longitude.values,hs_data.latitude.values)
            lon = (grid[0][:][:]+ 180) % 360 -180
            lat = grid[1][:][:]
            

        else:
            lon = None
            lat = None
            wave_hs = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        wave_hs = MPI.COMM_WORLD.bcast(wave_hs)

        return lon, lat, wave_hs
    
    def get_wave_mapstatus(self):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            hs_data = xr.open_dataset(
                self._WW3_hs_file 
            )
            mapstatus = hs_data.MAPSTA.values[:, :]
           

        else:
            mapstatus = None

      
        mapstatus = MPI.COMM_WORLD.bcast(mapstatus)

        return mapstatus