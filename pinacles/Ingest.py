import xarray as xr
import numpy as np
import os
from scipy import interpolate
from mpi4py import MPI
import datetime

class IngestBase:
    def __init__(self, namelist, Grid, TimesSteppingController):

        self._Grid = Grid
        self._TimeSteppingManager = TimesSteppingController

        return


class IngestEra5:

    def __init__(self, namelist, Grid, TimeSteppingController):

        IngestBase.__init__(self, namelist, Grid, TimeSteppingController)

        assert("real_data" not in namelist)
        self._real_data = namelist['meta']['real_data']
        assert(os.path.exists(self._real_data))

        self.start_time = datetime.datetime(namelist["time"]["year"],
            namelist["time"]["month"],
            namelist["time"]["day"],
            namelist["time"]["hour"])


        self.start_time = np.datetime64(self.start_time)    
        self.end_time = self.start_time + np.timedelta64(int(namelist["time"]["time_max"]), 's')


        return

    def initialize(self):



        self.get_times()

        return


    def get_times(self):

        print('Getting model times.')
        if MPI.COMM_WORLD.Get_rank() == 0:

            print('Simulation starts at: ', self.start_time)
            print('Simulation ends at ', self.end_time)

            print('Reading in timedata data from input files.')

            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            self.time_atm = atm_data.time.values
            self.time_sfc = sfc_data.time.values


            print('Checking input data covers the correct time range.')
            for td in [self.time_atm, self.time_sfc]:
                assert( (td[0] - self.start_time <= 0))
                assert( (td[-1] - self.end_time >= 0))
            print('Great, the data covers the correct range.')

            self.atm_timeindx = np.abs(self.time_atm - self.start_time).argmin()
            self.sfc_timeindx = np.abs(self.time_sfc - self.start_time).argmin()

            print('\t Starting index for surface: ', self.sfc_timeindx)
            print('\t Start index for atmosphere: ', self.atm_timeindx)
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
        
        return

    def get_u(self, shift=0):

        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            u10 = np.array(sfc_data.u10[self.sfc_timeindx + shift, :, :])
            u = np.array(atm_data.u[self.atm_timeindx + shift, ::-1, :, :])
            u = np.concatenate((u10.reshape(1,u10.shape[0], u10.shape[1]), u), axis=0)

            lat = sfc_data.latitude.values
            lon = sfc_data.longitude.values


        else:
            u = None
            lat = None
            lon = None
        u = MPI.COMM_WORLD.bcast(u)
        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)

        return lon, lat, u 


    def interp_height(self, lon, lat, shift=0):

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert(lat_shape == lon_shape)

        lon_hgt, lat_hgt, hgt = self.get_hgt(shift=shift)
        lon_hgt_grid, lat_hgt_grid = np.meshgrid(lon_hgt, lat_hgt)
        
        lon_hgt_grid = lon_hgt_grid.flatten()
        lat_hgt_grid = lat_hgt_grid.flatten()


        # Mask data to make interpolation more efficient
        mask = (lon_hgt_grid >= np.amin(lon) - 1.0) & (lon_hgt_grid <= np.amax(lon) + 1.0)
        
        mask = mask & (lat_hgt_grid >= np.amin(lat) - 1.0) & (lat_hgt_grid <= np.amax(lat) + 1.0)
        lon_lat = (lon_hgt_grid[mask], lat_hgt_grid[mask])


        hgt_horizontal = np.empty((hgt.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double)
        for i in range(hgt.shape[0]):
            hgt_horizontal[i,:,:] = interpolate.griddata(lon_lat, 
                                            hgt[i,:,:].flatten()[mask],
                                            (lon, lat), method='linear')

        return hgt_horizontal 

    def interp_T(self, lon, lat, height, shift=0):
            
        hgt_horizontal_interp = self.interp_height(self._Grid.lon_local, self._Grid.lat_local)

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert(lat_shape == lon_shape)
        assert(len(np.shape(height)) == 1)

        lon_T, lat_T, T = self.get_T(shift=shift)
        lon_T_grid, lat_T_grid = np.meshgrid(lon_T, lat_T)
        
        lon_T_grid = lon_T_grid.flatten()
        lat_T_grid = lat_T_grid.flatten()


        # Mask data to make interpolation more efficient
        mask = (lon_T_grid >= np.amin(lon) - 1.0) & (lon_T_grid <= np.amax(lon) + 1.0)
        
        mask = mask & (lat_T_grid >= np.amin(lat) - 1.0) & (lat_T_grid <= np.amax(lat) + 1.0)
        lon_lat = (lon_T_grid[mask], lat_T_grid[mask])

        
        T_horizontal = np.empty((T.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double)
        for i in range(T.shape[0]):

            lat_lon_array = np.vstack(lon_lat).T

            rbf = interpolate.RBFInterpolator(lat_lon_array, T[i,:,:].flatten()[mask], neighbors=6)

            field = rbf(np.vstack((lon.flatten(), lat.flatten())).T)

            T_horizontal[i,:,:] = field.reshape(T_horizontal[i,:,:].shape) #interpolate.griddata(lon_lat, 
                                  #          T[i,:,:].flatten()[mask],
                                  #          (lon, lat), method='linear')


        Ti = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double) 
        nh = self._Grid.n_halo
        for i in range(T_horizontal.shape[1]):
            for j in range(T_horizontal.shape[2]):

                z = hgt_horizontal_interp[:,i,j]
                z = np.concatenate(([2.0], z))

                interp = interpolate.Akima1DInterpolator(z, T_horizontal[:,i,j])
                Ti[i,j,:] =  np.pad(interp.__call__(height[nh[2]:-nh[2]]), 3, mode='edge')


        return Ti

    def interp_qv(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(self._Grid.lon_local, self._Grid.lat_local)

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert(lat_shape == lon_shape)
        assert(len(np.shape(height)) == 1)

        lon_qv, lat_qv, qv = self.get_qv(shift=shift)
        lon_qv_grid, lat_qv_grid = np.meshgrid(lon_qv, lat_qv)
        
        lon_qv_grid = lon_qv_grid.flatten()
        lat_qv_grid = lat_qv_grid.flatten()


        # Mask data to make interpolation more efficient
        mask = (lon_qv_grid >= np.amin(lon) - 1.0) & (lon_qv_grid <= np.amax(lon) + 1.0)
        
        mask = mask & (lat_qv_grid >= np.amin(lat) - 1.0) & (lat_qv_grid <= np.amax(lat) + 1.0)
        lon_lat = (lon_qv_grid[mask], lat_qv_grid[mask])


        qv_horizontal = np.empty((qv.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double)
        for i in range(qv.shape[0]):

            lat_lon_array = np.vstack(lon_lat).T

            rbf = interpolate.RBFInterpolator(lat_lon_array, qv[i,:,:].flatten()[mask], neighbors=6)
            field = rbf(np.vstack((lon.flatten(), lat.flatten())).T)


            qv_horizontal[i,:,:] = field.reshape(qv_horizontal[i,:,:].shape)
                                  #interpolate.griddata(lon_lat, 
                                   #         qv[i,:,:].flatten()[mask],
                                   #         (lon, lat), method='linear')


        qvi = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double) 
        nh = self._Grid.n_halo
        for i in range(qv_horizontal.shape[1]):
            for j in range(qv_horizontal.shape[2]):

                z = hgt_horizontal_interp[:,i,j]

                interp = interpolate.Akima1DInterpolator(z, qv_horizontal[:,i,j])
                qvi[i,j,:] =  np.pad(interp.__call__(height[nh[2]:-nh[2]]), 3, mode='edge')


        return qvi 


    def interp_u(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(lon, lat)

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert(lat_shape == lon_shape)
        assert(len(np.shape(height)) == 1)

        lon_u, lat_u, u = self.get_u(shift=shift)
        lon_u_grid, lat_u_grid = np.meshgrid(lon_u, lat_u)
        
        lon_v, lat_v, v = self.get_v(shift=shift)
        uu, vv = self._Grid.MapProj.rotate_wind(lon_u_grid, u, v)
        u[:,:,:] = uu[:,:,:]

        lon_u_grid = lon_u_grid.flatten()
        lat_u_grid = lat_u_grid.flatten()


        # Mask data to make interpolation more efficient
        mask = (lon_u_grid >= np.amin(lon) - 1.0) & (lon_u_grid <= np.amax(lon) + 1.0)
        
        mask = mask & (lat_u_grid >= np.amin(lat) - 1.0) & (lat_u_grid <= np.amax(lat) + 1.0)
        lon_lat = (lon_u_grid[mask], lat_u_grid[mask])


        u_horizontal = np.empty((u.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double)
        for i in range(u.shape[0]):
            
            
            lat_lon_array = np.vstack(lon_lat).T
            rbf = interpolate.RBFInterpolator(lat_lon_array, u[i,:,:].flatten()[mask], neighbors=6)

            field = rbf(np.vstack((lon.flatten(), lat.flatten())).T)

            

            #import sys; sys.exit()
            u_horizontal[i,:,:] = field.reshape(u_horizontal[i,:,:].shape) #interpolate.griddata(lon_lat, 
                                  #          u[i,:,:].flatten()[mask],
                                  #          (lon, lat), method='linear')


        ui = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double) 

        nh = self._Grid.n_halo

        for i in range(u_horizontal.shape[1]):
            for j in range(u_horizontal.shape[2]):

                z = hgt_horizontal_interp[:,i,j]
                z = np.concatenate(([10.0], z))

                interp = interpolate.Akima1DInterpolator(z[1:], u_horizontal[1:,i,j])
                ui[i,j,:] =  np.pad(interp.__call__(height[nh[2]:-nh[2]]), 3, mode='edge')


        return ui

    def interp_v(self, lon, lat, height, shift=0):

        hgt_horizontal_interp = self.interp_height(lon, lat)

        lat_shape = lat.shape
        lon_shape = lon.shape
        assert(lat_shape == lon_shape)
        assert(len(np.shape(height)) == 1)

        lon_v, lat_v, v = self.get_v(shift=shift)
        lon_v_grid, lat_v_grid = np.meshgrid(lon_v, lat_v)
        
        lon_u, lat_u, u = self.get_u(shift=shift)
        uu, vv = self._Grid.MapProj.rotate_wind(lon_v_grid, u, v)
        v[:,:,:] = vv[:,:,:]

        lon_v_grid = lon_v_grid.flatten()
        lat_v_grid = lat_v_grid.flatten()


        # Mask data to make interpolation more efficient
        mask = (lon_v_grid >= np.amin(lon) - 1.0) & (lon_v_grid <= np.amax(lon) + 1.0)
        
        mask = mask & (lat_v_grid >= np.amin(lat) - 1.0) & (lat_v_grid <= np.amax(lat) + 1.0)
        lon_lat = (lon_v_grid[mask], lat_v_grid[mask])


        v_horizontal = np.empty((v.shape[0], lon.shape[0], lon.shape[1]), dtype=np.double)
        for i in range(v.shape[0]):
            
            lat_lon_array = np.vstack(lon_lat).T
            rbf = interpolate.RBFInterpolator(lat_lon_array, v[i,:,:].flatten()[mask], neighbors=6)

            field = rbf(np.vstack((lon.flatten(), lat.flatten())).T)
            
            v_horizontal[i,:,:] = field.reshape(v_horizontal[i,:,:].shape)#interpolate.griddata(lon_lat, 
                                  #          v[i,:,:].flatten()[mask],
                                  #          (lon, lat), method='linear')


        vi = np.empty((lon.shape[0], lon.shape[1], height.shape[0]), dtype=np.double) 


        nh = self._Grid.n_halo

        for i in range(v_horizontal.shape[1]):
            for j in range(v_horizontal.shape[2]):

                z = hgt_horizontal_interp[:,i,j]
                z = np.concatenate(([10.0], z))

                interp = interpolate.Akima1DInterpolator(z[1:], v_horizontal[1:,i,j])
                vi[i,j,:] = np.pad(interp.__call__(height[nh[2]:-nh[2]]), 3, mode='edge')


        return vi



    def get_v(self, shift=0):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            v10 = np.array(sfc_data.v10[self.sfc_timeindx + shift, :, :])
            v = np.array(atm_data.v[self.atm_timeindx + shift, ::-1, :, :])
            v = np.concatenate((v10.reshape(1, v10.shape[0], v10.shape[1]), v), axis=0)

            lat = sfc_data.latitude.values
            lon = sfc_data.longitude.values


        else:
            v = None
            lat = None
            lon = None

        v = MPI.COMM_WORLD.bcast(v)
        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)

        return lon, lat, v


    def get_T(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            t2m = np.array(sfc_data.t2m[self.sfc_timeindx + shift, :, :])
            t = np.array(atm_data.t[self.sfc_timeindx + shift, ::-1, :, :])

            t2m = t2m.reshape(1, t2m.shape[0], t2m.shape[1])

            t = np.concatenate((t2m,t), axis=0)

            lat = sfc_data.latitude.values
            lon = sfc_data.longitude.values


        else:
            t = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        t = MPI.COMM_WORLD.bcast(t)

        return lon , lat, t


    def get_qv(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:
            
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            q = np.array(atm_data.q[self.sfc_timeindx + shift, ::-1, :, :])
            lat = sfc_data.latitude.values
            lon = sfc_data.longitude.values

        else:
            q = None
            lat = None
            lon = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        q = MPI.COMM_WORLD.bcast(q)

        return lon, lat, q

    def get_hgt(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            atm_data = xr.open_dataset(os.path.join(self._real_data, 'atm_data_to_pinacles.nc'))

            hgt = np.array(atm_data.hgt[self.sfc_timeindx + shift, ::-1, :, :])
            lat = sfc_data.latitude.values
            lon = sfc_data.longitude.values

        else:
            lon = None
            lat = None 
            hgt = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        hgt = MPI.COMM_WORLD.bcast(hgt)

        return lon, lat, hgt

    def get_skin_T(self, shift=0):
        if MPI.COMM_WORLD.Get_rank() == 0:
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            skin_T = sfc_data.skt.values[self.sfc_timeindx + shift,:,:]
            lon = sfc_data.longitude.values 
            lat = sfc_data.latitude.values 
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
            sfc_data = xr.open_dataset(os.path.join(self._real_data, 'sfc_data_to_pinacles.nc'))
            slp = sfc_data.sp.values[self.sfc_timeindx + shift,:,:]
            lon = sfc_data.longitude.values
            lat = sfc_data.latitude.values
        else:
            lon = None
            lat = None
            slp = None

        lon = MPI.COMM_WORLD.bcast(lon)
        lat = MPI.COMM_WORLD.bcast(lat)
        slp = MPI.COMM_WORLD.bcast(slp)

        return lon, lat, slp


class IngestMerra(IngestBase):
    def __init__(self, namelist, Grid):

        IngestBase.__init__(self, namelist, Grid)

        assert("real_data" not in namelist)
        self._real_data = namelist['meta']['real_data']
        assert(os.path.exists(self._real_data))

        self.get_times()


        return


    def get_times(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            ocn_xr = xr.open_mfdataset(os.path.join(self._real_data, '*ocn*'))
            asm_xr = xr.open_mfdataset(os.path.join(self._real_data, '*asm*'))

            for time in ocn_xr.time:
                print(time.values)

            for time in asm_xr.time:
                print(time.values)

        return
    
class IngestE3SM(IngestBase):
    def __init__(self, namelist, Grid):

        IngestBase.__init__(self, namelist, Grid)

        assert("real_data" not in namelist)
        self._real_data = namelist['meta']['real_data']
        assert(os.path.exists(self._real_data))

        for v in ["year", "month", "day", "hour"]:
            assert(v in namelist['time'])


        self._calendar_units = "seconds since 0001-01-01"

        self._start_datetime = cftime.datetime(
            namelist["time"]["year"],
            namelist["time"]["month"],
            namelist["time"]["day"],
            namelist["time"]["hour"],
            calendar='noleap')

        self._start_seconds = cftime.date2num(self._start_datetime, units=self._calendar_units, calendar='noleap')

        self.get_times()


        import sys; sys.exit()
        return


    def get_times(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            ds = xr.open_mfdataset(os.path.join(self._real_data, '*.cam.h3*'))

            self.time_indexes = ds.indexes['time']
            

            time_seconds = []
            for t in self.time_indexes:
                time_seconds.append(cftime.date2num(t, units=self._calendar_units, calendar='noleap'))

            self._time_seconds = np.array(time_seconds)
            print(self._time_seconds, self._start_seconds)


            #print(self._start_datetime,  cftime.date2num(self._start_datetime, units=self._calendar_units, calendar='noleap'))
            #print(self.time_indexes[0], cftime.date2num(self.time_indexes[0], units=self._calendar_units, calendar='noleap'))
            

            #print(self.time_indexes,self._start_datetime)
            #print(cftime.date2index(self.time_indexes, self._start_datetime, calendar='noleap'))


            #print(cftime.nc)
            #cftime.time2index(cftime.date2num, self.time_indexes )


        return