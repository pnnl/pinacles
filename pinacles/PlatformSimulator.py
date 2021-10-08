import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles.interpolation_impl import centered_second
import pickle 
class PlatformSimulator:

    def __init__(self, name, TimeSteppingController, Grid, Ref, 
        ScalarState, VelocityState, DiagnosticState):


        self._simulator_name = name
        self._TimeSteppingController = TimeSteppingController
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        self._path = './' + self._simulator_name + '.nc'
        self._location = np.zeros((3,), dtype=np.double)
        self.frequency = 1.0
        self._starttime = 1800.0
        self._endtime = 9600.0

        self._location[0] = 0.0
        self._location[1] = 100.0
        self._location[2] = 100.0

        self._up_or_down = 1

        self._count = 0
        self._starttime = 0
        self._endtime = 1000.0
        # with open('/Users/pres026/Desktop/HISCALE_airplane/2016/sgp/hiscale/mei-iwg1/fligh_path.pkl','rb') as f:
        #     self._flight_path = pickle.load(f)
        #     self._starttime = self._flight_path['start_time']
        #     self._endtime = self._flight_path['end_time']


        return

    def initialize(self):

        if MPI.COMM_WORLD.Get_rank() == 0:

            rt_grp = nc.Dataset(self._path, 'w')
            rt_grp.createDimension('time')
            rt_grp.createVariable('time', np.double, dimensions=('time',))

            rt_grp.createVariable('X', np.double, dimensions=('time',))
            rt_grp.createVariable('Y', np.double, dimensions=('time',))
            rt_grp.createVariable('Z', np.double, dimensions=('time',))

            for v in self._ScalarState.names:
                rt_grp.createVariable(v, np.double, ('time',))

            for v in self._VelocityState.names:
                rt_grp.createVariable(v, np.double, ('time',))

            for v in self._DiagnosticState.names:
                rt_grp.createVariable(v, np.double, ('time',))


            rt_grp.close()

        return

    def update(self):

        if (self._TimeSteppingController.time < self._starttime  or 
            self._TimeSteppingController.time   > self._endtime):
            return

        # self.update_position_file()
        self.update_position(wrap_x=True, wrap_y=True)

        platform_on_rank = self._Grid.point_on_rank(
            self._location[0],self._location[1],self._location[2]
            )
        
        if not platform_on_rank:
            return
        
        indices = self._Grid.point_indicies(
            self._location[0],self._location[1],self._location[2]
            )

      

    

        xind = indices[0]
        yind = indices[1]
        zind = indices[2]

        rt_grp = nc.Dataset(self._path, 'r+')
        time = rt_grp['time']
        rt_grp['time'][len(time)] = self._TimeSteppingController.time

        for name in self._ScalarState.names:
            data = self._ScalarState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind] 

        for name in self._VelocityState.names:
            data = self._VelocityState.get_field(name)
            if name == 'u':
                rt_grp[name][-1] = centered_second(data[xind-1, yind, zind],data[xind, yind, zind])
            elif name == 'v':
                rt_grp[name][-1] = centered_second(data[xind, yind-1, zind],data[xind, yind, zind])
            elif name == 'w':
                rt_grp[name][-1] = centered_second(data[xind, yind, zind-1],data[xind, yind, zind])

        for name in self._DiagnosticState.names:
            data = self._DiagnosticState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        rt_grp['X'][-1] = self._location[0]
        rt_grp['Y'][-1] = self._location[1]
        rt_grp['Z'][-1] = self._location[2]

        rt_grp.close()

        return

    def update_position(self,wrap_x=True, wrap_y=True):
        if self._location[2] < 0.0:
            self._up_or_down = 1.0
        if self._location[2] >  1300.0:
            self._up_or_down = -1.0


        self._location[0] = 2560.0 + 500.0 * np.sin(2.0 * np.pi * self._TimeSteppingController.time/360.0)
        self._location[1] = 2560.0 + 500.0 *  np.cos(2.0 * np.pi * self._TimeSteppingController.time/360.0)

        self._location[2] += 0.5 * self._up_or_down

        if wrap_x:
            x_range = self._Grid.x_range
            if self._location[0] < x_range[0]:
                self._location[0] = x_range[1] - (x_range[0]-self._location[0]) + 1.0
            elif self._location[0] > x_range[1]:
                self._location[0] = x_range[0] + (self._location[0] - x_range[1]) -1.0
        if wrap_y:
            y_range = self._Grid.y_range
            if self._location[1] < y_range[0]:
                self._location[1] = y_range[1] - (y_range[0]-self._location[1]) + 1.0
            elif self._location[1] > y_range[1]:
                self._location[1] = y_range[0] + (self._location[1] - y_range[1]) -1.0


        return

    def update_position_file(self):


        #self._location[0] = 2560.0 + 500.0 * np.sin(2.0 * np.pi * self._TimeSteppingController.time/360.0)
        #self._location[1] = 2560.0 + 500.0 *  np.cos(2.0 * np.pi * self._TimeSteppingController.time/360.0)#
        self._location[0] += self._flight_path['air_speed'][self._count]
        self._location[2] = self._flight_path['radar_alt'][self._count]

        self._count += 1
        return


