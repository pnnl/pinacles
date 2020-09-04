import numpy as np
import netCDF4 as nc
from mpi4py import MPI
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
        self._output_dt = 1.0
        self._starttime = 1800.0
        self._endtime = 9600.0

        self._location[0] = 0.0
        self._location[1] = 0.0
        self._location[2] = 200.0

        self._up_or_down = 1

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

        self._TimeSteppingController.set_dt(self._output_dt)
        self.update_position()

        nh = self._Grid.n_halo
        x = self._Grid.x_edge_global[nh[0]-1: -nh[0]]
        y = self._Grid.y_edge_global[nh[1]-1: -nh[1]]
        z = self._Grid.z_edge_global[nh[2]-1: -nh[2]]

        xpos = self._location[0]
        ypos = self._location[1]
        zpos = self._location[2]

        # Jump out of the function if this point is not on this processor
        if xpos < np.min(x) or xpos > np.max(x):
            return

        if ypos < np.min(y) or ypos > np.max(y):
            return

        xind = int(xpos//self._Grid.dx[0] - 1 + nh[0])
        yind = int(ypos//self._Grid.dx[1] - 1 + nh[1])
        zind = int(zpos//self._Grid.dx[2] - 1 + nh[2])


        rt_grp = nc.Dataset(self._path, 'r+')
        time = rt_grp['time']
        rt_grp['time'][len(time)] = self._TimeSteppingController.time

        for name in self._ScalarState.names:
            data = self._ScalarState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        for name in self._VelocityState.names:
            data = self._VelocityState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        for name in self._DiagnosticState.names:
            data = self._DiagnosticState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        rt_grp['X'][-1] = xpos
        rt_grp['Y'][-1] = ypos
        rt_grp['Z'][-1] = zpos

        rt_grp.close()

        return

    def update_position(self):
        if self._location[2] < 0.0:
            self._up_or_down = 1.0
        if self._location[2] >  1300.0:
            self._up_or_down = -1.0


        self._location[0] = 2560.0 + 500.0 * np.sin(2.0 * np.pi * self._TimeSteppingController.time/360.0)
        self._location[1] = 2560.0 + 500.0 *  np.cos(2.0 * np.pi * self._TimeSteppingController.time/360.0)

        self._location[2] += 0.5 * self._up_or_down

        return


