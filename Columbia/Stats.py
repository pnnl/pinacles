import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI

class Stats:
    def __init__(self, namelist, Grid, Ref, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref

        self._frequency = namelist['stats']['frequency']
        self._ouput_root = str(namelist['meta']['output_directory'])
        self._casename = str(namelist['meta']['casename'])

        self._classes = {}

        try:
            self._customname = str(namelist['meta']['customname'])
        except:
            self._customname = self._casename

        self._output_path = os.path.join(self._ouput_root, self._casename)
        self._stats_file = os.path.join(self._output_path, 'stats.nc')
        self._TimeSteppingController = TimeSteppingController

        self._rt_grp = None

        self.setup_directories()
        self._TimeSteppingController.add_timematch(self._frequency)
        self._last_io_time = 0


        return

    def add_class(self, aclass):
        assert(aclass not in self._classes)
        self._classes[aclass.name] = aclass
        return

    def initialize(self):
        #Todo: Follow CORADS standard?

        #We only need to initialzie the profiles on the rank 0
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        #Create groups for each class
        self._rt_grp = nc.Dataset(self._stats_file, 'w')

        nh = self._Grid.n_halo

        #Create group for reference class
        ref_grp = self._rt_grp.createGroup('reference')
        ref_grp.createDimension('z', size=self._Grid.n[2])
        vh = ref_grp.createVariable('z', np.double, dimensions=('z',))
        vh[:] = self._Grid.z_global[nh[2]:-nh[2]]

        ref_grp.createDimension('z_edge', size=self._Grid.n[2]+1)
        vh = ref_grp.createVariable('z_edge', np.double, dimensions=('z_edge',))
        vh[:] = self._Grid.z_edge_global[nh[2]-1:-nh[2]]

        #Now write the reference profiles
        self._Ref.write_stats(ref_grp)

        #Loop over classes and create grpus and dimensions
        for aclass in self._classes:
            this_grp = self._rt_grp.createGroup(aclass)

            #Create subgroups for profiles and timeseries
            timeseries_grp = this_grp.createGroup('timeseries')
            profiles_grp = this_grp.createGroup('profiles')

            #Create dimensions for timeseries
            timeseries_grp.createDimension('time')
            timeseries_grp.createVariable('time', np.double, dimensions=('time',))

            #Create dimensions for profiles
            profiles_grp.createDimension('time')
            profiles_grp.createVariable('time', np.double, dimensions=('time',))
            profiles_grp.createDimension('z', size=self._Grid.n[2])
            vh = profiles_grp.createVariable('z', np.double, dimensions=('z',))
            vh[:] = self._Grid.z_global[nh[2]:-nh[2]]

            profiles_grp.createDimension('z_edge', size=self._Grid.n[2]+1)
            vh = profiles_grp.createVariable('z_edge', np.double, dimensions=('z_edge',))
            vh[:] = self._Grid.z_edge_global[nh[2]-1:-nh[2]]

        self._rt_grp.sync()

        #Now loop over clases and init output files
        for aclass in self._classes:
            this_grp = self._rt_grp[aclass]
            self._classes[aclass].io_initialize(this_grp)

        self._rt_grp.sync()

        return

    def update(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        if  not (self._last_io_time == 0 or np.allclose(self._TimeSteppingController._time - self._last_io_time, self._frequency)):
            return


        #Increment time for all groups
        for aclass in self._classes:
            for grp in ['timeseries', 'profiles']:
                this_grp = self._rt_grp[aclass]
                time = this_grp[grp]['time']
                time[time.shape[0]] = self._TimeSteppingController._time


        self._last_io_time = self._TimeSteppingController._time

        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)


        return