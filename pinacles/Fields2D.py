import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI
import json 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class Fields2D:
    def __init__(self, namelist, Grid, Ref, TimesteppingController):

        self._Grid = Grid
        self._Ref = Ref

        self._frequency = namelist['stats']['frequency']
        self._ouput_root = str(namelist['meta']['output_directory'])
        self._casename = str(namelist['meta']['simname'])

        self._classes = {}

        self._output_path = os.path.join(self._ouput_root, self._casename)
        self._stats_file = os.path.join(self._output_path, 'stats.nc')
        self._TimeSteppingController = TimeSteppingController

        self._rt_grp = None

        self.setup_directories()
        MPI.COMM_WORLD.Barrier()
        self._TimeSteppingController.add_timematch(self._frequency)
        self._last_io_time = 0

        self._namelist = namelist
        
        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert(aclass not in self._classes)
        self._classes[aclass.name] = aclass
        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)


        return