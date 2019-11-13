import netCDF4 as nc
import os
from mpi4py import MPI

class Stats:
    def __init__(self, namelist, TimeSteppingController):

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

        return

    def add_class(self, aclass):
        assert(aclass not in self._classes)
        self._classes[aclass.name] = aclass
        return

    def initialize(self):
        self._rt_grp = nc.Dataset(self._stats_file, 'w')
        for aclass in self._classes:
            self._rt_grp.createGroup(aclass)

        self._rt_grp.sync()
        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)


        return