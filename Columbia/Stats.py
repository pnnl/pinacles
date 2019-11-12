import netCDF4 as nc
import os
from mpi4py import MPI

class Stats:
    def __init__(self, namelist, Grid, Ref, TimeSteppingController):

        self._frequency = namelist['stats']['frequency']
        self._ouput_root = namelist['meta']['output_directory']
        self._casename = namelist['meta']['casename']

        try:
            self._customname = namelist['meta']['customname']
        except:
            self._customname = self._casename

        self._output_path = os.path.join(self._ouput_root, self._casename)

        self._Grid = Grid
        self._Ref = Ref
        self._TimeSteppingController = TimeSteppingController


        self.setup_directories()

        return

    def update(self):


        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)


        return