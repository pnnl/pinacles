import pickle
import os
from mpi4py import MPI

class Restart:
    def __init__(self, namelist):

        # Remember the namelist
        self._namelist = namelist

        self._fequency = self._namelist['restart']['frequency']

        self._restart_simulation = self._namelist['restart']['restart_simulation']

        self._infile = None
        if self._restart_simulation:
            self._infile = self._namelist['restart']['infile']

        #Set-up reastart output path
        self._path = os.path.join(os.path.join(
            namelist['meta']['output_directory'], 
            namelist['meta']['simname']), 'Restart')

        #Create the path if it doesn't exist
        self.create_path()
        
        # This dictionary will store the data that is required for output
        self.data_dict = {}

        return

    def create_path(self):
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            os.makedirs(self._path)

        MPI.COMM_WORLD.Barrier()

        return

    def dump(self, time):

        time_path = os.path.join(self._path, str(time))
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            os.mkdir(time_path)
        MPI.COMM_WORLD.Barrier()
        
        self.data_dict['namelist'] = self._namelist

        with open(os.path.join(time_path, str(rank) + '.pkl'), 'wb') as f:
            pickle.dump(self.data_dict, f)
        
        self.data_dict = {}

        return

    def read(self):

        rank = MPI.COMM_WORLD.Get_rank()

        with open(os.path.join(self._infile, str(rank) + '.pkl'), 'rb') as f:
            self.data_dict = pickle.load(f)
        
        return

    def purge_data_dict(self):
        self.data_dict = {}
        return

    @property
    def frequency(self):
        return self._fequency

    @property
    def path(self):
        return self._path

    @property
    def infile(self):
        return self._infile

    @property
    def restart_simulation(self):
        return self._restart_simulation