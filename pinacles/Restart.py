import pickle
import os
from mpi4py import MPI
import time
from datetime import datetime as dt
from pinacles import parameters 
class Restart:
    def __init__(self, namelist):

        # Remember the namelist
        self._namelist = namelist

        self._fequency = parameters.LARGE
        if 'frequency' in self._namelist['restart']:
            self._fequency = self._namelist['restart']['frequency']

        self._restart_simulation = False
        if 'restart_simulation' in self._namelist['restart']:
            self._restart_simulation = self._namelist['restart']['restart_simulation']

        self._walltime_restart = parameters.LARGE
        self._do_walltime_restart = False
        if 'walltime_restart' in self._namelist['restart']:
            self._walltime_restart = self._namelist['restart']['walltime_restart']
            self._do_walltime_restart = True

        self._infile = None
        if self._restart_simulation:
            self._infile = self._namelist['restart']['infile']

        sim_path = os.path.join(
            namelist['meta']['output_directory'], 
            namelist['meta']['simname'])
        
        # If the case already exits create a new directory and time and date it
        if MPI.COMM_WORLD.Get_rank() == 0:
            if os.path.exists(sim_path):

                sim_path = os.path.join(
                namelist['meta']['output_directory'], 
                namelist['meta']['simname'])
        
                # If the simulation path exists, create another
                sim_path = sim_path.split('_started_')            
                namelist['meta']['simname'] =  sim_path[0] + '_started_' + dt.now().strftime("%Y_%m_%d-%I_%M_%S_%p")     

        #Broadcast the directory name that was just created on rank0
        namelist['meta']['simname'] =  MPI.COMM_WORLD.bcast(namelist['meta']['simname'])


        #Set-up reastart output path
        self._path = os.path.join(os.path.join(
            namelist['meta']['output_directory'], 
            namelist['meta']['simname']), 'Restart')

        #Create the path if it doesn't exist
        self.create_path()
        
        # This dictionary will store the data that is required for output
        self.data_dict = {}

        # This is the list of classes we want to restart
        self._classes_to_restart = []

        return

    def create_path(self):
        """ Function that check to see if the restart path exists, if it does not, 
        create it. 
        """
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._path):
                os.makedirs(self._path)

        MPI.COMM_WORLD.Barrier()

        return

    def dump(self, out_time):
        """ This function creates a restart file on each rank, and the writes data dict to it.
        Args:
            out_time {float}: the model time. This is used to for the output directory name. 
        """
        t0 = time.perf_counter()

        # Create the path for this output time
        time_path = os.path.join(self._path, str(out_time))
        rank = MPI.COMM_WORLD.Get_rank()
        
        # We only need to create the directory on the file system once, so 
        # do it on rank 0
        if rank == 0:
            # Announce that a restart file is being written
            print('\t Writing restart files @ ' + str(out_time) + 's.')
            
            # Create needed directory
            os.mkdir(time_path)
        MPI.COMM_WORLD.Barrier()
        
        self.data_dict['namelist'] = self._namelist

        with open(os.path.join(time_path, str(rank) + '.pkl'), 'wb') as f:
            pickle.dump(self.data_dict, f)
        
        self.data_dict = {}
        MPI.COMM_WORLD.Barrier()
        t1 = time.perf_counter()
        
        #Print
        if rank == 0:
            print('\t Finished writing restart file in ' + str(t1 - t0) + 's.')


        return

    def read(self):

        rank = MPI.COMM_WORLD.Get_rank()

        with open(os.path.join(self._infile, str(rank) + '.pkl'), 'rb') as f:
            self.data_dict = pickle.load(f)
        
        return

    def add_class_to_restart(self, class_to_add):
        """ This function adds a class to the list of classes that will either 
        contribute to restart files or read from redstart files. These classes must contain
        the class methods restart and dump_restart.

        Args:
            class_to_add {class}: a class to dump restarts or restart must have correct methods
        """

        #Check to make sure that the class has a restart attribute
        assert hasattr(class_to_add, 'restart')
        assert hasattr(class_to_add, 'dump_restart')

        # Add this class
        self._classes_to_restart.append(class_to_add)

        return

    def restart(self):
        """[summary]
        """

        t0 = time.perf_counter()
        rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print('\t Restarting simulation from ' + self._infile + '.')
        # First read the restart files from  disk
        self.read()

        # Now loop over the calles and call the restart method
        for item in self._classes_to_restart:
            item.restart(self.data_dict)

        MPI.COMM_WORLD.Barrier()
        t1 = time.perf_counter()

        if rank == 0:
            print('\t Finished restarting simulation in ' + str(t1 - t0) + 's.')


        return

    def dump_restart(self, time):
        """ This is the top level method that actually writes the restart files. 
        
        1) It iterates through all of the classes that will be restarted calling dump_restart which adds data
        to data_dict for each of the classes

        2) It calls the dump method of this class which actually writes the data_dict to disk.

        3) It calls purge_data_dict which frees up memory.

        Args:
            time (float): the model time associated for this restart dump
        """

        for item in self._classes_to_restart:
            item.dump_restart(self.data_dict)

        self.dump(time)
        self.purge_data_dict()

        return

    def purge_data_dict(self):
        """ This method frees memory associated with the dictionary (dict_data) used to store data that is
        then writte to the restart files. 
        """
        self.data_dict = {}
        return

    @property
    def frequency(self):
        """ Returns the frequency with which restart files are to be output. 

        Returns:
            float: frequency in seconds
        """
        return self._fequency

    @property
    def path(self):
        """ Returns the top-level path where restart files will be written.

        Returns:
            str: restart path
        """
        return self._path

    @property
    def infile(self):
        """ Returns the input file path for a restarted simulation. If the simulation 
        is not to be restarted then the default value of None is returned.  

        Returns:
            str: path to the restart file
        """
        return self._infile

    @property
    def restart_simulation(self):
        """ Returns a bool indicating if this simulation is or was to be restarted. 

        Returns:
            bool: True if simulation is to restarted
        """
        return self._restart_simulation

    @property
    def n_classes(self):
        """ Returns the number of classes that are to be restated. 

        Returns:
            integer: number of classes to be restarted
        """
        return len(self._classes_to_restart)

    @property
    def walltime_restart(self):
        return self._walltime_restart

    @property
    def do_walltime_restart(self):
        return self._do_walltime_restart