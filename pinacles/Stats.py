import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI
import json

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Stats:
    def __init__(self, namelist, Timers, Grid, Ref, TimeSteppingController):

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref

        self._frequency = namelist["stats"]["frequency"]
        self._ouput_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])

        self._classes = {}

        self._output_path = os.path.join(self._ouput_root, self._casename)
        self._stats_file = os.path.join(self._output_path, "stats.nc")
        self._TimeSteppingController = TimeSteppingController

        self._rt_grp = None

        self.setup_directories()
        MPI.COMM_WORLD.Barrier()
        self._TimeSteppingController.add_timematch(self._frequency)
        self._last_io_time = 0

        self._namelist = namelist

        self._Timers.add_timer("Stats")
        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def initialize(self):
        # Todo: Follow CORADS standard?

        # We only need to initialzie the profiles on the rank 0
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        # Create groups for each class
        self._rt_grp = nc.Dataset(self._stats_file, "w")

        # Copy in the input files
        self._rt_grp.input_json = json.dumps(self._namelist)
        self._rt_grp.uuid = self._namelist["meta"]["unique_id"]
        self._rt_grp.wall_time = self._namelist["meta"]["wall_time"]

        with open(os.path.join(self._output_path, "input.json"), "w") as input_file_out:
            json.dump(self._namelist, input_file_out, sort_keys=True, indent=4)

        nh = self._Grid.n_halo

        # Create group for reference class
        ref_grp = self._rt_grp.createGroup("reference")
        ref_grp.createDimension("z", size=self._Grid.n[2])
        vh = ref_grp.createVariable("z", np.single, dimensions=("z",))
        vh[:] = self._Grid.z_global[nh[2] : -nh[2]]
        vh.long_name = "height of cell-center above surface"
        vh.units = "m"
        vh.standard_name = "cell-center height"

        ref_grp.createDimension("z_edge", size=self._Grid.n[2] + 1)
        vh = ref_grp.createVariable("z_edge", np.single, dimensions=("z_edge",))
        vh[:] = self._Grid.z_edge_global[nh[2] - 1 : -nh[2]]
        vh.long_name = "height of cell-edge above surface"
        vh.units = "m"
        vh.standard_name = "cell-edge height"

        # Now write the reference profiles
        self._Ref.write_stats(ref_grp)

        # Loop over classes and create grpus and dimensions
        for aclass in self._classes:
            this_grp = self._rt_grp.createGroup(aclass)

            # Create subgroups for profiles and timeseries
            timeseries_grp = this_grp.createGroup("timeseries")
            profiles_grp = this_grp.createGroup("profiles")

            # Create dimensions for timeseries
            timeseries_grp.createDimension("time")
            time = timeseries_grp.createVariable(
                "time", np.single, dimensions=("time",)
            )
            time.long_name = "time since beginning of simulation"
            time.units = "s"
            time.standard_name = "time"

            # Create dimensions for profiles
            profiles_grp.createDimension("time")
            profiles_grp.createVariable("time", np.single, dimensions=("time",))
            profiles_grp.createDimension("z", size=self._Grid.n[2])
            vh = profiles_grp.createVariable("z", np.single, dimensions=("z",))
            vh[:] = self._Grid.z_global[nh[2] : -nh[2]]
            vh.long_name = "height of cell-center above surface"
            vh.units = "m"
            vh.standard_name = "cell-center height"

            profiles_grp.createDimension("z_edge", size=self._Grid.n[2] + 1)
            vh = profiles_grp.createVariable(
                "z_edge", np.single, dimensions=("z_edge",)
            )
            vh[:] = self._Grid.z_edge_global[nh[2] - 1 : -nh[2]]
            vh.long_name = "height of cell-edge above surface"
            vh.units = "m"
            vh.standard_name = "cell-edge height"

        self._rt_grp.sync()

        # Now loop over clases and init output files
        for aclass in self._classes:
            this_grp = self._rt_grp[aclass]
            self._classes[aclass].io_initialize(this_grp)

        self._rt_grp.sync()
        self._rt_grp.close()

        return

    def update(self):

        if not np.allclose(self._TimeSteppingController._time % self._frequency, 0.0):
            return
        self._Timers.start_timer("Stats")

        if MPI.COMM_WORLD.Get_rank() == 0:
            self._rt_grp = nc.Dataset(self._stats_file, "r+")
        else:
            self._rt_grp = None

        # Increment time for all groups
        for aclass in self._classes:
            for grp in ["timeseries", "profiles"]:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    this_grp = self._rt_grp[aclass]
                    time = this_grp[grp]["time"]
                    time[time.shape[0]] = self._TimeSteppingController._time
        if MPI.COMM_WORLD.Get_rank() == 0:
            self._rt_grp.sync()

        # Call io for all of the classes
        for aclass in self._classes:
            if MPI.COMM_WORLD.Get_rank() == 0:
                this_grp = self._rt_grp[aclass]
            else:
                this_grp = None
            self._classes[aclass].io_update(this_grp)

        if MPI.COMM_WORLD.Get_rank() == 0:
            self._rt_grp.sync()
            self._rt_grp.close()

        self._last_io_time = self._TimeSteppingController._time

        self._Timers.end_timer("Stats")
        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        return
