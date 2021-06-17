import numpy as np
import time
import os
from mpi4py import MPI
import netCDF4 as nc


class Timer:
    def __init__(self, namelist, TimeSteppingController):

        self._TimeSteppingController = TimeSteppingController

        self._timer_data = {}
        self._timer_start = {}
        self._n_calls = {}
        self._n_timesteps = 0

        self._frequency = namelist["stats"]["frequency"]
        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])

        self._output_path = os.path.join(self._output_root, self._casename)
        self._stats_file = os.path.join(self._output_path, "timers.nc")

        self._setup_directories()
        MPI.COMM_WORLD.barrier()

        return

    def initialize(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset(self._stats_file, "w")

            time_dim = rt_grp.createDimension("time")
            time = rt_grp.createVariable("time", np.single, ("time",))
            time.units = "s"

            for timer_name in self._timer_data:
                timer_nc = rt_grp.createVariable(
                    timer_name + "_max", np.single, ("time",)
                )
                timer_nc.units = "s"

                timer_nc = rt_grp.createVariable(
                    timer_name + "_min", np.single, ("time",)
                )
                timer_nc.units = "s"

                timer_nc = rt_grp.createVariable(
                    timer_name + "_pertimestep_max", np.single, ("time",)
                )
                timer_nc.units = "s"

                timer_nc = rt_grp.createVariable(
                    timer_name + "_pertimestep_min", np.single, ("time",)
                )
                timer_nc.units = "s"

            rt_grp.close()

        return

    def update(self):

        rt_grp = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset(self._stats_file, "r+")

            # Update time
            time = rt_grp["time"]
            time[time.shape[0]] = self._TimeSteppingController._time

        # Do communications to compute statistics
        max_value = np.empty((1,), dtype=np.single)
        min_value = np.empty((1,), dtype=np.single)

        for var in self._timer_data:

            MPI.COMM_WORLD.Reduce(
                np.array([self._timer_data[var]], dtype=np.single), max_value, MPI.MAX
            )
            if MPI.COMM_WORLD.Get_rank() == 0:
                var_nc = rt_grp[var + "_max"]
                var_nc[-1] = max_value

            MPI.COMM_WORLD.Reduce(
                np.array([self._timer_data[var]], dtype=np.single), min_value, MPI.MIN
            )

            if MPI.COMM_WORLD.Get_rank() == 0:
                var_nc = rt_grp[var + "_min"]
                var_nc[-1] = min_value

            if self._n_timesteps == 0:
                max_value[0] = 0.0
            else:
                MPI.COMM_WORLD.Reduce(
                    np.array(
                        [self._timer_data[var] / self._n_timesteps], dtype=np.single
                    ),
                    max_value,
                    MPI.MAX,
                )

            if MPI.COMM_WORLD.Get_rank() == 0:
                var_nc = rt_grp[var + "_pertimestep_max"]
                var_nc[-1] = max_value

            if self._n_timesteps == 0:
                min_value[0] = 0.0
            else:
                MPI.COMM_WORLD.Reduce(
                    np.array(
                        [self._timer_data[var] / self._n_timesteps], dtype=np.single
                    ),
                    min_value,
                    MPI.MIN,
                )

            if MPI.COMM_WORLD.Get_rank() == 0:
                var_nc = rt_grp[var + "_pertimestep_min"]
                var_nc[-1] = min_value

        if MPI.COMM_WORLD.Get_rank() == 0:

            rt_grp.close()

        # Reset the data to zeros on all ranks
        self._timer_data = dict.fromkeys(self._timer_data, 0.0)
        self._n_calls = dict.fromkeys(self._timer_data, 0.0)

        self._n_timesteps = 0
        return

    def _setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        return

    @property
    def frequency(self):
        return self._frequency

    def add_timer(self, name):

        self._timer_data[name] = 0.0
        self._timer_start[name] = 0.0
        self._n_calls[name] = 0

        return

    def start_timer(self, name):

        self._timer_start[name] = time.perf_counter()

        return

    def end_timer(self, name):

        t_end = time.perf_counter()
        self._timer_data[name] += t_end - self._timer_start[name]
        self._n_calls[name] += 1

        return

    def get_accumulated_time(self, name):
        return self._timer_data[name]

    def finish_timestep(self):
        self._n_timesteps += 1
        return

    @property
    def n_timers(self):
        return len(self._timer_data)
