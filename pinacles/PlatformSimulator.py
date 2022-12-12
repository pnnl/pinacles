import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles.interpolation_impl import centered_second
from pinacles import UtilitiesParallel
import pickle
import os


class PlatformSimulator:
    def __init__(
        self,
        namelist,
        name,
        startloc,
        datafile,
        TimeSteppingController,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
    ):

        self._simulator_name = name
        self._TimeSteppingController = TimeSteppingController
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._location = np.zeros((3,), dtype=np.double)
        self._location[0] = startloc[0]
        self._location[1] = startloc[1]
        self._location[2] = startloc[2]

        output_root = str(namelist["meta"]["output_directory"])
        casename = str(namelist["meta"]["simname"])
        self._path = os.path.join(output_root, casename)
        self._path = os.path.join(self._path, self._simulator_name + ".nc")

        self.frequency = 1.0
        self._count = 0
        self._up_or_down = 1

        with open(datafile, "rb") as f:
            self._flight_path = pickle.load(f)
            self._starttime = self._flight_path["time_offset"][0]
            self._endtime = self._flight_path["time_offset"][-1]

        # "Fast-forward the location if the time is greater than the start time of the platform
        if (
            self._TimeSteppingController.time > self._starttime
            and self._TimeSteppingController.time < self._endtime
        ):
            while (
                self._flight_path["time_offset"][self._count]
                <= self._TimeSteppingController.time
            ):
                self.update_position_file(
                    self._flight_path["time_offset"][self._count],
                    wrap_x=True,
                    wrap_y=True,
                )

        return

    def initialize(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if os.path.exists(self._path):
                old_rt = nc.Dataset(self._path, "r")
                t_end = old_rt.variables["time"][-1]
                t_start = old_rt.variables["time"][0]
                os.rename(
                    self._path,
                    self._path[:-3] + "_" + str(t_start) + "_" + str(t_end) + ".nc",
                )

            rt_grp = nc.Dataset(self._path, "w")
            rt_grp.createDimension("time")
            rt_grp.createVariable("time", np.double, dimensions=("time",))

            rt_grp.createVariable("X", np.double, dimensions=("time",))
            rt_grp.createVariable("Y", np.double, dimensions=("time",))
            rt_grp.createVariable("Z", np.double, dimensions=("time",))

            for v in self._ScalarState.names:
                rt_grp.createVariable(v, np.double, ("time",))

            for v in self._VelocityState.names:
                rt_grp.createVariable(v, np.double, ("time",))

            for v in self._DiagnosticState.names:
                rt_grp.createVariable(v, np.double, ("time",))

            rt_grp.close()

        return

    def update(self):

        if (
            self._TimeSteppingController.time < self._starttime
            or self._TimeSteppingController.time > self._endtime
        ):
            return

        self.update_position_file(
            self._TimeSteppingController.time, wrap_x=True, wrap_y=True
        )

        platform_on_rank = self._Grid.point_on_rank(
            self._location[0], self._location[1], self._location[2]
        )

        if not platform_on_rank:
            return

        indices = self._Grid.point_indicies(
            self._location[0], self._location[1], self._location[2]
        )

        xind = indices[0]
        yind = indices[1]
        zind = indices[2]

        rt_grp = nc.Dataset(self._path, "r+")
        time = rt_grp["time"]
        rt_grp["time"][len(time)] = self._TimeSteppingController.time

        for name in self._ScalarState.names:
            data = self._ScalarState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        for name in self._VelocityState.names:
            data = self._VelocityState.get_field(name)
            if name == "u":
                rt_grp[name][-1] = centered_second(
                    data[xind - 1, yind, zind], data[xind, yind, zind]
                )
            elif name == "v":
                rt_grp[name][-1] = centered_second(
                    data[xind, yind - 1, zind], data[xind, yind, zind]
                )
            elif name == "w":
                rt_grp[name][-1] = centered_second(
                    data[xind, yind, zind - 1], data[xind, yind, zind]
                )

        for name in self._DiagnosticState.names:
            data = self._DiagnosticState.get_field(name)
            rt_grp[name][-1] = data[xind, yind, zind]

        rt_grp["X"][-1] = self._location[0]
        rt_grp["Y"][-1] = self._location[1]
        rt_grp["Z"][-1] = self._location[2]

        rt_grp.close()

        return

    def update_position(self, time, wrap_x=True, wrap_y=True):

        self._location[0] = 2560.0 + 500.0 * np.sin(2.0 * np.pi * time / 360.0)
        self._location[1] = 2560.0 + 500.0 * np.cos(2.0 * np.pi * time / 360.0)

        if self._location[2] > 1300.0:
            self._up_or_down = -1
        elif self._location[2] < 100.0:
            self._up_or_down = 1

        self._location[2] += 0.5 * self._up_or_down

        if wrap_x:
            x_range = self._Grid.x_range
            if self._location[0] < x_range[0]:
                self._location[0] += x_range[1] - x_range[0]
            elif self._location[0] > x_range[1]:
                self._location[0] += x_range[0] - x_range[1]
        if wrap_y:
            y_range = self._Grid.y_range
            if self._location[1] < y_range[0]:
                self._location[1] += y_range[1] - y_range[0]
            elif self._location[1] > y_range[1]:
                self._location[1] += y_range[0] - y_range[1]

        return

    def update_position_file(self, time, wrap_x=True, wrap_y=True):

        dx_current = np.interp(
            time, self._flight_path["time_offset"][:], self._flight_path["dx"][:]
        )
        dy_current = np.interp(
            time, self._flight_path["time_offset"][:], self._flight_path["dy"][:]
        )
        z_current = np.interp(
            time, self._flight_path["time_offset"][:], self._flight_path["z"][:]
        )

        self._location[0] += dx_current
        self._location[1] += dy_current
        self._location[2] = z_current

        self._count += 1

        if wrap_x:
            x_range = self._Grid.x_range
            if self._location[0] < x_range[0]:
                self._location[0] += x_range[1] - x_range[0]
            elif self._location[0] > x_range[1]:
                self._location[0] += x_range[0] - x_range[1]
        if wrap_y:
            y_range = self._Grid.y_range
            if self._location[1] < y_range[0]:
                self._location[1] += y_range[1] - y_range[0]
            elif self._location[1] > y_range[1]:
                self._location[1] += y_range[0] - y_range[1]

        return


class PlatformSimulators:
    def __init__(
        self,
        namelist,
        TimeSteppingController,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
    ):

        self._Grid = Grid
        self._Ref = Ref
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState
        self._flight_data_files = None
        self._startlocations = None

        self._n = 0

        self.name = "PlatformSimulators"

        self._list_of_platforms = []

        if "platforms" in namelist:
            self._startlocations = namelist["platforms"]["start_locations"]
            self._flight_data_files = namelist["platforms"]["flight_data_files"]
            self.frequency = 1.0

        else:
            self.frequency = 86400.0 * 100
            # Don't limit the time step due to the (non)existence of this simulator
            return

        self._n = len(self._startlocations)

        for i, startloc in enumerate(self._startlocations):
            self._list_of_platforms.append(
                PlatformSimulator(
                    namelist,
                    "aaf_" + str(i),
                    startloc,
                    self._flight_data_files[i],
                    TimeSteppingController,
                    Grid,
                    Ref,
                    ScalarState,
                    VelocityState,
                    DiagnosticState,
                )
            )
        return

    def initialize(self):
        if self._n == 0:
            return

        for platform_i in self._list_of_platforms:
            platform_i.initialize()

        return

    def update(self):
        if self._n == 0:
            return

        for platform_i in self._list_of_platforms:
            platform_i.update()

        return
