import numpy as np
import netCDF4 as nc
import time
import os
from mpi4py import MPI
from pinacles import UtilitiesParallel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Fields2D:
    def __init__(self, namelist, Grid, Ref, VelocityState, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState

        """Set the output frequency, default it to the stats frequency 
        but allow in to be overridden """

        try:
            self._frequency = namelist["fields2d"]["frequency"]
        except:
            self._frequency = namelist["stats"]["frequency"]

        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])

        self._classes = {}

        self._output_path = os.path.join(self._output_root, self._casename)
        self._output_path = os.path.join(self._output_path, "fields2d")

        self._TimeSteppingController = TimeSteppingController

        self._rt_grp = None

        self.setup_directories()
        MPI.COMM_WORLD.Barrier()
        self._TimeSteppingController.add_timematch(self._frequency)
        self._last_io_time = 0

        self._this_rank = MPI.COMM_WORLD.Get_rank()

        self._namelist = namelist

        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def initialize(self):

        return

    def update(self):

        if not np.allclose(self._TimeSteppingController._time % self._frequency, 0.0):
            return

        t0 = time.perf_counter()

        output_here = os.path.join(
            self._output_path, str(np.round(self._TimeSteppingController.time))
        )
        MPI.COMM_WORLD.barrier()
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)
        MPI.COMM_WORLD.barrier()

        rt_grp = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset(
                os.path.join(output_here, str(self._this_rank) + ".nc"),
                "w",
                format="NETCDF4_CLASSIC",
            )

            # Add some metadata
            rt_grp.unique_id = self._namelist["meta"]["unique_id"]
            rt_grp.wall_time = self._namelist["meta"]["wall_time"]
            rt_grp.fequency = self.frequency

            self.setup_nc_dims(rt_grp)

        for aclass in self._classes:
            self._classes[aclass].io_fields2d_update(rt_grp)

        self.output_velocities(rt_grp)

        # Sync and closue netcdf file
        if rt_grp is not None:
            rt_grp.sync()
            rt_grp.close()

        self._last_io_time = self._TimeSteppingController._time

        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t  2D IO Finished in: " + str(t1 - t0) + " seconds."
        )

        return

    def output_velocities(self, nc_grp):

        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo

        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        for v in ["u", "v", "w"]:

            if nc_grp is not None:
                var_nc = nc_grp.createVariable(
                    v,
                    np.double,
                    dimensions=(
                        "X",
                        "Y",
                    ),
                )

            var = self._VelocityState.get_field(v)
            send_buffer.fill(0.0)
            send_buffer[start[0] : end[0], start[1] : end[1]] = var[
                nh[0] : -nh[0], nh[1] : -nh[1], nh[2]
            ]
            MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
            if nc_grp is not None:
                var_nc[:, :] = recv_buffer

            if nc_grp is not None:
                nc_grp.sync()

        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        return

    def setup_nc_dims(self, rt_grp):

        rt_grp.createDimension("time", size=1)
        rt_grp.createDimension("X", size=self._Grid.n[0])
        rt_grp.createDimension("Y", size=self._Grid.n[1])
        rt_grp.createDimension("Z", size=self._Grid.n[2])

        nhalo = self._Grid.n_halo

        T = rt_grp.createVariable("time", np.double, dimensions=("time"))
        T[0] = self._TimeSteppingController.time
        T.units = "s"
        T.long_name = "time"
        T.standard_name = "t"

        X = rt_grp.createVariable("X", np.double, dimensions=("X"))
        X.units = "m"
        X.long_name = "x-coordinate"
        X.standard_name = "x"
        X[:] = self._Grid.x_global[nhalo[0] : -nhalo[0]]

        Y = rt_grp.createVariable("Y", np.double, dimensions=("Y"))
        Y.units = "m"
        Y.long_name = "y-coordinate"
        Y.standard_name = "y"
        Y[:] = self._Grid.y_global[nhalo[1] : -nhalo[1]]

        return
