import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI
import json

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Fields2D:
    def __init__(self, namelist, Grid, Ref, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref

        self._frequency = namelist["stats"]["frequency"]
        self._ouput_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])

        self._classes = {}

        self._output_path = os.path.join(self._ouput_root, self._casename)
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

        output_here = os.path.join(
            self._output_path, str(np.round(self._TimeSteppingController.time))
        )
        MPI.COMM_WORLD.barrier()
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)
        MPI.COMM_WORLD.barrier()

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

        # Sync and closue netcdf file
        rt_grp.sync()
        rt_grp.close()

        self._last_io_time = self._TimeSteppingController._time
        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        return

    def setup_nc_dims(self, rt_grp):

        rt_grp.createDimension("time", size=1)
        rt_grp.createDimension("X", size=self._Grid.local_shape[0])
        rt_grp.createDimension("Y", size=self._Grid.local_shape[1])
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
        X[:] = self._Grid.x_local[nhalo[0] : -nhalo[0]]

        Y = rt_grp.createVariable("Y", np.double, dimensions=("Y"))
        Y.units = "m"
        Y.long_name = "y-coordinate"
        Y.standard_name = "y"
        Y[:] = self._Grid.y_local[nhalo[1] : -nhalo[1]]



        nh = self._Grid.n_halo
        lat = rt_grp.createVariable("lat", np.double, dimensions=("X", "Y"))
        lat.units = "degrees"
        lat.long_name = "degress latitude"
        lat.standard_name = "latitude"
        lat[:,:] = self._Grid.lat_local[nh[0]:-nh[0], nh[1]:-nh[1]]

        lon = rt_grp.createVariable("lon", np.double, dimensions=("X", "Y"))
        lon.units = "degrees"
        lon.long_name = "degrees longitude"
        lon.standard_name = "longitude"
        lon[:,:] = self._Grid.lon_local[nh[0]:-nh[0], nh[1]:-nh[1]]

        return
