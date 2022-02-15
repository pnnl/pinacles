import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI

import time
from pinacles import UtilitiesParallel


class DumpFields:
    def __init__(self, namelist, Timers, Grid, TimeSteppingController):

        self._Timers = Timers
        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController

        self._this_rank = MPI.COMM_WORLD.Get_rank()
        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])
        self._output_path = self._output_path = os.path.join(
            self._output_root, self._casename
        )
        self._output_path = os.path.join(self._output_path, "fields")

        try:
            self._frequency = namelist["fields"]["frequency"]
        except:
            self._frequency = 1e9

        if self._this_rank == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        self._classes = {}
        self._namelist = namelist

        self._Timers.add_timer("DumpFields")
        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def update(self):
        self._Timers.start_timer("DumpFields")

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

        self.setup_nc_dims(rt_grp)

        nhalo = self._Grid.n_halo
        for ac in self._classes:
            # Loop over all variables
            ac = self._classes[ac]
            for v in ac._dofs:

                if (
                    "ff8" not in v
                ):  # Exclude the bins from the 3D fields for the same of storage
                    v_nc = rt_grp.createVariable(
                        v, np.double, dimensions=("time", "X", "Y", "Z")
                    )
                    v_nc.units = ac.get_units(v)
                    v_nc.long_names = ac.get_long_name(v)
                    v_nc.standard_name = ac.get_standard_name(v)
                    v_nc[0, :, :, :] = ac.get_field(v)[
                        nhalo[0] : -nhalo[0], nhalo[1] : -nhalo[1], nhalo[2] : -nhalo[2]
                    ]

            rt_grp.sync()

        rt_grp.close()
        self._Timers.end_timer("DumpFields")
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

        Z = rt_grp.createVariable("Z", np.double, dimensions=("Z"))
        Z.units = "m"
        Z.long_name = "z-coordinate"
        Z.standard_name = "z"
        Z[:] = self._Grid.z_local[nhalo[2] : -nhalo[2]]
        rt_grp.sync()

        return
