import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI
import h5py

import time
from pinacles import UtilitiesParallel


def DumpFieldsFactory(namelist, Timers, Grid, TimeSteppingController):

    assert "fields" in namelist
    if "io_type" not in namelist["fields"]:
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "NETCDF":
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5":
        return DumpFields_hdf(namelist, Timers, Grid, TimeSteppingController)

    return


class DumpFields_hdf:
    def __init__(self, namelist, Timers, Grid, TimeSteppingController):

        self._Timers = Timers
        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController

        self.collective = True
        try:
            if "collective" in namelist["fields"]["hdf5"]:
                self.collective = namelist["fields"]["hdf5"]["collective"]
        except:
            pass

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

        t0 = time.perf_counter()
        self._Timers.start_timer("DumpFields")

        output_here = self._output_path  # os.path.join(
        # self._output_path, str(np.round(self._TimeSteppingController.time))
        # )

        output_here = os.path.join(output_here)

        MPI.COMM_WORLD.barrier()
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)

        MPI.COMM_WORLD.barrier()

        fx = h5py.File(
            os.path.join(
                output_here, str(np.round(self._TimeSteppingController.time)) + ".h5"
            ),
            "w",
            driver="mpio",
            comm=MPI.COMM_WORLD,
        )

        nhalo = self._Grid.n_halo
        local_start = self._Grid._local_start
        local_end = self._Grid._local_end
        for i, v in enumerate(["X", "Y", "Z"]):
            dset = fx.create_dataset(v, (self._Grid.n[i],), dtype="d")
            dset.make_scale()
            if MPI.COMM_WORLD.rank == 0:
                dset[:] = self._Grid._global_axes[i][nhalo[i] : -nhalo[i]]

        for ac in self._classes:
            # Loop over all variables
            ac = self._classes[ac]
            for v in ac._dofs:

                if "ff" not in v:

                    dset = fx.create_dataset(
                        v,
                        (self._Grid.n[0], self._Grid.n[1], self._Grid.n[2]),
                        dtype="d",
                    )
                    for i, d in enumerate(["X", "Y", "Z"]):
                        dset.dims[i].attach_scale(fx[d])

                    if self.collective:
                        with dset.collective:
                            dset[
                                local_start[0] : local_end[0],
                                local_start[1] : local_end[1],
                                :,
                            ] = ac.get_field(v)[
                                nhalo[0] : -nhalo[0],
                                nhalo[1] : -nhalo[1],
                                nhalo[2] : -nhalo[2],
                            ]
                    else:
                        dset[
                            local_start[0] : local_end[0],
                            local_start[1] : local_end[1],
                            :,
                        ] = ac.get_field(v)[
                            nhalo[0] : -nhalo[0],
                            nhalo[1] : -nhalo[1],
                            nhalo[2] : -nhalo[2],
                        ]

                    dset.attrs["units"] = ac.get_units(v)
                    dset.attrs["long_name"] = ac.get_long_name(v)
                    dset.attrs["standar_name"] = ac.get_standard_name(v)

        fx.close()
        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t Parallel IO of 3D fields finished in: " + str(t1 - t0) + " seconds"
        )
        self._Timers.end_timer("DumpFields")
        return


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
                    "ff" not in v
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
