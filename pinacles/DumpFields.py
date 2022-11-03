import numpy as np
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


import time
from pinacles import UtilitiesParallel


def DumpFieldsFactory(namelist, Timers, Grid, TimeSteppingController):
    assert "fields" in namelist
    if "io_type" not in namelist["fields"]:
        from pinacles.DumpFieldsNetCDF import DumpFields
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "NETCDF":
        from pinacles.DumpFieldsNetCDF import DumpFields
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5":
        try:
            import h5py
        except:
            UtilitiesParallel.print_root('No H5PY--Reverting to NETCDF fields output')
            from pinacles.DumpFieldsNetCDF import DumpFields
            return DumpFields(namelist, Timers, Grid, TimeSteppingController)

        from pinacles.DumpFieldsHDF5 import DumpFields_hdf
        return DumpFields_hdf(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5_GATHER":
        from pinacles.DumpFieldsGatherHDF5 import DumpFields_hdf
        return DumpFields_hdf(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5_GATHER_PERPROC":
        from pinacles.DumpFieldsGatherHDF5PerProc import DumpFields_hdf_perproc
        return DumpFields_hdf_perproc(namelist, Timers, Grid, TimeSteppingController)


    return
