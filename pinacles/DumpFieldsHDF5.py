import numpy as np
import os
from mpi4py import MPI
try:
    import h5py
except:
    pass

import time
from pinacles import UtilitiesParallel


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

        self.compression = "gzip"
        self.shuffle = True

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

        s = self._TimeSteppingController.time
        days = s // 86400.
        s = s - (days * 86400.)
        hours = s // 3600.
        s = s - (hours * 3600.)
        minutes = s // 60.
        s = s - (minutes * 60.)
        seconds = s
        s = s - int(seconds)
        ms = s * 1000.0

        fx = h5py.File(
            os.path.join(
                output_here,
                "{:02}d-{:02}h-{:02}m-{:02}s-{:03}ms".format(
                    int(days), int(hours), int(minutes), int(seconds), int(ms)
                )
                + ".h5",
            ),
            "w",
            driver="mpio",
            comm=MPI.COMM_WORLD,
        )
        fx.attrs['dt'] = self._TimeSteppingController.dt
        
        nhalo = self._Grid.n_halo
        local_start = self._Grid._local_start
        local_end = self._Grid._local_end
        for i, v in enumerate(["X", "Y", "Z"]):
            dset = fx.create_dataset(v, (self._Grid.n[i],), dtype="d")
            dset.make_scale()
            if MPI.COMM_WORLD.rank == 0:
                dset[:] = self._Grid._global_axes[i][nhalo[i] : -nhalo[i]]
        dset = fx.create_dataset("time", 1, dtype="d")
        dset.make_scale()
        dset[:] = self._TimeSteppingController.time

        # import sys; sys.exit()
        for ac in self._classes:
            # Loop over all variables
            ac = self._classes[ac]

            for v in ac._dofs:
                if "ff8" not in v:

                    dset = fx.create_dataset(
                        v,
                        (1, self._Grid.n[0], self._Grid.n[1], self._Grid.n[2]),
                        dtype=np.double,
                        compression=self.compression,
                        shuffle=self.shuffle,
                        chunks=(
                            1,
                            np.max(self._Grid.rank_nx),
                            np.max(self._Grid.rank_ny),
                            self._Grid.n[2],
                        ),
                    )

                    for i, d in enumerate(["time", "X", "Y", "Z"]):
                        dset.dims[i].attach_scale(fx[d])

                    if self.collective:

                        with dset.collective:
                            dset[
                                0,
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
                    dset.attrs["standard_name"] = ac.get_standard_name(v)

                MPI.COMM_WORLD.barrier()

        fx.close()
        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t Parallel IO of 3D fields finished in: " + str(t1 - t0) + " seconds"
        )
        self._Timers.end_timer("DumpFields")
        return
