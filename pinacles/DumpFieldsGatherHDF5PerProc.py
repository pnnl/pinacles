from pinacles.DumpFieldsGatherHDF5 import DumpFields_hdf
from mpi4py import MPI
import os
import time
from pinacles import UtilitiesParallel
import h5py
import numpy as np


class DumpFields_hdf_perproc(DumpFields_hdf):
    def update(self):

        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def update(self):

        t0 = time.perf_counter()
        self._Timers.start_timer("DumpFields")

        output_here = self._output_path  # os.path.join(

        s = self._TimeSteppingController.time
        days = s // 86400
        s = s - (days * 86400)
        hours = s // 3600
        s = s - (hours * 3600)
        minutes = s // 60
        seconds = s - (minutes * 60)

        MPI.COMM_WORLD.barrier()

        output_here = os.path.join(
            output_here,
            "{:02}d-{:02}h-{:02}m-{:02}s".format(
                int(days), int(hours), int(minutes), int(seconds)
            ),
        )

        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)

        MPI.COMM_WORLD.barrier()

        nhalo = self._Grid.n_halo
        local_start = self._Grid._local_start
        local_end = self._Grid._local_end
        if self.write:

            output_here = os.path.join(output_here, str(self.iocomm.Get_rank()) + ".h5")
            fx = h5py.File(output_here, "w")

            fx.attrs["dt"] = self._TimeSteppingController.dt

            n = (
                self.xw_end - self.xw_start,
                self.yw_end - self.yw_start,
                self._Grid.n[2],
            )
            range = (
                (self.xw_start, self.xw_end),
                (self.yw_start, self.yw_end),
                (nhalo[2], nhalo[2] + self._Grid.n[2]),
            )
            for i, v in enumerate(["X", "Y", "Z"]):
                dset = fx.create_dataset(v, (n[i],), dtype="d")
                dset.make_scale()
                dset[:] = self._Grid._global_axes[i][range[i][0] : range[i][1]]
            dset = fx.create_dataset("time", 1, dtype="d")
            dset.make_scale()
            dset[:] = self._TimeSteppingController.time

        for ac in self._classes:
            #     # Loop over all variables
            ac = self._classes[ac]

            for v in ac._dofs:
                if "ff8" not in v:

                    if self.write:
                        dset = fx.create_dataset(
                            v,
                            (1, n[0], n[1], n[2]),
                            dtype=np.double,
                            compression=self.compression,
                            shuffle=self.shuffle,
                            chunks=self.chunks,
                        )

                        for i, d in enumerate(["time", "X", "Y", "Z"]):
                            dset.dims[i].attach_scale(fx[d])

                    vv = self.gather.call(ac.get_field(v))

                    if self.write:
                        dset[0, :, :, :] = vv

        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t Parallel IO of 3D fields finished in: " + str(t1 - t0) + " seconds"
        )
        self._Timers.end_timer("DumpFields")

        return
