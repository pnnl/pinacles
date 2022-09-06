import numpy as np
import time
import os
from mpi4py import MPI
from pinacles import UtilitiesParallel

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
try:
    import h5py
except:
    pass


class Fields2D:
    def __init__(
        self,
        namelist,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        
        if "qad" in self._ScalarState._dofs:
            self._scalar_list = ["qc","qnc","qad","qnad","qad2","qnad2"]
        else:
            self._scalar_list = ["qc","qnc"]

        """Set the output frequency, default it to the stats frequency 
        but allow in to be overridden """

        try:
            self._frequency = namelist["fields2d"]["frequency"]
        except:
            self._frequency = namelist["stats"]["frequency"]

        # specify vertical levels (indices) at which to output velocity planes
        try:
            self._output_levels = namelist["fields2d"]["output_levels"]
        except:
            self._output_levels = [0]

        try:
            self._x_locs = namelist["fields2d"]["x_locs"]
        except:
            self._x_locs = []

        try:
            self._y_locs = namelist["fields2d"]["y_locs"]
        except:
            self._y_locs = []

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

        output_here = self._output_path  # os.path.join(
        #    self._output_path, str(np.round(self._TimeSteppingController.time))
        # )

        MPI.COMM_WORLD.barrier()
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)
        MPI.COMM_WORLD.barrier()

        fx = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            s = self._TimeSteppingController.time
            days = s // 86400.0
            s = s - (days * 86400.0)
            hours = s // 3600.0
            s = s - (hours * 3600.0)
            minutes = s // 60.0
            s = s - (minutes * 60.0)
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
            )

            # Add some metadata
            fx.attrs["unique_id"] = self._namelist["meta"]["unique_id"]
            fx.attrs["wall_time"] = self._namelist["meta"]["wall_time"]
            fx.attrs["frequency"] = self.frequency

            self.setup_dims(fx)

        for aclass in self._classes:
            self._classes[aclass].io_fields2d_update(fx)

        self.output_horizontal(
            fx, [self._ScalarState, self._DiagnosticState, self._VelocityState]
        )
        self.output_velocities(fx)
        self.output_scalars(fx)

        # Sync and close netcdf file
        if fx is not None:
            fx.close()

        self._last_io_time = self._TimeSteppingController._time

        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t  2D IO Finished in: " + str(t1 - t0) + " seconds."
        )

        return

    def output_velocities(self, fx):

        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo

        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        for v in ["u", "v"]:
            for k in self._output_levels:
                z = self._Grid.z_global[nh[2] + k]

                if fx is not None:
                    var_fx = fx.create_dataset(
                        v + "_" + str(z),
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Y"]):
                        var_fx.dims[i].attach_scale(fx[d])

                var = self._VelocityState.get_field(v)
                send_buffer.fill(0.0)
                send_buffer[start[0] : end[0], start[1] : end[1]] = var[
                    nh[0] : -nh[0], nh[1] : -nh[1], nh[2] + k
                ]
                MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
                if fx is not None:
                    var_fx[:, :] = recv_buffer

        for v in ["w"]:
            for k in self._output_levels:
                z = self._Grid.z_global[nh[2] + k]

                if fx is not None:
                    var_fx = fx.create_dataset(
                        v + "_" + str(z),
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Y"]):
                        var_fx.dims[i].attach_scale(fx[d])

                var = self._VelocityState.get_field(v)
                send_buffer.fill(0.0)
                send_buffer[start[0] : end[0], start[1] : end[1]] = (
                    np.add(
                        var[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] + k],
                        var[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] + k - 1],
                    )
                    * 0.5
                )
                MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
                if fx is not None:
                    var_fx[:, :] = recv_buffer

        return

    def output_horizontal(self, fx, container):
        xla = self._Grid.global_axes[0]
        yla = self._Grid.global_axes[1]
        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo    
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[2]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        for con in container:
            for v in con._dofs:
                for xl in self._x_locs:

                    xi = np.min(np.argmin(np.abs(yla - xl)))
                    data = con.get_field_slice_h(v, xi)

                    if fx is not None:
                        var_fx = fx.create_dataset(
                            v + "_x_" + str(xl),
                            (1, self._Grid.n[1], self._Grid.n[2]),
                            dtype=np.double,
                        )

                        for i, d in enumerate(["time", "Y", "Z"]):
                            var_fx.dims[i].attach_scale(fx[d])

                        var_fx[:, :] = data

                for yl in self._y_locs:

                    yi = np.min(np.argmin(np.abs(xla - yl)))
                    data = con.get_field_slice_h(v, yi, y=True)

                    if fx is not None:
                        var_fx = fx.create_dataset(
                            v + "_y_" + str(xl),
                            (1, self._Grid.n[0], self._Grid.n[2]),
                            dtype=np.double,
                        )

                        for i, d in enumerate(["time", "X", "Z"]):
                            var_fx.dims[i].attach_scale(fx[d])

                        var_fx[:, :] = data
                                               
                if fx is not None:
                    var_fx = fx.create_dataset(
                        v + "_mean",
                        (1, self._Grid.n[0], self._Grid.n[2]),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Z"]):
                        var_fx.dims[i].attach_scale(fx[d])
                    
                    # for i in range(start[0],end[0]):
                    #     for k in range(start[2],end[2]):
                    #        send_buffer[i, k] = data_3d[i, start[1] : end[1], k].sum()
                
                data_3d = con.get_field(v)
                send_buffer.fill(0.0) 
                send_buffer[start[0]:end[0], start[2]:end[2]] = np.sum(data_3d[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] : -nh[2]], axis = 1)
                MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)    
                
                if fx is not None:
                    var_fx[:, :] = recv_buffer/self._Grid.n[1]
        return

    def output_scalars(self, fx):

        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo

        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        for v in self._scalar_list:
            for k in self._output_levels:
                z = self._Grid.z_global[nh[2] + k]

                if fx is not None:
                    var_fx = fx.create_dataset(
                                v + "_" + str(z),
                                (1, self._Grid.n[0], self._Grid.n[1]),
                                dtype=np.double,
                            )

                    for i, d in enumerate(["time", "X", "Y"]):
                        var_fx.dims[i].attach_scale(fx[d])

                var = self._ScalarState.get_field(v)
                send_buffer.fill(0.0)
                send_buffer[start[0] : end[0], start[1] : end[1]] = var[
                    nh[0] : -nh[0], nh[1] : -nh[1], nh[2] + k
                ]
                MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
                if fx is not None:
                    var_fx[:, :] = recv_buffer


        return

    def setup_directories(self):

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        return

    def setup_dims(self, fx):

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

        return


class Fields2DNone:
    def __init__(self):

        self._frequency = 1.0e10
        self._classes = {}

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

        return


def factory(
    namelist,
    Grid,
    Ref,
    ScalarState,
    VelocityState,
    DiagnosticState,
    TimeSteppingController,
):
    try:
        import h5py

        dofields2d = True
    except:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Cannot import h5py, will not provide 2d Fields!")
        dofields2d = False

    if dofields2d:
        return Fields2D(
            namelist,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    else:
        return Fields2DNone()
