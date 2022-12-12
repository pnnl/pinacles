import numpy as np
import numba
import os
from mpi4py import MPI
from pinacles import UtilitiesParallel

try:
    import h5py
except:
    pass


class CoarseGrainerBase:
    def __init__(self, namelist):

        self._namelist = namelist
        self._classes = {}
        self._frequency = 1e9

        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def update(self):
        return


class CoarseGrain:
    def __init__(
        self,
        namelist,
        TimeSteppingController,
        Grid,
        ScalarState,
        VelocityState,
        DiagnosticState,
        Micro,
        resolution,
        frequency,
    ):

        self._namelist = namelist
        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._Micro = Micro

        self._resolution = resolution
        self._frequency = frequency

        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])
        self._output_path = self._output_path = os.path.join(
            self._output_root, self._casename
        )
        self._output_path = os.path.join(self._output_path, "CoarseGrainedFields")
        self._output_path = os.path.join(self._output_path, str(resolution))

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)
        MPI.COMM_WORLD.barrier()

        if self._Grid.l[0] % self._resolution == 0.0:
            self.X_left = np.arange(0.0, self._Grid.l[0], self._resolution)
        else:
            self.X_left = np.arange(
                0.0, self._Grid.l[0] - self._resolution, self._resolution
            )

        if self._Grid.l[1] % self._resolution == 0.0:
            self.Y_left = np.arange(0.0, self._Grid.l[1], self._resolution)
        else:
            self.Y_left = np.arange(
                0.0, self._Grid.l[1] - self._resolution, self._resolution
            )

        self.X_right = self.X_left + self._resolution
        self.Y_right = self.Y_left + self._resolution

        self._coarse_shape = (
            self.X_left.shape[0],
            self.Y_right.shape[0],
            self._Grid.n[2],
        )

        return

    @staticmethod
    @numba.njit()
    def compute_mean(x, y, nh, var, resolution, count_array, local_array):

        shape = var.shape
        local_array.fill(0.0)
        count_array.fill(0)
        for i in range(nh[0], shape[0] - nh[0]):
            ii = int(x[i] // resolution)
            for j in range(nh[1], shape[1] - nh[1]):
                jj = int(y[j] // resolution)
                for k in range(nh[2], shape[2] - nh[2]):
                    local_array[ii, jj, k - nh[2]] += var[i, j, k]
                    count_array[ii, jj, k - nh[2]] += 1

    @staticmethod
    @numba.njit()
    def compute_frac(
        x, y, nh, var, resolution, count_array, local_array, threshold=1e-10
    ):

        shape = var.shape
        local_array.fill(0.0)
        count_array.fill(0)
        for i in range(nh[0], shape[0] - nh[0]):
            ii = int(x[i] // resolution)
            for j in range(nh[1], shape[1] - nh[1]):
                jj = int(y[j] // resolution)
                for k in range(nh[2], shape[2] - nh[2]):
                    if var[i, j, k] >= threshold:
                        local_array[ii, jj, k - nh[2]] += 1.0
                    count_array[ii, jj, k - nh[2]] += 1

    @staticmethod
    @numba.njit()
    def compute_2nd_moments(
        x, y, nh, var1, mean1, var2, mean2, resolution, local_array
    ):

        local_array.fill(0.0)
        shape = var1.shape
        for i in range(nh[0], shape[0] - nh[0]):
            ii = int(x[i] // resolution)
            for j in range(nh[1], shape[1] - nh[1]):
                jj = int(y[j] // resolution)
                for k in range(nh[2], shape[2] - nh[2]):
                    local_array[ii, jj, k - nh[2]] += (
                        var1[i, j, k] - mean1[ii, jj, k - nh[2]]
                    ) * (var2[i, j, k] - mean2[ii, jj, k - nh[2]])

        return

    @staticmethod
    @numba.njit()
    def compute_3rd_moments(
        x, y, nh, var1, mean1, var2, mean2, var3, mean3, resolution, local_array
    ):

        local_array.fill(0.0)
        shape = var1.shape
        for i in range(nh[0], shape[0] - nh[0]):
            ii = int(x[i] // resolution)
            for j in range(nh[1], shape[1] - nh[1]):
                jj = int(y[j] // resolution)
                for k in range(nh[2], shape[2] - nh[2]):
                    local_array[ii, jj, k - nh[2]] += (
                        (var1[i, j, k] - mean1[ii, jj, k - nh[2]])
                        * (var2[i, j, k] - mean2[ii, jj, k - nh[2]])
                        * (var3[i, j, k] - mean3[ii, jj, k - nh[2]])
                    )

        return

    def update(self):
        if not np.allclose(self._TimeSteppingController._time % self._frequency, 0.0):
            return

        local_array = np.zeros(self._coarse_shape, dtype=np.double)
        count_array = np.zeros(self._coarse_shape, dtype=np.int)

        global_array = np.zeros_like(local_array)
        global_count = np.zeros_like(count_array)

        local_axes = self._Grid._local_axes
        local_axes_edge = self._Grid._local_axes_edge
        nh = self._Grid.n_halo

        my_rank = MPI.COMM_WORLD.Get_rank()

        if my_rank == 0:
            s = self._TimeSteppingController.time
            days = s // 86400
            s = s - (days * 86400)
            hours = s // 3600
            s = s - (hours * 3600)
            minutes = s // 60
            seconds = s - (minutes * 60)

            fx = h5py.File(
                os.path.join(
                    self._output_path,
                    "{:02}d-{:02}h-{:02}m-{:02}s".format(
                        int(days), int(hours), int(minutes), int(seconds)
                    )
                    + ".h5",
                ),
                "w",
            )

            # Add some metadata
            fx.attrs["unique_id"] = self._namelist["meta"]["unique_id"]
            fx.attrs["wall_time"] = self._namelist["meta"]["wall_time"]
            fx.attrs["frequency"] = self._frequency

            for i, v in enumerate(["X", "Y", "Z"]):
                dset = fx.create_dataset(v, (self._coarse_shape[i],), dtype="d")
                dset.make_scale()

                if v == "X":
                    dset[:] = 0.5 * (self.X_left + self.X_right)
                if v == "Y":
                    dset[:] = 0.5 * (self.Y_left + self.Y_right)
                if v == "Z":
                    dset[:] = self._Grid._global_axes[i][nh[i] : -nh[i]]

            dset = fx.create_dataset("time", 1, dtype="d")
            dset.make_scale()
            dset[:] = self._TimeSteppingController.time

        u = np.copy(self._VelocityState.get_field("u"))
        u[:-1, :, :] = 0.5 * (u[:-1, :, :] + u[1:, :, :])

        v = np.copy(self._VelocityState.get_field("v"))
        v[:, :-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])

        w = np.copy(self._VelocityState.get_field("w"))
        w[:, :, :-1] = 0.5 * (w[:, :, :-1] + w[:, :, 1:])

        u_mean = None
        v_mean = None
        w_mean = None

        for c in [self._DiagnosticState, self._ScalarState, self._VelocityState]:
            for vn in c.dofs:
                var = c.get_field(vn)

                if vn == "w":
                    var = w
                if vn == "v":
                    var = v
                if vn == "u":
                    var = u

                x = local_axes[0]
                y = local_axes[1]

                self.compute_mean(
                    x, y, nh, var, self._resolution, count_array, local_array
                )

                # Now we have the mean on every rank
                MPI.COMM_WORLD.Allreduce(count_array, global_count, op=MPI.SUM)
                MPI.COMM_WORLD.Allreduce(local_array, global_array, op=MPI.SUM)

                if vn == "u":
                    u_mean = np.divide(global_array, global_count)
                if vn == "v":
                    v_mean = np.divide(global_array, global_count)
                if vn == "w":
                    w_mean = np.divide(global_array, global_count)

                var_mean = np.divide(global_array, global_count)
                if my_rank == 0:
                    var_fx = fx.create_dataset(
                        vn,
                        (
                            1,
                            self._coarse_shape[0],
                            self._coarse_shape[1],
                            self._coarse_shape[2],
                        ),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Y", "Z"]):
                        var_fx.dims[i].attach_scale(fx[d])

                    var_mean = np.divide(global_array, global_count)
                    var_fx[0, :, :, :] = var_mean

                # We don't need correlations for the SBM variables so exit loop.
                if "ff" in vn:
                    continue

                # Now compute 2nd moments
                self.compute_2nd_moments(
                    x,
                    y,
                    nh,
                    var,
                    var_mean,
                    var,
                    var_mean,
                    self._resolution,
                    local_array,
                )
                MPI.COMM_WORLD.Reduce(local_array, global_array, op=MPI.SUM)

                if my_rank == 0:
                    var_fx = fx.create_dataset(
                        vn + "_" + vn + "_mom",
                        (
                            1,
                            self._coarse_shape[0],
                            self._coarse_shape[1],
                            self._coarse_shape[2],
                        ),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Y", "Z"]):
                        var_fx.dims[i].attach_scale(fx[d])

                    var_fx[0, :, :, :] = global_array / global_count

                self.compute_3rd_moments(
                    x,
                    y,
                    nh,
                    var,
                    var_mean,
                    var,
                    var_mean,
                    var,
                    var_mean,
                    self._resolution,
                    local_array,
                )
                MPI.COMM_WORLD.Reduce(local_array, global_array, op=MPI.SUM)

                if my_rank == 0:
                    var_fx = fx.create_dataset(
                        vn + "_" + vn + "_" + vn + "_mom",
                        (
                            1,
                            self._coarse_shape[0],
                            self._coarse_shape[1],
                            self._coarse_shape[2],
                        ),
                        dtype=np.double,
                    )

                    for i, d in enumerate(["time", "X", "Y", "Z"]):
                        var_fx.dims[i].attach_scale(fx[d])

                    var_fx[0, :, :, :] = global_array / global_count

        # Compute fluxes
        for c in [self._ScalarState, self._VelocityState, self._DiagnosticState]:
            for vn in c.dofs:

                # Don't want to compute the per bin fluxes
                if "ff" in vn:
                    continue

                var = c.get_field(vn)

                if vn == "w":
                    var = w
                if vn == "v":
                    var = v
                if vn == "u":
                    var = u

                x = local_axes[0]
                y = local_axes[1]

                self.compute_mean(
                    x, y, nh, var, self._resolution, count_array, local_array
                )

                # Now we have the mean on every rank
                MPI.COMM_WORLD.Allreduce(count_array, global_count, op=MPI.SUM)
                MPI.COMM_WORLD.Allreduce(local_array, global_array, op=MPI.SUM)

                var_mean = np.divide(global_array, global_count)

                # First compute the z-fluxes
                if vn != "w":
                    self.compute_2nd_moments(
                        x,
                        y,
                        nh,
                        w,
                        w_mean,
                        var,
                        var_mean,
                        self._resolution,
                        local_array,
                    )

                    MPI.COMM_WORLD.Reduce(local_array, global_array, op=MPI.SUM)

                    if my_rank == 0:
                        var_fx = fx.create_dataset(
                            "w" + "_" + vn + "_mom",
                            (
                                1,
                                self._coarse_shape[0],
                                self._coarse_shape[1],
                                self._coarse_shape[2],
                            ),
                            dtype=np.double,
                        )

                        for i, d in enumerate(["time", "X", "Y", "Z"]):
                            var_fx.dims[i].attach_scale(fx[d])

                        var_fx[0, :, :, :] = global_array / global_count

        # Compute cloud, rain, and ice fraction
        qc_cloud = self._Micro.get_qcloud()
        qc = self._Micro.get_qc()
        qr = qc - qc_cloud

        qi = self._Micro.get_qi()

        x = local_axes[0]
        y = local_axes[1]

        self.compute_frac(
            x, y, nh, qc_cloud, self._resolution, count_array, local_array
        )

        # Now we have the mean on every rank
        MPI.COMM_WORLD.Allreduce(count_array, global_count, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(local_array, global_array, op=MPI.SUM)

        if my_rank == 0:
            var_fx = fx.create_dataset(
                "cf",
                (
                    1,
                    self._coarse_shape[0],
                    self._coarse_shape[1],
                    self._coarse_shape[2],
                ),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y", "Z"]):
                var_fx.dims[i].attach_scale(fx[d])

            var_fx[0, :, :, :] = global_array / global_count

        self.compute_frac(x, y, nh, qr, self._resolution, count_array, local_array)

        # Now we have the mean on every rank
        MPI.COMM_WORLD.Allreduce(count_array, global_count, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(local_array, global_array, op=MPI.SUM)

        if my_rank == 0:
            var_fx = fx.create_dataset(
                "rf",
                (
                    1,
                    self._coarse_shape[0],
                    self._coarse_shape[1],
                    self._coarse_shape[2],
                ),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y", "Z"]):
                var_fx.dims[i].attach_scale(fx[d])

            var_fx[0, :, :, :] = global_array / global_count

        self.compute_frac(x, y, nh, qi, self._resolution, count_array, local_array)

        # Now we have the mean on every rank
        MPI.COMM_WORLD.Allreduce(count_array, global_count, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(local_array, global_array, op=MPI.SUM)

        if my_rank == 0:
            var_fx = fx.create_dataset(
                "if",
                (
                    1,
                    self._coarse_shape[0],
                    self._coarse_shape[1],
                    self._coarse_shape[2],
                ),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y", "Z"]):
                var_fx.dims[i].attach_scale(fx[d])

            var_fx[0, :, :, :] = global_array / global_count

        return


class CoarseGrainer(CoarseGrainerBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        TimeSteppingController,
        ScalarState,
        VelocityState,
        DiagnosticState,
        Micro,
    ):

        CoarseGrainerBase.__init__(self, namelist)

        self._Timers = Timers
        self._Grid = Grid
        self._TimesteppingController = TimeSteppingController
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._Micro = Micro

        try:
            self._frequency = namelist["coarse_grainers"]["frequency"]
        except:
            self._frequency = 1e9

        try:
            self._resolutions = namelist["coarse_grainers"]["resolutions"]
        except:
            self._resolutions = []

        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])
        self._output_path = self._output_path = os.path.join(
            self._output_root, self._casename
        )
        self._output_path = os.path.join(self._output_path, "CoarseGrainedFields")

        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)
        MPI.COMM_WORLD.barrier()

        self.coarse_grainers = []
        for res in self._resolutions:
            self.coarse_grainers.append(
                CoarseGrain(
                    namelist,
                    TimeSteppingController,
                    Grid,
                    ScalarState,
                    VelocityState,
                    DiagnosticState,
                    Micro,
                    res,
                    self._frequency,
                )
            )

    def update(self):
        for cg in self.coarse_grainers:
            cg.update()

        return


def CoarseGrainFactory(
    namelist,
    Timers,
    Grid,
    TimeSteppingController,
    ScalarState,
    VelocityState,
    DiagnosticState,
    Micro,
):

    if "coarse_grainers" not in namelist:
        return CoarseGrainerBase(namelist)
    else:
        try:
            import h5py
        except:
            UtilitiesParallel.print_root("No H5PY-Disabling Coarse Grain output")
            return CoarseGrainerBase(namelist)

        return CoarseGrainer(
            namelist,
            Timers,
            Grid,
            TimeSteppingController,
            ScalarState,
            VelocityState,
            DiagnosticState,
            Micro,
        )

    return
