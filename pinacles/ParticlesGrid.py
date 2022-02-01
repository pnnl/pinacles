import numpy as np
import numba
from numba.core import types
from numba.typed import Dict, List
import time
import os
from mpi4py import MPI
import h5py


class ParticlesBase:
    def __init__(
        self,
        namelist,
        Grid,
        Ref,
        TimeSteppingController,
        VelocityState,
        ScalarState,
        DiagnosticState,
    ):

        # Initialize data
        self._namelist = namelist
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])
        self._output_path = self._output_path = os.path.join(
            self._output_root, self._casename
        )
        self._output_path = os.path.join(self._output_path, "particles")
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        # Add a variable for the number of lagrangian particles
        self._DiagnosticState.add_variable("n_lagrangian")
        self._DiagnosticState.add_variable("n_particles")

        # Set the particle output frequency
        assert "frequency" in self._namelist["particles"]
        self.frequency = self._namelist["particles"]["frequency"]

        # Set the number of particles to be injected
        assert "n_inject" in self._namelist["particles"]
        self.n_inject = self._namelist["particles"]["n_inject"]

        # Set inject time
        assert "time_inject" in self._namelist["particles"]
        self.time_inject = self._namelist["particles"]["time_inject"]

        # Set injection type
        assert "inject_type" in self._namelist["particles"]
        self.inject_type = self._namelist["particles"]["inject_type"]

        if self.inject_type == "point":
            assert "point_location" in self._namelist["particles"]
            self.point_location = self._namelist["particles"]["point_location"]
            assert len(self.point_location) == 3

            assert "point_time_range" in self._namelist["particles"]
            self.point_time_range = tuple(
                self._namelist["particles"]["point_time_range"]
            )
            assert len(self.point_time_range) == 2

        if self.inject_type == "surface random":
            assert "inject_height" in self._namelist["particles"]
            self.inject_height = self._namelist["particles"]["inject_height"]
            assert type(self.inject_height) is int or type(self.inject_height) is float

        # Get boundary type
        assert "boundary_type" in self._namelist["particles"]
        self.boundary_type = self._namelist["particles"]["boundary_type"]
        assert self.boundary_type in ["periodic", "exit"]

        self.injected = False
        self._particle_dofs = 0  # The first three dofs are always position
        self._interp_particle_dofs = 0
        self._nointerp_particle_dofs = 0
        self._particle_data = None
        self._initialzied = False
        self._n_buffer = 1024  # The size of the buffer
        self._minimum_particles = 2

        # These are numba dictionaries
        self._particle_varnames = Dict.empty(
            key_type=types.unicode_type, value_type=types.int64
        )
        self._interp_particle_varnames = Dict.empty(
            key_type=types.unicode_type, value_type=types.int64
        )
        self._nointerp_particle_varnames = Dict.empty(
            key_type=types.unicode_type, value_type=types.int64
        )

        self.add_particle_variable("valid")
        self.add_particle_variable("id")
        self.add_particle_variable("x")
        self.add_particle_variable("y")
        self.add_particle_variable("z")

        self.add_particle_variable("u")
        self.add_particle_variable("v")
        self.add_particle_variable("w")
        self.add_particle_variable("dummy")
        self.add_particle_variable("n_particles")  # Number of particles

        n = 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            xp = np.random.uniform(0.0, 5120, n)
            yp = np.random.uniform(0.0, 5120, n)
            zp = np.random.uniform(50, 1000.0, n)
        else:
            xp = np.empty((n,), dtype=np.double)
            yp = np.empty((n,), dtype=np.double)
            zp = np.empty((n,), dtype=np.double)

        MPI.COMM_WORLD.Bcast(xp, root=0)
        MPI.COMM_WORLD.Bcast(yp, root=0)
        MPI.COMM_WORLD.Bcast(zp, root=0)

        self._allocate_memory()
        self.initialize_particles(xp, yp, zp)
        self.call_count = 1000000000

        nglob = np.empty((1,), dtype=np.int)
        MPI.COMM_WORLD.Allreduce(
            np.array(np.sum(self._n), dtype=np.int), nglob, op=MPI.SUM
        )

        return

    def _allocate_memory(self):
        local_shape = self._Grid.local_shape

        # This is list of floatng point numpy arrays (that are not necessarily of the same size)
        self._particle_data = List.empty_list(item_type=types.float64[:, :])

        # Store the valid particle here
        self._n = np.zeros(
            (local_shape[0], local_shape[1], local_shape[2]), dtype=np.int64
        )

        return

    def initialize_particles(self, xp, yp, zp):
        assert xp.shape == yp.shape
        assert xp.shape == zp.shape
        low_corner_local = (
            self._Grid.x_range_local[0],
            self._Grid.y_range_local[0],
            self._Grid.z_range_local[0],
        )
        high_corner_local = (
            self._Grid.x_range_local[1],
            self._Grid.y_range_local[1],
            self._Grid.z_range_local[1],
        )
        local_shape = self._Grid.local_shape
        dx = self._Grid.dx
        n_per_cell = np.zeros(local_shape, dtype=np.int64)

        self.map_particles_allocate(
            low_corner_local,
            high_corner_local,
            dx,
            xp,
            yp,
            zp,
            self._particle_varnames,
            self._n_buffer,
            self._particle_dofs,
            self._n,
            self._particle_data,
        )

        return

    @staticmethod
    @numba.njit
    def map_particles_allocate(
        low_corner_local,
        high_corner_local,
        dx,
        xp,
        yp,
        zp,
        particle_varnames,
        n_buffer,
        particle_dofs,
        n,
        particle_data,
    ):

        # First determine how many points are valid
        valid_dof = particle_varnames["valid"]
        x_dof = particle_varnames["x"]
        y_dof = particle_varnames["y"]
        z_dof = particle_varnames["z"]
        npart = xp.shape[0]
        for pi in range(npart):
            if (
                xp[pi] >= low_corner_local[0]
                and xp[pi] < high_corner_local[0]
                and yp[pi] >= low_corner_local[1]
                and yp[pi] < high_corner_local[1]
            ):
                i = int((xp[pi] - low_corner_local[0]) // dx[0])
                j = int((yp[pi] - low_corner_local[1]) // dx[1])
                k = int((zp[pi] - low_corner_local[2]) // dx[2])
                n[i, j, k] += 1

        # Loop over the grid and allocate particle arrays
        shape = n.shape
        ishift = shape[1] * shape[2]
        jshift = shape[2]
        for i in range(shape[0]):
            ii = i * ishift
            for j in range(shape[1]):
                jj = j * jshift
                for k in range(shape[2]):
                    particle_data.append(
                        np.zeros((particle_dofs, max(0, n[i, j, k])), dtype=np.double)
                    )
                    arr = particle_data[ii + jj + k]
                    arr[valid_dof, :] = 0.0

        for pi in range(npart):
            if (
                xp[pi] >= low_corner_local[0]
                and xp[pi] < high_corner_local[0]
                and yp[pi] >= low_corner_local[1]
                and yp[pi] < high_corner_local[1]
            ):
                i = int((xp[pi] - low_corner_local[0]) // dx[0])
                j = int((yp[pi] - low_corner_local[1]) // dx[1])
                k = int((zp[pi] - low_corner_local[2]) // dx[2])
                arr = particle_data[i * ishift + j * jshift + k]

                fits = False
                for n in range(arr.shape[1]):

                    if arr[valid_dof, n] == 0.0:
                        arr[valid_dof, n] = 1.0
                        arr[x_dof, n] = xp[pi]
                        arr[y_dof, n] = yp[pi]
                        arr[z_dof, n] = zp[pi]
                        fits = True
                    break

                if not fits:
                    n = arr.shape[1]
                    new_buf = np.zeros((arr.shape[0], n_buffer), dtype=np.double)
                    particle_data[i * ishift + j * jshift + k] = np.concatenate(
                        (arr, new_buf), axis=1
                    )
                    arr = particle_data[i * ishift + j * jshift + k]
                    arr[valid_dof, n] = 1.0
                    arr[x_dof, n] = xp[pi]
                    arr[y_dof, n] = yp[pi]
                    arr[z_dof, n] = zp[pi]

        return

    @staticmethod
    def surface_random_inject(
        n_inject,
        l,
        x_range,
        y_range,
        low_corner_local,
        high_corner_local,
        dx,
        nhalo,
        particle_varnames,
        particle_data,
        inject_height,
        n,
    ):

        if MPI.COMM_WORLD.Get_rank() == 0:
            x = np.random.uniform(low=0.0, high=l[0], size=n_inject)
            y = np.random.uniform(low=0.0, high=l[1], size=n_inject)
        else:
            x = None
            y = None

        x = MPI.COMM_WORLD.bcast(x, root=0)
        y = MPI.COMM_WORLD.bcast(y, root=0)

        # Figure out which new particles are on this MPI rank
        mask = (x_range[0] <= x) & (x_range[1] > x)
        mask = mask & (y_range[0] <= y) & (y_range[1] > y)
        x = x[mask]
        y = y[mask]

        # Compute the grid index of each point on this rank
        iindx = (x - low_corner_local[0]) // dx[0]
        jindx = (y - low_corner_local[1]) // dx[1]
        kindx = (inject_height - low_corner_local[2]) // dx[2]

        iindx = iindx.astype(int)
        jindx = jindx.astype(int)
        kindx = int(kindx)

        # Get the particle  dofs for the position
        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]
        ndof = particle_varnames["n_particles"]
        iddof = particle_varnames["id"]

        # Now let check an see if we need to allocate more memory
        assert len(iindx) == len(jindx)

        shape = n.shape
        ishift = shape[1] * shape[2]
        jshift = shape[2]

        for i in range(len(iindx)):
            ii = int(iindx[i] * ishift)
            jj = int(jindx[i] * jshift)
            k = kindx

            arr = particle_data[ii + jj + k]
            n_arr = arr.shape[1]

            if n_arr < n[iindx[i], jindx[i], k] + 1:
                new_buf = np.zeros((arr.shape[0], 1), dtype=np.double)
                particle_data[ii + jj + k] = np.concatenate((arr, new_buf), axis=1)
                arr = particle_data[ii + jj + k]

            # if n[ii, jj, k] < n_total:
            for pi in range(arr.shape[1]):
                if arr[valid, pi] == 0.0:
                    arr[xdof, pi] = x[i]
                    arr[ydof, pi] = y[i]
                    arr[zdof, pi] = inject_height
                    arr[iddof, pi] = np.random.uniform(-1e9, 1e9)
                    arr[valid, pi] = 1.0
                    arr[ndof, pi] = 0.0
                    n[iindx[i], jindx[i], k] += 1

                    # if n[ii, jj, k] >= n_total:
                    #    break

        # import sys

        # sys.exit()

        return

    @staticmethod
    @numba.njit()
    def point_inject(
        low_corner_local,
        high_corner_local,
        local_shape,
        dx,
        n,
        dt,
        particle_varnames,
        particle_data,
        indicies,
        n_total,
    ):

        i, j, k = indicies
        shape = n.shape

        # Get the particle array at this point

        ishift = shape[1] * shape[2]
        jshift = shape[2]
        ii = i * ishift
        jj = j * jshift
        arr = particle_data[ii + jj + k]
        n_arr = arr.shape[1]

        # Get the particle  dofs for the position
        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]
        ndof = particle_varnames["n_particles"]
        iddof = particle_varnames["id"]

        # Now let check an see if we need to allocate more memory
        if n_arr < n_total:
            new_buf = np.zeros((arr.shape[0], n_total - n_arr), dtype=np.double)
            particle_data[ii + jj + k] = np.concatenate((arr, new_buf), axis=1)
            arr = particle_data[ii + jj + k]

        if n[i, j, k] < n_total:
            for pi in range(arr.shape[1]):
                if arr[valid, pi] == 0.0:
                    arr[xdof, pi] = np.random.uniform(
                        low_corner_local[0] + dx[0] * i,
                        low_corner_local[0] + dx[0] * (i + 1),
                    )
                    arr[ydof, pi] = np.random.uniform(
                        low_corner_local[1] + dx[1] * j,
                        low_corner_local[1] + dx[1] * (j + 1),
                    )
                    arr[zdof, pi] = np.random.uniform(
                        low_corner_local[2] + dx[2] * k,
                        low_corner_local[2] + dx[2] * (k + 1),
                    )
                    arr[iddof, pi] = np.random.uniform(-1e9, 1e9)
                    arr[valid, pi] = 1.0

                    arr[ndof, pi] = 0.0
                    n[i, j, k] += 1

                    if n[i, j, k] >= n_total:
                        break

        # Now divide flux among particles
        for pi in range(arr.shape[1]):
            if arr[valid, pi] == 1.0:
                arr[ndof, pi] += 1.0 * dt / n[i, j, k]

        return

    def get_particle_var(self, name):
        if name in self._interp_particle_varnames:
            indx = self._interp_particle_varnames[name]
            return indx
        elif name in self._nointerp_particle_varnames:
            indx = self._nointerp_particle_varnames[name] + self._interp_particle_dofs
            return indx

    def update(self):

        return

    def update_position(self):

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        tke_sgs = self._DiagnosticState.get_field("tke_sgs")

        local_shape = self._Grid.local_shape
        n_halo = self._Grid.n_halo
        low_corner = (
            self._Grid.x_range[0],
            self._Grid.y_range[0],
            self._Grid.z_range[0],
        )
        high_corner = (
            self._Grid.x_range[1],
            self._Grid.y_range[1],
            self._Grid.z_range[1],
        )

        low_corner_local = (
            self._Grid.x_range_local[0],
            self._Grid.y_range_local[0],
            self._Grid.z_range_local[0],
        )
        high_corner_local = (
            self._Grid.x_range_local[1],
            self._Grid.y_range_local[1],
            self._Grid.z_range_local[1],
        )

        l = self._Grid.l

        self.compute_new_position(
            local_shape,
            tke_sgs[
                n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1], n_halo[2] : -n_halo[2]
            ],
            self._n,
            self._particle_varnames,
            self._particle_data,
            self._TimeSteppingController.dt,
        )

        # Bounce particles at the surface

        self._surface_bounce(
            local_shape, self._n, self._particle_varnames, self._particle_data
        )

        if MPI.COMM_WORLD.Get_size() == 0:
            if self.boundary_type == "exit":
                self._boundary_exit_serial(
                    local_shape,
                    l,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                )

            if self.boundary_type == "periodic":
                self._boundary_periodic_serial(
                    local_shape,
                    high_corner,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    self._Grid.subcomms[0].Get_size(),
                    self._Grid.subcomms[1].Get_size(),
                )
        else:

            # Apply global lateral boundaries here. I think we can use the serial code
            if self.boundary_type == "exit":
                self._boundary_exit_serial(
                    local_shape,
                    l,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                )

            if self.boundary_type == "periodic":
                self._boundary_periodic_serial(
                    local_shape,
                    high_corner,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    self._Grid.subcomms[0].Get_size(),
                    self._Grid.subcomms[1].Get_size(),
                )

            n_send = np.zeros(
                (local_shape[0], local_shape[1], local_shape[2], 2), dtype=np.int
            )

            # First do x
            # Get the x sub communicator
            comm = self._Grid.subcomms[0]
            comm_size = comm.Get_size()

            if comm_size > 1:
                self._find_x_crossing_particles(
                    local_shape,
                    low_corner_local,
                    high_corner_local,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    n_send,
                )
                n_send_right = np.sum(n_send[:, :, :, 0])
                n_send_left = np.sum(n_send[:, :, :, 1])

                send_buffer_left = np.empty(
                    (self._particle_dofs, n_send_left), dtype=np.double
                )
                send_buffer_right = np.empty(
                    (self._particle_dofs, n_send_right), dtype=np.double
                )

                # In the future we can take this out of the loop when the number to send are zero
                self._pack_x_buffers(
                    local_shape,
                    low_corner_local,
                    high_corner_local,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    n_send,
                    send_buffer_left,
                    send_buffer_right,
                )

                # Send from left to right
                source, dest = comm.Shift(0, 1)
                if source == MPI.PROC_NULL:
                    source = comm_size - 1

                if dest == MPI.PROC_NULL:
                    dest = 0

                # First send left
                n_recv_left = np.array(0, dtype=np.int)
                comm.Sendrecv(n_send_right, dest, recvbuf=n_recv_left, source=source)

                # Now we need to allocate the receive buffer
                recv_buffer_left = np.empty(
                    (self._particle_dofs, n_recv_left), dtype=np.double
                )
                comm.Sendrecv(
                    send_buffer_right, dest, recvbuf=recv_buffer_left, source=source
                )
                self.unpack_buffer(
                    low_corner_local,
                    high_corner_local,
                    high_corner,
                    self._Grid.dx,
                    self._n,
                    self._n_buffer,
                    recv_buffer_left,
                    self._particle_varnames,
                    self._particle_data,
                    "x",
                )

                source, dest = comm.Shift(0, -1)
                if source == MPI.PROC_NULL:
                    source = 0

                if dest == MPI.PROC_NULL:
                    dest = comm_size - 1

                n_recv_right = np.array(0, dtype=np.int)
                comm.Sendrecv(n_send_left, dest, recvbuf=n_recv_right, source=source)
                recv_buffer_right = np.empty(
                    (self._particle_dofs, n_recv_right), dtype=np.double
                )
                comm.Sendrecv(
                    send_buffer_left, dest, recvbuf=recv_buffer_right, source=source
                )
                self.unpack_buffer(
                    low_corner_local,
                    high_corner_local,
                    high_corner,
                    self._Grid.dx,
                    self._n,
                    self._n_buffer,
                    recv_buffer_right,
                    self._particle_varnames,
                    self._particle_data,
                    "x",
                )

            # Get the y sub communicator and it size
            comm = self._Grid.subcomms[1]
            comm_size = comm.Get_size()

            MPI.COMM_WORLD.Barrier()

            # No need to do anything
            if comm_size > 1:
                # Now do y
                n_send.fill(0.0)
                self._find_y_crossing_particles(
                    local_shape,
                    low_corner_local,
                    high_corner_local,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    n_send,
                )
                n_send_right = np.sum(n_send[:, :, :, 0])
                n_send_left = np.sum(n_send[:, :, :, 1])

                send_buffer_left = np.empty(
                    (self._particle_dofs, n_send_left), dtype=np.double
                )
                send_buffer_right = np.empty(
                    (self._particle_dofs, n_send_right), dtype=np.double
                )

                # In the future we can take this out of the loop when the number to send are zero
                self._pack_y_buffers(
                    local_shape,
                    low_corner_local,
                    high_corner_local,
                    self._n,
                    self._particle_varnames,
                    self._particle_data,
                    n_send,
                    send_buffer_left,
                    send_buffer_right,
                )

                # Send from left to right
                source, dest = comm.Shift(0, 1)
                if source == MPI.PROC_NULL:
                    source = comm_size - 1

                if dest == MPI.PROC_NULL:
                    dest = 0

                # First send left
                n_recv_left = np.array(0, dtype=np.int)
                comm.Sendrecv(n_send_right, dest, recvbuf=n_recv_left, source=source)

                # Now we need to allocate the receive buffer
                recv_buffer_left = np.empty(
                    (self._particle_dofs, n_recv_left), dtype=np.double
                )
                comm.Sendrecv(
                    send_buffer_right, dest, recvbuf=recv_buffer_left, source=source
                )

                self.unpack_buffer(
                    low_corner_local,
                    high_corner_local,
                    high_corner,
                    self._Grid.dx,
                    self._n,
                    self._n_buffer,
                    recv_buffer_left,
                    self._particle_varnames,
                    self._particle_data,
                    "y",
                )
                source, dest = comm.Shift(0, -1)

                if source == MPI.PROC_NULL:
                    source = 0

                if dest == MPI.PROC_NULL:
                    dest = comm_size - 1

                n_recv_right = np.array(0, dtype=np.int)
                comm.Sendrecv(n_send_left, dest, recvbuf=n_recv_right, source=source)
                recv_buffer_right = np.empty(
                    (self._particle_dofs, n_recv_right), dtype=np.double
                )
                comm.Sendrecv(
                    send_buffer_left, dest, recvbuf=recv_buffer_right, source=source
                )
                self.unpack_buffer(
                    low_corner_local,
                    high_corner_local,
                    high_corner,
                    self._Grid.dx,
                    self._n,
                    self._n_buffer,
                    recv_buffer_right,
                    self._particle_varnames,
                    self._particle_data,
                    "y",
                )

        # Move particles to new eulerian grid cells
        self.move_particles_on_grid(
            low_corner_local,
            local_shape,
            self._Grid.dx,
            self._n_buffer,
            self._n,
            self._particle_varnames,
            self._particle_data,
        )

        if self.inject_type == "surface random":
            if (
                self._TimeSteppingController.time >= self.time_inject
                and not self.injected
            ):
                self.surface_random_inject(
                    self.n_inject,
                    self._Grid.l,
                    self._Grid.x_range_local,
                    self._Grid.y_range_local,
                    low_corner_local,
                    high_corner_local,
                    self._Grid.dx,
                    self._Grid.n_halo,
                    self._particle_varnames,
                    self._particle_data,
                    self.inject_height,
                    self._n,
                )

                self.injected = True

        # Inject new particles
        if self.inject_type == "point":
            if (
                self._TimeSteppingController.time >= self.point_time_range[0]
                and self._TimeSteppingController.time <= self.point_time_range[1]
            ):

                point = tuple(self.point_location)
                plume_on_rank = self._Grid.point_on_rank(point[0], point[1], point[2])
                if plume_on_rank:
                    ih = self._Grid.point_indicies(point[0], point[1], point[2])

                    indicies = (ih[0] - n_halo[0], ih[1] - n_halo[1], ih[2] - n_halo[2])

                    self.point_inject(
                        low_corner_local,
                        high_corner_local,
                        local_shape,
                        self._Grid.dx,
                        self._n,
                        self._TimeSteppingController.dt,
                        self._particle_varnames,
                        self._particle_data,
                        indicies,
                        self.n_inject,
                    )

        # Interpolate velocities onto the particles
        self.interpolate_pt(
            local_shape,
            self._Grid.n_halo,
            low_corner_local,
            self._Grid.dx,
            self._n,
            self._particle_varnames,
            self._particle_data,
            u,
            "u",
            0,
        )
        self.interpolate_pt(
            local_shape,
            self._Grid.n_halo,
            low_corner_local,
            self._Grid.dx,
            self._n,
            self._particle_varnames,
            self._particle_data,
            v,
            "v",
            1,
        )
        self.interpolate_pt(
            local_shape,
            self._Grid.n_halo,
            low_corner_local,
            self._Grid.dx,
            self._n,
            self._particle_varnames,
            self._particle_data,
            w,
            "w",
            2,
        )

        return

    @staticmethod
    @numba.njit()
    def compute_new_position(
        local_shape, tke_sgs, n, particle_varnames, particle_data, dt
    ):

        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]

        udof = particle_varnames["u"]
        vdof = particle_varnames["v"]
        wdof = particle_varnames["w"]

        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                arr[xdof, p] += (arr[udof, p]) * dt + np.random.normal(
                                    loc=0.0, scale=0.67 * np.sqrt(tke_sgs[i, j, k])
                                ) * dt
                                arr[ydof, p] += (arr[vdof, p]) * dt + np.random.normal(
                                    loc=0.0, scale=0.67 * np.sqrt(tke_sgs[i, j, k])
                                ) * dt
                                arr[zdof, p] += (arr[wdof, p]) * dt + np.random.normal(
                                    loc=0.0, scale=0.67 * np.sqrt(tke_sgs[i, j, k])
                                ) * dt

        return

    @staticmethod
    @numba.njit()
    def _surface_bounce(local_shape, n, particle_varnames, particle_data):
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                arr[zdof, p] = np.abs(arr[zdof, p])
        return

    @staticmethod
    @numba.njit()
    def _boundary_periodic_serial(
        local_shape,
        high_corner,
        n,
        particle_varnames,
        particle_data,
        mpi_size_x,
        mpi_size_y,
    ):
        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if mpi_size_x == 1:
                                    arr[xdof, p] = arr[xdof, p] % high_corner[0]
                                else:
                                    arr[xdof, p] = arr[xdof, p]
                                if mpi_size_y == 1:
                                    arr[ydof, p] = arr[ydof, p] % high_corner[1]
                                else:
                                    arr[ydof, p] = arr[ydof, p]
        return

    @staticmethod
    @numba.njit()
    def _boundary_exit_serial(
        local_shape, high_corner, n, particle_varnames, particle_data
    ):
        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if (
                                    arr[xdof, p] >= high_corner[0]
                                    or arr[xdof, p] <= 0
                                    or arr[ydof, p] >= high_corner[1]
                                    or arr[ydof, p] <= 0
                                ):
                                    arr[valid, p] = 0.0
                                    n[i, j, k] -= 1

        return

    @staticmethod
    @numba.njit()
    def _find_x_crossing_particles(
        local_shape,
        low_corner_local,
        high_corner_local,
        n,
        particle_varnames,
        particle_data,
        n_send,
    ):
        xdof = particle_varnames["x"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[xdof, p] >= high_corner_local[0]:
                                    n_send[i, j, k, 0] += 1
                                elif arr[xdof, p] < low_corner_local[0]:
                                    n_send[i, j, k, 1] += 1
        return

    @staticmethod
    @numba.njit()
    def _pack_x_buffers(
        local_shape,
        low_corner_local,
        high_corner_local,
        n,
        particle_varnames,
        particle_data,
        n_send,
        send_buffer_left,
        send_buffer_right,
    ):
        xdof = particle_varnames["x"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        n_left_count = 0
        n_right_count = 0
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n_send[i, j, k, 1] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[xdof, p] <= low_corner_local[0]:
                                    for di in range(arr.shape[0]):
                                        send_buffer_left[di, n_left_count] = arr[di, p]
                                    arr[valid, p] = 0.0
                                    n[i, j, k] -= 1
                                    n_left_count += 1

                    elif n_send[i, j, k, 0] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[xdof, p] > high_corner_local[0]:
                                    for di in range(arr.shape[0]):
                                        send_buffer_right[di, n_right_count] = arr[
                                            di, p
                                        ]
                                    arr[valid, p] = 0.0
                                    n[i, j, k] -= 1
                                    n_right_count += 1

        return

    @staticmethod
    @numba.njit()
    def unpack_buffer(
        low_corner_local,
        high_corner_local,
        high_corner_global,
        dx,
        n,
        n_buffer,
        recv_buffer,
        particle_varnames,
        particle_data,
        dir,
    ):

        valid_dof = particle_varnames["valid"]
        x_dof = particle_varnames["x"]
        y_dof = particle_varnames["y"]
        z_dof = particle_varnames["z"]

        shape = n.shape
        ishift = shape[1] * shape[2]
        jshift = shape[2]

        for bi in range(recv_buffer.shape[1]):

            # Here we reset the
            if dir == "x":
                recv_buffer[x_dof, bi] = recv_buffer[x_dof, bi] % high_corner_global[0]
            elif dir == "y":
                recv_buffer[y_dof, bi] = recv_buffer[y_dof, bi] % high_corner_global[1]

            i = min(
                max(int((recv_buffer[x_dof, bi] - low_corner_local[0]) // dx[0]), 0),
                shape[0] - 1,
            )
            j = min(
                max(int((recv_buffer[y_dof, bi] - low_corner_local[1]) // dx[1]), 0),
                shape[1] - 1,
            )
            k = int((recv_buffer[z_dof, bi] - low_corner_local[2]) // dx[2])

            arr = particle_data[i * ishift + j * jshift + k]

            fits = False
            for ni in range(arr.shape[1]):
                if arr[valid_dof, ni] == 0.0:
                    for d in range(recv_buffer.shape[0]):
                        arr[d, ni] = recv_buffer[d, bi]
                        fits = True
                    n[i, j, k] += 1
                    break

            if not fits:
                ni = arr.shape[1]
                new_buf = np.zeros((arr.shape[0], n_buffer), dtype=np.double)
                particle_data[i * ishift + j * jshift + k] = np.concatenate(
                    (arr, new_buf), axis=1
                )
                arr = particle_data[i * ishift + j * jshift + k]
                for d in range(recv_buffer.shape[0]):
                    arr[d, ni] = recv_buffer[d, bi]

                n[i, j, k] += 1

        return

    @staticmethod
    @numba.njit()
    def _pack_y_buffers(
        local_shape,
        low_corner_local,
        high_corner_local,
        n,
        particle_varnames,
        particle_data,
        n_send,
        send_buffer_left,
        send_buffer_right,
    ):
        ydof = particle_varnames["y"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        n_left_count = 0
        n_right_count = 0
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n_send[i, j, k, 1] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[ydof, p] <= low_corner_local[1]:
                                    for di in range(arr.shape[0]):
                                        send_buffer_left[di, n_left_count] = arr[di, p]
                                    arr[valid, p] = 0.0
                                    n[i, j, k] -= 1
                                    n_left_count += 1
                    elif n_send[i, j, k, 0] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[ydof, p] > high_corner_local[1]:
                                    for di in range(arr.shape[0]):
                                        send_buffer_right[di, n_right_count] = arr[
                                            di, p
                                        ]
                                    arr[valid, p] = 0.0
                                    n[i, j, k] -= 1
                                    n_right_count += 1

        return

    @staticmethod
    @numba.njit()
    def _find_y_crossing_particles(
        local_shape,
        low_corner_local,
        high_corner_local,
        n,
        particle_varnames,
        particle_data,
        n_send,
    ):
        ydof = particle_varnames["y"]
        valid = particle_varnames["valid"]
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                if arr[ydof, p] >= high_corner_local[1]:
                                    n_send[i, j, k, 0] += 1
                                elif arr[ydof, p] < low_corner_local[1]:
                                    n_send[i, j, k, 1] += 1
        return

    @staticmethod
    @numba.njit()
    def interpolate_pt(
        local_shape,
        n_halo,
        low_corner_local,
        dx,
        n,
        particle_varnames,
        particle_data,
        u,
        var,
        loc,
    ):

        # Compute points
        if loc == 0:
            xpos_shift = 0.0
            ypos_shift = 0.5 * dx[1]
            zpos_shift = 0.5 * dx[2]
        elif loc == 1:
            xpos_shift = 0.5 * dx[0]
            ypos_shift = 0.0
            zpos_shift = 0.5 * dx[2]
        elif loc == 2:
            xpos_shift = 0.5 * dx[0]
            ypos_shift = 0.5 * dx[1]
            zpos_shift = 0.0
        else:
            xpos_shift = 0.0
            ypos_shift = 0.0
            zpos_shift = 0.0

        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]
        var_dof = particle_varnames[var]

        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid, p] != 0.0:
                                xl = arr[xdof, p] - low_corner_local[0] - xpos_shift
                                yl = arr[ydof, p] - low_corner_local[1] - ypos_shift
                                zl = arr[zdof, p] - low_corner_local[2] - zpos_shift

                                ix = int(xl // dx[0]) - 1 + n_halo[0]
                                iy = int(yl // dx[1]) - 1 + n_halo[1]
                                iz = int(zl // dx[2]) - 1 + n_halo[2]

                                xd = (xl % dx[0]) / dx[0]
                                yd = (yl % dx[1]) / dx[1]
                                zd = (zl % dx[2]) / dx[2]

                                c00 = (1.0 - xd) * u[ix, iy, iz] + xd * u[
                                    ix + 1, iy, iz
                                ]
                                c01 = (1.0 - xd) * u[ix, iy, iz + 1] + xd * u[
                                    ix + 1, iy, iz + 1
                                ]
                                c10 = (1.0 - xd) * u[ix, iy + 1, iz] + xd * u[
                                    ix + 1, iy + 1, iz
                                ]
                                c11 = (1.0 - xd) * u[ix, iy + 1, iz + 1] + xd * u[
                                    ix + 1, iy + 1, iz + 1
                                ]

                                c0 = c00 * (1.0 - yd) + c10 * yd
                                c1 = c01 * (1.0 - yd) + c11 * yd

                                arr[var_dof, p] = c0 * (1.0 - zd) + c1 * zd

        return

    @staticmethod
    @numba.njit()
    def move_particles_on_grid(
        low_corner_local, local_shape, dx, n_buffer, n, particle_varnames, particle_data
    ):

        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]

        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]

        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i, j, k] > 0:
                        # if n[i,j,k] > 0:
                        #    print(i,j,k, n[i,j,k],np.max(n))
                        # if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for pi in range(arr.shape[1]):
                            if arr[valid, pi] != 0.0:

                                # Compute the location of  the particle on the grid
                                inew = int(
                                    (arr[xdof, pi] - low_corner_local[0]) // dx[0]
                                )
                                jnew = int(
                                    (arr[ydof, pi] - low_corner_local[1]) // dx[1]
                                )
                                knew = int(
                                    (arr[zdof, pi] - low_corner_local[2]) // dx[2]
                                )

                                # If the new grid cell is different than the old, move it!
                                if i != inew or j != jnew or k != knew:

                                    # Get the particle array at the new grid point
                                    arr_new = particle_data[
                                        inew * ishift + jnew * jshift + knew
                                    ]

                                    # Bool to hold if the moving aprticle can fit on the array at the new point
                                    fits = False

                                    # Loop over all points in the new array
                                    for pi_new in range(arr_new.shape[1]):

                                        # Find if there is a point that is currently empty
                                        if arr_new[valid, pi_new] == 0.0:
                                            # If we are here we found an empty location

                                            # Loop over all degress of freedom
                                            for d in range(arr.shape[0]):
                                                # print('Moving Particle')
                                                # Copy all degrees of freedom into the new array
                                                arr_new[d, pi_new] = arr[d, pi]

                                            # If we made it here, it must fit.
                                            fits = True
                                            break
                                    if not fits:
                                        # Get the index in the resized array
                                        n_arr = arr_new.shape[1]

                                        # Create a new buffer
                                        new_buf = np.zeros(
                                            (arr_new.shape[0], n_buffer),
                                            dtype=np.double,
                                        )

                                        # Write the particle data
                                        particle_data[
                                            inew * ishift + jnew * jshift + knew
                                        ] = np.concatenate((arr_new, new_buf), axis=1)
                                        arr_new = particle_data[
                                            inew * ishift + jnew * jshift + knew
                                        ]
                                        for d in range(arr.shape[0]):
                                            arr_new[d, n_arr] = arr[d, pi]

                                    n[i, j, k] -= 1
                                    n[inew, jnew, knew] += 1

                                    arr[valid, pi] = 0.0
        # print(imin, imax, jmin, jmax, kmin, kmax, local_shape)
        return

    @staticmethod
    @numba.jit()
    def distill_dof(particle_varnames, n, var, particle_data, data):
        dof = particle_varnames[var]
        valid = particle_varnames["valid"]
        count = 0

        for arr in particle_data:
            for p in range(arr.shape[1]):
                if arr[valid, p] != 0.0:
                    data[count] = arr[dof, p]
                    count += 1

        return

    @staticmethod
    @numba.njit()
    def particle_sum_on_grid(particle_varnames, n, var, particle_data, data):

        shape = data.shape

        dof = particle_varnames[var]
        valid = particle_varnames["valid"]

        ishift = shape[1] * shape[2]
        jshift = shape[2]

        for i in range(shape[0]):
            ii = i * ishift
            for j in range(shape[1]):
                jj = j * jshift
                for k in range(shape[2]):
                    arr = particle_data[ii + jj + k]
                    data[i, j, k] = 0
                    for pi in range(arr.shape[1]):
                        if arr[valid, pi] != 0.0:
                            data[i, j, k] += arr[dof, pi]

    def add_particle_variable(self, name, interp=False):
        # Assert if variable names are duplicated
        assert name not in self._interp_particle_varnames
        assert name not in self._nointerp_particle_varnames

        if interp:
            self._interp_particle_varnames[name] = self._interp_particle_dofs
            self._interp_particle_dofs += 1
        else:
            self._nointerp_particle_varnames[name] = self._nointerp_particle_dofs
            self._nointerp_particle_dofs += 1

        self._particle_varnames[name] = self._particle_dofs

        self._particle_dofs += 1
        return

    @staticmethod
    @numba.njit()
    def _fulfill_minimum_particles(n, n_minimum, particle_varnames, particle_data):

        shape = n.shape

        xdof = particle_varnames["x"]
        ydof = particle_varnames["y"]
        zdof = particle_varnames["z"]
        valid = particle_varnames["valid"]
        npart_dof = particle_varnames["n_particles"]

        ishift = shape[1] * shape[2]
        jshift = shape[2]

        for i in range(shape[0]):
            ii = i * ishift
            for j in range(shape[1]):
                jj = j * jshift
                for k in range(shape[2]):
                    if n[i, j, k] >= 1 and n[i, j, k] < n_minimum:
                        arr = particle_data[ii + jj + k]

                        # Loop over all particles that are valid and determine their locations
                        x_pos_mean = 0.0
                        y_pos_mean = 0.0
                        z_pos_mean = 0.0
                        npart_mean = 0.0
                        mean_count = 0.0
                        for pi in range(arr.shape[1]):
                            if arr[valid, pi] != 0.0:
                                x_pos_mean += arr[xdof, pi]
                                y_pos_mean += arr[ydof, pi]
                                z_pos_mean += arr[zdof, pi]
                                npart_mean += arr[npart_dof, pi]
                                mean_count += 1.0

                        # print(i,j,k,
                        #    mean_count, n[i,j,k], arr[valid,:],
                        #    x_pos_mean, y_pos_mean, z_pos_mean)

                        n_old = n[i, j, k]
                        new_points = int(n_minimum - n[i, j, k])
                        new_added = 0
                        # Nucleate
                        for pi in range(arr.shape[1]):
                            if arr[valid, pi] == 0.0 and new_points < new_added:
                                arr[xdof, pi] = x_pos_mean / mean_count
                                arr[ydof, pi] = y_pos_mean / mean_count
                                arr[zdof, pi] = z_pos_mean / mean_count
                                arr[npart_dof, pi] = npart_mean / n_minimum
                                arr[valid, pi] = 1.0

                                new_added += 1
                                n[i, j, k] += 1.0
                            elif arr[valid, pi] != 0.0:
                                arr[npart_dof, pi] = (
                                    npart_mean / n_minimum
                                )  # Set the mean value to the

                            # if new_added == new_points:
                            #    break

        return

    def fulfill_minimum_particles(self):

        self._fulfill_minimum_particles(
            self._n,
            self._minimum_particles,
            self._particle_varnames,
            self._particle_data,
        )

        return


class ParticlesSimple(ParticlesBase):
    def __init__(
        self,
        namelist,
        Grid,
        Ref,
        TimeSteppingController,
        VelocityState,
        ScalarState,
        DiagnsoticState,
    ):

        ParticlesBase.__init__(
            self,
            namelist,
            Grid,
            Ref,
            TimeSteppingController,
            VelocityState,
            ScalarState,
            DiagnsoticState,
        )

        return

    def update(self):

        self.update_position()

        self.fulfill_minimum_particles()

        # print(np.sum(self._n))

        n_halo = self._Grid.n_halo
        n_lagrangian = self._DiagnosticState.get_field("n_lagrangian")
        n_lagrangian[
            n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1], n_halo[2] : -n_halo[2]
        ] = self._n[:, :, :]

        n_particles = self._DiagnosticState.get_field("n_particles")
        self.particle_sum_on_grid(
            self._particle_varnames,
            self._n,
            "n_particles",
            self._particle_data,
            n_particles[
                n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1], n_halo[2] : -n_halo[2]
            ],
        )

        # Missing a factor of density here
        n_particles[:, :, :] = (
            n_particles[:, :, :]
            / (self._Grid.dx[0] * self._Grid.dx[1] * self._Grid.dx[2])
            / self._Ref.rho0[np.newaxis, np.newaxis, :]
        )

        return

    def output(self):
        self.particle_output()
        return

    def particle_output(self):

        path = os.path.join(
            self._output_path,
            str(10000000000 + np.round(self._TimeSteppingController.time, 2)) + ".h5",
        )

        n_on_ranks = MPI.COMM_WORLD.allgather(np.sum(self._n))
        n_total = np.sum(n_on_ranks)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\t Number of particles: ", n_total)
        if n_total == 0:
            return

        my_rank = MPI.COMM_WORLD.Get_rank()
        my_start = 0
        if my_rank != 0:
            my_start = np.sum(n_on_ranks[:my_rank])

        my_end = my_start + n_on_ranks[my_rank]

        local_shape = self._Grid.local_shape
        low_corner_local = (
            self._Grid.x_range_local[0],
            self._Grid.y_range_local[0],
            self._Grid.z_range_local[0],
        )

        fx = h5py.File(
            path,
            "w",
            driver="mpio",
            comm=MPI.COMM_WORLD,
        )

        fx.attrs["time"] = self._TimeSteppingController.time

        for v in ["id", "x", "y", "z", "u", "v", "w", "n_particles"]:

            dset = fx.create_dataset(
                v,
                (n_total,),
                dtype="d",
            )

            var_local = np.empty((np.sum(self._n),), dtype=np.double)
            self.distill_dof(
                self._particle_varnames, self._n, v, self._particle_data, var_local
            )
            dset[my_start:my_end] = var_local

        # Write prognostic sclaar fields
        for v in self._ScalarState.dofs:
            dset = fx.create_dataset(
                v,
                (n_total,),
                dtype="d",
            )
            try:
                vf = self._ScalarState.get_field(v)
            except:
                continue
            self.interpolate_pt(
                local_shape,
                self._Grid.n_halo,
                low_corner_local,
                self._Grid.dx,
                self._n,
                self._particle_varnames,
                self._particle_data,
                vf,
                "dummy",
                0,
            )
            var_local = np.empty((np.sum(self._n),), dtype=np.double)
            self.distill_dof(
                self._particle_varnames,
                self._n,
                "dummy",
                self._particle_data,
                var_local,
            )
            dset[my_start:my_end] = var_local

        for v in ["T", "buoyancy", "horizontal divergence"]:
            dset = fx.create_dataset(
                v,
                (n_total,),
                dtype="d",
            )
            try:
                vf = self._DiagnosticState.get_field(v)
            except:
                continue
            self.interpolate_pt(
                local_shape,
                self._Grid.n_halo,
                low_corner_local,
                self._Grid.dx,
                self._n,
                self._particle_varnames,
                self._particle_data,
                vf,
                "dummy",
                0,
            )
            var_local = np.empty((np.sum(self._n),), dtype=np.double)
            self.distill_dof(
                self._particle_varnames,
                self._n,
                "dummy",
                self._particle_data,
                var_local,
            )
            dset[my_start:my_end] = var_local

        fx.close()

        return
