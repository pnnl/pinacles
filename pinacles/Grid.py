from mpi4py import MPI
import mpi4py_fft
import numpy as np
from numba import stencil
from mpi4py_fft.pencil import Subcomm
from pinacles import ProjectionLCC
import numba


class GridBase:
    def __init__(self, namelist, llx, lly, llz):

        self._namelist = namelist

        # List of class attributes that will be restarted
        self._restart_attributes = []

        self._ll_corner = (llx, lly, llz)

        # print(self._ll_corner)
        # The total number of points in the domain NOT including halo/ghost points
        self._n = np.array(namelist["grid"]["n"], dtype=np.int)
        self._restart_attributes.append("_n")
        assert len(self._n) == 3

        # The number of halo points
        self._n_halo = np.array(namelist["grid"]["n_halo"], dtype=np.int)
        self._restart_attributes.append("_n_halo")
        assert len(self._n_halo) == 3

        # The total number of points in the CRM domain including halo/ghost points
        self._ngrid = self._n + 2 * self._n_halo
        self._restart_attributes.append("_ngrid")

        self._ngrid_local = None
        self._restart_attributes.append("_ngrid_local")

        self._local_shape = None
        self._restart_attributes.append("_local_shape")

        self._local_start = None
        self._restart_attributes.append("_local_start")

        self._local_end = None
        self._restart_attributes.append("_local_end")

        # Length of each CRM domain side
        self._l = np.array(namelist["grid"]["l"], dtype=np.double)
        self._restart_attributes.append("_l")
        assert len(self._l) == 3

        # The global x,y,z coordinates
        self._global_axes = None
        self._global_axes_edge = None
        self._local_axes = None
        self._local_axes_edge = None

        self._restart_attributes.append("_global_axes")
        self._restart_attributes.append("_global_axes_edge")
        self._restart_attributes.append("_local_axes")
        self._restart_attributes.append("_local_axes_edge")

        # Set the grid spacing
        self._dx = None
        self._restart_attributes.append("_dx")
        self._dxi = None
        self._restart_attributes.append("_dxi")

        # Store sub-communicators created by mpi4py_fft
        self.subcomms = None
        self._create_subcomms()

        # Get local grid information
        self._get_local_grid_indicies()

        self._compute_index_bounds()



        return

    def _compute_index_bounds(self):

        """compute the local indicies of the non-halo grid points. For the case of vairables defined on
        the grid edges, the indicies include the variabiles on the domain boundary.
        """

        # Low-index for edge arrays
        self._ibl_edge = tuple(np.array(self.n_halo) - 1)

        # High-index for edge array
        self._ibu_edge = tuple(np.array(self._ibl_edge) + np.array(self.local_shape))

        # Low-index for cell-center arrays
        self._ibl = tuple(self.n_halo)

        # High index for cell-center arrays
        self._ibu = tuple(np.array(self._ibl) + np.array(self.local_shape) - 1)
        return

    def restart(self, data_dict, **kwargs):

        return

    def dump_restart(self, data_dict):

        return

    @property
    def subcomm_rank(self):
        return self._subcomm_rank

    @property
    def subcomm_size(self):
        return self._subcomm_size

    @property
    def low_rank(self):
        return self._low_rank

    @property
    def high_rank(self):
        return self._high_rank

    @property
    def ibl(self):
        return self._ibl

    @property
    def ibu(self):
        return self._ibu

    @property
    def ibl_edge(self):
        return self._ibl_edge

    @property
    def ibu_edge(self):
        return self._ibu_edge

    @property
    def n(self):
        return self._n
        """ Returns the number of points in the domain w/o halos. 
        This corresponds to namelist['grid']['n'] in the input namelist.

        Returns: 
            n: float ndarray of shape (3,)
        """

    @property
    def n_halo(self):
        """
        Returns an array containing the number of halo points in
        each of the coordinate directions.

        Returns:
            n_halo : float ndarray of shape (3,)
        """
        return self._n_halo

    @property
    def l(self):
        """The length of the LES domain in each of the coordinate
        directions.

        Returns:
            l : float ndarray of shape (3,)
        """
        return self._l

    @property
    def ngrid(self):
        """Returns the total number of points in the global
        domain including ghost points.

        Returns:
            ngrid : float ndarray of shape (3,)
        """
        return self._ngrid

    @property
    def ngrid_local(self):
        return self._ngrid_local

    @property
    def local_shape(self):
        return self._local_shape

    @property
    def nl(self):
        # TODO replace all instances of local_shape with nl
        return self._local_shape

    @property
    def local_start(self):
        return self._local_start

    @property
    def local_end(self):
        return self._local_end

    @property
    def dx(self):
        return self._dx

    @property
    def dxi(self):
        return self._dxi

    @property
    def x_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[0])

    @property
    def x_edge_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[0])

    @property
    def y_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[1])

    @property
    def y_edge_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[1])

    @property
    def z_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[2])

    @property
    def z_edge_global(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[2])

    @property
    def global_axes(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return self._global_axes.copy()

    @property
    def x_local(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[0]
        end = self._local_end[0] + 2 * self._n_halo[0]
        return np.copy(self._global_axes[0][start:end])

    @property
    def y_local(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[1]
        end = self._local_end[1] + 2 * self._n_halo[1]
        return np.copy(self._global_axes[1][start:end])

    @property
    def x_edge_local(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[0]
        end = self._local_end[0] + 2 * self._n_halo[0]
        return np.copy(self._global_axes_edge[0][start:end])

    @property
    def y_edge_local(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[1]
        end = self._local_end[1] + 2 * self._n_halo[1]
        return np.copy(self._global_axes_edge[1][start:end])

    @property
    def z_local(self):
        """Copy here is forced to keep _global_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[2]
        end = self._local_end[2] + 2 * self._n_halo[2]
        return np.copy(self._local_axes[2][start:end])

    @property
    def local_axes(self):
        """Copy here is forced to keep _local_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._local_axes)

    @property
    def local_axes_edge(self):
        """
        Copy here is forced to keep _local_axes externally immutable,
        if performance becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._local_axes_edge)

    @property
    def x_range(self):
        return (
            self._global_axes_edge[0][self._n_halo[0] - 1],
            self._global_axes_edge[0][-self._n_halo[0] - 1],
        )

    @property
    def y_range(self):
        return (
            self._global_axes_edge[1][self._n_halo[1] - 1],
            self._global_axes_edge[1][-self._n_halo[1] - 1],
        )

    @property
    def z_range(self):
        return (
            self._global_axes_edge[2][self._n_halo[2] - 1],
            self._global_axes_edge[2][-self._n_halo[2] - 1],
        )

    @property
    def x_range_local(self):
        return (
            self._local_axes_edge[0][self._n_halo[0] - 1],
            self._local_axes_edge[0][-self._n_halo[0] - 1],
        )

    @property
    def y_range_local(self):
        return (
            self._local_axes_edge[1][self._n_halo[1] - 1],
            self._local_axes_edge[1][-self._n_halo[1] - 1],
        )

    @property
    def z_range_local(self):
        return (
            self._local_axes_edge[2][self._n_halo[2] - 1],
            self._local_axes_edge[2][-self._n_halo[2] - 1],
        )

    def get_DistArray(self):
        return mpi4py_fft.DistArray(self._n, self.subcomms)

    def _create_subcomms(self):
        self.subcomms = Subcomm(MPI.COMM_WORLD, dims=[0, 0, 1])

        self._subcomm_size = []
        self._subcomm_rank = []

        self._low_rank = []
        self._high_rank = []

        for i, comm in enumerate(self.subcomms):
            self._subcomm_size.append(comm.Get_size())
            self._subcomm_rank.append(comm.Get_rank())
            self._low_rank.append(self._subcomm_rank[-1] == 0)
            self._high_rank.append(self._subcomm_size[-1] - 1 == self._subcomm_rank[-1])

        self._subcomm_rank = tuple(self._subcomm_size)
        self._subcomm_size = tuple(self._subcomm_size)
        self._low_rank = tuple(self._low_rank)
        self._high_rank = tuple(self._high_rank)

        return

    def _get_local_grid_indicies(self):

        # Create a dummy array using the given subcomms and the
        # global domain size
        dum_array = mpi4py_fft.DistArray(self._n, self.subcomms)

        # Get shape of dum_array
        self._local_shape = np.array(dum_array.shape)

        # Get the starting index and ending of this ranks part of the dist array
        # these are global indicies
        self._local_start = np.array(dum_array.substart)
        self._local_end = self._local_start + self._local_shape

        # Get the local shape
        self._ngrid_local = self._local_shape + 2 * self._n_halo

        self.rank_nx = np.array(MPI.COMM_WORLD.allgather(self._local_shape[0]))
        self.rank_ny = np.array(MPI.COMM_WORLD.allgather(self._local_shape[1]))

        return

    @stencil
    def upt_to_vpt(u):
        return 0.25 * (u[0, 0, 0] + u[1, 0, 0] + u[0, -1, 0] + u[1, -1, 0])

    @stencil
    def vpt_to_upt(v):
        return 0.25 * (v[0, 0, 0] + v[0, 1, 0] + v[-1, 0, 0] + v[-1, 1, 0])


class RegularCartesian(GridBase):
    def __init__(self, namelist, llx=0.0, lly=0.0, llz=0.0):

        GridBase.__init__(self, namelist, llx=llx, lly=lly, llz=llz)

        self._compute_globalcoordinates()

        self.lat_lon = False

        if "center_latlon" in namelist["grid"]:
            self._center_latlon = tuple(namelist["grid"]["center_latlon"])
            self._conic_intersection = tuple(namelist["grid"]["conic_intersection"])

            self.MapProj = ProjectionLCC.LambertConformal(
                6.3781e6,
                self._conic_intersection[0],
                self._conic_intersection[1],
                self._center_latlon[0],
                self._center_latlon[1],
            )

            self.compute_latlon()
            self.lat_lon = True




        nh = self.n_halo

        x_dist_edge = np.minimum(self.x_local[nh[0]:-nh[0]],np.abs(self.x_local[nh[0]:-nh[0]] - self.l[0]))
        y_dist_edge = np.minimum(self.y_local[nh[1]:-nh[1]],np.abs(self.y_local[nh[1]:-nh[1]] - self.l[1]))  
        
        self._edge_dist = np.minimum(x_dist_edge[:,np.newaxis], y_dist_edge[np.newaxis,:])

        x_dist_edge = np.minimum(self.x_edge_local[nh[0]:-nh[0]],np.abs(self.x_edge_local[nh[0]:-nh[0]] - self.l[0]))
        y_dist_edge = np.minimum(self.y_local[nh[1]:-nh[1]],np.abs(self.y_local[nh[1]:-nh[1]] - self.l[1]))  

        self._edge_dist_u = np.minimum(x_dist_edge[:,np.newaxis], y_dist_edge[np.newaxis,:])

        x_dist_edge = np.minimum(self.x_local[nh[0]:-nh[0]],np.abs(self.x_local[nh[0]:-nh[0]] - self.l[0]))
        y_dist_edge = np.minimum(self.y_edge_local[nh[1]:-nh[1]],np.abs(self.y_edge_local[nh[1]:-nh[1]] - self.l[1]))  
        
        self._edge_dist_v = np.minimum(x_dist_edge[:,np.newaxis], y_dist_edge[np.newaxis,:])
 
 
#       import sys; sys.exit()


        #start =self._ibu_edge[0]-5 #-nh[1] - self.nudge_width - 1
        #end = self._ibu_edge[0] + 1 #-nh[1] - 1
        
        #print( self.x_edge_local)
        #print(start, end, self.x_edge_local[start:end])
  
        #end = self._ibu[0]
        
        #start = end -  5
        #print(start, end, self.x_local[start:end])
        #import sys; sys.exit()
        #return
       # print(self._global_axes_edge[0]); import sys; sys.exit()
        
    def compute_latlon(self):


        # Compute domain half width
        halfwidth = None
        if 'parent_grid' not in self._namelist:
            halfwidth = (self.l[0] / 2.0, self.l[1] / 2.0)
        else:
            lp = self._namelist['parent_grid']['l']
            halfwidth = (lp[0] / 2.0, lp[1] / 2.0)

        local_axis = self._local_axes
        local_axis_edge = self._local_axes_edge

        self.halfwidth = halfwidth 

        #print(local_axis[0])
        #print(local_axis_edge[0])
        #import sys; sys.exit()

        x_global_mesh, y_global_mesh = np.meshgrid(
            self.x_global[:] - halfwidth[0],
            self.y_global[:] - halfwidth[1],
            indexing="ij",
        )

        x_local_mesh, y_local_mesh = np.meshgrid(
            local_axis[0] - halfwidth[0], local_axis[1] - halfwidth[1], indexing="ij"
        )

        # print(np.shape(x_local_mesh), np.shape(local_axis[0]))
        # import sys; sys.exit()

        # Longitude-Latitude at the u point
        x_local_mesh_edge_x, y_local_mesh_edge_x = np.meshgrid(
            local_axis_edge[0] - halfwidth[0],
            local_axis[1] - halfwidth[1],
            indexing="ij",
        )

        # Longitude-Latitude at the v point
        x_local_mesh_edge_y, y_local_mesh_edge_y = np.meshgrid(
            local_axis[0] - halfwidth[0],
            local_axis_edge[1] - halfwidth[1],
            indexing="ij",
        )

        self.lon_global, self.lat_global = self.MapProj.compute_latlon(
            x_global_mesh, y_global_mesh
        )
        self.lon_local, self.lat_local = self.MapProj.compute_latlon(
            x_local_mesh, y_local_mesh
        )
        self.lon_local_edge_x, self.lat_local_edge_x = self.MapProj.compute_latlon(
            x_local_mesh_edge_x, y_local_mesh_edge_x
        )
        self.lon_local_edge_y, self.lat_local_edge_y = self.MapProj.compute_latlon(
            x_local_mesh_edge_y, y_local_mesh_edge_y
        )

        # u = np.ones_like(self.lon_local)
        # v = np.zeros_like(self.lon_local)

        # u = np.ones((3,3,3))

        # uv = self.upt_to_vpt(u)
        # print(uv)

        # urot, vrot = self.MapProj.rotate_wind(self.lon_local, u, v)
        # import pylab as plt
        # plt.quiver(self.lon_local, self.lat_local, urot, vrot)
        # plt.colorbar()
        # plt.show()

        self.lon_max = MPI.COMM_WORLD.allreduce(
            np.max(self.lon_local_edge_x), op=MPI.MAX
        )
        self.lon_min = MPI.COMM_WORLD.allreduce(
            np.min(self.lon_local_edge_x), op=MPI.MIN
        )

        self.lat_max = MPI.COMM_WORLD.allreduce(
            np.max(self.lat_local_edge_y), op=MPI.MAX
        )
        self.lat_min = MPI.COMM_WORLD.allreduce(
            np.min(self.lat_local_edge_y), op=MPI.MIN
        )


        return

    def latlon_to_xy(self, lat, lon):
        x, y = self.MapProj.compute_xy(lat, lon)
        return x + self.halfwidth[0], y + self.halfwidth[1]

    def _compute_globalcoordinates(self):

        self._global_axes = []
        self._global_axes_edge = []

        self._local_axes_edge = []
        self._local_axes = []
        dx_list = []
        for i in range(3):
            dx = self._l[i] / self._n[i]
            dx_list.append(dx)
            # Location of lowest most halo point
            lx = (-self._n_halo[i] + 0.5) * dx

            # Location of upper most halo point
            ux = ((self._n[i] + self._n_halo[i]) - 0.5) * dx

            # Generate an axis based on upper and lower points
            self._global_axes.append(
                np.linspace(lx, ux, self.ngrid[i]) + self._ll_corner[i]
            )
            self._global_axes_edge.append(self._global_axes[i] + 0.5 * dx)

            # Compute the local axes form the global axes
            start = self._local_start[i]
            end = self._local_end[i] + 2 * self._n_halo[i]
            self._local_axes.append(self._global_axes[i][start:end])
            self._local_axes_edge.append(self._global_axes_edge[i][start:end])

        self._dx = np.array(dx_list)
        self._dxi = 1.0 / self._dx
        return

    @property
    def ll_corner(self):
        return self._ll_corner

    def restart(self, data_dict, **kwargs):
        """
        Here we just do checks for domain decomposition consistency with the namelist file
        # currently, we require that a restarted simulation have exactly the same domain
        # as the simulation from which it is being restarted.
        """

        if "restart_type" in data_dict:
            return

        key = "RegularCartesianGrid"
        assert np.array_equal(self._dx, data_dict[key]["_dx"])
        assert np.array_equal(self._dxi, data_dict[key]["_dxi"])

        for i in range(3):
            assert np.array_equal(self._local_axes[i], data_dict[key]["_local_axes"][i])
            assert np.array_equal(
                self._local_axes_edge[i], data_dict[key]["_local_axes_edge"][i]
            )

            assert np.array_equal(
                self._global_axes[i], data_dict[key]["_global_axes"][i]
            )
            assert np.array_equal(
                self._global_axes_edge[i], data_dict[key]["_global_axes_edge"][i]
            )

        for item in [
            "_l",
            "_n",
            "_n_halo",
            "_ngrid",
            "_ngrid_local",
            "_local_shape",
            "_local_start",
            "_local_end",
        ]:
            assert np.all(self.__dict__[item] == data_dict[key][item])

        return

    def dump_restart(self, data_dict):

        # Loop through all attributes storing them
        key = "RegularCartesianGrid"
        data_dict[key] = {}
        for item in self._restart_attributes:
            data_dict[key][item] = self.__dict__[item]

        return

    def point_on_rank(self, x, y, z):

        x_range = self.x_range_local
        y_range = self.y_range_local

        on_rank = False

        if x_range[0] <= x and x_range[1] >= x:
            if y_range[0] <= y and y_range[1] >= y:
                on_rank = True

        return on_rank

    def point_indicies(self, x, y, z):

        # Get the indicies of the x, y, and z points
        x_index = np.argmin(np.abs(self.x_local - x))
        y_index = np.argmin(np.abs(self.y_local - y))
        z_index = np.argmin(np.abs(self.z_local - z))

        return (x_index, y_index, z_index)



    def CreateGather(self, xrange, yrange, x_edge=False, y_edge=False):
        return self._Gather(self, xrange, yrange, x_edge, y_edge)

    class _Gather:
        def __init__(self, ModelGrid, xrange, yrange, x_edge=False, y_edge=False):

            self.xrange = xrange
            self.yrange = yrange

            self.ModelGrid = ModelGrid
            assert type(xrange) is tuple
            assert type(yrange) is tuple

            my_rank = MPI.COMM_WORLD.Get_rank()

            self.xranges_to_get = MPI.COMM_WORLD.allgather(xrange)
            self.yranges_to_get = MPI.COMM_WORLD.allgather(yrange)

            local_starts = MPI.COMM_WORLD.allgather(ModelGrid._local_start)

            self.local_starts = np.array(local_starts)
            local_ends = MPI.COMM_WORLD.allgather(ModelGrid._local_end)
            self.local_ends = np.array(local_ends)

            self.n_to_send = []
            self.x_start = []
            self.x_end = []
            self.y_start = []
            self.y_end = []

            n = len(self.xranges_to_get)
            for i in range(n):
                if x_edge:
                    local_ends[i][0] += 1
                if y_edge:
                    local_ends[i][1] += 1

            self.local_ends = np.array(local_ends)

            # First compute what I need to send

            for i in range(n):

                xi_start = 0
                xi_end = 0

                x = np.arange(local_starts[my_rank][0], local_ends[my_rank][0])
                mask = (x >= self.xranges_to_get[i][0]) & (
                    x < self.xranges_to_get[i][1]
                )
                indicies = np.where(mask)[0]
                nx = len(indicies)
                if nx > 0:
                    xi_start = np.amin(indicies)
                    xi_end = np.amax(indicies) + 1

                yi_start = 0
                yi_end = 0
                y = np.arange(local_starts[my_rank][1], local_ends[my_rank][1])
                mask = (y >= self.yranges_to_get[i][0]) & (
                    y < self.yranges_to_get[i][1]
                )
                indicies = np.where(mask)[0]
                ny = len(indicies)
                if ny > 0:
                    yi_start = np.amin(indicies)
                    yi_end = np.amax(indicies) + 1

                self.n_to_send.append((nx * ny) * ModelGrid.n[2])
                self.x_start.append(xi_start)
                self.x_end.append(xi_end)
                self.y_start.append(yi_start)
                self.y_end.append(yi_end)

            self.n_to_send = np.array(self.n_to_send)
            self.send_size = np.sum(self.n_to_send)
            self.x_start = np.array(self.x_start)
            self.x_end = np.array(self.x_end)
            self.y_start = np.array(self.y_start)
            self.y_end = np.array(self.y_end)

            disp = 0
            self.send_disp = []
            for i in range(self.n_to_send.shape[0]):
                self.send_disp.append(disp)
                disp += self.n_to_send[i]

            n = np.empty((1,), dtype=np.int)
            self.n_to_recv = []
            for i in range(self.n_to_send.shape[0]):
                MPI.COMM_WORLD.Scatter(self.n_to_send, n, root=i)
                self.n_to_recv.append(n[0])

            self.n_to_recv = np.array(self.n_to_recv)

            # Now we scatter the global start
            tmp = np.empty((1,), dtype=np.int)
            self.x_start_recv = []
            self.x_end_recv = []

            self.y_start_recv = []
            self.y_end_recv = []

            for i in range(self.x_start.shape[0]):
                MPI.COMM_WORLD.Scatter(
                    self.x_start + ModelGrid._local_start[0], tmp, root=i
                )
                self.x_start_recv.append(tmp[0])

                MPI.COMM_WORLD.Scatter(
                    self.x_end + ModelGrid._local_start[0], tmp, root=i
                )
                self.x_end_recv.append(tmp[0])

                MPI.COMM_WORLD.Scatter(
                    self.y_start + ModelGrid._local_start[1], tmp, root=i
                )
                self.y_start_recv.append(tmp[0])

                MPI.COMM_WORLD.Scatter(
                    self.y_end + ModelGrid._local_start[1], tmp, root=i
                )
                self.y_end_recv.append(tmp[0])

            self.x_start_recv = np.array(self.x_start_recv)
            self.x_end_recv = np.array(self.x_end_recv)
            self.y_start_recv = np.array(self.y_start_recv)
            self.y_end_recv = np.array(self.y_end_recv)

            disp = 0
            self.recv_disp = []
            for i in range(self.n_to_recv.shape[0]):
                self.recv_disp.append(disp)
                disp += self.n_to_recv[i]

            self.recv_disp = np.array(self.recv_disp)

            return

        @staticmethod
        @numba.njit()
        def pack_send(nh, x_start, x_end, y_start, y_end, var, send_array):

            shape = var.shape
            n_procs = len(x_start)

            count = 0
            for n in range(n_procs):
                for i in range(nh[0] + x_start[n], nh[0] + x_end[n]):
                    for j in range(nh[1] + y_start[n], nh[1] + y_end[n]):
                        for k in range(nh[2], shape[2] - nh[2]):
                            send_array[count] = var[i, j, k]
                            count += 1

            return

        @staticmethod
        @numba.njit()
        def unpack_send(
            n_to_recv,
            xrange_start,
            yrange_start,
            x_start_recv,
            x_end_recv,
            y_start_recv,
            y_end_recv,
            recv_array,
            gathered_array,
        ):

            shape = gathered_array.shape

            count = 0
            for i in range(n_to_recv.shape[0]):

                if n_to_recv[i] > 0:

                    istart = x_start_recv[i] - xrange_start
                    iend = x_end_recv[i] - xrange_start

                    jstart = y_start_recv[i] - yrange_start
                    jend = y_end_recv[i] - yrange_start

                    for ii in range(istart, iend):
                        for j in range(jstart, jend):
                            for k in range(shape[2]):
                                gathered_array[ii, j, k] = recv_array[count]
                                count += 1

            return

        def call(self, var):

            xrange = self.xrange
            yrange = self.yrange

            gathered_array = np.empty(
                (xrange[1] - xrange[0], yrange[1] - yrange[0], self.ModelGrid.n[2]),
                dtype=np.double,
            )

            nh = self.ModelGrid.n_halo
            send_array = np.empty(self.send_size, dtype=np.double)
            recv_array = np.empty(
                (xrange[1] - xrange[0]) * (yrange[1] - yrange[0]) * self.ModelGrid.n[2],
                dtype=np.double,
            )

            self.pack_send(
                nh, self.x_start, self.x_end, self.y_start, self.y_end, var, send_array
            )

            sm = [send_array, (self.n_to_send, self.send_disp), MPI.DOUBLE]
            rm = [recv_array, (self.n_to_recv, self.recv_disp), MPI.DOUBLE]

            MPI.COMM_WORLD.Alltoallv(sm, rm)

            self.unpack_send(
                self.n_to_recv,
                xrange[0],
                yrange[0],
                self.x_start_recv,
                self.x_end_recv,
                self.y_start_recv,
                self.y_end_recv,
                recv_array,
                gathered_array,
            )

            return gathered_array
