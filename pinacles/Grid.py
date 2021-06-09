from mpi4py import MPI
import mpi4py_fft
import numpy as np
from mpi4py_fft.pencil import Subcomm


class GridBase:
    def __init__(self, namelist):

        # List of class atributes that will be restarted
        self._restart_attributes = []

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

        # Lenght of each CRM domain side
        self._l = np.array(namelist["grid"]["l"], dtype=np.double)
        self._restart_attributes.append("_l")
        assert len(self._l) == 3

        # The global x,y,z coordiantes
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

        # Store sub-cummunicators created by mpi4py_fft
        self.subcomms = None
        self._create_subcomms()

        # Get local grid information
        self._get_local_grid_indicies()

        self._compute_index_bounds()

        return

    def _compute_index_bounds(self):

        """ compute the local indicies of the non-halo grid points. For the case of vairables defined on 
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

    def restart(self):

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
        This corresponds to namelist['grid']['n'] in the input namelsit.

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
        # TODO replace all instanes of local_shape with nl
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
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[0])

    @property
    def x_edge_global(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[0])

    @property
    def y_global(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[1])

    @property
    def y_edge_global(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[1])

    @property
    def z_global(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes[2])

    @property
    def z_edge_global(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._global_axes_edge[2])

    @property
    def global_axes(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return self._global_axes.copy()

    @property
    def x_local(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[0]
        end = self._local_end[0] + 2 * self._n_halo[0]
        return np.copy(self._global_axes[0][start:end])

    @property
    def y_local(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[1]
        end = self._local_end[1] + 2 * self._n_halo[1]
        return np.copy(self._global_axes[1][start:end])

    @property
    def z_local(self):
        """ Copy here is forced to keep _global_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        start = self._local_start[2]
        end = self._local_end[2] + 2 * self._n_halo[2]
        return np.copy(self._global_axes[2][start:end])

    @property
    def local_axes(self):
        """ Copy here is forced to keep _local_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
        view so that copy occurs.
        """
        return np.copy(self._local_axes)

    @property
    def local_axes_edge(self):
        """
        Copy here is forced to keep _local_axes externally immutable,
        if performace becomes an issue we can provide a property that return a
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
            self._high_rank.append(self._subcomm_size[-1]-1 == self._subcomm_rank[-1])

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

        return


class RegularCartesian(GridBase):
    def __init__(self, namelist):

        GridBase.__init__(self, namelist)
        self._compute_globalcoordiantes()

        return

    def _compute_globalcoordiantes(self):

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
            self._global_axes.append(np.linspace(lx, ux, self.ngrid[i]))
            self._global_axes_edge.append(self._global_axes[i] + 0.5 * dx)

            # Compute the local axes form the global axes
            start = self._local_start[i]
            end = self._local_end[i] + 2 * self._n_halo[i]
            self._local_axes.append(self._global_axes[i][start:end])
            self._local_axes_edge.append(self._global_axes_edge[i][start:end])

        self._dx = np.array(dx_list)
        self._dxi = 1.0 / self._dx
        return

    def restart(self, data_dict):
        """ 
        Here we just do checks for domain decomposition consistency with the namelist file
        # currently, we require that a restarted simulation have exactly the same domain 
        # as the simulation from which it is being restarted.
        """

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
