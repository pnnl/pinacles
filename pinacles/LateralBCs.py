# This class will be used to implment the various types of open boundary conditions.
import numpy as np
import numba
from pinacles import UtilitiesParallel


def LateralBCsFactory(namelist, Grid, State, VelocityState):
    try:
        lbc = namelist["lbc"]
    except:
        return LateralBCsDummy()

    if lbc["type"].lower() == "periodic":
        return LateralBCsDummy()
    elif lbc["type"].lower() == "open":

        lbc_class = LateralBCs(Grid, State, VelocityState)

        try:
            boundary_treatment = lbc["open_boundary_treatment"]
            if boundary_treatment.lower() == "mean":
                UtilitiesParallel.print_root("Using mean boundary treatment.")
                lbc_class._LBC_set_function = lbc_class.set_vars_on_boundary_to_mean
            if boundary_treatment.lower() == "recycle":
                UtilitiesParallel.print_root("Using recycle boundary conditions.")

                lbc_class._LBC_set_function = lbc_class.set_vars_on_boundary_recycle
                assert "recycle_plane_pct" in lbc
                recycle_plane_loc = lbc["recycle_plane_pct"]  # Units are %
                lbc_class.set_recycle_plane(recycle_plane_loc[0], recycle_plane_loc[1])

        except:
            UtilitiesParallel.print_root("Usinge mean boundary treatment.")
            lbc_class._LBC_set_function = lbc_class.set_vars_on_boundary_to_mean

        return lbc_class


class LateralBCsDummy:
    def __init__(self):
        return

    def init_vars_on_boundary(self):
        return

    def set_vars_on_boundary(self):
        return

    def get_vars_on_boundary(self):
        UtilitiesParallel("No Boundaries Implemented for LateralBCsDummy")
        return

    def update(self, normal=None):

        return


class LateralBCs:
    def __init__(self, Grid, State, VelocityState):

        self._Grid = Grid
        self._State = State
        self._VelocityState = VelocityState

        self._var_on_boundary = {}
        self._LBC_set_function = None

        # Variables used only for recycling condiitons
        self._ix_recycle_plane = None
        self._iy_recycle_plane = None

        return

    def init_vars_on_boundary(self):
        """ Allocate memory to store large scale conditions to set lateral boundary conditions under inflow conditions. 
        These can be taken from reanalysis, prescribed, or taken from a nest.
        """
        ng = self._Grid.ngrid_local

        for var_name in self._State._dofs:
            self._var_on_boundary[var_name] = {}

            self._var_on_boundary[var_name]["x_low"] = np.zeros(
                (ng[1], ng[2]), dtype=np.double
            )
            self._var_on_boundary[var_name]["x_high"] = np.zeros(
                (ng[1], ng[2]), dtype=np.double
            )
            self._var_on_boundary[var_name]["y_low"] = np.zeros(
                (ng[0], ng[2]), dtype=np.double
            )
            self._var_on_boundary[var_name]["y_high"] = np.zeros(
                (ng[0], ng[2]), dtype=np.double
            )
        return

    def set_vars_on_boundary(self):
        self._LBC_set_function()
        return

    def set_vars_on_boundary_to_mean(self):

        for var_name in self._State._dofs:
            # Compute the domain mean of the variables
            var_mean = self._State.mean(var_name)

            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            x_low[:, :] = var_mean[np.newaxis, :]
            x_high[:, :] = var_mean[np.newaxis, :]
            y_low[:, :] = var_mean[np.newaxis, :]
            y_high[:, :] = var_mean[np.newaxis, :]

        return

    def set_recycle_plane(self, x_percent, y_percent):

        self._ix_recycle_plane = int(x_percent * self._Grid.n[0]) + self._Grid.n_halo[0]
        self._iy_recycle_plane = int(y_percent * self._Grid.n[1]) + self._Grid.n_halo[1]

        return

    def set_vars_on_boundary_recycle(self):

        nh = self._Grid.n_halo
        nl = self._Grid.nl
        ls = self._Grid._local_start
        le = self._Grid._local_end

        for var_name in self._State._dofs:
            # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            slab_x = self._State.get_slab_x(
                var_name, (self._ix_recycle_plane, self._ix_recycle_plane + 1)
            )

            # if var_name == "s":
            #    slab_x[0, ls[1] : le[1], :6] += np.random.randn(nl[1], 6) * 0.5

            # print(x_low.shape ,slab_x.shape )
            x_low[nh[1] : -nh[1], nh[2] : -nh[2]] = slab_x[0, ls[1] : le[1], :]

            slab_x = self._State.get_slab_x(
                var_name, (self._ix_recycle_plane+100, self._ix_recycle_plane+100 + 1)
            )

            x_high[nh[1] : -nh[1], nh[2] : -nh[2]] = slab_x[0, ls[1] : le[1], :]

            slab_y = self._State.get_slab_y(
                var_name, (self._iy_recycle_plane, self._iy_recycle_plane + 1)
            )
            # if var_name == "s":
            #    slab_y[ls[0] : le[0], 0, :6] += np.random.randn(nl[0], 6) * 0.5

            y_low[nh[0] : -nh[0], nh[2] : -nh[2]] = slab_y[ls[0] : le[0], 0, :]
            y_high[nh[0] : -nh[0], nh[2] : -nh[2]] = slab_y[ls[0] : le[0], 0, :]

        return

    def get_vars_on_boundary(self, var_name):
        """ Return arrays pointing to the externally prescribed boundary data

        Args:
            var_name (string): variable field name

        Returns:
            tuple of arrays containing the x_low, x_high, y_low, and y_high boundary arrays
        """

        return (
            self._var_on_boundary[var_name]["x_low"],
            self._var_on_boundary[var_name]["x_high"],
            self._var_on_boundary[var_name]["y_low"],
            self._var_on_boundary[var_name]["y_high"],
        )

    def update(self, normal=True):

        self.all_scalars(normal)

        return

    def all_scalars(self, normal):

        for var_name in self._State._dofs:

            self.open_x(var_name, normal)
            self.open_y(var_name, normal)

        return

    def open_x(self, var_name, normal):

        # u is the normal velocity component on a lateral boundary in x
        u = self._VelocityState.get_field("u")
        var = self._State.get_field(var_name)

        ibl = self._Grid.ibl[0]
        ibl_edge = self._Grid.ibl_edge[0]

        if self._Grid.low_rank[0]:
            if var_name != "u":
                self.open_x_impl_low(
                    ibl, ibl_edge, u, self._var_on_boundary[var_name]["x_low"], var
                )
            elif normal:
                # Set the lbc on the normal velocity component
                self.normal_x_impl_low(
                    ibl_edge, u, self._var_on_boundary[var_name]["x_low"],
                )

        ibu = self._Grid.ibu[0]
        ibu_edge = self._Grid.ibu_edge[0]

        if self._Grid.high_rank[0]:
            if var_name != "u":
                self.open_x_impl_high(
                    ibu, ibu_edge, u, self._var_on_boundary[var_name]["x_high"], var
                )
            elif normal:
                # Set the lbc on the normal velocity component
                self.normal_x_impl_high(
                    ibu_edge, u, self._var_on_boundary[var_name]["x_high"]
                )

        return

    @staticmethod
    @numba.njit()
    def normal_x_impl_low(ibl_edge, u, var_on_boundary):
        shape = u.shape
        for j in range(shape[1]):
            for k in range(shape[2]):
                u[: ibl_edge + 1, j, k] = var_on_boundary[j, k]

        return

    @staticmethod
    @numba.njit()
    def normal_x_impl_high(ibu_edge, u, var_on_boundary):
        shape = u.shape
        for j in range(shape[1]):
            for k in range(shape[2]):
                u[ibu_edge:, j, k] = var_on_boundary[j, k]
        return

    @staticmethod
    @numba.njit()
    def open_x_impl_low(ibl, ibl_edge, u, var_on_boundary, var):

        shape = var.shape
        for j in range(shape[1]):
            for k in range(shape[2]):
                ul = u[ibl_edge, j, k]

                if ul < 0:  # Outflow condition:
                    var[:ibl, j, k] = 2.0 * var[ibl, j, k] - var[ibl + 1, j, k]
                else:  # Inflow condition
                    var[:ibl, j, k] = var_on_boundary[j, k]

        return

    @staticmethod
    @numba.njit()
    def open_x_impl_high(ibu, ibu_edge, u, var_on_boundary, var):
        shape = var.shape

        for j in range(shape[1]):
            for k in range(shape[2]):
                ul = u[ibu_edge, j, k]

                if ul > 0:  # Outflow condition
                    var[ibu + 1 :, j, k] = 2.0 * var[ibu, j, k] - var[ibu - 1, j, k]
                else:  # Inflow condition
                    var[ibu + 1 :, j, k] = var_on_boundary[j, k]

        return

    @staticmethod
    @numba.njit()
    def normal_y_impl_low(ibl_edge, v, var_on_boundary):
        shape = v.shape
        for i in range(shape[0]):
            for k in range(shape[2]):
                v[i, : ibl_edge + 1, k] = var_on_boundary[i, k]

        return

    @staticmethod
    @numba.njit()
    def normal_y_impl_high(ibu_edge, v, var_on_boundary):
        shape = v.shape
        for i in range(shape[0]):
            for k in range(shape[2]):
                v[i, ibu_edge:, k] = var_on_boundary[i, k]
        return

    def open_y(self, var_name, normal):

        # v is the normal velocity component on a lateral boundary in y
        v = self._VelocityState.get_field("v")
        var = self._State.get_field(var_name)

        ibl = self._Grid.ibl[1]
        ibl_edge = self._Grid.ibl_edge[1]

        if self._Grid.low_rank[1]:
            if var_name != "v":
                self.open_y_impl_low(
                    ibl, ibl_edge, v, self._var_on_boundary[var_name]["y_low"], var
                )
            elif normal:
                # Set the lbc on the normal velocity component
                self.normal_y_impl_low(
                    ibl_edge, v, self._var_on_boundary[var_name]["y_low"],
                )

        ibu = self._Grid.ibu[1]
        ibu_edge = self._Grid.ibu_edge[1]

        if self._Grid.high_rank[1]:
            if var_name != "v":
                self.open_y_impl_high(
                    ibu, ibu_edge, v, self._var_on_boundary[var_name]["y_high"], var
                )
            elif normal:
                # Set the lbc on the normal velocity component
                self.normal_y_impl_high(
                    ibu_edge, v, self._var_on_boundary[var_name]["y_high"]
                )
        return

    @staticmethod
    @numba.njit()
    def open_y_impl_low(ibl, ibl_edge, v, var_on_boundary, var):

        shape = var.shape
        for i in range(shape[0]):
            for k in range(shape[2]):
                vl = v[i, ibl_edge, k]

                if vl < 0:  # Outflow condition:
                    var[i, :ibl, k] = 2.0 * var[i, ibl, k] - var[i, ibl + 1, k]
                else:  # Inflow condition
                    var[i, :ibl, k] = var_on_boundary[i, k]

        return

    @staticmethod
    @numba.njit()
    def open_y_impl_high(ibu, ibu_edge, v, var_on_boundary, var):

        shape = var.shape

        for i in range(shape[0]):
            for k in range(shape[2]):
                vl = v[i, ibu_edge, k]

                if vl > 0:  # Outflow condition
                    var[i, ibu + 1 :, k] = 2.0 * var[i, ibu, k] - var[i, ibu - 1, k]
                else:  # Inflow condition
                    var[i, ibu + 1 :, k] = var_on_boundary[i, k]

        return
