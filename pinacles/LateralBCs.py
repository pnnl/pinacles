# This class will be used to implment the various types of open boundary conditions.
import numpy as np
import numba


class LateralBCs:
    def __init__(self, Grid, State, VelocityState):

        self._Grid = Grid
        self._State = State
        self._VelocityState = VelocityState

        self._var_on_boundary = {}

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

    def update(self):

        self.all_scalars()

        return

    def all_scalars(self):

        for var_name in self._State._dofs:

            self.open_x(var_name)
            self.open_y(var_name)

        return

    def open_x(self, var_name):

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
            else:
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
            else:
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

    def open_y(self, var_name):

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
            else:
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
            else:
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
