from tkinter.tix import CheckList
from turtle import update
import numpy as np
from pinacles.LateralBCs import LateralBCsBase
from mpi4py import MPI
import numba

from scipy import ndimage


class LateralBCsNest(LateralBCsBase):
    def __init__(self, namelist, Grid, State, VelocityState, Parent, NestState):
        LateralBCsBase.__init__(self, Grid, State, VelocityState)
        assert "nest" in namelist
        nest_namelist = namelist["nest"]
        assert "factor" in nest_namelist
        self.factor = nest_namelist["factor"]

        assert "parent_pts" in nest_namelist
        self.parent_pts = nest_namelist["parent_pts"]
        assert "root_point" in nest_namelist
        self.root_point = nest_namelist["root_point"]

        self._NestState = NestState
        self._Parent = Parent

        # Set up parent gather from nest
        parent_start = np.array(Parent.ModelGrid._local_start)
        parent_end = np.array(Parent.ModelGrid._local_end)
        parent_n = np.array(Parent.ModelGrid.n)

        local_start = self._Grid._local_start
        local_end = self._Grid._local_end

        fa = np.array(self.factor)
        ra = np.array(self.root_point)

        xi_local = np.arange(
            parent_start[0] * fa[0] - fa[0] * ra[0],
            parent_end[0] * fa[0] - fa[0] * ra[0] + 1,
        )

        try:
            xi_low = np.min(xi_local[xi_local >= 0])
            xi_hi = np.max(xi_local[xi_local <= fa[0] * self.parent_pts[0]])
        except:
            xi_low = 0
            xi_hi = 0

        yi_local = np.arange(
            parent_start[1] * fa[1] - fa[1] * ra[1],
            parent_end[1] * fa[1] - fa[1] * ra[1] + 1,
        )

        try:
            yi_low = np.min(yi_local[yi_local >= 0])
            yi_hi = np.max(yi_local[yi_local <= fa[1] * self.parent_pts[1]])
        except:
            yi_low = 0
            yi_hi = 0

        self.gather_to_parent = self._Grid.CreateGather(
            (xi_low, xi_hi), (yi_low, yi_hi)
        )
        self.gather_to_parent_v = self._Grid.CreateGather(
            (xi_low, xi_hi), (yi_low, yi_hi), y_edge=False
        )
        self.gather_to_parent_u = self._Grid.CreateGather(
            (xi_low, xi_hi), (yi_low, yi_hi), x_edge=False
        )

        xi = np.empty(self._Grid.ngrid_local, dtype=np.double)
        yi = np.empty(self._Grid.ngrid_local, dtype=np.double)

        xi[:, :, :] = self._Grid.x_local[:, np.newaxis, np.newaxis]
        yi[:, :, :] = self._Grid.y_local[np.newaxis, :, np.newaxis]

        self.x_nest_parent = self.gather_to_parent.call(xi)
        self.y_nest_parent = self.gather_to_parent.call(yi)

        try:
            self.x_indx_in_parent = np.where(
                np.isin(
                    np.round(Parent.ModelGrid.x_local, 6),
                    np.round(self.x_nest_parent[:, 0, 0], 6),
                )
            )[0]
            self.x_indx_in_nest = np.where(
                np.isin(
                    np.round(self.x_nest_parent[:, 0, 0], 6),
                    np.round(Parent.ModelGrid.x_local, 6),
                )
            )[0]
        except:
            self.x_indx_in_parent = np.empty((0,), dtype=np.int)
            self.x_indx_in_nest = np.empty((0,), dtype=np.int)

        try:
            self.y_indx_in_parent = np.where(
                np.isin(
                    np.round(Parent.ModelGrid.y_local, 6),
                    np.round(self.y_nest_parent[0, :, 0], 6),
                )
            )[0]
            self.y_indx_in_nest = np.where(
                np.isin(
                    np.round(self.y_nest_parent[0, :, 0], 6),
                    np.round(Parent.ModelGrid.y_local, 6),
                )
            )[0]
        except:
            self.y_indx_in_parent = np.empty((0,), dtype=np.int)
            self.y_indx_in_nest = np.empty((0,), dtype=np.int)

        xi[:, :, :] = self._Grid.x_edge_local[:, np.newaxis, np.newaxis]
        yi[:, :, :] = self._Grid.y_edge_local[np.newaxis, :, np.newaxis]

        self.x_edge_nest_parent = self.gather_to_parent.call(xi)
        self.y_edge_nest_parent = self.gather_to_parent.call(yi)

        try:
            self.x_edge_indx_in_parent = np.where(
                np.isin(
                    np.round(Parent.ModelGrid.x_edge_local, 6),
                    np.round(self.x_edge_nest_parent[:-1, 0, 0], 6),
                )
            )[0]
            self.x_edge_indx_in_nest = np.where(
                np.isin(
                    np.round(self.x_edge_nest_parent[:-1, 0, 0], 6),
                    np.round(Parent.ModelGrid.x_edge_local, 6),
                )
            )[0]
        except:
            self.x_edge_indx_in_parent = np.empty((0,), dtype=np.bool)
            self.x_edge_indx_in_nest = np.empty((0,), dtype=np.bool)

        try:
            self.y_edge_indx_in_parent = np.where(
                np.isin(
                    np.round(Parent.ModelGrid.y_edge_local, 6),
                    np.round(self.y_edge_nest_parent[0, :-1, 0], 6),
                )
            )[0]
            self.y_edge_indx_in_nest = np.where(
                np.isin(
                    np.round(self.y_edge_nest_parent[0, :-1, 0], 6),
                    np.round(Parent.ModelGrid.y_edge_local, 6),
                )
            )[0]
        except:
            self.y_edge_indx_in_parent = np.empty((0,), dtype=np.int)
            self.y_edge_indx_in_nest = np.empty((0,), dtype=np.int)

        return

    @staticmethod
    @numba.njit()
    def update_parent_field(
        pnh,
        dt,
        x_indx_in_parent,
        y_indx_in_parent,
        x_indx_in_nest,
        y_indx_in_nest,
        child_data,
        parent_data,
    ):

        shape = parent_data.shape

        for i in range(x_indx_in_parent.shape[0]):
            i_n = x_indx_in_nest[i]
            i_p = x_indx_in_parent[i]
            for j in range(y_indx_in_parent.shape[0]):
                j_n = y_indx_in_nest[j]
                j_p = y_indx_in_parent[j]
                for k_p in range(pnh[2], shape[2] - pnh[2]):
                    k_n = k_p - pnh[2]
                    parent_data[i_p, j_p, k_p] -= (
                        dt
                        * (1.0 / (40.0 * dt))
                        * (parent_data[i_p, j_p, k_p] - child_data[i_n, j_n, k_n])
                    )

        return

    def update_parent(self, dt):

        parent_start = np.array(self._Parent.ModelGrid._local_start)
        parent_end = np.array(self._Parent.ModelGrid._local_end)
        parent_nh = self._Parent.ModelGrid.n_halo

        for var_name in self._State._dofs:

            if var_name == "u":
                child_data = self.gather_to_parent_u.call(
                    self._State.get_field(var_name)
                )

            elif var_name == "v":
                child_data = self.gather_to_parent_v.call(
                    self._State.get_field(var_name)
                )
            else:

                child_data = self.gather_to_parent.call(self._State.get_field(var_name))

            ndimage.uniform_filter(child_data, size=(3, 3, 0), output=child_data)

            parent_data = self._NestState.get_field(var_name)

            if var_name == "u":
                x_indx_in_parent = self.x_edge_indx_in_parent
                y_indx_in_parent = self.y_indx_in_parent

                x_indx_in_nest = self.x_edge_indx_in_nest
                y_indx_in_nest = self.y_indx_in_nest
            elif var_name == "v":
                x_indx_in_parent = self.x_indx_in_parent
                y_indx_in_parent = self.y_edge_indx_in_parent

                x_indx_in_nest = self.x_indx_in_nest
                y_indx_in_nest = self.y_edge_indx_in_nest

            else:
                x_indx_in_parent = self.x_indx_in_parent
                y_indx_in_parent = self.y_indx_in_parent

                x_indx_in_nest = self.x_indx_in_nest
                y_indx_in_nest = self.y_indx_in_nest

            self.update_parent_field(
                parent_nh,
                dt,
                x_indx_in_parent,
                y_indx_in_parent,
                x_indx_in_nest,
                y_indx_in_nest,
                child_data,
                parent_data,
            )

        return

    def set_vars_on_boundary(self, **kwargs):
        assert "ParentNest" in kwargs
        ParentNest = kwargs["ParentNest"]

        # Get info
        parent_nhalo = ParentNest.ModelGrid.n_halo
        nh = self._Grid.n_halo

        local_start = self._Grid._local_start
        local_end = self._Grid._local_end

        for var_name in self._State._dofs:

            # if var_name == 'w':
            #    continue

            # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            center_point_x = parent_nhalo[0] + self.root_point[0]
            center_point_y = parent_nhalo[1] + self.root_point[1]

            # Now get the indicies of the subset on this rank
            start = (local_start[0]) // self.factor[0] + self.root_point[0]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[0] / self.factor[0])),
            )

            if var_name == "v":
                slab_range = (center_point_y, center_point_y + 1)
            else:
                slab_range = (center_point_y, center_point_y + 1)

            # First we get y_low
            y_low[nh[0] : -nh[0], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_y(var_name, slab_range)[
                    local_part_of_parent[0] : local_part_of_parent[1], :, :
                ],
                self.factor[1],
                axis=0,
            )[: self._Grid._local_shape[0], 0, :]

            if var_name == "v":
                slab_range = (
                    center_point_y + self.parent_pts[1],
                    center_point_y + self.parent_pts[1] + 1,
                )
            else:
                slab_range = (
                    center_point_y + self.parent_pts[1] + 1,
                    center_point_y + self.parent_pts[1] + 2,
                )

            y_high[nh[0] : -nh[0], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_y(var_name, slab_range)[
                    local_part_of_parent[0] : local_part_of_parent[1], :, :
                ],
                self.factor[1],
                axis=0,
            )[: self._Grid._local_shape[0], 0, :]

            # Now get the indicies of the subset on this rank
            start = (local_start[1]) // self.factor[1] + self.root_point[1]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[1] / self.factor[1])),
            )

            if var_name == "u":
                slab_range = (center_point_x, center_point_x + 1)
            else:
                slab_range = (center_point_x, center_point_x + 1)

            x_low[nh[1] : -nh[1], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_x(var_name, slab_range)[
                    :, local_part_of_parent[0] : local_part_of_parent[1], :
                ],
                self.factor[0],
                axis=1,
            )[0, : self._Grid._local_shape[1], :]

            if var_name == "u":
                slab_range = (
                    center_point_x + self.parent_pts[0],
                    center_point_x + self.parent_pts[0] + 1,
                )
            else:
                slab_range = (
                    center_point_x + self.parent_pts[0] + 1,
                    center_point_x + self.parent_pts[0] + 2,
                )

            x_high[nh[1] : -nh[1], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_x(var_name, slab_range)[
                    :, local_part_of_parent[0] : local_part_of_parent[1], :
                ],
                self.factor[0],
                axis=1,
            )[0, : self._Grid._local_shape[1], :]

        return
