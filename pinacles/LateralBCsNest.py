import numpy as np
from pinacles.LateralBCs import LateralBCsBase
from mpi4py import MPI


class LateralBCsNest(LateralBCsBase):
    def __init__(self, namelist, Grid, State, VelocityState, NestState):
        print("B4 Base init")
        LateralBCsBase.__init__(self, Grid, State, VelocityState)
        print("After Base init")
        assert "nest" in namelist
        nest_namelist = namelist["nest"]
        assert "factor" in nest_namelist
        self.factor = nest_namelist["factor"]
        # print(self.factor)

        assert "parent_pts" in nest_namelist
        self.parent_pts = nest_namelist["parent_pts"]
        assert "root_point" in nest_namelist
        self.root_point = nest_namelist["root_point"]

        self._NestState = NestState

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

            # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            center_point_x = parent_nhalo[0] + self.root_point[0]
            center_point_y = parent_nhalo[1] + self.root_point[1]

            # Now get the indicies of the subset on this rank
            start = (local_start[0]) // self.factor[0] + self.root_point[0]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[0] / 3)),
            )

            slab_range = (center_point_y, center_point_y + 1)

            # First we get y_low

            # import sys; sys.exit()
            y_low[nh[0] : -nh[0], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_y(var_name, slab_range)[
                    local_part_of_parent[0] : local_part_of_parent[1], :, :
                ],
                self.factor[1],
                axis=0,
            )[: self._Grid._local_shape[0], 0, :]

            slab_range = (
                center_point_y + self.parent_pts[1] - 1,
                center_point_y + self.parent_pts[1],
            )

            y_high[nh[0] : -nh[0], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_y(var_name, slab_range)[
                    local_part_of_parent[0] : local_part_of_parent[1], :, :
                ],
                self.factor[1],
                axis=0,
            )[: self._Grid._local_shape[0], 0, :]

            # local_part_of_parent = (
            #    (local_start[1]) // self.factor[1] + self.root_point[1],
            #    (local_end[1]) // self.factor[1] + self.root_point[1],
            # )

            # Now get the indicies of the subset on this rank
            start = (local_start[1]) // self.factor[1] + self.root_point[1]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[1] / 3)),
            )

            slab_range = (center_point_x, center_point_x + 1)

            x_low[nh[1] : -nh[1], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_x(var_name, slab_range)[
                    :, local_part_of_parent[0] : local_part_of_parent[1], :
                ],
                self.factor[0],
                axis=1,
            )[0, : self._Grid._local_shape[1], :]

            slab_range = (
                center_point_x + self.parent_pts[0] - 1,
                center_point_x + self.parent_pts[0],
            )

            x_high[nh[1] : -nh[1], nh[2] : -nh[2]] = np.repeat(
                self._NestState.get_slab_x(var_name, slab_range)[
                    :, local_part_of_parent[0] : local_part_of_parent[1], :
                ],
                self.factor[0],
                axis=1,
            )[0, : self._Grid._local_shape[1], :]

        return
