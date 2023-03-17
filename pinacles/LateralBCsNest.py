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
        assert "two_way" in nest_namelist
        self.two_way = nest_namelist["two_way"]

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
    # @numba.njit()
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
                # for k_p in range(pnh[2], shape[2] - pnh[2]):
                # k_n = k_p - pnh[2]
                # parent_data[i_p, j_p, k_p] -= (
                #    dt
                #    * (1.0 / (40.0 * dt))
                #    * (parent_data[i_p, j_p, k_p] - child_data[i_n, j_n, k_n])
                # )

        return

    def update_parent(self, dt):
        if not self.two_way:
            return

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

            ndimage.uniform_filter(
                child_data, size=(self.factor[0], self.factor[1], 0), output=child_data
            )

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

            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            # Compute location of the lower edge to extract from the parent domain
            # these are indicies in the parent nest's domain

            center_point_x = parent_nhalo[0] + self.root_point[0]
            center_point_y = parent_nhalo[1] + self.root_point[1]

            # Now get the indicies of the subset on this rank.
            start = (local_start[0]) // self.factor[0] + self.root_point[0]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[0] / self.factor[0])),
            )

            # We must handle u and v components differently
            if var_name == "v":
                slab_range = (center_point_y - 1, center_point_y)
            else:
                slab_range = (center_point_y - 1, center_point_y)

            # First we get y_low

            slab = self._NestState.get_slab_y(var_name, slab_range)

            slab_repeate = np.repeat(
                    slab,
                    self.factor[1],
                    axis=0,
                )

            slab_filter = ndimage.uniform_filter(
                slab_repeate,
                size=(self.factor[0], 1, 1),
            )
            
            #slab_filter = slab_repeate

            odd_shift = 0
            if var_name == 'u':
                odd_shift = self.factor[0]//2

            si = local_part_of_parent[0] * self.factor[0] - nh[0] - odd_shift
            ei = si + y_low.shape[0] 
            

            
            y_low[:, nh[2] : -nh[2]] = slab_filter[si:ei,
                :,
                :,
            ][
                :, 0, :
            ]

            if var_name == "v":
                slab_range = (
                    center_point_y + self.parent_pts[1],
                    center_point_y + self.parent_pts[1] + 1,
                )
            else:
                slab_range = (
                    center_point_y + self.parent_pts[1],
                    center_point_y + self.parent_pts[1] + 1,
                )


            slab = self._NestState.get_slab_y(var_name, slab_range)

            slab_repeate = np.repeat(
                    slab,
                    self.factor[1],
                    axis=0,
                )

            slab_filter = ndimage.uniform_filter(
                slab_repeate,
                size=(self.factor[0], 1, 1),
            )
            #import time
            #time.sleep(MPI.COMM_WORLD.Get_rank()+0.01)
            #print(local_part_of_parent, si, ei)
            #print('ll_corner:', self._Grid._ll_corner)
            #print('RANK:', MPI.COMM_WORLD.Get_rank())
            #print('slab:', slab_filter[si:ei,0,0][:6]/2.0)
            #print('local_axes:', self._Grid.local_axes_edge[0][:6])
            #MPI.COMM_WORLD.Barrier()
            
            #import sys; sys.exit()

            #slab_filter = slab_repeate


            y_high[:, nh[2] : -nh[2]] = slab_filter[
                si:ei,
                :,
                :,
            ][
                :, 0, :
            ]



            # Now get the indicies of the subset on this rank
            start = (local_start[1]) // self.factor[1] + self.root_point[1]
            local_part_of_parent = (
                start,
                start + int(np.ceil(self._Grid._local_shape[1] / self.factor[1])),
            )

            if var_name == "u":
                slab_range = (center_point_x - 1, center_point_x)
            else:
                slab_range = (center_point_x - 1, center_point_x)

            slab = self._NestState.get_slab_x(var_name, slab_range)
            slab_repeate = np.repeat(slab,
                    self.factor[0],
                    axis=1,
                )

            slab_filter = ndimage.uniform_filter(slab_repeate,
                size=(1, self.factor[1], 1),
            )

            #slab_filter = slab_repeate
        
            odd_shift = 0
            if var_name == 'u':
                odd_shift = self.factor[0]//2

            
            si = local_part_of_parent[0] * self.factor[1] - nh[1] - odd_shift
            ei = si + x_low.shape[0]


            if var_name == 'v':
                si +=  1#self.factor[1]
                ei +=  1 #self.factor[1]


            x_low[:, nh[2] : -nh[2]] = slab_filter[
                :,
                si:ei,
                :,
            ][
                0, :, :
            ]

            if var_name == "u":
                slab_range = (
                    center_point_x + self.parent_pts[0],
                    center_point_x + self.parent_pts[0] + 1,
                )
            else:
                slab_range = (
                    center_point_x + self.parent_pts[0],
                    center_point_x + self.parent_pts[0] + 1,
                )

            slab = self._NestState.get_slab_x(var_name, slab_range)
            slab_repeate = np.repeat(slab,
                    self.factor[0],
                    axis=1,
                )

            slab_filter = ndimage.uniform_filter(slab_repeate,
                size=(1, self.factor[1], 1),
            )

            #slab_filter = slab_repeate

            x_high[:, nh[2] : -nh[2]] = slab_filter[
                :,
                si:ei,
                :,
            ][
                0, :, :
            ]

        return 

    def inflow_pert(self, LBCVel):
        
        nh = self._Grid.n_halo
        x_low, x_high, y_low, y_high = self.get_vars_on_boundary('s')
        u_x_low, u_x_high, u_y_low, u_y_high = LBCVel.get_vars_on_boundary('u')
        v_x_low, v_x_high, v_y_low, v_y_high = LBCVel.get_vars_on_boundary('v')
    
        k_depth = 1  + nh[2]
        Ek = 0.16
        
        speed = (u_x_low[:, nh[2]:k_depth]**2.0 + v_x_low[:, nh[2]:k_depth]**2.0)
        s_p = (speed)/(1250.0 * Ek)
        x_low[:, nh[2]:k_depth] += np.random.uniform(-1.0, 1.0, size=(x_low.shape[0], 1)) * s_p
        
        speed = (u_x_high[:, nh[2]:k_depth]**2.0 + v_x_high[:, nh[2]:k_depth]**2.0)
        s_p = (speed)/(1250.0 * Ek)
        x_high[:, nh[2]:k_depth] += np.random.uniform(-1.0, 1.0, size=(x_high.shape[0], 1)) * s_p
        
        speed = (u_y_low[:, nh[2]:k_depth]**2.0 + v_y_low[:, nh[2]:k_depth]**2.0)
        s_p = (speed)/(1250.0 * Ek)
        y_low[:, nh[2]:k_depth] += np.random.uniform(-1.0, 1.0, size=(y_low.shape[0], 1)) * s_p
        
        speed = (u_y_high[:, nh[2]:k_depth]**2.0 + v_y_high[:, nh[2]:k_depth]**2.0)
        s_p = (speed)/(1250.0 * Ek)
        y_high[:, nh[2]:k_depth] += np.random.uniform(-1.0, 1.0, size=(y_high.shape[0], 1)) * s_p

        return
