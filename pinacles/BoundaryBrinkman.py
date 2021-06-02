import numpy as np
import numba


class BoundaryBrinkman:
    @staticmethod
    @numba.njit()
    def add_boundary(terrain_fraction, u, v, w, ut, vt, wt):

        shape = terrain_fraction.shape
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                for k in range(shape[2] - 1):

                    if terrain_fraction[i, j, k] > 0.0:
                        for surr in [0, 1]:

                            ut[
                                i + surr, j, k
                            ] = 0.0  # 0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i+1,j,k])  * u[i,j,k]
                            vt[
                                i, j + surr, k
                            ] = 0.0  # 0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i,j+1,k])  * v[i,j,k]
                            wt[
                                i, j, k + surr
                            ] = 0.0  # 0.5 * (terrain_fraction[i,j,k] + terrain_fraction[i,j,k+1])  * w[i,j,k]

                            u[i + surr, j, k] = 0.0
                            v[i, j + surr, k] = 0.0
                            w[i, j, k + surr] = 0.0

        return

    @staticmethod
    @numba.njit()
    def _compute_masks(
        terrain_fraction, x_edge_mask, y_edge_mask, z_edge_mask, cc_mask
    ):
        shape = terrain_fraction.shape

        # The value 10 is indicative of the free stream (make this a paramter)
        x_edge_mask.fill(1.0)
        y_edge_mask.fill(1.0)
        z_edge_mask.fill(1.0)

        # Compute the masks
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                for k in range(shape[2] - 1):

                    if terrain_fraction[i, j, k] > 0.0:
                        for surr in [0, 1]:
                            x_edge_mask[i + surr, j, k] = 0.0
                            y_edge_mask[i, j + surr, k] = 0.0
                            z_edge_mask[i, j, k + surr] = 0.0

        cc_mask[:, :, :] = 1.0 - terrain_fraction[:, :, :]

        return

    @staticmethod
    @numba.njit()
    def _compute_dists(
        x_edge_mask,
        y_edge_mask,
        z_edge_mask,
        cc_mask,
        x_mask_dist,
        y_mask_dist,
        z_mask_dist,
        cc_mask_dist,
    ):
        shape = x_edge_mask.shape

        # Compute the dists
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                for k in range(1, shape[2] - 1):
                    for surr in [-1, 1]:
                        if x_edge_mask[i + surr, j, k] == 0.0:
                            x_mask_dist[i, j, k] = 1.0
                        if y_edge_mask[i, j + surr, k] == 0.0:
                            y_mask_dist[i, j, k] = 1.0
                        if z_edge_mask[i, j, k + surr] == 0.0:
                            z_mask_dist[i, j, k] = 1.0

        return

    def __init__(self, Grid, DiagnosticState, VelocityState, TimeSteppingController):

        self._Grid = Grid
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._DiagnosticState.add_variable("terrain_fraction")
        self._DiagnosticState.add_variable("x_edge_mask")
        self._DiagnosticState.add_variable("y_edge_mask")
        self._DiagnosticState.add_variable("z_edge_mask")
        self._DiagnosticState.add_variable("cc_mask")

        self._DiagnosticState.add_variable("cc_mask_dist")
        self._DiagnosticState.add_variable("x_mask_dist")
        self._DiagnosticState.add_variable("y_mask_dist")
        self._DiagnosticState.add_variable("z_mask_dist")

        return

    def initialize(self):

        terrain_fraction = self._DiagnosticState.get_field("terrain_fraction")
        x_edge_mask = self._DiagnosticState.get_field("x_edge_mask")
        y_edge_mask = self._DiagnosticState.get_field("y_edge_mask")
        z_edge_mask = self._DiagnosticState.get_field("z_edge_mask")
        cc_mask = self._DiagnosticState.get_field("cc_mask")

        x_mask_dist = self._DiagnosticState.get_field("x_mask_dist")
        y_mask_dist = self._DiagnosticState.get_field("y_mask_dist")
        z_mask_dist = self._DiagnosticState.get_field("z_mask_dist")
        cc_mask_dist = self._DiagnosticState.get_field("cc_mask_dist")

        n_halo = self._Grid.n_halo
        # terrain_fraction[:,:,:n_halo[2]] = 1.0
        terrain_fraction[14:22, 60:68, :14] = 1.0  # /self._TimeSteppingController.dt

        terrain_fraction[14 + 12 : 22 + 12, 60 - 12 : 68 - 12, :14] = 1.0

        terrain_fraction[14 + 12 : 22 + 12, 60 + 12 : 68 + 12, :14] = 1.0
        terrain_fraction[14 + 12 : 22 + 12, 60:68, :20] = 1.0

        self._compute_masks(
            terrain_fraction, x_edge_mask, y_edge_mask, z_edge_mask, cc_mask
        )
        self._compute_dists(
            x_edge_mask,
            y_edge_mask,
            z_edge_mask,
            cc_mask,
            x_mask_dist,
            y_mask_dist,
            z_mask_dist,
            cc_mask_dist,
        )

        self._x_edge_mask_bool = x_edge_mask == 0.0
        self._y_edge_mask_bool = y_edge_mask == 0.0
        self._z_edge_mask_bool = z_edge_mask == 0.0
        self._cc_mask_bool = cc_mask == 0.0

        return

    def update(self):

        terrain_fraction = self._DiagnosticState.get_field("terrain_fraction")
        x_edge_mask = self._DiagnosticState.get_field("x_edge_mask")
        y_edge_mask = self._DiagnosticState.get_field("y_edge_mask")
        z_edge_mask = self._DiagnosticState.get_field("z_edge_mask")
        cc_mask = self._DiagnosticState.get_field("cc_mask")

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        wt = self._VelocityState.get_tend("w")

        # Could make the masks bools and store them, then we would have too loop over the arrays
        u[self._x_edge_mask_bool] = 0.0
        v[self._y_edge_mask_bool] = 0.0
        w[self._z_edge_mask_bool] = 0.0

        ut[self._x_edge_mask_bool] = 0.0
        vt[self._y_edge_mask_bool] = 0.0
        wt[self._z_edge_mask_bool] = 0.0

        # ut[-10:,:,:] += (4.0 - u[-10:,:,:])*(1.0/30.0)
        # vt[-10:,:,:] += (0.0 - v[-10:,:,:])*(1.0/30.0)
        # wt[-10:,:,:] += (0.0 - w[-10:,:,:])*(1.0/30.0)
        # Add boundary
        # ut[:10,:,:] += (-1.0 - u[:10,:,:])*(1.0/30.0)
        # vt[:10,:,:] += (0.0 - v[:10,:,:])*(1.0/30.0)
        # wt[:10,:,:] += (0.0 - w[:10,:,:])*(1.0/30.0)
        return
