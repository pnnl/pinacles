import numpy as np
from pinacles.LateralBCs import LateralBCsBase

class LateralBCsRecycle(LateralBCsBase):

    def __init__(self, namelist, Grid, State, VelocityState):
        
        LateralBCsBase.__init__(self, Grid, State, VelocityState)

        assert "lbc" in namelist
        lbc = namelist["lbc"]
        
        assert "type" in namelist["lbc"]
        assert "recycle_plane_pct" in namelist["lbc"]

        recycle_plane_loc = lbc["recycle_plane_pct"]  # Units are %
        self.set_recycle_plane(recycle_plane_loc[0], recycle_plane_loc[1])

        return

    def set_recycle_plane(self, x_percent, y_percent):

        self._ix_recycle_plane = int(x_percent * self._Grid.n[0]) + self._Grid.n_halo[0]
        self._iy_recycle_plane = int(y_percent * self._Grid.n[1]) + self._Grid.n_halo[1]

        return

    def set_vars_on_boundary(self, **kwargs):
        print('Calling recycle')

        nh = self._Grid.n_halo
        nl = self._Grid.nl
        ls = self._Grid._local_start
        le = self._Grid._local_end
        
        if not self.count%2 == 0:
            self.count += 1
            return
        self.count = 0

        for var_name in self._State._dofs:
            # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            slab_x = self._State.get_slab_x(
                var_name, (self._ix_recycle_plane, self._ix_recycle_plane + 1)
            )

            x_low[nh[1] : -nh[1], nh[2] : -nh[2]] = slab_x[0, ls[1] : le[1], :]

            x_high[nh[1] : -nh[1], nh[2] : -nh[2]] = slab_x[0, ls[1] : le[1], :]

            slab_y = self._State.get_slab_y(
                var_name, (self._iy_recycle_plane, self._iy_recycle_plane + 1)
            )

            y_low[nh[0] : -nh[0], nh[2] : -nh[2]] = slab_y[ls[0] : le[0], 0, :]

            y_high[nh[0] : -nh[0], nh[2] : -nh[2]] = slab_y[ls[0] : le[0], 0, :]

        return
        
