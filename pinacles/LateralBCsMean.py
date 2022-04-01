import numpy as np
from pinacles.LateralBCs import LateralBCsBase


class LateralBCsMean(LateralBCsBase):
    def __init__(self, Grid, State, VelocityState):
        LateralBCsBase.__init__(self, Grid, State, VelocityState)

        return

    def set_vars_on_boundary(self, **kwargs):

        for var_name in self._State._dofs:
            # Compute the domain mean of the variables
            var_mean = self._State.mean(var_name)

            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var_name)

            x_low[:, :] = var_mean[np.newaxis, :]
            x_high[:, :] = var_mean[np.newaxis, :]
            y_low[:, :] = var_mean[np.newaxis, :]
            y_high[:, :] = var_mean[np.newaxis, :]

        return
