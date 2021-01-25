import numpy as np

class MeanStateAcceleration:
    def __init__(self, namelist, Grid, ScalarState, VelocityState):

        self._Grid = Grid
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState

        # TODO Make this a runtime option
        self._acceleration_factor = 4.0

        return

    def update(self):

        for v in ['qv', 's']:
            tend_array = self._ScalarState.get_tend(v)
            tend_mean = self._ScalarState.mean(v, tend=True)
            tend_array += np.multiply(tend_mean[np.newaxis, np.newaxis, :], self._acceleration_factor)


        for v in ['u', 'v']:
            tend_array = self._VelocityState.get_tend(v)
            tend_mean = self._VelocityState.mean(v, tend=True)
            tend_array += np.multiply(tend_mean[np.newaxis, np.newaxis, :], self._acceleration_factor)           

        return

    


