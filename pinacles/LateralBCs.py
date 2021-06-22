# This class will be used to implment the various types of open boundary conditions.
import numpy 
import numba
class LateralBCs:

    def __init__(self, Grid, State, VelocityState):

        self._Grid = Grid
        self._State = State
        self._VelocityState = VelocityState

        return


    def update(self):

        self.open_x()
        self.open_y()

        return


    def all_scalars(self):

        return

    def all_velocities(self):

        return



    def open_x(self):

        # u is the normal velocity component on a lateral boundary in x
        print(self._VelocityState)

        u = self._VelocityState.get_field('u')

        self.open_x_impl()

        return


    @staticmethod
    @numba.njit()
    def open_x_impl():


        return


    def open_y(self):

        self.open_y_impl()

        return

    @staticmethod
    @numba.njit()
    def open_y_impl():

        return 

    


