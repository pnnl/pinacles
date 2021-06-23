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
        u = self._VelocityState.get_field('u')
    

        ibl = self._Grid.ibl
        ibl_edge = self._Grid.ibl_edge

        if self._Grid.low_rank[0]:
            self.open_x_impl_low()



        ibu = self._Grid.ibu
        ibu_edge = self._Grid.ibu_edge


        if self._Grid.high_rank[0]:
            self.open_x_impl_high()



        return


    @staticmethod
    @numba.njit()
    def open_x_impl_low():



        return

    @staticmethod
    @numba.njit()
    def open_x_impl_high():

        return


    def open_y(self):

        self.open_y_impl_low()
        self.open_y_impl_high()

        return


    @staticmethod
    @numba.njit()
    def open_y_impl_low():

        return 


    @staticmethod
    @numba.njit()
    def open_y_impl_high():

        return 
    


