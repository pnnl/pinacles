import numpy as np
import numba

class ParticlesBase:

    def __init__(self, Grid, VelocityState, ScalarState, DiagnosticState):

        # Initialize data
        self._Grid = Grid
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        # Number of arrays
        self._n_buffer = 10
        self._n_particles = 0
        self._particle_dofs = 3

        self._initialzied = False

        return

    def add_particle_dimension():
        assert(self._initilzied == False)
        self._particle_dofs += 1
        return 

    def update_position(self):

        if self._n_particles == 0:
            # Do nothing if there are no particles here
            return


        return

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def n_memory(self):
        return self._n_buffer

    @property
    def initialized(self):
        return self._initialzied

   # @staticmethod
   # @numba.njit():
   # def compute_new_positions(n_particles, dx, x, y, z, u, v, w):

    #    for pi in self._n_particles:
    #        ix =
    #        iy =
    #        iz =

    #    return




class ParticlesSimple(ParticlesBase):

    def __init__(self, Grid, VelocityState, ScalarState, DiagnsoticState):

        ParticlesBase.__init__(self,Grid, VelocityState, ScalarState, DiagnsoticState)

        return

    def update(self):

        self.update_position()


        return