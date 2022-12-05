import numpy as np
from pinacles import Damping_impl


class Damping:
    def __init__(self, namelist, Timers, Grid, DiagnosticState):

        self._vars = namelist["damping"]["vars"]
        self._states = []
        self._Timers = Timers
        self._Grid = Grid
        self._DiagnosticState = DiagnosticState
        

        return

    def update(self):

        return
    
    def init_means(self):
        return

    def add_state(self, State):
        self._states.append(State)
        return

    @property
    def vars(self):
        return self._vars


class Rayleigh(Damping):
    def __init__(self, namelist, Timers, Grid, DiagnosticState):
        Damping.__init__(self, namelist, Timers, Grid, DiagnosticState)

        self._depth = namelist["damping"]["depth"]
        self._timescale = namelist["damping"]["timescale"]

        self._timescale_profile = None
        self._timescale_profile_edge = None

        self._compute_timescale_profile()

        self._Timers.add_timer("Rayleigh_update")

        return

    def update(self):

        self._Timers.start_timer("Rayleigh_update")

        # First loop over all of the variables
        for var in self._vars:
            for state in self._states:
                if var in state.names:
                    field = state.get_field(var)
                    tend = state.get_tend(var)
                    loc = state.get_loc(var)

                    mean = state.mean(var)

                    if var == "w":
                        mean.fill(0.0)

                        if loc == "c":
                            Damping_impl.rayleigh(
                                self._timescale_profile, mean, field, tend
                            )
                        elif loc == "z":
                            Damping_impl.rayleigh(
                                self._timescale_profile_edge, mean, field, tend
                            )
                            
                        #N2 = self._DiagnosticState.get_field('bvf')
                        #Damping_impl.rayleigh_N2(1.0/1800.0, N2,  field, tend)    
                            
                            

        self._Timers.end_timer("Rayleigh_update")

        return

    def _compute_timescale_profile(self):

        self._timescale_profile = np.zeros(self._Grid.ngrid[2], dtype=np.double)
        self._timescale_profile_edge = np.zeros_like(self._timescale_profile)

        z = self._Grid.z_global
        z_edge = self._Grid.z_edge_global

        z_top = self._Grid.l[2]
        for k in range(self._Grid.ngrid[2]):
            if z[k] >= z_top - self._depth:
                self._timescale_profile[k] = (1.0 / self._timescale) * np.sin(
                    (np.pi / 2.0) * (1.0 - (z_top - z[k]) / self._depth)
                ) ** 2.0
                self._timescale_profile_edge[k] = (1.0 / self._timescale) * np.sin(
                    (np.pi / 2.0) * (1.0 - (z_top - z_edge[k]) / self._depth)
                ) ** 2.0

        return

    @property
    def depth(self):
        return self._depth


class RayleighInitial(Damping):
    def __init__(self, namelist, Timers, Grid):
        Damping.__init__(self, namelist, Timers, Grid)
        self._depth = namelist["damping"]["depth"]
        self._timescale = namelist["damping"]["timescale"]

        self._timescale_profile = None
        self._timescale_profile_edge = None

        self._compute_timescale_profile()
        self.means = {}

        self._Timers.add_timer("RayleighInitial_update")

        return

    def init_means(self):
        # First loop over all of the variables
        for var in self._vars:
            for state in self._states:
                if var in state.names:
                    mean = state.mean(var)
                    self.means[var] = mean

        return

    def update(self):

        self._Timers.start_timer("RayleighInitial_update")

        # First loop over all of the variables
        for var in self._vars:
            for state in self._states:
                if var in state.names:
                    field = state.get_field(var)
                    tend = state.get_tend(var)
                    loc = state.get_loc(var)

                    mean = self.means[var]

                    if loc == "c":
                        Damping_impl.rayleigh(
                            self._timescale_profile, mean, field, tend
                        )
                    elif loc == "z":
                        Damping_impl.rayleigh(
                            self._timescale_profile_edge, mean, field, tend
                        )

        self._Timers.end_timer("RayleighInitial_update")

        return

    def _compute_timescale_profile(self):

        self._timescale_profile = np.zeros(self._Grid.ngrid[2], dtype=np.double)
        self._timescale_profile_edge = np.zeros_like(self._timescale_profile)

        z = self._Grid.z_global
        z_edge = self._Grid.z_edge_global

        z_top = self._Grid.l[2]
        for k in range(self._Grid.ngrid[2]):
            if z[k] >= z_top - self._depth:
                self._timescale_profile[k] = (1.0 / self._timescale) * np.sin(
                    (np.pi / 2.0) * (1.0 - (z_top - z[k]) / self._depth)
                ) ** 2.0
                self._timescale_profile_edge[k] = (1.0 / self._timescale) * np.sin(
                    (np.pi / 2.0) * (1.0 - (z_top - z_edge[k]) / self._depth)
                ) ** 2.0

        return

    @property
    def depth(self):
        return self._depth
