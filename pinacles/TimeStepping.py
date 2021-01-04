import numpy as np
from pinacles import TimeStepping_impl as TS_impl
from pinacles import UtilitiesParallel
from mpi4py import MPI

class RungeKuttaBase:

    def __init__(self, namelist, Grid, PrognosticState):
        self._Grid = Grid
        self._PrognosticState = PrognosticState
        self._t = 0
        self.n_rk_step = 0
        self._rk_step = 0
        self.cfl_target = namelist['time']['cfl']
        self._dt = 0.0


        return

    def initialize(self):

        return

    def update(self):

        return

class RungeKutta2ndSSP(RungeKuttaBase):
    def __init__(self, namelist, Grid, PrognosticState):
        RungeKuttaBase.__init__(self, namelist, Grid, PrognosticState)
        self.Tn = None
        self.n_rk_step = 2
        self._rk_step = 0
        return

    def initialize(self):
        self._Tn = np.empty_like(self._PrognosticState.state_array)
        self._Tn[:,:,:,:]=0.0
        return

    def update(self):
        present_state = self._PrognosticState.state_array
        present_tend = self._PrognosticState.tend_array

        if self._rk_step == 0:
            self._Tn = np.copy(present_state) #TODO: Replace this copy
            TS_impl.rk2ssp_s0(present_state, present_tend, self._dt )
            self._rk_step = 1

        else:
            TS_impl.rk2ssp_s1(self._Tn, present_state,
                present_tend, self._dt)
            self._rk_step = 0
            self._t += self._dt

        return

    @property 
    def dt(self): 
        return self._dt

def factory(namelist, Grid, PrognosticState):
    return RungeKutta2ndSSP(namelist, Grid, PrognosticState)

class TimeSteppingController:
    def __init__(self, namelist, Grid, VelocityState):
        self._Grid = Grid
        self._VelocityState = VelocityState
        self._TimeStepper = []
        self._times_to_match = []
        self._dt = 0.0
        self._dt_max = 2.0
        self._cfl_target = namelist['time']['cfl']
        self._time_max = namelist['time']['time_max']
        self._time = 0.0
        return

    def add_timestepper(self, TimeStepper):
        self._TimeStepper.append(TimeStepper)
        return

    def add_timematch(self, delta_time):
        self._times_to_match.append(delta_time)
        return

    def adjust_timestep(self, n_rk_step):
        if n_rk_step == 0:
            #Get the current model time
            self._time += self._dt

            nhalo = self._Grid.n_halo
            dxi = self._Grid.dxi

            #Get velocity components
            u = self._VelocityState.get_field('u')
            v = self._VelocityState.get_field('v')
            w = self._VelocityState.get_field('w')

            cfl_max = 0.0
            cfl_max_local, umax, vmax, wmax = TS_impl.comput_local_cfl_max(nhalo, dxi, u, v, w)


            umax = UtilitiesParallel.ScalarAllReduce(umax, op=MPI.MAX)
            vmax = UtilitiesParallel.ScalarAllReduce(vmax, op=MPI.MAX)
            wmax = UtilitiesParallel.ScalarAllReduce(wmax, op=MPI.MAX)

            recv_buffer = np.zeros((1,), dtype=np.double)
            MPI.COMM_WORLD.Allreduce(np.array([cfl_max_local], dtype=np.double), recv_buffer, op=MPI.MAX)
            cfl_max = recv_buffer[0]
            self._cfl_current = self._dt * cfl_max
            self._dt = min(self._cfl_target / cfl_max, self._dt_max)
            self.match_time()

            for Stepper in self._TimeStepper:
                Stepper._dt = self._dt

            if MPI.COMM_WORLD.Get_rank() == 0:
                print('Time:', self._time, 'CFL Before Adjustment:', self._cfl_current,
                    'CFL After Adjustment:', cfl_max * self._dt, 'dt:', self._dt )
                print('\t umax: ', umax, '\t vmax:', vmax, '\t wmax:', wmax)

        return

    def match_time(self):
        #Must be called after dt is computed
        for match in self._times_to_match:
            if self._time//match < (self._time + self._dt)//match:
                self._dt = min(match*(1.0+self._time//match) - self._time, self._dt)

        return

    @property
    def dt(self):
        return self._dt

    @property
    def time(self):
        return self._time

    @property
    def time_max(self):
        return self._time_max
