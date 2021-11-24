import numpy as np
from pinacles import TimeStepping_impl as TS_impl
from pinacles import UtilitiesParallel
from mpi4py import MPI


class RungeKuttaBase:
    def __init__(self, namelist, Timers, Grid, PrognosticState):
        self._Timers = Timers
        self._Grid = Grid
        self._PrognosticState = PrognosticState
        self.n_rk_step = 0
        self._rk_step = 0
        self.cfl_target = namelist["time"]["cfl"]
        self._dt = 0.0

        return

    def initialize(self):

        return

    def update(self):

        return


class RungeKutta2ndSSP(RungeKuttaBase):
    def __init__(self, namelist, Timers, Grid, PrognosticState):
        RungeKuttaBase.__init__(self, namelist, Timers, Grid, PrognosticState)
        self.Tn = None
        self.n_rk_step = 2
        self._rk_step = 0

        self._Timers.add_timer("RungeKutta2ndSSP_update")

        return

    def initialize(self):
        self._Tn = np.empty_like(self._PrognosticState.state_array)
        self._Tn[:, :, :, :] = 0.0
        return

    def update(self):
        self._Timers.start_timer("RungeKutta2ndSSP_update")

        present_state = self._PrognosticState.state_array
        present_tend = self._PrognosticState.tend_array

        if self._rk_step == 0:
            np.copyto(self._Tn, present_state)
            TS_impl.rk2ssp_s0(present_state, present_tend, self._dt)
            self._rk_step = 1

        else:
            TS_impl.rk2ssp_s1(self._Tn, present_state, present_tend, self._dt)
            self._rk_step = 0

        self._Timers.end_timer("RungeKutta2ndSSP_update")
        return

    @property
    def dt(self):
        return self._dt


def factory(namelist, Timers, Grid, PrognosticState):
    return RungeKutta2ndSSP(namelist, Timers, Grid, PrognosticState)


class TimeSteppingController:
    def __init__(self, namelist, Grid, DiagnosticState, VelocityState):

        self._restart_atts = []

        self._Grid = Grid
        self._DiagnosticState = DiagnosticState
        self._VelocityState = VelocityState
        self._TimeStepper = []
        self._times_to_match = []
        self._dt = 0.0
        self._restart_atts.append("_dt")
        self._dt_max = 10.0
        self._cfl_target = namelist["time"]["cfl"]

        try:
            self.diff_num_target = namelist["time"]["diff_num"]
        except:
            self.diff_num_target = 0.8

        self._time_max = namelist["time"]["time_max"]
        self._time = 0.0
        self._restart_atts.append("_time")
        return

    def add_timestepper(self, TimeStepper):
        self._TimeStepper.append(TimeStepper)
        return

    def add_timematch(self, delta_time):
        self._times_to_match.append(delta_time)
        return

    def adjust_timestep(self, n_rk_step, end_time):
        if n_rk_step == 0:

            nhalo = self._Grid.n_halo
            dxi = self._Grid.dxi

            # Get velocity components
            u = self._VelocityState.get_field("u")
            v = self._VelocityState.get_field("v")
            w = self._VelocityState.get_field("w")

            eddy_diffusivity = self._DiagnosticState.get_field("eddy_diffusivity")

            cfl_max = 0.0
            cfl_max_local, umax, vmax, wmax = TS_impl.comput_local_cfl_max(
                nhalo, dxi, u, v, w
            )

            umax = UtilitiesParallel.ScalarAllReduce(umax, op=MPI.MAX)
            vmax = UtilitiesParallel.ScalarAllReduce(vmax, op=MPI.MAX)
            wmax = UtilitiesParallel.ScalarAllReduce(wmax, op=MPI.MAX)

            recv_buffer = np.zeros((1,), dtype=np.double)
            MPI.COMM_WORLD.Allreduce(
                np.array([cfl_max_local], dtype=np.double), recv_buffer, op=MPI.MAX
            )
            cfl_max = recv_buffer[0]
            self._cfl_current = self._dt * cfl_max
            dt_from_cfl = self._cfl_target / max(cfl_max, 0.001)

            diff_num_max_local = TS_impl.compute_local_diff_num_max(
                nhalo, dxi, self._dt, eddy_diffusivity
            )
            MPI.COMM_WORLD.Allreduce(
                np.array([diff_num_max_local], dtype=np.double), recv_buffer, op=MPI.MAX
            )
            diff_num_max_time_div_dt = recv_buffer[0] / max(self._dt, 0.001)
            self._diff_num_current = recv_buffer[0]
            self._dt = self.diff_num_target / max(diff_num_max_time_div_dt, 0.0001)

            self._dt = min(self._dt, self._dt_max)
            self._dt = min(self._dt, dt_from_cfl)

            if self._time + self._dt > end_time:
                self._dt = end_time - self._time

            for Stepper in self._TimeStepper:
                Stepper._dt = self._dt

            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Time:", self._time)
                print(
                    "\tCFL Before Adjustment:",
                    np.round(self._cfl_current, 5),
                    "CFL After Adjustment:",
                    np.round(cfl_max * self._dt, 5),
                )
                print(
                    "\tDiffusion Number Before Adjustment:",
                    np.round(self._diff_num_current, 5),
                    "Diffusion Number After Adjustment:",
                    np.round(diff_num_max_time_div_dt * self._dt, 5),
                )
                print(
                    "\tdt:",
                    self._dt,
                )
                print(
                    "\tumax: ",
                    np.round(umax, 5),
                    "\t vmax:",
                    np.round(vmax, 5),
                    "\t wmax:",
                    np.round(wmax, 5),
                )

        return

    def match_time(self):
        # Must be called after dt is computed
        for match in self._times_to_match:
            if self._time // match < (self._time + self._dt) // match:
                self._dt = min(
                    match * (1.0 + self._time // match) - self._time, self._dt
                )

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

    def restart(self, data_dict, **kwargs):

        key = "TimeStepManager"

        for atts in self._restart_atts:
            self.__dict__[atts] = data_dict[key][atts]

        return

    def dump_restart(self, data_dict):

        key = "TimeStepManager"
        data_dict[key] = {}

        for atts in self._restart_atts:
            data_dict[key][atts] = self.__dict__[atts]

        return
