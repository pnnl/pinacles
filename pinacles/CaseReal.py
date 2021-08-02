import numpy as np
import mpi4py as mpi 
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel


class SurfaceReanalysis(Surface.SurfaceBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        Surface.SurfaceBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        return

    def io_initialize(self, rt_grp):

        return

    def io_update(self, rt_grp):

        return

    def update(self):

        return

class ForcingReanalysis(Forcing.ForcingBase):

    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):
        Forcing.ForcingBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )
        
        return

    def update(self):

        return

    