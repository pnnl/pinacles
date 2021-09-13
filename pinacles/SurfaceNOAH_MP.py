import pinacles.Surface as Surface
import pinacles.externals.wrf_noahmp_wrapper.noahmp_via_cffi as NoahMP


class SurfaceNoahMP(Surface.SurfaceBase):

    def __init__(self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState):

        
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

    def initialize(self):
        return super().initialize()

    def update(self):
        return super().update()

    def io_initialize(self, rt_grp):
        return super().io_initialize(rt_grp)

    def io_update(self, rt_grp):
        return super().io_update(rt_grp)

        