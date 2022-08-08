from pinacles.ScalarAdvection import ScalarWENO


def factory(namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeStepping):
    return ScalarWENO(
        namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeStepping
    )
