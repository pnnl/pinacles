from pinacles.ScalarAdvection import ScalarWENO
from pinacles.SLScalarAdvection import CTU


def factory(namelist, Timers, Grid, Ref, ScalarState, VelocityState, TimeStepping):

    try:
        adv_type = namelist["scalar_advection"]["type"]
    except:
        adv_type = "weno"

    if adv_type == "sl2":
        return CTU(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping)
    else:
        return ScalarWENO(
            namelist, Timers, Grid, Ref, ScalarState, VelocityState, TimeStepping
        )
