from pinacles import SGS
from pinacles import SGSSmagorinsky
from pinacles import SGSTKE

from mpi4py import MPI


def factory(namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
    try:
        sgs_model = namelist["sgs"]["model"]
    except:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Looks like there is no SGS model specified in the namelist!")
        sgs_model = None

    if sgs_model is None:
        return SGS.SGSBase(namelist, Timers, Grid, Ref, VelocityState, DiagnosticState)
    elif sgs_model == "smagorinsky":
        return SGSSmagorinsky.Smagorinsky(
            namelist, Timers, Grid, Ref, VelocityState, DiagnosticState
        )
    elif sgs_model == "tke":
        return SGSTKE.SGSTKE(
            namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState
        )
