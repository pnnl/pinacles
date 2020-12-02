from pinacles.Radiation import RRTMG


from mpi4py import MPI

def factory(namelist, Grid, Ref, ScalarState, DiagnosticState, Surf, TimeSteppingController):
    return RRTMG(namelist, Grid, Ref, ScalarState, DiagnosticState, Surf, TimeSteppingController)
