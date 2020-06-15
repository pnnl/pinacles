from Columbia import SGS
from Columbia import SGSSmagorinsky

from mpi4py import MPI

def factory(namelist, Grid, Parallel, Ref, VelocityState, DiagnosticState):
    try:
        sgs_model = namelist['sgs']['model']
    except:
        if Parallel.rank == 0:
            print('Looks like there is no SGS model specified in the namelist!')
        sgs_model = None


    if sgs_model is None:
        return SGS.SGSBase(namelist, Grid, Ref, VelocityState, DiagnosticState)
    elif sgs_model == 'smagorinsky':
        return SGSSmagorinsky.Smagorinsky(namelist, Grid, Ref, VelocityState, DiagnosticState)