from Columbia import Microphysics
from Columbia import WRF_Micro_Kessler
from Columbia import WRF_Micro_P3
from Columbia import WRF_Micro_SBM

def factory(namelist, Grid, Parallel, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

    try:
        scheme = namelist['microphysics']['scheme']
    except:
        scheme = 'base'

    if scheme == 'base':
        return Microphysics.MicrophysicsBase(Grid,  Parallel, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'kessler':
        return WRF_Micro_Kessler.MicroKessler(Grid, Parallel, Ref,  ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'p3':
        return WRF_Micro_P3.MicroP3(Grid, Parallel, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'sbm':
        return  WRF_Micro_SBM.MicroSBM(Grid, Parallel, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)


    return