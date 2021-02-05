from pinacles import Microphysics
from pinacles import WRF_Micro_Kessler
from pinacles import WRF_Micro_P3
from pinacles import WRF_Micro_SBM

def factory(namelist, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

    try:
        scheme = namelist['microphysics']['scheme']
    except:
        scheme = 'base'

    if scheme == 'base':
        return Microphysics.MicrophysicsBase(Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'kessler':
        return WRF_Micro_Kessler.MicroKessler(Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'p3':
        return WRF_Micro_P3.MicroP3(namelist,Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    elif scheme == 'sbm':
        return  WRF_Micro_SBM.MicroSBM(namelist,Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)


    return
