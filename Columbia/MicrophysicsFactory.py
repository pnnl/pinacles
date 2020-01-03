from Columbia import Microphysics
from Columbia import WRF_Micro_Kessler

def factory(namelist, Grid, Ref, ScalarState, DiagnosticState, TimeSteppingController):

    try:
        scheme = namelist['microphysics']['scheme']
    except:
        scheme = 'base'

    if scheme == 'base':
        return Microphysics.MicrophysicsBase(Grid, Ref, ScalarState, DiagnosticState, TimeSteppingController)
    elif scheme == 'kessler':
        return WRF_Micro_Kessler.MicroKessler(Grid, Ref, ScalarState, DiagnosticState, TimeSteppingController)

    return