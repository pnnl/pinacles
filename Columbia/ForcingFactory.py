from Columbia import CaseSullivanAndPatton
from Columbia import CaseBOMEX
from Columbia import Forcing

def factory(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
    casename = namelist['meta']['casename']
    if casename == 'sullivan_and_patton':
        return CaseSullivanAndPatton.ForcingSullivanAndPatton(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    if casename == 'bomex': 
        return CaseBOMEX.ForcingBOMEX(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    else: 
        return Forcing.ForcingBase(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)