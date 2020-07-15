from Columbia import CaseSullivanAndPatton
from Columbia import CaseBOMEX
from Columbia import CaseRICO
from Columbia import CaseATEX
from Columbia import Forcing

def factory(namelist, Grid, Ref, Microphysics, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):
    casename = namelist['meta']['casename']
    if casename == 'sullivan_and_patton':
        return CaseSullivanAndPatton.ForcingSullivanAndPatton(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    if casename == 'bomex': 
        return CaseBOMEX.ForcingBOMEX(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    if casename == 'rico': 
        return CaseRICO.ForcingRICO(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    if casename == 'atex':
        return CaseATEX.ForcingATEX(namelist, Grid, Ref, Microphysics, VelocityState, ScalarState, DiagnosticState, TimeSteppingController)
    else: 
        return Forcing.ForcingBase(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)