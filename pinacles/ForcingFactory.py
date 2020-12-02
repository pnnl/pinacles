from pinacles import CaseSullivanAndPatton
from pinacles import CaseBOMEX
from pinacles import CaseRICO
from pinacles import CaseATEX
from pinacles import CaseTestbed
from pinacles import Forcing

def factory(namelist, Grid, Ref, Microphysics, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):
    casename = namelist['meta']['casename']
    if casename == 'sullivan_and_patton':
        return CaseSullivanAndPatton.ForcingSullivanAndPatton(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'bomex': 
        return CaseBOMEX.ForcingBOMEX(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'rico': 
        return CaseRICO.ForcingRICO(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'atex':
        return CaseATEX.ForcingATEX(namelist, Grid, Ref, Microphysics, VelocityState, ScalarState, DiagnosticState, TimeSteppingController)
    elif casename == 'testbed':
        return CaseTestbed.ForcingTestbed(namelist, Grid, Ref,  VelocityState, ScalarState, DiagnosticState, TimeSteppingController)
    else: 
        return Forcing.ForcingBase(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)