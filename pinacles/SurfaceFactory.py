from Columbia import CaseSullivanAndPatton
from Columbia import CaseBOMEX
from Columbia import CaseStableBubble
from Columbia import CaseRICO
from Columbia import CaseATEX
from Columbia import CaseTestbed
from Columbia import Surface


def factory(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):
    casename = namelist['meta']['casename']
    if casename == 'sullivan_and_patton':
        return CaseSullivanAndPatton.SurfaceSullivanAndPatton(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'stable_bubble': 
        return CaseStableBubble.SurfaceStableBubble(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState) 
    elif casename == 'bomex': 
        return CaseBOMEX.SurfaceBOMEX(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'rico':
        return CaseRICO.SurfaceRICO(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'atex':
        return CaseATEX.SurfaceATEX(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    elif casename == 'testbed':
        return CaseTestbed.SurfaceTestbed(namelist,Grid, Ref, VelocityState,ScalarState,DiagnosticState,TimeSteppingController)
    else:
        return Surface.SurfaceBase(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
