from pinacles import CaseSullivanAndPatton
from pinacles import CaseBOMEX
from pinacles import CaseStableBubble
from pinacles import CaseRICO
from pinacles import CaseATEX
from pinacles import CaseTestbed
from pinacles import Surface


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
