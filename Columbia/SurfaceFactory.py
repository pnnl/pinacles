from Columbia import CaseSullivanAndPatton
from Columbia import Surface

def factory(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):
    casename = namelist['meta']['casename']
    if casename == 'sullivan_and_patton':
        return CaseSullivanAndPatton.SurfaceSullivanAndPatton(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)
    else:
        return Surface.SurfaceBase(namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState)