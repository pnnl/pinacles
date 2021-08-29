from pinacles import CaseSullivanAndPatton
from pinacles import CaseBOMEX
from pinacles import CaseStableBubble
from pinacles import CaseRICO
from pinacles import CaseATEX
from pinacles import CaseTestbed
from pinacles import Surface
from pinacles import CaseReal


def factory(
    namelist,
    Timers,
    Grid,
    Ref,
    VelocityState,
    ScalarState,
    DiagnosticState,
    TimeSteppingController,
    Ingest,
):
    casename = namelist["meta"]["casename"]
    if casename == "sullivan_and_patton":
        return CaseSullivanAndPatton.SurfaceSullivanAndPatton(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "stable_bubble":
        return CaseStableBubble.SurfaceStableBubble(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "bomex":
        return CaseBOMEX.SurfaceBOMEX(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "rico":
        return CaseRICO.SurfaceRICO(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "atex":
        return CaseATEX.SurfaceATEX(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "testbed":
        return CaseTestbed.SurfaceTestbed(
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif casename == "real":
        return CaseReal.SurfaceReanalysis(
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
            Ingest,
        )
    else:
        return Surface.SurfaceBase(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
