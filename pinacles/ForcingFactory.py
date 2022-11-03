from pinacles import CaseSullivanAndPatton
from pinacles import CaseBOMEX
from pinacles import CaseDYCOMS
from pinacles import CaseRICO
from pinacles import CaseATEX
from pinacles import CaseTestbed
from pinacles import Forcing


def factory(
    namelist,
    Timers,
    Grid,
    Ref,
    Microphysics,
    VelocityState,
    ScalarState,
    DiagnosticState,
    TimeSteppingController,
):
    casename = namelist["meta"]["casename"]
    if casename == "sullivan_and_patton":
        return CaseSullivanAndPatton.ForcingSullivanAndPatton(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "bomex":
        return CaseBOMEX.ForcingBOMEX(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif (casename == "dycoms" or casename == "dycoms_rotated"):
        return CaseDYCOMS.ForcingDYCOMS(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "rico":
        return CaseRICO.ForcingRICO(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "atex":
        return CaseATEX.ForcingATEX(
            namelist,
            Timers,
            Grid,
            Ref,
            Microphysics,
            VelocityState,
            ScalarState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif casename == "testbed":
        return CaseTestbed.ForcingTestbed(
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
            TimeSteppingController,
        )
    else:
        return Forcing.ForcingBase(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
