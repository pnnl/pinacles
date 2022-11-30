
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
    from pinacles import CaseSullivanAndPatton
    if casename == "sullivan_and_patton":
        return CaseSullivanAndPatton.ForcingSullivanAndPatton(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "bomex":
        from pinacles import CaseBOMEX
        return CaseBOMEX.ForcingBOMEX(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif (casename == "dycoms" or casename == "dycoms_rotated"):
        from pinacles import CaseDYCOMS
        return CaseDYCOMS.ForcingDYCOMS(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "rico":
        from pinacles import CaseRICO   
        return CaseRICO.ForcingRICO(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
    elif casename == "atex":
        from pinacles import CaseATEX
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
        from pinacles import CaseTestbed
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
    elif casename == "real":
        from pinacles import CaseReal
        return CaseReal.ForcingReanalysis(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )

    else:
        return Forcing.ForcingBase(
            namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
        )
