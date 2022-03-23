from pinacles import Microphysics
from pinacles import WRF_Micro_Kessler
from pinacles import WRF_Micro_P3
from pinacles import SAM_Micro_M2005_MA
from pinacles import WRF_Micro_SBM
from pinacles import Microphysics_SA


def factory(
    namelist,
    Timers,
    Grid,
    Ref,
    ScalarState,
    VelocityState,
    DiagnosticState,
    TimeSteppingController,
):

    try:
        scheme = namelist["microphysics"]["scheme"]
    except:
        scheme = "base"

    if scheme == "base":
        return Microphysics.MicrophysicsBase(
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif scheme == "sa":
        return Microphysics_SA.MicroSA(
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif scheme == "kessler":
        return WRF_Micro_Kessler.MicroKessler(
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif scheme == "p3":
        return WRF_Micro_P3.MicroP3(
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif scheme == "m2005_ma":
        return SAM_Micro_M2005_MA.Micro_M2005_MA(
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    elif scheme == "sbm":
        return WRF_Micro_SBM.MicroSBM(
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )

    return
