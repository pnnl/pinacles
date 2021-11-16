from pinacles import Microphysics
from pinacles import Microphysics_SA
from pinacles import WRF_Micro_Kessler
from pinacles import WRF_Micro_P3
from pinacles import WRF_Micro_SBM


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
