from pinacles.Radiation import RRTMG, RadiationDycoms


from mpi4py import MPI


def factory(
    namelist,
    Grid,
    Ref,
    ScalarState,
    DiagnosticState,
    Surf,
    Micro,
    TimeSteppingController,
):
    try:
        rad_type = namelist["radiation"]["type"]
        if rad_type == "dycoms":
            return RadiationDycoms(
                namelist,
                Grid,
                Ref,
                ScalarState,
                DiagnosticState,
                Micro,
                TimeSteppingController,
            )

    except:
        return RRTMG(
            namelist,
            Grid,
            Ref,
            ScalarState,
            DiagnosticState,
            Surf,
            Micro,
            TimeSteppingController,
        )
