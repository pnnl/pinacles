from pinacles.Radiation import RRTMG, RadiationDycoms
from mpi4py import MPI


def factory(
    namelist,
    Timers,
    Grid,
    Ref,
    ScalarState,
    DiagnosticState,
    Surf,
    Micro,
    TimeSteppingController,
):
    if "radiation" in namelist:
        try:
            rad_type = namelist["radiation"]["type"]
        except:
            return RRTMG(
                namelist,
                Timers,
                Grid,
                Ref,
                ScalarState,
                DiagnosticState,
                Surf,
                Micro,
                TimeSteppingController,
            )

        if rad_type == "dycoms":
            return RadiationDycoms(
                namelist,
                Timers,
                Grid,
                Ref,
                ScalarState,
                DiagnosticState,
                Micro,
                TimeSteppingController,
            )
        else:
            return RRTMG(
                namelist,
                Timers,
                Grid,
                Ref,
                ScalarState,
                DiagnosticState,
                Surf,
                Micro,
                TimeSteppingController,
            )
    else:
        return DummyRad()


class DummyRad:
    def __init__(self):
        self.frequency = 1e20
        self.name = "DummyRad"

    def update(self, force=False, time_loop=False):
        return

    def init_profiles(self):
        return

    def io_initialize(self, nc_grp):
        return

    def io_update(self, nc_grp):
        return

    def io_fields2d_update(self, nc_grp):
        return

    def restart(self, data_dict):
        return

    def dump_restart(self, data_dict):
        return
