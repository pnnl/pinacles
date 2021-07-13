from pinacles import PressureSolver
from pinacles import PressureSolverNonPeriodic
from pinacles import UtilitiesParallel
import sys


def factory(namelist, Timers, Grid, Ref, VelocityState, DiagnosticState):

    try:
        solver_type = namelist["pressure"]
    except:
        solver_type = "periodic"

    if solver_type == "periodic":
        return PressureSolver.PressureSolver(
            Timers, Grid, Ref, VelocityState, DiagnosticState
        )

    elif solver_type == "open":
        return PressureSolver.PressureSolverNonPeriodic(
            Grid, Ref, VelocityState, DiagnosticState
        )
    else:
        UtilitiesParallel.print_root("No compatible pressure solver available.")
        sys.exit()
