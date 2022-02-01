import numba
import numpy as np


class ThermodynamicsBase:
    def __init__(
        self, Timer, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro
    ):

        self._Timers = Timer
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._VelocityState = VelocityState
        self._Micro = Micro

        # Add prognostic fields that are required by all thermodynamic classes
        self._ScalarState.add_variable(
            "s", units="K", latex_name="s", long_name="static energy"
        )  # TODO Move this elsewhere

        # Add diagnostic fields that are required by all thermodynamic classes
        self._DiagnosticState.add_variable(
            "T", units="K", latex_name="T", long_name="Temperature"
        )
        self._DiagnosticState.add_variable(
            "alpha",
            units="m^3 K^{-1}",
            latex_name="\alpha",
            long_name="Specific Volume",
        )
        self._DiagnosticState.add_variable(
            "buoyancy", units="m s^{-1}", latex_name="b", long_name="buoyancy"
        )

        self._DiagnosticState.add_variable(
            "buoyancy_gradient_mag",
            units="1/s^{-2}",
            latex_name="b",
            long_name="buoyancy",
        )

        self.name = "ThermodynamicsBase"

        return

    def get_qc(self):
        return np.zeros((self._Grid.ngrid_local), dtype=np.double)

    def get_qi(self):
        return np.zeros((self._Grid.ngrid_local), dtype=np.double)

    @staticmethod
    @numba.njit(fastmath=True)
    def compute_buoyancy_gradient(dxi, b, buoyancy_gradient_mag):

        shape = b.shape
        for i in range(1, shape[0]-1):
            for j in range(1, shape[1]-1):
                for k in range(1, shape[2]-1):
                    buoyancy_gradient_mag[i, j, k] = np.sqrt(
                        ((b[i + 1, j, k] - b[i - 1, j, k]) * 0.5 * dxi[0]) ** 2.0
                        + ((b[i, j + 1, k] - b[i, j - 1, k]) * 0.5 * dxi[1]) ** 2.0
                        + ((b[i, j, k + 1] - b[i, j, k - 1]) * 0.5 * dxi[2]) ** 2.0
                    )

        return


from pinacles import ThermodynamicsDry
from pinacles import ThermodynamicsMoist


def factory(
    namelist, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro
):
    try:
        thermo_type = namelist["Thermodynamics"]["type"]
    except:
        thermo_type = "moist"

    if thermo_type == "moist":
        return ThermodynamicsMoist.ThermodynamicsMoist(
            Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro
        )
    else:
        return ThermodynamicsDry.ThermodynamicsDry(
            Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState
        )
