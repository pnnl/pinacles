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

        return

    def get_qc(self):
        return np.zeros((self._Grid.ngrid_local), dtype=np.double)

    def get_qi(self):
        return np.zeros((self._Grid.ngrid_local), dtype=np.double)


from pinacles import ThermodynamicsDry_Thetav
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
    elif thermo_type == 'dry_thetav':
        return ThermodynamicsDry_Thetav.ThermodynamicsDry_Thetav(
            Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState
        )
    else:
        return ThermodynamicsDry.ThermodynamicsDry(
            Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState
        )
