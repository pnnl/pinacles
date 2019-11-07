class ThermodynamicsBase:
    def __init__(self, Grid, Ref,  ScalarState, VelocityState, DiagnosticState): 

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._VelocityState = VelocityState

        #Add prognostic fields that are required by all thermodynamic classes
        self._ScalarState.add_variable('s') #TODO Move this elsewhere

        #Add diagnostic fields that are required by all thermodynamic classes
        self._DiagnosticState.add_variable('T')
        self._DiagnosticState.add_variable('alpha')
        self._DiagnosticState.add_variable('buoyancy')

        return

from Columbia import ThermodynamicsDry
def factory(namelist, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
    return ThermodynamicsDry.ThermodynamicsDry(Grid, Ref, ScalarState, VelocityState,
                                                DiagnosticState)
