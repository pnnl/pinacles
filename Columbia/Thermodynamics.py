class ThermodynamicsBase: 
    def __init__(self, Grid, ScalarState, DiagnosticState): 
        
        self._Grid = Grid 
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        #Add prognostic fields that are required by all thermodynamic classes 
        self._ScalarState.add_variable('h') #TODO Move this elsewhere 

        #Add diagnostic fields that are required by all thermodynamic classes 
        self._DiagnosticState.add_variable('T')
        self._DiagnosticState.add_variable('alpha')

        return 

from Columbia import ThermodynamicsDry 
def factory(namelist, Grid, ScalarState, DiagnosticState, ): 
    return ThermodynamicsDry.ThermodynamicsDry(Grid, ScalarState, 
                                                DiagnosticState)