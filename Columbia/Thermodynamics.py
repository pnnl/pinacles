class ThermodynamicsBase: 
    def __init__(self, Grid, PrognosticState, DiagnosticState): 
        
        self._Grid = Grid 
        self._PrognosticState = PrognosticState
        self._DiagnosticState = DiagnosticState

        #Add prognostic fields that are required by all thermodynamic classes 
        self._PrognosticState.add_variable('h') #TODO Move this elsewhere 

        #Add diagnostic fields that are required by all thermodynamic classes 
        self._DiagnosticState.add_variable('T')
        self._DiagnosticState.add_variable('alpha')

        return 

from Columbia import ThermodynamicsDry 
def factory(namelist, Grid, PrognosticState, DiagnosticState, ): 
    return ThermodynamicsDry.ThermodynamicsDry(Grid, PrognosticState, 
                                                DiagnosticState)