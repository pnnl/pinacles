from Columbia.wrf_physics import kessler

class MicroKessler():
    def __init__(self, Grid, Ref, ScalarState, DiagnosticState):
       
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        self._ScalarState.add_variable('qv')
        self._ScalarState.add_variable('qc') 
        self._ScalarState.add_variable('qr')

        return 

    def update(self): 

        T = self._DiagnosticState.get_field('T')
        qv = self._ScalarState.get_field('qv')
        qc = self._ScalarState.get_field('qc')
        qr = self._ScalarState.get_field('qr')

        import sys; sys.exit()


        return