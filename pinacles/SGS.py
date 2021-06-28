class SGSBase:
    def __init__(self, namelist, Timers, Grid, Ref, VelocityState, DiagnosticState):

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        return

    def update(self):
        return
