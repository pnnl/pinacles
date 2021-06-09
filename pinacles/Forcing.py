class ForcingBase:
    def __init__(
        self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        return

    def update(self):
        return

    def restart(self, data_dict):
        return

    def dump_restart(self, data_dict):
        return
