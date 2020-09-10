from Columbia import parameters
class SurfaceBase:

    def __init__(self, namelist, Grid, Ref, VelocityState, ScalarState, DiagnosticState):

        self._name = 'Surface'

        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        self._theta_flux = None
        self._buoyancy_flux = None

        self._lhf_sfc = None
        self._shf_sfc = None
        self._taux_sfc = None
        self._tauy_sfc = None
        self._windspeed_sfc = None
        self.gustiness = 0.1
        self.T_surface = None

        self._z0 = None


        return

    def update(self):

        return

    def bflux_from_thflux(self):
        assert(self._theta_flux is not None)

        nh = self._Grid.n_halo
        nh2 = nh[2]
        self._buoyancy_flux = self._theta_flux  * parameters.G/self._Ref.T0_edge[nh2-1]

        return

    @property
    def name(self):
        return self._name

    def io_initialize(self, rt_grp):
        return

    def io_update(self, rt_grp):
        return