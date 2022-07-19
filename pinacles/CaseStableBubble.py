import numpy as np
import numba
from mpi4py import MPI
from pinacles import Surface, Surface_impl
from pinacles import parameters
import pinacles.ThermodynamicsDry_impl as DryThermo
import pinacles.UtilitiesParallel as UtilitiesParallel


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing Stable-bubble Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")

    xl = ModelGrid.local_axes[0]
    yl = ModelGrid.local_axes[1]
    zl = ModelGrid.local_axes[2]
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    # Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    # dista = np.zeros_like(u)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt(
                    ((xl[i] / 1000.0 - 25.6) / 4.0) ** 2.0
                    + ((zl[k] / 1000.0 - 3.0) / 2.0) ** 2.0
                )
                t = 300.0 - zl[k] * parameters.G * parameters.ICPD
                if dist <= 1.0:
                    t -= 15.0 * (np.cos(np.pi * dist) + 1.0) / 2.0

                s[i, j, k] = DryThermo.s(zl[k], t)


class SurfaceStableBubble(Surface.SurfaceBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):
        Surface.SurfaceBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self._theta_flux = 0.0
        self._z0 = 0.0
        self.bflux_from_thflux()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc) + self._buoyancy_flux
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)

        self._Timers.add_timer("SurfaceStableBubble_update")

    def update(self):
        # No surface implementation for this case
        pass
