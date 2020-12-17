import numpy as np
import time
from pinacles import MomentumAdvection_impl

class MomentumAdvectionBase:
    def __init__(self, Grid, Ref, ScalarState, VelocityState):
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState

        return

    def update(self):

        return

class MomentumWENO5(MomentumAdvectionBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState):
        MomentumAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState)

    def update(self):

        #Get values from thermodynmic reference state
        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge

        dxi = self._Grid.dxi

        #Retrieve velocities from container
        u = np.copy(self._VelocityState.get_field('u')) + 10.0
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')


        u_t = self._VelocityState.get_tend('u')
        v_t = self._VelocityState.get_tend('v')
        w_t = self._VelocityState.get_tend('w')

        #TODO Move the allocation somewhere else.
        fluxx = np.empty_like(u)
        fluxy = np.empty_like(v)
        fluxz = np.empty_like(w)

        #Here we return the fluxes. We could capture these for output
        #U Component
        MomentumAdvection_impl.u_advection_weno5(rho0, rho0_edge,
            u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.uv_flux_div(dxi[0], dxi[1], dxi[2], alpha0,
            fluxx, fluxy, fluxz, u_t)

        #V Component
        MomentumAdvection_impl.v_advection_weno5(rho0, rho0_edge,
            u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.uv_flux_div(dxi[0], dxi[1], dxi[2], alpha0,
            fluxx, fluxy, fluxz, v_t)

        # W Component
        MomentumAdvection_impl.w_advection_weno5(rho0, rho0_edge
            ,u, v, w, fluxx, fluxy, fluxz)

        MomentumAdvection_impl.w_flux_div(dxi[0], dxi[1], dxi[2], alpha0_edge,
            fluxx, fluxy, fluxz, w_t)

        return

def factory(namelist, Grid, Ref, ScalarState, VelocityState):
    return MomentumWENO5(Grid, Ref, ScalarState, VelocityState)
