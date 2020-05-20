import numpy as np
import numba

@numba.njit()
def compute_fluxes(dxi, rho0, rho0_edge, phi, eddy_diffusivity, fluxx, fluxy, fluxz, phi_t):

    shape = phi.shape

    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            for k in range(shape[2]-1):

                fluxx[i,j,k] = 0.5*(eddy_diffusivity[i,j,k] + eddy_diffusivity[i+1,j,k])*(phi[i+1,j,k] - phi[i,j,k])*dxi[0] * rho0[k]
                fluxy[i,j,k] = 0.5*(eddy_diffusivity[i,j,k] + eddy_diffusivity[i,j+1,k])*(phi[i,j+1,k] - phi[i,j,k])*dxi[1] * rho0[k] 
                fluxz[i,j,k] = 0.5*(eddy_diffusivity[i,j,k] + eddy_diffusivity[i,j,k+1])*(phi[i,j,k+1] - phi[i,j,k])*dxi[1] * rho0_edge[k]

    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):
                phi_t[i,j,k] -= (fluxx[i,j,k] - fluxx[i-1,j,k])*dxi[0]/rho0[k]                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                phi_t[i,j,k] -= (fluxy[i,j,k] - fluxy[i,j-1,k])*dxi[1]/rho0[k]    
                phi_t[i,j,k] -= (fluxz[i,j,k] - fluxz[i,j,k-1])*dxi[2]/rho0[k] 
    return



class ScalarDiffusion:
    def __init__(self, namelist, Grid, Ref, DiagnosticState, ScalarState):

        self._Grid = Grid
        self._Ref = Ref
        self._DiagnosticState = DiagnosticState
        self._ScalarState = ScalarState

        return

    def update(self):

        dxi = self._Grid.dxi

        rho0 = self._Ref.rho0
        alpha0 = self._Ref.alpha0
        rho0_edge = self._Ref.rho0_edge

        eddy_diffusivity = self._DiagnosticState.get_field('eddy_diffusivity')

        fluxx = np.zeros_like(eddy_diffusivity)
        fluxy = np.zeros_like(eddy_diffusivity)
        fluxz = np.zeros_like(eddy_diffusivity)

        for var in self._ScalarState.names:
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)


            compute_fluxes(dxi, rho0, rho0_edge, phi, eddy_diffusivity, fluxx, fluxy, fluxz, phi_t)




        return