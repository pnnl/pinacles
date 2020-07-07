import numba
import numpy as np
from Columbia.interpolation_impl import interp_weno5


class ScalarAdvectionBase:

    def __init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping):
        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._Ref = Ref
        self._TimeStepping = TimeStepping

        return

    def update(self):

        return

@numba.njit
def weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi.shape
    for i in range(2,phi_shape[0]-3):
        for j in range(2,phi_shape[1]-3):
            for k in range(2,phi_shape[2]-3):
                #First compute x-advection
                if u[i,j,k] >= 0:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(
                                                     phi[i-2,j,k],
                                                     phi[i-1,j,k],
                                                     phi[i,j,k],
                                                     phi[i+1,j,k],
                                                     phi[i+2,j,k])
                else:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(
                                                     phi[i+3,j,k],
                                                     phi[i+2, j, k],
                                                     phi[i+1, j, k],
                                                     phi[i,j,k],
                                                     phi[i-1,j,k])

                #First compute y-advection
                if v[i,j,k] >= 0:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(
                                                     phi[i,j-2,k],
                                                     phi[i,j-1,k],
                                                     phi[i,j,k],
                                                     phi[i,j+1,k],
                                                     phi[i,j+2,k])
                else:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(
                                                     phi[i,j+3,k],
                                                     phi[i, j+2, k],
                                                     phi[i, j+1, k],
                                                     phi[i,j,k],
                                                     phi[i,j-1,k])

                #First compute y-advection
                if w[i,j,k] >= 0:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(
                                                     phi[i,j,k-2],
                                                     phi[i,j,k-1],
                                                     phi[i,j,k],
                                                     phi[i,j,k+1],
                                                     phi[i,j,k+2])
                else:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(
                                                     phi[i,j,k+3],
                                                     phi[i, j, k+2],
                                                     phi[i, j, k+1],
                                                     phi[i,j,k],
                                                     phi[i,j,k-1])
    return

theta = 1.0
@numba.njit
def weno5_advection_flux_limit(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi.shape
    for i in range(2,phi_shape[0]-3):
        for j in range(2,phi_shape[1]-3):
            for k in range(2,phi_shape[2]-3):
                #First compute x-advection
                if u[i,j,k] >= 0:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(phi[i-2,j,k],
                                                     phi[i-1,j,k],
                                                     phi[i,j,k],
                                                     phi[i+1,j,k],
                                                     phi[i+2,j,k])

                    fluxlow = rho0[k] * u[i,j,k] * phi[i,j,k]
                else:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(phi[i+3,j,k],
                                                     phi[i+2, j, k],
                                                     phi[i+1, j, k],
                                                     phi[i,j,k],
                                                     phi[i-1,j,k])
                    fluxlow = rho0[k] * u[i,j,k] * phi[i+1,j,k]

                denom = phi[i+1,j,k] - phi[i,j,k]
                if denom  != 0.0:
                    r = (phi[i,j,k] - phi[i-1,j,k])/denom
                    #flim = np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(0.0, np.minimum(theta * r, np.minimum((1 + r)/2.0, theta)))
                    fluxx[i,j,k] = fluxlow - flim*(fluxlow - fluxx[i,j,k])

                #First compute y-advection
                if v[i,j,k] >= 0:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(phi[i,j-2,k],
                                                     phi[i,j-1,k],
                                                     phi[i,j,k],
                                                     phi[i,j+1,k],
                                                     phi[i,j+2,k])
                    fluxlow = rho0[k] * v[i,j,k] * phi[i,j,k]
                else:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(phi[i,j+3,k],
                                                     phi[i, j+2, k],
                                                     phi[i, j+1, k],
                                                     phi[i,j,k],
                                                     phi[i,j-1,k])
                    fluxlow = rho0[k] * v[i,j,k] * phi[i,j+1,k]
                denom = phi[i,j+1,k] - phi[i,j,k]
                if denom  != 0.0:
                    r = (phi[i,j,k] - phi[i,j-1,k])/denom
                    #flim =  np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(0.0, np.minimum(theta * r, np.minimum((1 + r)/2.0, theta)))
                    fluxy[i,j,k] = fluxlow - flim*(fluxlow - fluxy[i,j,k])


                #First compute y-advection
                if w[i,j,k] >= 0:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(phi[i,j,k-2],
                                                     phi[i,j,k-1],
                                                     phi[i,j,k],
                                                     phi[i,j,k+1],
                                                     phi[i,j,k+2])
                    fluxlow = rho0_edge[k] * w[i,j,k] * phi[i,j,k]
                else:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(phi[i,j,k+3],
                                                     phi[i, j, k+2],
                                                     phi[i, j, k+1],
                                                     phi[i,j,k],
                                                     phi[i,j,k-1])
                    fluxlow = rho0_edge[k] * w[i,j,k] * phi[i,j,k+1]

                denom = phi[i,j,k+1] - phi[i,j,k]
                if denom  != 0.0:
                    r = (phi[i,j,k] - phi[i,j,k-1])/denom
                    #flim =  np.maximum(0.0, np.minimum(1,r)) #minmod
                    flim = np.maximum(0.0, np.minimum(theta * r, np.minimum((1 + r)/2.0, theta)))
                    fluxz[i,j,k] = fluxlow - flim*(fluxlow - fluxz[i,j,k])

    return


@numba.njit
def first_order(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):

    phi_shape = phi.shape
    for i in range(2,phi_shape[0]-3):
        for j in range(2,phi_shape[1]-3):
            for k in range(2,phi_shape[2]-3):
                #First compute x-advection
                if u[i,j,k] >= 0:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * phi[i,j,k]
                else:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * phi[i+1,j,k]

                #First compute y-advection
                if v[i,j,k] >= 0:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * phi[i,j,k]
                else:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * phi[i,j+1,k]

                #First compute y-advection
                if w[i,j,k] >= 0:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * phi[i,j,k]
                else:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * phi[i,j,k+1]

    return

@numba.njit
def flux_divergence(nhalo, idx, idy, idzi, alpha0, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi_t.shape
    #TODO Tighten range of loops
    for i in range(1,phi_shape[0] -1):
        for j in range(1,phi_shape[1] -1):
            for k in range(1,phi_shape[2] - 1):
                phi_t[i,j,k] -= alpha0[k]*((fluxx[i,j,k] - fluxx[i-1,j,k])*idx
                                            + (fluxy[i,j,k] - fluxy[i,j-1,k])*idy
                                            + (fluxz[i,j,k] - fluxz[i,j,k-1])*idzi)
    return


@numba.njit
def flux_divergence_bounded(nhalo, idx, idy, idzi, alpha0, fluxx, fluxy, fluxz,
                            fluxx_low, fluxy_low, fluxz_low, dt, phi, phi_t):
    phi_shape = phi_t.shape
    #TODO Tighten range of loops
    for i in range(1,phi_shape[0] -1):
        for j in range(1,phi_shape[1] -1):
            for k in range(1,phi_shape[2] - 1):

                tend_tmp = -alpha0[k]*((fluxx[i,j,k] - fluxx[i-1,j,k])*idx
                                            + (fluxy[i,j,k] - fluxy[i,j-1,k])*idy
                                            + (fluxz[i,j,k] - fluxz[i,j,k-1])*idzi)

                if phi[i,j,k] + tend_tmp*dt < 0.0:

                    fluxx[i-1,j,k] = fluxx_low[i-1,j,k]
                    fluxx[i,j,k] = fluxx_low[i,j,k]

                    fluxy[i,j-1,k] = fluxy_low[i,j-1,k]
                    fluxy[i,j,k] = fluxy_low[i,j,k]

                    fluxz[i,j,k-1] = fluxz_low[i,j,k-1]
                    fluxz[i,j,k]  = fluxz_low[i,j,k]

    for i in range(1,phi_shape[0] -1):
        for j in range(1,phi_shape[1] -1):
            for k in range(1,phi_shape[2] - 1):
                phi_t[i,j,k] -= alpha0[k]*((fluxx[i,j,k] - fluxx[i-1,j,k])*idx
                                            + (fluxy[i,j,k] - fluxy[i,j-1,k])*idy
                                            + (fluxz[i,j,k] - fluxz[i,j,k-1])*idzi)

    return


class ScalarWENO5(ScalarAdvectionBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping):
        ScalarAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping)
        return

    def update(self):

        # For now we assume that all scalars are advected with this scheme. This doesn't have to
        # remain true.

        #Ge the velocities (No copy done here)
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        #Get the releveant reference variables
        #TODO there is acopy hiding here
        rho0 = self._Ref.rho0
        alpha0 = self._Ref.alpha0
        rho0_edge = self._Ref.rho0_edge

        dt = self._TimeStepping.dt

        #Allocate arrays for storing fluxes
        # TODO define these as class data
        fluxx = np.zeros_like(u)
        fluxy = np.zeros_like(v)
        fluxz = np.zeros_like(w)

        fluxx_low = np.zeros_like(u)
        fluxy_low = np.zeros_like(v)
        fluxz_low = np.zeros_like(w)


        nhalo = self._Grid.n_halo
        #Now iterate over the scalar variables
        for var in self._ScalarState.names:

            #Get a scalar field (No copy done here)
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)

            #Now compute the WENO fluxes
            if 'ff' in var or var in ['qc', 'qr']:
                #TODO This could probably be made faster
                # First compute the higher order fluxes, for now we do it with WENO
                weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)

                #Now compute the lower order upwind fluxes these are used if high-order fluxes 
                # break boundness. 
                first_order(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx_low, fluxy_low, fluxz_low, phi_t)

                # Now insure the that the advection does not violate boundeness of scalars.
                flux_divergence_bounded(nhalo, self._Grid.dxi[0], self._Grid.dxi[1], self._Grid.dxi[2],
                    alpha0, fluxx, fluxy, fluxz, fluxx_low, fluxy_low, fluxz_low, dt, phi, phi_t)
             #   weno5_advection_flux_limit(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)
            else:
                weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)

                #Now compute the flux divergences
                flux_divergence(nhalo, self._Grid.dxi[0], self._Grid.dxi[1], self._Grid.dxi[2],
                    alpha0, fluxx, fluxy, fluxz, phi_t)


        return
