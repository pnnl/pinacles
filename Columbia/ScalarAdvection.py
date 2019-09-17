import numba
import numpy as np

def factory(namelist, Grid, Ref, ScalarState, VelocityState):

    return ScalarWENO5(Grid, Ref, ScalarState, VelocityState)

    return


class ScalarAdvectionBase:

    def __init__(self, Grid, Ref, ScalarState, VelocityState):
        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._Ref = Ref

        return

    def update(self):

        return

@numba.njit
def centered_second(phi, phip1):
    return 0.5*(phi, phip1)

@numba.njit()
def interp_weno5(phim2, phim1,  phi, phip1, phip2):
    p0 = (1.0/3.0)*phim2 - (7.0/6.0)*phim1 + (11.0/6.0)*phi
    p1 = (-1.0/6.0) * phim1 + (5.0/6.0)*phi + (1.0/3.0)*phip1
    p2 = (1.0/3.0) * phi + (5.0/6.0)*phip1 - (1.0/6.0)*phip2

    beta2 = (13.0/12.0 * (phi - 2.0 * phip1 + phip2)*(phi - 2.0 * phip1 + phip2)
                        + 0.25 * (3.0 * phi - 4.0 * phip1 + phip2)*(3.0 * phi - 4.0 * phip1 + phip2))
    beta1 = (13.0/12.0 * (phim1 - 2.0 * phi + phip1)*(phim1 - 2.0 * phi + phip1)
                        + 0.25 * (phim1 - phip1)*(phim1 - phip1))
    beta0 = (13.0/12.0 * (phim2 - 2.0 * phim1 + phi)*(phim2 - 2.0 * phim1 + phi)
                        + 0.25 * (phim2 - 4.0 * phim1 + 3.0 * phi)*(phim2 - 4.0 * phim1 + 3.0 * phi))

    alpha0 = 0.1/((beta0 + 1e-10) * (beta0 + 1e-10))
    alpha1 = 0.6/((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = 0.3/((beta2 + 1e-10) * (beta2 + 1e-10))

    alpha_sum_inv = 1.0/(alpha0 + alpha1 + alpha2)
    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2

@numba.njit
def weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi.shape
    for k in range(nhalo[2],phi_shape[2]-1):
        for j in range(nhalo[1],phi_shape[1]-nhalo[1]):
            for i in range(nhalo[0],phi_shape[0]-nhalo[0]):
                #First compute x-advection
                if u[i,j,k] >= 0:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(phi[i-2,j,k],
                                                     phi[i-1,j,k],
                                                     phi[i,j,k],
                                                     phi[i+1,j,k],
                                                     phi[i+2,j,k])
                else:
                    fluxx[i,j,k] = rho0[k] * u[i,j,k] * interp_weno5(phi[i+3,j,k],
                                                     phi[i+2, j, k],
                                                     phi[i+1, j, k],
                                                     phi[i,j,k],
                                                     phi[i-1,j,k])

                #First compute y-advection
                if v[i,j,k] >= 0:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(phi[i,j-2,k],
                                                     phi[i,j-1,k],
                                                     phi[i,j,k],
                                                     phi[i,j+1,k],
                                                     phi[i,j+2,k])
                else:
                    fluxy[i,j,k] = rho0[k] * v[i,j,k] * interp_weno5(phi[i,j+3,k],
                                                     phi[i, j+2, k],
                                                     phi[i, j+1, k],
                                                     phi[i,j,k],
                                                     phi[i,j-1,k])

                #First compute y-advection
                if w[i,j,k] >= 0:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(phi[i,j,k-2],
                                                     phi[i,j,k-2],
                                                     phi[i,j,k],
                                                     phi[i,j,k+1],
                                                     phi[i,j,k+2])
                else:
                    fluxz[i,j,k] = rho0_edge[k] * w[i,j,k] * interp_weno5(phi[i,j,k+3],
                                                     phi[i, j, k+2],
                                                     phi[i, j, k+1],
                                                     phi[i,j,k],
                                                     phi[i,j,k-1])

    return

@numba.njit
def flux_divergence(nhalo, idx, idy, idzi, alpha0, fluxx, fluxy, fluxz, phi_t):
    phi_shape = phi_t.shape
    #TODO Tighten range of loops
    for k in range(nhalo[2],phi_shape[2] - nhalo[2]):
        for j in range(nhalo[1],phi_shape[1] - nhalo[1]):
            for i in range(nhalo[0],phi_shape[0] - nhalo[0]):
                phi_t[i,j,k] -= alpha0[k]*((fluxx[i,j,k] - fluxx[i-1,j,k])*idx
                                            + (fluxy[i,j,k] - fluxy[i,j-1,k])*idy
                                            + (fluxz[i,j,k] - fluxz[i,j,k-1])*idzi)

    return

class ScalarWENO5(ScalarAdvectionBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState):
        ScalarAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState)
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

        #Allocate arrays for storing fluxes
        # TODO define these as class data
        fluxx = np.zeros_like(u)
        fluxy = np.zeros_like(v)
        fluxz = np.zeros_like(w)

        nhalo = self._Grid.n_halo
        #Now iterate over the scalar variables
        for var in self._ScalarState.names:
            #Get a scalar field (No copy done here)
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)

            #Now compute the WENO fluxes
            weno5_advection(nhalo, rho0, rho0_edge, u, v, w, phi, fluxx, fluxy, fluxz, phi_t)

            

            #Now compute the flux divergences
            flux_divergence(nhalo, self._Grid.dx[0], self._Grid.dx[1], self._Grid.dx[2],
                alpha0, fluxx, fluxy, fluxz, phi_t)


        return
