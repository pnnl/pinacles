import numpy as np
import numba
import time
from Columbia import ScalarAdvection

@numba.njit()
def compute_coeff_tensor(u, v, w, dxi, dt, coeff_tensor):

    #Set up some local storage
    alpha = np.empty((3), dtype=np.double)
    beta = np.empty((3,), dtype=np.double)
    gamma = np.empty((3,), dtype=np.double)

    shape =  coeff_tensor.shape

    # Compute for u
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):

                ut = 0.5 * (u[i,j,k] + u[i-1,j,k])
                vt = 0.5 * (v[i,j,k] + v[i,j-1,k])
                wt = 0.5 * (w[i,j,k] + w[i,j,k-1])

                um = 0.0; vm = 0.0; wm = 0.0
                up = 0.0; vp = 0.0; wp = 0.0

                if ut >= 0: up = 1
                if ut < 0: um = 1

                if vt >= 0: vp = 1
                if vt < 0: vm = 1

                if wt >= 0: wp = 1
                if wt < 0: wm = 1


                alpham1 = up*dxi[0] * np.abs(ut) * dt
                alphap1  = um*dxi[0] * np.abs(ut) * dt

                alpha[0] = alpham1
                alpha[1] = 1.0 - alpham1 - alphap1
                alpha[2] = alphap1

                betam1 = vp*dxi[1] * np.abs(vt) * dt
                betap1  = vm*dxi[1] * np.abs(vt) * dt
                beta[0] = betam1
                beta[1] = 1.0 - betam1 - betap1
                beta[2] = betap1

                gammam1 = wp*dxi[2] * np.abs(wt) * dt
                gammap1  = wm*dxi[2] * np.abs(wt) * dt
                gamma[0] = gammam1
                gamma[1] = 1.0 - gammam1 - gammap1
                gamma[2] = gammap1

                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            coeff_tensor[i,j,k,p,q,r] = alpha[p] * beta[q] * gamma[r]

    return


@numba.njit()
def compute_coeff_tensor_biquad(u, v, w, dxi,dt, coeff_tensor):

    shape =  coeff_tensor.shape

    #Set up some local storage
    alpha = np.empty((3), dtype=np.double)
    beta = np.empty((3,), dtype=np.double)
    gamma = np.empty((3,), dtype=np.double)


    # Compute for u
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):

                xt = -0.5 * (u[i,j,k] + u[i-1,j,k]) * dt
                yt = -0.5 * (v[i,j,k] + v[i,j-1,k]) * dt
                zt = -0.5 * (w[i,j,k] + w[i,j,k-1]) * dt

                alpham1 = 0.5 * (xt*xt*dxi[0]*dxi[0] - xt*dxi[0])
                alphap1  = 0.5 * (xt*xt*dxi[0]*dxi[0] + xt*dxi[0])

                alpha[0] = alpham1
                alpha[1] = 1.0 - alpham1 - alphap1
                alpha[2] = alphap1

                betam1 = 0.5 * (yt*yt*dxi[1]*dxi[1] - yt*dxi[1])
                betap1  = 0.5 * (yt*yt*dxi[1]*dxi[1] + yt*dxi[1])

                beta[0] = betam1
                beta[1] = 1.0 - betam1 - betap1
                beta[2] = betap1

                gammam1 = 0.5 * (zt*zt*dxi[2]*dxi[2] - zt*dxi[2])
                gammap1  = 0.5 * (zt*zt*dxi[2]*dxi[2] + zt*dxi[2])

                gamma[0] = gammam1
                gamma[1] = 1.0 - gammam1 - gammap1
                gamma[2] = gammap1

                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            coeff_tensor[i,j,k,p,q,r] = alpha[p] * beta[q] * gamma[r]

    return


@numba.njit()
def compute_SL_tend(phi, phi_t, dt, coeff_tensor):
    shape = phi.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                phi_old = phi[i,j,k]
                phi_new = 0.0

                for p in range(3):
                    for r in range(3):
                        for s in range(3):
                            phi_new += coeff_tensor[i,j,k,p,r,s]*phi[i+p-1,j+r-1,k+s-1]
                phi_t[i,j,k] += (phi_new-phi_old)/dt


@numba.njit(fastmath=True)
def compute_SL_tend_bounded(phi, phi_t, dt, coeff_tensor1, coeff_tensor2, gamma=0.5):
    shape = phi.shape

    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):
                phi_old = phi[i,j,k]
                phi_new = 0.0

                phi_max = -1e9
                phi_min = 1e9

                first = 0.0
                second = 0.0 
                #print(np.sum(coeff_tensor[i,j,k,:]),np.sum(coeff_tensor2[i,j,k,:]))
                for p in range(3):
                    for r in range(3):
                        for s in range(3):
                            phi_loc = phi[i+p-1,j+r-1,k+s-1]
                            first += coeff_tensor1[i,j,k,p,r,s]*phi_loc
                            second += coeff_tensor2[i,j,k,p,r,s]*phi_loc

                            phi_max = max(phi_max, phi_loc)
                            phi_min = min(phi_min, phi_loc)

                phi_new = (1.0 - gamma)*first  +  gamma * second
                #sgn = np.sign(phi_max - phi_new)
                if phi_new > phi_max or phi_new < phi_min:
                    phi_new = first
                phi_t[i,j,k] += (phi_new-phi_old)/dt



class CTU(ScalarAdvection.ScalarAdvectionBase):

    def __init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping):

        ScalarAdvection.ScalarAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping)
        coeff_shape = (self._Grid.ngrid_local[0], self._Grid.ngrid_local[1], self._Grid.ngrid_local[2],3,3,3)
        self._coeff_tensor2 = np.zeros(coeff_shape, dtype=np.double, order='C')
        self._coeff_tensor1 = np.zeros(coeff_shape, dtype=np.double, order='C')
        print(coeff_shape)

        return


    def update(self):

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        dt = self._TimeStepping.dt
        dxi = self._Grid.dxi


        t0 = time.time() 
        compute_coeff_tensor_biquad(u,v,w,dxi, dt, self._coeff_tensor2)
        compute_coeff_tensor(u,v,w,dxi, dt, self._coeff_tensor1)
        t1 = time.time() 
        print('Computing coefficients', t1 - t0)


        #Now iterate over the scalar variables
        t0 = time.time()
        for var in self._ScalarState.names:

            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)

            compute_SL_tend_bounded(phi, phi_t, dt, self._coeff_tensor1, self._coeff_tensor2)
            #compute_SL_tend(phi, phi_t, dt, self._coeff_tensor1)
        t1  = time.time()
        print('Scalar Update', t1-t0) 
        return