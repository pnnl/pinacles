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
def compute_coeff_tensor_biquad(u, v, w, dxi,dt, x_quad, y_quad, z_quad, coeff_tensor):

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


                if xt > 0.0:
                    x_quad[i,j,k] = 1
                else:
                    x_quad[i,j,k] = 0
                if yt > 0.0:
                    y_quad[i,j,k] = 1
                else:
                    y_quad[i,j,k] = 0
                if zt > 0.0:
                    z_quad[i,j,k] = 1
                else:
                    z_quad[i,j,k] = 0


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
def compute_SL_tend_bounded(phi, phi_t, dt, coeff_tensor1, coeff_tensor2, gamma=1.0):
    shape = phi.shape
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):

                #Initialize values
                phi_old = phi[i,j,k]
                phi_new = 0.0
                phi_max = -1e9
                phi_min = 1e9
                first = 0.0
                second = 0.0

                for p in range(3):
                    for r in range(3):
                        for s in range(3):

                            #Grab the value of the scalar
                            phi_loc = phi[i+p-1,j+r-1,k+s-1]

                            #CTU Interpolation stage
                            first += coeff_tensor1[i,j,k,p,r,s]*phi_loc

                            #SL2 Interpolation stage
                            second += coeff_tensor2[i,j,k,p,r,s]*phi_loc

                            #Compute the SL2 stencil bounds
                            phi_max = max(phi_max, phi_loc)
                            phi_min = min(phi_min, phi_loc)

                #Convex combination of the CTU and SL2 schemes
                phi_new = (1.0 - gamma)*first  +  gamma * second


                #Here we apply the correction to ensure boundedness
                if phi_new > phi_max or phi_new < phi_min:
                    phi_new = first

                #Back out the tendencies and apply!
                phi_t[i,j,k] += (phi_new-phi_old)/dt


@numba.njit(fastmath=True)
def compute_SL_tend_bounded_tvd(u, v, w, phi, phi_t, dt, coeff_tensor1, coeff_tensor2, x_quad, y_quad, z_quad, gamma=1.0):
    shape = phi.shape

    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            for k in range(1,shape[2]-1):
                #Initialize values
                phi_old = phi[i,j,k]
                phi_new = 0.0
                phi_max = -1e9
                phi_min = 1e9
                first = 0.0
                second = 0.0

                xq = x_quad[i,j,k]
                yq = y_quad[i,j,k]
                zq = z_quad[i,j,k]

                for p in range(3):
                    for r in range(3):
                        for s in range(3):

                            #Grab the value of the scalar
                            phi_loc = phi[i+p-1,j+r-1,k+s-1]

                            #CTU Interpolation stage
                            first += coeff_tensor1[i,j,k,p,r,s]*phi_loc

                            #SL2 Interpolation stage
                            second += coeff_tensor2[i,j,k,p,r,s]*phi_loc

                #Compute the SL2 stencil bounds
                for p in range(2):
                    for r in range(2):
                        for s in range(2):
                            phi_loc = phi[xq + i+p-1,yq+j+r-1,zq+k+s-1]
                            phi_max = max(phi_max, phi_loc)
                            phi_min = min(phi_min, phi_loc)

                #Convex combination of the CTU and SL2 schemes
                phi_new = (1.0 - gamma)*first  +  gamma * second

                #Here we apply the correction to ensure boundedness
                if phi_new > phi_max or phi_new < phi_min:
                    phi_new = first

                #Back out the tendencies and apply!
                phi_t[i,j,k] += (phi_new-phi_old)/dt

class CTU(ScalarAdvection.ScalarAdvectionBase):

    def __init__(self, namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping):

        ScalarAdvection.ScalarAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState, TimeStepping)
        coeff_shape = (self._Grid.ngrid_local[0], self._Grid.ngrid_local[1], self._Grid.ngrid_local[2],3,3,3)

        #See if any relevant parametes are set in the input file
        try:
            self._gamma = namelist['scalar_advection']['sl2']['gamma']
        except:
            self._gamma = 0.5

        #Initialze two tensors for storing the coefficients
        self._coeff_tensor2 = np.zeros(coeff_shape, dtype=np.double, order='C')
        self._coeff_tensor1 = np.zeros(coeff_shape, dtype=np.double, order='C')


        return


    def update(self):

        # Grab the velocities
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')


        #For storing the quadrants
        x_quad = np.empty(u.shape, dtype=np.int64)
        y_quad = np.empty(v.shape, dtype=np.int64)
        z_quad = np.empty(w.shape, dtype=np.int64)


        # Grab the time an spatial resolution
        dt = self._TimeStepping.dt
        dxi = self._Grid.dxi

        #Compute the coefficient tensors
        compute_coeff_tensor_biquad(u,v,w,dxi, dt, x_quad, y_quad, z_quad, self._coeff_tensor2)
        compute_coeff_tensor(u,v,w,dxi, dt, self._coeff_tensor1)

        #Now iterate over the scalar variables
        for var in self._ScalarState.names:

            #Grab the sclars and tendencies for each field
            phi = self._ScalarState.get_field(var)
            phi_t = self._ScalarState.get_tend(var)

            compute_SL_tend_bounded_tvd(u,v,w,phi, phi_t, dt, self._coeff_tensor1, self._coeff_tensor2, x_quad, y_quad, z_quad, gamma=self._gamma)
            #compute_SL_tend(phi, phi_t, dt, self._coeff_tensor1)

        return