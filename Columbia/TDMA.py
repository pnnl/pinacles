import numpy as np
import numba

@numba.njit()
def Thomas(x, a, b, c):
    """ a generic Thomas algorithm tridiagonal solver. 
    
    Arguments:
        x {[type]} -- [description]
        a {[type]} -- [description]
        b {[type]} -- [description]
        c {[type]} -- [description]
    """
    shape = x.shape
    scratch = np.empty(shape[2], dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            #Upward sweep
            scratch[0] = c[0]/b[0]
            x[i,j,0] = x[i,j,0]/b[0]
            for k in range(1,shape[2]):
                m = 1.0/(b[k] - a[k] * scratch[k-1])
                scratch[k] = c[k] * m
                x[i,j,k] = (x[i,j,k] - a[k] * x[i,j,k-1])*m
            #Downward sweep
            for k in range(shape[2]-2,-1,-1):
                x[i,j,k] = x[i,j,k] - scratch[k] * x[i,j,k+1]
    return
 
@numba.njit()
def PressureThomas(n_halo, dxs, rho0, rho0_edge, kx2, ky2, x, a, c): 
    shape = x.shape 
    scratch = np.empty(shape[2], dtype=np.double)
    b = np.empty(shape[2], dtype=np.double)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            #For each i and j build the diagonal
            b[0] = (rho0[n_halo[2]] * (kx2[i] + ky2[j])
                         - (rho0[n_halo[2]])/dxs[2]/dxs[2])
            for k in range(1,shape[2]-1):
                b[k] = (rho0[k + n_halo[2]] * (kx2[i] + ky2[j])
                     - (rho0_edge[k + n_halo[2]] + rho0_edge[k + n_halo[2] -1])/dxs[2]/dxs[2])
            k = shape[2]-1
            b[k] = (rho0[k + n_halo[2]] * (kx2[i] + ky2[j])
                    - (rho0_edge[k + n_halo[2] -1])/dxs[2]/dxs[2])

            #Now begin the actual algorithm solve
            
            #Upward sweep
            scratch[0] = c[0]/b[0]
            x[i,j,0] = x[i,j,0]/b[0]
            for k in range(1,shape[2]):
                m = 1.0/(b[k] - a[k] * scratch[k-1])
                scratch[k] = c[k] * m
                x[i,j,k] = (x[i,j,k] - a[k] * x[i,j,k-1])*m

            #Downward sweep
            for k in range(shape[2]-2,-1,-1):
                x[i,j,k] = x[i,j,k] - scratch[k] * x[i,j,k+1]

    return 

class PressureTDMA: 
    def __init__(self, Grid, Ref):

        self._Grid = Grid
        self._Ref = Ref

        #Set up the diagonals for the solve
        self._a = None
        self._b = None
        self._c = None


        self._compute_modified_wavenumbers()
        self._set_upperlower_diagonals()

        return 

    def _set_upperlower_diagonals(self):
        self._a = np.zeros(self._Grid.n[2], dtype=np.double)
        self._c = np.zeros(self._Grid.n[2], dtype=np.double)
        return

    def _compute_modified_wavenumbers(self):
        nl = self._Grid.nl
        local_start = self._Grid.local_start
        n_h = self._Grid.n_halo
        dx = self._Grid.dx
        n = self._Grid.n 
        xl = self._Grid.x_local[n_h[0]:-n_h[0]]
        yl = self._Grid.y_local[n_h[1]:-n_h[1]]

        self._kx2 = np.zeros(nl[0], dtype=np.double)
        self._ky2 = np.zeros(nl[1], dtype=np.double)
        #TODO the code below feels a bit like boilerplate
        for ii in range(nl[0]): 
            i = local_start[0] + ii 
            if i <= n[0]/2: 
                xi = np.double(i)
            else: 
                xi = np.double(i - n[0])
            self._kx2[ii] = (2.0 * np.cos((2.0 * np.pi/n[0]) * xi)-2.0)/dx[0]/dx[0]        

        for jj in range(nl[1]): 
            j = local_start[1] + jj
            if j <= n[1]/2: 
                yi = np.double(jj)
            else: 
                yi = np.double(j - n[1])
            self._ky2[jj] = (2.0 * np.cos((2.0 * np.pi/n[1]) * yi)-2.0)/dx[1]/dx[1]     

        #Remove the odd-ball
        if local_start[0] == 0: 
            self._kx2[0] = 0.0
        if local_start[1] == 0: 
            self._ky2[1] == 0.0

        return

    def solve(self, x): 
        n_halo = self._Grid.n_halo 
        dxs = self._Grid.dx
        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge 
        PressureThomas(n_halo, dxs, rho0, rho0_edge, self._kx2, self._ky2, x, self._a, self._c)

        return 