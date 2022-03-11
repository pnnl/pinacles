import numba
import numpy as np

class Canopy:
    
    def __init__(self, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController):
        
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._TimeSteppingController = TimeSteppingController
    
        self._DiagnosticState = DiagnosticState
    
        self._DiagnosticState.add_variable('LAI')
    
        
        return
    
    
    @staticmethod
    @numba.njit()
    def fill_lai(x_local, y_local, z_local, LAI):
        shape = LAI.shape
        for i in range(shape[0]):
            if (x_local[i] > 3200-1200.0) and (x_local[i] <= 3200+1200.0):
                for j in range(shape[1]):
                    if (y_local[j]> 3200-1200.0 ) and (y_local[j] <= 3200+1200.0):
                        for k in range(shape[2]):
                            if (z_local[k] <= 50.0):
                                LAI[i,j,k] = 0.05 + 0.29 * (z_local[k]/50.0)

    @staticmethod
    @numba.njit()
    def apply_canopy(u0, v0, u, v, w, LAI, dt, ut, vt, wt):
        shape = u.shape
        
        for i in range(1, shape[0]-1):
            for j in range(1, shape[1]-1):
                for k in range(1, shape[1]-1):
                    _u = u[i,j,k] + u0
                    _v = v[i,j,k] + v0
                    
                    _ut =  0.2 * 0.5 * (LAI[i,j,k] +LAI[i-1,j,k]) * np.sqrt(_u **2.0) *_u 
                    ut[i,j,k] -= np.sign(_u) * min(np.abs(_ut * dt), np.abs(0.8 * _u/dt))
                    _vt = 0.2 * 0.5 * (LAI[i,j,k] +LAI[i,j-1,k]) * np.sqrt(_v **2.0) *_v
                    vt[i,j,k] -= np.sign(_v) *  min(np.abs(_vt * dt),np.abs( 0.8 * _v/dt))
                    _wt = 0.2 * 0.5 * (LAI[i,j,k] +LAI[i,j,k-1]) * np.sqrt(w[i,j,k] **2.0) * w[i,j,k]
                    wt[i,j,k] -= np.sign(w[i,j,k]) * min(np.abs(_wt * dt), np.abs(0.8 * w[i,j,k]/dt))
        
        return

    def update(self):
        
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')
        
        
        ut = self._VelocityState.get_tend('u')
        vt = self._VelocityState.get_tend('v')
        wt = self._VelocityState.get_tend('w')
        
        LAI = self._DiagnosticState.get_field('LAI')
        x_local = self._Grid.x_local
        y_local = self._Grid.y_local 
        z_local = self._Grid.z_local
        
        dt = self._TimeSteppingController.dt
    
    
        self.fill_lai(x_local, y_local, z_local, LAI)
 
    
        u0 = self._Ref.u0
        v0 = self._Ref.v0
        
        self.apply_canopy(u0 , v0, u, v, w, LAI, dt,  ut, vt, wt)
    
        return
    