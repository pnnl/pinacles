from Columbia import ThermodynamicsDry, parameters
import numpy as np 
from scipy.integrate import odeint
import numba 


class ReferenceBase: 
    def __init__(self, Grid, Thermo):
        
        self._Grid = Grid
        self._Thermo = Thermo 
        self._ssfc = None 

        #Allocate memory for reference state profiles 
        self._P0 = np.empty_like(self._Grid.z_global)
        self._rho0 = np.empty_like(self._P0)
        self._alpha0 = np.empty_like(self._P0)
        self._T0 = np.empty_like(self._P0)

        self._P0_edge = np.empty_like(self._P0)
        self._rho0_edge = np.empty_like(self._P0)
        self._alpha0_edge = np.empty_like(self._P0)
        self._T0_edge = np.empty_like(self._P0)

        return 

    def set_surface(self, Psfc=1e5, Tsfc=293.15, qsfc=0.0):
        
        self._Psfc = Psfc 
        self._Tsfc = Tsfc 
        self._qsfc = qsfc 

        return 

    def update_ref_boundaries(self): 

        #Set the ghostpoint values for the cell-center reference profiles 
        self._boundary_update(self._P0)
        self._boundary_update(self._rho0)
        self._boundary_update(self._alpha0)
        self._boundary_update(self._T0)

        #Set the ghostpoitn values for the cell-edge reference profiles 
        self._boundary_update(self._P0_edge)
        self._boundary_update(self._rho0_edge)
        self._boundary_update(self._alpha0_edge)
        self._boundary_update(self._T0_edge)

        return 

    def _boundary_update(self, prof_array):
        n_halo = self._Grid.n_halo[2]
        prof_array[:n_halo] = prof_array[n_halo:2*n_halo][::-1]
        prof_array[-n_halo:] = prof_array[-2*n_halo:-n_halo][::-1]


        return 

    @property
    def Psfc(self): 
        return self._Psfc

    @property
    def Tsfc(self): 
        return self._Tsfc

    @property
    def qsfc(self): 
        return self.qsfc 

    @property
    def ssfc(self): 
        return self._ssfc

    @property
    def p0(self): 
        return np.copy(self._P0)

    @property
    def T0(self):
        return np.copy(self._T0)

    @property
    def rho0(self): 
        return np.copy(self._rho0)

    @property
    def alpha0(self): 
        return np.copy(self._alpha0)


def _integrate_dry(z, lnpsfc, ssfc, n=250): 
    p0_out = np.empty_like(z)
    p0_out[0] = lnpsfc 
    for i in range(z.shape[0]-1): 
        zis  = z[i]
        zie  = z[i+1]
        dz = (zie - zis)/n
        lnpi = p0_out[i]
        for li in range(n): 
            T = ThermodynamicsDry.T(z[i] + dz*li, ssfc)
            dlnp = -parameters.G / (parameters.RD * T)
            lnpi = lnpi + dlnp * dz 
        p0_out[i+1] = lnpi
            
    return np.exp(p0_out)

class ReferenceDry(ReferenceBase): 
    def __init__(self, namelist, Grid, Thermo):
        
        ReferenceBase.__init__(self, Grid, Thermo)

        return 


    def integrate(self): 

        self._ssfc = ThermodynamicsDry.s(0.0, self._Tsfc) 

        lnp_sfc = np.log(self._Psfc)
        nhalo = self._Grid.n_halo[2]
        
        z = np.append([0.0],self._Grid.z_global[nhalo:-nhalo] )

        self._P0[nhalo:-nhalo] = _integrate_dry(z, lnp_sfc, self.ssfc)[1:]
        self._P0_edge[nhalo:-nhalo] = _integrate_dry(self._Grid.z_global_edge[nhalo:-nhalo], lnp_sfc, self.ssfc)
        
        self._T0[nhalo:-nhalo] = ThermodynamicsDry.T(z, self.ssfc)[1:]
        self._T0_edge[nhalo:-nhalo] = ThermodynamicsDry.T(self._Grid.z_global_edge[nhalo:-nhalo], self.ssfc)

        self._rho0[nhalo:-nhalo] = ThermodynamicsDry.rho(self._P0[nhalo:-nhalo], self._T0[nhalo:-nhalo])

        self._rho0_edge[nhalo:-nhalo]=ThermodynamicsDry.rho(self._P0_edge[nhalo:-nhalo], self._T0_edge[nhalo:-nhalo])
        
        self._alpha0[nhalo:-nhalo] = ThermodynamicsDry.alpha(self._P0[nhalo:-nhalo], self._T0[nhalo:-nhalo])
        self._alpha0_edge[nhalo:-nhalo] = ThermodynamicsDry.alpha(self._P0_edge[nhalo:-nhalo], self._T0_edge[nhalo:-nhalo])

        self.update_ref_boundaries() 

        print(self.rho0)
        return 

def factory(namelist, Grid, Thermo): 
    return ReferenceDry(namelist, Grid, Thermo)