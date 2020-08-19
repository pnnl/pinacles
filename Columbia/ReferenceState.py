from Columbia import ThermodynamicsDry_impl, parameters
import numpy as np
from scipy.integrate import odeint
import numba

class ReferenceBase:
    def __init__(self, Grid):

        self._Grid = Grid
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

        self._exner = None
        self._exner_edge = None

        return

    def set_surface(self, Psfc=1e5, Tsfc=293.15, qsfc=0.0, u0=0.0, v0=0.0):

        self._Psfc = Psfc
        self._Tsfc = Tsfc
        self._qsfc = qsfc
        self._u0 = u0
        self._v0 = v0

        self._u0 = u0
        self._v0 = v0

        return

    def update_ref_boundaries(self):

        #Set the ghostpoint values for the cell-center reference profiles
        self._boundary_update(self._P0)
        self._boundary_update(self._rho0)
        self._boundary_update(self._alpha0)
        self._boundary_update(self._T0)

        #Set the ghostpoitn values for the cell-edge reference profiles
        self._boundary_update_edge(self._P0_edge)
        self._boundary_update_edge(self._rho0_edge)
        self._boundary_update_edge(self._alpha0_edge)
        self._boundary_update_edge(self._T0_edge)

        return

    def _boundary_update(self, prof_array):
        n_halo = self._Grid.n_halo[2]
        prof_array[:n_halo] = prof_array[2 * n_halo - 1:n_halo - 1:-1]
        prof_array[-n_halo:] = prof_array[-n_halo - 2:-2 * n_halo -2:-1]

        return

    def _boundary_update_edge(self, prof_array):
        n_halo = self._Grid.n_halo[2]

        prof_array[:n_halo - 1] = prof_array[2*n_halo - 2:n_halo - 1:-1]
        prof_array[-n_halo:] = prof_array[-2*n_halo-1:-2*n_halo+n_halo-1][::-1]


        return

    def _compute_exner(self):
        self._exner = (self._P0/parameters.P00)**(parameters.KAPPA)
        self._exner_edge = (self._P0/parameters.P00)**(parameters.KAPPA)
        return

    @property
    def u0(self):
        return self._u0

    @property
    def v0(self):
        return self._v0

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
    def p0_edge(self):
        return np.copy(self.p0_edge)

    @property
    def T0(self):
        return np.copy(self._T0)

    @property
    def T0_edge(self):
        return np.copy(self._T0_edge)

    @property
    def rho0(self):
        return np.copy(self._rho0)

    @property
    def rho0_edge(self):
        return np.copy(self._rho0_edge)

    @property
    def alpha0(self):
        return np.copy(self._alpha0)

    @property
    def alpha0_edge(self):
        return np.copy(self._alpha0_edge)

    @property
    def exner(self):
        return np.copy(self._exner)

    @property
    def exner_edge(self):
        return np.copy(self._exner_edge)

def _integrate_dry(z, lnpsfc, ssfc, n=250):
    p0_out = np.empty_like(z)
    p0_out[0] = lnpsfc
    for i in range(z.shape[0]-1):
        zis  = z[i]
        zie  = z[i+1]
        dz = (zie - zis)/n
        lnpi = p0_out[i]
        for li in range(n+1):
            T = ThermodynamicsDry_impl.T(z[i] + dz*li, ssfc)
            dlnp = -parameters.G / (parameters.RD * T)
            lnpi = lnpi + dlnp * dz
        p0_out[i+1] = lnpi

    return np.exp(p0_out)

def _integrate_dry_bouss(z, lnpsfc, ssfc, n=250):
    p0_out = np.empty_like(z)
    p0_out[0] = np.exp(lnpsfc)
    for i in range(z.shape[0]-1):
        zis  = z[i]
        zie  = z[i+1]
        dz = (zie - zis)/n
        lnpi = p0_out[i]
        for li in range(n):
            dlnp = -parameters.G 
            lnpi = lnpi + dlnp * dz
        p0_out[i+1] = lnpi
    return p0_out

class ReferenceDry(ReferenceBase):
    def __init__(self, namelist, Grid):

        ReferenceBase.__init__(self, Grid)

        return

    def integrate(self):

        self._ssfc = ThermodynamicsDry_impl.s(0.0, self._Tsfc)

        lnp_sfc = np.log(self._Psfc)
        nhalo = self._Grid.n_halo[2]

        z = np.append([0.0],self._Grid.z_global[nhalo:-nhalo] )

        #Compute reference pressure profiles
        self._P0[nhalo:-nhalo] = _integrate_dry_bouss(z, lnp_sfc, self.ssfc)[1:]
        self._P0_edge[nhalo-1:-nhalo+1] = _integrate_dry_bouss(self._Grid.z_edge_global[nhalo-1:-nhalo+1], lnp_sfc, self.ssfc)

        #Compute reference temperature profiles
        self._T0[nhalo:-nhalo] = ThermodynamicsDry_impl.T(z, self.ssfc)[1:]
        self._T0_edge[nhalo-1:-nhalo+1] = ThermodynamicsDry_impl.T(self._Grid.z_edge_global[nhalo-1:-nhalo+1], self.ssfc)

        #Cmopute reference density profiles
        self._rho0[nhalo:-nhalo] = ThermodynamicsDry_impl.rho(self._P0[nhalo:-nhalo], self._T0[nhalo:-nhalo])
        self._rho0_edge[nhalo-1:-nhalo+1] = ThermodynamicsDry_impl.rho(self._P0_edge[nhalo-1:-nhalo+1], self._T0_edge[nhalo-1:-nhalo+1])

        #Compute reference specifi volume profiles
        self._alpha0[nhalo:-nhalo] = ThermodynamicsDry_impl.alpha(self._P0[nhalo:-nhalo], self._T0[nhalo:-nhalo])
        self._alpha0_edge[nhalo-1:-nhalo+1]= ThermodynamicsDry_impl.alpha(self._P0_edge[nhalo-1:-nhalo+1], self._T0_edge[nhalo-1:-nhalo+1])

        #Set the ghostpoint for the reference profiles
        self.update_ref_boundaries()

        self._compute_exner()
        return


    def write_stats(self, nc_ref_grp):

        nh = self._Grid.n_halo
        P0 = nc_ref_grp.createVariable('P0', np.double, dimensions=('z'))
        P0[:] = self._P0[nh[2]:-nh[2]]

        P0_edge = nc_ref_grp.createVariable('P0_edge', np.double, dimensions=('z_edge'))
        P0_edge[:] = self._P0_edge[nh[2]-1:-nh[2]]

        T0 = nc_ref_grp.createVariable('T0', np.double, dimensions=('z'))
        T0[:] = self._T0[nh[2]:-nh[2]]

        T0_edge = nc_ref_grp.createVariable('T0_edge', np.double, dimensions=('z_edge'))
        T0_edge[:] = self._T0_edge[nh[2]-1:-nh[2]]

        rho0 = nc_ref_grp.createVariable('rho0', np.double, dimensions=('z'))
        rho0[:] = self._rho0[nh[2]:-nh[2]]

        rho0_edge = nc_ref_grp.createVariable('rho0_edge', np.double, dimensions=('z_edge'))
        rho0_edge[:] = self._rho0_edge[nh[2]-1:-nh[2]]

        alpha0 = nc_ref_grp.createVariable('alpha0', np.double, dimensions=('z'))
        alpha0[:] = self._alpha0[nh[2]:-nh[2]]

        alpha0_edge = nc_ref_grp.createVariable('alpha0_edge', np.double, dimensions=('z_edge'))
        alpha0_edge[:] = self._alpha0_edge[nh[2]-1:-nh[2]]

        exner = nc_ref_grp.createVariable('exner', np.double, dimensions=('z'))
        exner[:] = self._exner[nh[2]:-nh[2]]

        exner_edge = nc_ref_grp.createVariable('exner_edge', np.double, dimensions=('z_edge'))
        exner_edge[:] = self._exner_edge[nh[2]-1:-nh[2]]

        return

def factory(namelist, Grid):
    return ReferenceDry(namelist, Grid)