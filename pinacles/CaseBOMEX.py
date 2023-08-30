import numpy as np
from mpi4py import MPI
import numba 
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel
import pinacles.ThermodynamicsDry_impl as DryThermo


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing BOMEX Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Psfc=1015e2, Tsfc=300.4, u0=-8.75, v0=0.0)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    # Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    perts = np.random.uniform(-0.01, 0.01, (shape[0], shape[1], shape[2])) 
    for i in range(shape[0]):
        for j in range(shape[1]):
            u700 = 0
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]
                if z < 520.0:
                    t = 298.7
                    qv[i, j, k] = 17.0 + z * (16.3 - 17.0) / 520.0
                elif z >= 520.0 and z <= 1480.0:
                    t = 298.7 + (z - 520) * (302.4 - 298.7) / (1480.0 - 520.0)
                    qv[i, j, k] = 16.3 + (z - 520.0) * (10.7 - 16.3) / (1480.0 - 520.0)
                elif z > 1480.0 and z <= 2000:
                    t = 302.4 + (z - 1480.0) * (308.2 - 302.4) / (2000.0 - 1480.0)
                    qv[i, j, k] = 10.7 + (z - 1480.0) * (4.2 - 10.7) / (2000.0 - 1480.0)
                elif z > 2000.0:
                    t = 308.2 + (z - 2000.0) * (311.85 - 308.2) / (3000.0 - 2000.0)
                    qv[i, j, k] = 4.2 + (z - 2000.0) * (3.0 - 4.2) / (3000.0 - 2000.0)

                t *= exner[k]
                if zl[k] < 400.0:
                    t += perts[i, j, k]
                s[i, j, k] = DryThermo.s(zl[k], t)

                if z <= 700.0:
                    u[i, j, k] = -8.75
                else:
                    u[i, j, k] = -8.75 + (z - 700.0) * 1.8e-3

    u -= Ref.u0
    v -= Ref.v0

    # u.fill(0.0)
    qv /= 1000.0

    return


class SurfaceBOMEX(Surface.SurfaceBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        Surface.SurfaceBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self._theta_flux = 8.0e-3  # K m/s
        self._qv_flux = 5.2e-5
        self._ustar = 0.28  # m/s
        self._theta_surface = 299.1  # K
        self.T_surface = 299.1
        self.bflux_from_thflux()

        nl = self._Grid.ngrid_local


        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc) + self._buoyancy_flux
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc) + self._ustar
        self._theta_flux_sfc = np.zeros_like(self._windspeed_sfc) + self._theta_flux
        self._qv_flux_sfc = np.zeros_like(self._windspeed_sfc) + self._qv_flux

        self._Timers.add_timer("SurfaceBomex_update")

        return

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp["timeseries"]

        # Add surface windspeed
        v = timeseries_grp.createVariable(
            "wind_horizontal", np.double, dimensions=("time",)
        )
        v.long_name = "Surface layer wind speed"
        v.unts = "m s^{-1}"
        v.standard_name = "surface wind"

        # Add surface stresses
        v = timeseries_grp.createVariable("ustar", np.double, dimensions=("time",))
        v.long_name = "friction velocity"
        v.units = "m s^{-1}"
        v.standard_name = "u^{\star}"

        v = timeseries_grp.createVariable("taux", np.double, dimensions=("time",))
        v.long_name = "surface shear stress x-component"
        v.unts = "m^2 s^{-2}"
        v.standard_name = "\tau{13}"

        v = timeseries_grp.createVariable("tauy", np.double, dimensions=("time",))
        v.long_name = "surface shear stress y-component"
        v.units = "m^2 s^{-2}"
        v.standard_name = "\tau{23}"

        # Add thermodynamic fluxes
        v = timeseries_grp.createVariable("tflx", np.double, dimensions=("time",))
        v.long_name = "surface temperature flux"
        v.units = "K m s^{-2}"
        v.standard_name = "surface temperature flux"

        v = timeseries_grp.createVariable("shf", np.double, dimensions=("time",))
        v.long_name = "surface sensible heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "shf"

        v = timeseries_grp.createVariable("lhf", np.double, dimensions=("time",))
        v.long_name = "surface latent heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "lhf"

        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        windspeed = (
            np.sum(self._windspeed_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        windspeed = UtilitiesParallel.ScalarAllReduce(windspeed)

        taux = (
            np.sum(self._taux_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        taux = UtilitiesParallel.ScalarAllReduce(taux)

        tauy = (
            np.sum(self._taux_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        tauy = UtilitiesParallel.ScalarAllReduce(tauy)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["wind_horizontal"][-1] = windspeed
            timeseries_grp["ustar"][-1] = self._ustar
            timeseries_grp["taux"][-1] = taux
            timeseries_grp["tauy"][-1] = tauy
            timeseries_grp["tflx"][-1] = (
                self._theta_flux * self._Ref.exner_edge[n_halo[2] - 1]
            )
            timeseries_grp["shf"][-1] = (
                self._theta_flux
                * self._Ref.exner_edge[n_halo[2] - 1]
                * parameters.CPD
                * self._Ref.rho0_edge[n_halo[2] - 1]
            )
            timeseries_grp["lhf"][-1] = (
                self._qv_flux * parameters.LV * self._Ref.rho0_edge[n_halo[2] - 1]
            )
        return

    def update(self):

        self._Timers.start_timer("SurfaceBomex_update")
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge

        # Get Fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        # Get Tendnecies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        utsfc = ut[:, :, nh[2]]
        vtsfc = vt[:, :, nh[2]]
        stsfc = st[:, :, nh[2]]
        qvtsfc = qvt[:, :, nh[2]]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        Surface_impl.tau_given_ustar(
            self._ustar_sfc,
            usfc,
            vsfc,
            self._Ref.u0,
            self._Ref.v0,
            self._windspeed_sfc,
            self._taux_sfc,
            self._tauy_sfc,
        )

        tflx = np.zeros_like(self._taux_sfc) + self._theta_flux * exner_edge[nh[2] - 1]
        qv_flx_sf = np.zeros_like(self._taux_sfc) + self._qv_flux
        #Surface_impl.iles_surface_flux_application(
        #   1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._taux_sfc, ut
        #)
        #Surface_impl.iles_surface_flux_application(
        #   1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._tauy_sfc, vt
        #)
        #Surface_impl.iles_surface_flux_application(
        #  1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, tflx, st
        #)
        #Surface_impl.iles_surface_flux_application(
        #  1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, qv_flx_sf, qvt
        #)

        self._Timers.end_timer("SurfaceBomex_update")
        return


class ForcingBOMEX(Forcing.ForcingBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):
        Forcing.ForcingBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self._f = 0.376e-4

        zl = self._Grid.z_local

        # Set Geostrophic wind
        self._ug = np.zeros_like(self._Grid.z_global)
        for k in range(zl.shape[0]):
            self._ug[k] = -10.0 + (1.8e-3) * zl[k]
        self._vg = np.zeros_like(self._ug)

        # Set heating rate
        self._heating_rate = np.zeros_like(self._Grid.z_global)
        self._subsidence = np.zeros_like(self._Grid.z_global)

        # Convert given form of tendencies (theta) to temperature tendency
        for k in range(zl.shape[0]):
            if zl[k] <= 1500.0:
                self._heating_rate[k] = -2.0 / (3600 * 24.0)  # K/s
            if zl[k] > 1500.0:
                self._heating_rate[k] = -2.0 / (3600 * 24.0) + (zl[k] - 1500.0) * (
                    0.0 - -2.0 / (3600 * 24.0)
                ) / (3000.0 - 1500.0)
            if zl[k] <= 1500.0:
                self._subsidence[k] = 0.0 + zl[k] * (-0.65 / 100.0 - 0.0) / (
                    1500.0 - 0.0
                )
            if zl[k] > 1500.0 and zl[k] <= 2100.0:
                self._subsidence[k] = -0.65 / 100 + (zl[k] - 1500.0) * (
                    0.0 - -0.65 / 100.0
                ) / (2100.0 - 1500.0)

        self._Timers.add_timer("ForcingBomex_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingBomex_update")
        exner = self._Ref.exner

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        st += (self._heating_rate * exner)[np.newaxis, np.newaxis, :]

        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], s, st)
        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], qv, qvt)

        self._Timers.end_timer("ForcingBomex_update")
        return
    
@numba.njit()
def compute_zi(nh, z_edge, thv):
    shape = thv.shape
    
    zi = np.zeros((shape[0], shape[1]), dtype=np.double)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            k_grad_max = 0
            grad = -999999.9
            for k in range(nh[2], shape[2]-nh[2]-1):
                tmp_grad = thv[i,j,k+1] - thv[i,j,k]
                if tmp_grad > grad and z_edge[k] >= 500.0:
                    k_grad_max = k
                    grad = tmp_grad
                
            zi[i,j] = z_edge[k_grad_max]   
    
    return zi
    
class BomexDiagnostics:
    
    def __init__(self, Grid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState):
    
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._Micro = Micro
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        
        
    def io_initialize(self, this_grp):
        
        my_rank = MPI.COMM_WORLD.Get_rank()
        
        if my_rank != 0:
            return
        
        timeseries_grp = this_grp["timeseries"]
        profiles_grp = this_grp["profiles"]
        
        
        # Add surface windspeed
        v = timeseries_grp.createVariable(
            "zi", np.double, dimensions=("time",)
        )
        v.long_name = "Inversion height based on potential temperature"
        v.unts = "m"
        v.standard_name = "Inversion height"

        
        # Add surface windspeed
        v = timeseries_grp.createVariable(
            "zi_s", np.double, dimensions=("time",)
        )
        v.long_name = "Inversion height based on static energy"
        v.unts = "m"
        v.standard_name = "Inversion height"

        
        return
        
    def io_update(self, this_grp):

    
        #Compute the height of the maximum gradient in potential temperature
        nh = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        z_edge = self._Grid.z_edge_global
        thv = self._DiagnosticState.get_field('thetav')
        s = self._ScalarState.get_field('s')


        # PBL Height based on thv
        zi = compute_zi(nh, z_edge, thv)
        
        zi_loc = (
            np.sum(zi[nh[0] : -nh[0], nh[1] : -nh[1]])
            / npts
        )
        zi_glob = UtilitiesParallel.ScalarAllReduce(zi_loc)
                
        MPI.COMM_WORLD.barrier()
        my_rank = MPI.COMM_WORLD.Get_rank()
        if my_rank == 0:
            
            timeseries_grp = this_grp["timeseries"]
            profiles_grp = this_grp["profiles"]
        
            timeseries_grp["zi"][-1] = zi_glob
        
        #PBL Height based on s
        zi = compute_zi(nh, z_edge, s)
        
        zi_loc = (
            np.sum(zi[nh[0] : -nh[0], nh[1] : -nh[1]])
            / npts
        )
        zi_glob = UtilitiesParallel.ScalarAllReduce(zi_loc)
                
        MPI.COMM_WORLD.barrier()
        my_rank = MPI.COMM_WORLD.Get_rank()
        if my_rank == 0:
            
            timeseries_grp = this_grp["timeseries"]
            profiles_grp = this_grp["profiles"]
        
            timeseries_grp["zi_s"][-1] = zi_glob
        
        
        return
