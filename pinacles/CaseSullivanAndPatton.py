from pinacles.Surface_impl import compute_ustar
import numpy as np
import numpy as np
import numba
from mpi4py import MPI
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel
from pinacles import parameters
import pinacles.ThermodynamicsDry_impl as DryThermo


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

def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing Sullivan and Patton Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0, u0=0.0, v0=0.0)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    # Wind is initially uniform
    u.fill(1.0)
    v.fill(0.0)
    w.fill(0.0)

    u -= Ref.u0
    v -= Ref.v0

    shape = s.shape
    perts = np.random.uniform(-0.001, 0.001, (shape[0], shape[1], shape[2]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                if zl[k] < 974.0:
                    t = 300.0
                    t *= exner[k]
                elif 974.0 <= zl[k] and zl[k] < 1074.0:
                    t = 300.0 + (zl[k] - 974.0) * 0.08
                    t *= exner[k]
                else:
                    t = 308.0 + (zl[k] - 1074.0) * 0.003
                    t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i, j, k]
                s[i, j, k] = DryThermo.s(zl[k], t)

    return



        

class SurfaceSullivanAndPatton(Surface.SurfaceBase):
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

        self._theta_flux = 0.24
        self._z0 = 0.1

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc)
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)
        self._tflx = np.zeros_like(self._windspeed_sfc)

        self._Timers.add_timer("SurfaceSullivanAndPatton_update")

        return

    def update(self):

        self._Timers.start_timer("SurfaceSullivanAndPatton_update")
        self.bflux_from_thflux()

        self._bflx_sfc[:, :] = self._buoyancy_flux
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

        u0 = self._Ref.u0
        v0 = self._Ref.v0

        # Get Tendencies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        utsfc = ut[:, :, nh[2]]
        vtsfc = vt[:, :, nh[2]]
        stsfc = st[:, :, nh[2]]

        # Compute the windspeed, friction velocity, and surface stresses
        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, u0, v0, self.gustiness, self._windspeed_sfc
        )
        Surface_impl.compute_ustar_sfc(
            self._windspeed_sfc,
            self._bflx_sfc,
            self._z0,
            self._Grid.dx[2] / 2.0,
            self._ustar_sfc,
        )
        Surface_impl.tau_given_ustar(
            self._ustar_sfc,
            usfc,
            vsfc,
            u0,
            v0,
            self._windspeed_sfc,
            self._taux_sfc,
            self._tauy_sfc,
        )

        # Compute the surface temperature flux
        self._tflx[:, :] = self._theta_flux * exner_edge[nh[2] - 1]

        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tflx, st
        )

        self._Timers.end_timer("SurfaceSullivanAndPatton_update")
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

        ustar = (
            np.sum(self._ustar_sfc[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        ustar = UtilitiesParallel.ScalarAllReduce(ustar)

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

        tflx = np.sum(self._tflx[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]]) / npts
        tflx = UtilitiesParallel.ScalarAllReduce(tflx)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["wind_horizontal"][-1] = windspeed
            timeseries_grp["ustar"][-1] = ustar
            timeseries_grp["taux"][-1] = taux
            timeseries_grp["tauy"][-1] = tauy
            timeseries_grp["tflx"][-1] = tflx
            timeseries_grp["shf"][-1] = (
                tflx * parameters.CPD * self._Ref.rho0_edge[n_halo[2] - 1]
            )
        return
    
    def io_fields2d_update(self, fx):
        
        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        # Compute the height of the maximum gradient in potential temperature
        nh = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        z_edge = self._Grid.z_edge_global
        s = self._ScalarState.get_field('s')


        # PBL Height based on thv
        zi = compute_zi(nh, z_edge, s)

        if fx is not None:
            zi_s = fx.create_dataset(
                "zi_s",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                zi_s.dims[i].attach_scale(fx[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = zi[nh[0]:-nh[0],nh[1]:-nh[1]]
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        
        
        if fx is not None:
            zi_s[:, :] = recv_buffer

        
        return


class ForcingSullivanAndPatton(Forcing.ForcingBase):
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

        self._f = 1.0e-4

        self._ug = np.zeros_like(self._Grid.z_global) + 5.0
        self._vg = np.zeros_like(self._ug)

        self._Timers.add_timer("ForcingSullivanAndPatton_update")

        return

    def update(self):

        self._Timers.start_timer("ForcingSullivanAndPatton_update")

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")

        u0 = self._Ref.u0
        v0 = self._Ref.v0

        Forcing_impl.large_scale_pgf(self._ug, self._vg, self._f, u, v, u0, v0, vt, ut)

        self._Timers.end_timer("ForcingSullivanAndPatton_update")
        return

class SullivanAndPattonDiagnostics:
    
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

        # Add PBL height based on thv
        v = timeseries_grp.createVariable(
            "zi", np.double, dimensions=("time",)
        )
        v.long_name = "Inversion height based on potential temperature"
        v.unts = "m"
        v.standard_name = "Inversion height"

        # Add PBL height based on s
        v = timeseries_grp.createVariable(
            "zi_s", np.double, dimensions=("time",)
        )
        v.long_name = "Inversion height based on static energy"
        v.unts = "m"
        v.standard_name = "Inversion height"

        
        return
        
    def io_update(self, this_grp):

        # Compute the height of the maximum gradient in potential temperature
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

        # PBL Height based on s
        zi = compute_zi(nh, z_edge, s)

        zi_loc = (
                np.sum(zi[nh[0]: -nh[0], nh[1]: -nh[1]])
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
    
    

