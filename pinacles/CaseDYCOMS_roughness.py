import numpy as np
from mpi4py import MPI
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel
import pinacles.ThermodynamicsMoist_impl as MoistThermo
from pinacles.WRF_Micro_Kessler import compute_qvs

def compute_thetal(p, T, ql, exner):
    return (T - parameters.LV * ql / parameters.CPD) / exner


def qv_star(p0, qt, pv):
    return parameters.EPSV * (1.0 - qt) * pv / (p0 - pv)


def pv_star(T):
    Tc = T - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def sat_adjust(p_, thetal_, qt_, exner_):
    """
    Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
    :param p: pressure [Pa]
    :param thetal: liquid water potential temperature  [K]
    :param qt:  total water specific humidity
    :return: T, ql
    """

    t_1 = thetal_ * exner_
    # Compute saturation vapor pressure
    pv_star_1 = pv_star(t_1)
    # Compute saturation mixing ratio
    qs_1 = qv_star(p_, qt_, pv_star_1)

    if qt_ <= qs_1:
        return t_1, 0.0  # Not saturated - return temperature and ql = 0.0
    else:
        ql_1 = qt_ - qs_1  # Get the initial q_l guess
        f_1 = thetal_ - compute_thetal(
            p_, t_1, ql_1, exner_
        )  # Difference in theta_l after computing
        t_2 = (
            t_1 + parameters.LV * ql_1 / parameters.CPD
        )  # Get a second temperature guess and new q_l guess

        pv_star_2 = pv_star(t_2)
        qs_2 = qv_star(p_, qt_, pv_star_2)
        ql_2 = qt_ - qs_2

        # iterate to convergence

        while np.abs(t_2 - t_1) >= 1e-9:
            pv_star_2 = pv_star(t_2)
            qs_2 = qv_star(p_, qt_, pv_star_2)
            ql_2 = qt_ - qs_2
            f_2 = thetal_ - compute_thetal(p_, t_2, ql_2, exner_)
            t_n = t_2 - f_2 * (t_2 - t_1) / (f_2 - f_1)
            t_1 = t_2
            t_2 = t_n
            f_1 = f_2

        return t_2, ql_2


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing DYCOMS Case, Roughness Test Version")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Psfc=1017.8e2, Tsfc=289.76, u0=6.22, v0=-4.8)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")
    qc = ScalarState.get_field("qc")

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    # Wind is uniform initially
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    perts = np.random.uniform(-0.1, 0.1, (shape[0], shape[1], shape[2]))


    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):

                z = zl[k]
                if z < 795.0:
                    thetal = 288.3
                    qt = 9.45
                else:
                    thetal = 295.0 + (z - 795.0) ** (1.0 / 3.0)
                    qt = 5.0 - 3.0 * (1.0 - np.exp(-(z - 795.0) / 500.0))

                qt /= 1000.0
                t, ql = sat_adjust(Ref.p0[k], thetal, qt, exner[k])
                qv[i, j, k] = qt - ql
                qc[i, j, k] = ql

                if zl[k] < 200.0:
                    t += perts[i, j, k]

                s[i, j, k] = MoistThermo.s(zl[k], t, ql, 0.0)

                u[i, j, k] = 3.0 + z * 4.3e-3
                v[i, j, k] = -9.0 + z * 5.6e-3

    u -= Ref.u0
    v -= Ref.v0

    return


class SurfaceDYCOMS_roughness(Surface.SurfaceBase):
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

        # self._shf = 16.0  # W/m^2
        # self._lhf = 93.0  # W/m^2
        # self._ustar = 0.25  # m/s
        # self._theta_surface = 290.0  # K
        # self.T_surface = 289.76

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._qvflx = np.zeros_like(self._windspeed_sfc)
        self._tflx = np.zeros_like(self._windspeed_sfc)
        self._lhf = np.ones_like(self._windspeed_sfc) * 93
        self._shf = np.ones_like(self._windspeed_sfc) * 16.0
        self._Ri = np.zeros_like(self._windspeed_sfc)
        self._N = np.zeros_like(self._windspeed_sfc)
        self._psi_m = np.zeros_like(self._windspeed_sfc)
        self._psi_h = np.zeros_like(self._windspeed_sfc)

        self._cm = np.zeros_like(self._windspeed_sfc)
        self._ch = np.zeros_like(self._windspeed_sfc)
        self._cq = np.zeros_like(self._windspeed_sfc)
        self._z0 = np.zeros_like(self._windspeed_sfc)
        self._ustar = np.zeros_like(self._windspeed_sfc)

        self.T_surface = np.ones_like(self._windspeed_sfc) * 289.76
        self._TSKIN =  np.ones_like(self._windspeed_sfc) * 289.76

        try:
            self._roughness_option = namelist['surface']['roughness']
            # Wave Steepness = Taylor Yelland 2001 with limiter
            
        except:
            self._roughness_option = 'charnock'
        if 'wave' in self._roughness_option: 
            self._Hs = np.ones_like(self._windspeed_sfc) * 2.0
            self._Lw = np.ones_like(self._windspeed_sfc) * 225.0
    
        if self._roughness_option == 'wave_steepness': # Taylor and Yelland 2001
            self._z0_prefactor  = 1200.0
            self._z0_exponent = 4.5
        # elif self._roughness_option == 'wave_age_windsea': #Drennan 2003
        #     self._z0_prefactor = 3.35
        #     self._z0_exponent = 3.4
        elif self._roughness_option == 'wave_age_COARE3.5': # Edson et al 2013
            self._z0_prefactor = 0.09
            self._z0_exponent = 2.0
        elif self._roughness_option == 'wave_age_Lmean': #Sauvage et al 2022
            self._z0_prefactor = 0.39
            self._z0_exponent = 2.6

        self._Timers.add_timer("SurfaceDycoms_roughness_update")

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
        # v = timeseries_grp.createVariable("tflx", np.double, dimensions=("time",))
        # v.long_name = "surface temperature flux"
        # v.units = "K m s^{-2}"
        # v.standard_name = "surface temperature flux"

        v = timeseries_grp.createVariable("shf", np.double, dimensions=("time",))
        v.long_name = "surface sensible heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "shf"

        v = timeseries_grp.createVariable("lhf", np.double, dimensions=("time",))
        v.long_name = "surface latent heat flux"
        v.units = "W m^{-2}"
        v.standard_name = "lhf"

        v = timeseries_grp.createVariable("zrough", np.double, dimensions=("time",))
        v.long_name = "surface roughness length"
        v.units = "m"
        v.standard_name = "zrough"

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


        ustar = (
            np.sum(self._ustar[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        ustar = UtilitiesParallel.ScalarAllReduce(ustar)

        shf = (
            np.sum(self._shf[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        shf = UtilitiesParallel.ScalarAllReduce(shf)

        lhf = (
            np.sum(self._lhf[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        lhf = UtilitiesParallel.ScalarAllReduce(lhf)

        zrough = (
            np.sum(self._z0[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
            / npts
        )
        zrough = UtilitiesParallel.ScalarAllReduce(zrough)

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["wind_horizontal"][-1] = windspeed
            timeseries_grp["ustar"][-1] = ustar
            timeseries_grp["taux"][-1] = taux
            timeseries_grp["tauy"][-1] = tauy
            # timeseries_grp["tflx"][-1] = self._shf / (
            #     parameters.CPD * self._Ref.rho0_edge[n_halo[2] - 1]
            # )
            timeseries_grp["shf"][-1] = shf
            timeseries_grp["lhf"][-1] = lhf
            timeseries_grp["zrough"][-1] = zrough
        return

    def update(self):
        
        self._Timers.start_timer("SurfaceDycoms_roughness_update")
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global 
        zsfc = self._Grid.z_local[nh[2]]

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge
        p0_edge = self._Ref.p0_edge
        exner = self._Ref.exner


        # Get Fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        qv = self._ScalarState.get_field("qv")
        s = self._ScalarState.get_field("s")
        T = self._DiagnosticState.get_field("T")
        N2 = self._DiagnosticState.get_field("bvf")

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
        qvsfc = qv[:, :, nh[2]]
        Tsfc = T[:, :, nh[2]]
        N2sfc = N2[:, :, nh[2]]
        self._N[:, :] = N2sfc

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        # Surface_impl.tau_given_ustar(
        #     self._ustar_sfc,
        #     usfc,
        #     vsfc,
        #     self._Ref.u0,
        #     self._Ref.v0,
        #     self._windspeed_sfc,
        #     self._taux_sfc,
        #     self._tauy_sfc,
        # )


        Surface_impl.compute_surface_layer_Ri_N2_passed(
            nh,
            z_edge[nh[2]] / 2.0,
            self._TSKIN,
            exner_edge[nh[2] - 1],
            p0_edge[nh[2] - 1],
            qvsfc,
            Tsfc,
            exner[nh[2]],
            qvsfc,
            self._windspeed_sfc,
            self._N,
            self._Ri,
        )

        self._qv0 = compute_qvs(self._TSKIN, self._Ref.Psfc)

        self._z0[self._z0 < 0.0002] = 0.0002

        if self._roughness_option == 'wave_steepness': # Taylor and Yelland 2001
            Surface_impl.compute_exchange_coefficients_wave_steepness(
                self._Ri, 
                z_edge[nh[2]] / 2.0, 
                self._z0,
                self._windspeed_sfc, 
                self._cm, 
                self._ch, 
                self._psi_m,
                self._psi_h, 
                self._Hs, 
                self._Lw, 
                self._z0_prefactor,
                self._z0_exponent,
                self._TSKIN,
                Tsfc)

           
        # elif self._roughness_option == 'wave_age_windsea': #Drennan 2003
   
        elif self._roughness_option == 'wave_age_COARE3.5': # Edson et al 2013
            Surface_impl.compute_exchange_coefficients_wave_age(
                self._Ri,
                z_edge[nh[2]] / 2.0,  
                self._z0, 
                self._windspeed_sfc, 
                self._cm, 
                self._ch,
                self._psi_m,
                self._psi_h,
                self._Hs, 
                self._Lw, 
                self._z0_prefactor,
                self._z0_exponent)
            
        elif self._roughness_option == 'wave_age_Lmean': #Sauvage et al 2022
                Surface_impl.compute_exchange_coefficients_wave_age(
                    self._Ri,
                    z_edge[nh[2]] / 2.0,  
                    self._z0, 
                    self._windspeed_sfc, 
                    self._cm, 
                    self._ch,
                    self._psi_m,
                    self._psi_h,
                    self._Hs, 
                    self._Lw, 
                    self._z0_prefactor,
                    self._z0_exponent)
                
        else: # Charnock
        
            Surface_impl.compute_exchange_coefficients_charnock(
                self._Ri,
                z_edge[nh[2]] / 2.0,
                self._z0,
                self._windspeed_sfc,
                self._cm,
                self._ch,
                self._psi_m,
                self._psi_h,
            )


        self._cq[:, :] = self._ch[:, :]


        self._deltas_sfc = ((Tsfc + parameters.G*parameters.ICPD * zsfc) - self._TSKIN)
        self._deltaqv_sfc = (qvsfc - self._qv0)
        self._tflx = -self._ch * self._windspeed_sfc * self._deltas_sfc
        self._qvflx = -self._cq * self._windspeed_sfc * self._deltaqv_sfc


        self._lhf = self._qvflx * parameters.LV * rho0_edge[nh[0] - 1]
        self._shf = self._tflx * parameters.CPD * rho0_edge[nh[0] - 1]

        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        self._ustar = (self._taux_sfc ** 2.0 + self._tauy_sfc ** 2.0) ** (1.0 / 4.0)

        tflx = np.zeros_like(self._taux_sfc) + self._shf / (
            parameters.CPD * self._Ref.rho0_edge[self._Grid.n_halo[2] - 1]
        )
        qv_flx_sf = np.zeros_like(self._taux_sfc) + self._lhf / (
            parameters.LV * self._Ref.rho0_edge[self._Grid.n_halo[2] - 1]
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, tflx, st
        )
        Surface_impl.iles_surface_flux_application(
            1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, qv_flx_sf, qvt
        )

        self._Timers.end_timer("SurfaceDycoms_roughness_update")
        return


class ForcingDYCOMS_roughness(Forcing.ForcingBase):
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
        self._vg = np.zeros_like(self._ug)
        for k in range(zl.shape[0]):
            self._ug[k] = 3.0 + (4.3e-3) * zl[k]
            self._vg[k] = -9.0 + (5.6e-3) * zl[k]

        # Set subsidence
        self._subsidence = np.zeros_like(self._Grid.z_global)

        for k in range(zl.shape[0]):
            self._subsidence[k] = -zl[k] * 3.75e-6

        self._Timers.add_timer("ForcingDycoms_roughness_update")
        return

    def update(self):

        self._Timers.start_timer("ForcingDycoms_roughness_update")
        exner = self._Ref.exner

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # st += (self._heating_rate * exner)[np.newaxis, np.newaxis, :]

        Forcing_impl.large_scale_pgf(
            self._ug, self._vg, self._f, u, v, self._Ref.u0, self._Ref.v0, ut, vt
        )

        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], s, st)
        Forcing_impl.apply_subsidence(self._subsidence, self._Grid.dxi[2], qv, qvt)

        self._Timers.end_timer("ForcingDycoms_roughness_update")
        return
