import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import Surface, Surface_impl, Forcing, Forcing_impl
from pinacles import parameters
from scipy import interpolate
from pinacles import UtilitiesParallel
import pinacles.ThermodynamicsMoist_impl as MoistThermo
import sys

def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing testbed Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    file = namelist["testbed"]["input_filepath"]
    try:
        micro_scheme = namelist["microphysics"]["scheme"]
    except:
        micro_scheme = "base"
    try:
        sbm_init_type = namelist["testbed"]["sbm_init_type"]
        # options are 'assume_distribution', 'all_vapor',
        if sbm_init_type not in ["assume_distribution", "all_vapor"]:
            UtilitiesParallel.print_root(
                " Warning: sbm_init_type is unknown. Defaulting to all_vapor"
            )
    except:
        sbm_init_type = "all_vapor"
    try:
        sbm_init_nc = namelist["testbed"]["sbm_init_nc"]
    except:
        sbm_init_nc = 55.0e6
    
    try:
        ref_init_type = namelist["testbed"]["reference_init_type"]
    except:
        UtilitiesParallel.print_root('must specify init_type')
        sys.exit()
        # ref_init_type = 'integrate'


    try:
        ref_init_type = namelist["testbed"]["reference_init_type"]
    except:
        UtilitiesParallel.print_root("must specify init_type")
        sys.exit()
        # ref_init_type = 'integrate'

    data = nc.Dataset(file, "r")
    try:
        init_data = data.groups["initialization_sonde"]
        UtilitiesParallel.print_root("\t \t Initializing from the sonde profile.")
        # init_data = data.groups['initialization_varanal]
        # print('Initializing from the analysis profile')
    except:
        init_data = data.groups["initialization"]

    psfc = init_data.variables["surface_pressure"][0]
    if psfc < 1.0e4:
        psfc *= 100.0  # Convert from hPa to Pa
    tsfc = init_data.variables["surface_temperature"][0]
    u0 = init_data.variables["reference_u0"][0]
    v0 = init_data.variables["reference_v0"][0]

    Ref.set_surface(Psfc=psfc, Tsfc=tsfc, u0=u0, v0=v0)
    if ref_init_type == "integrate":
        Ref.integrate()
    elif ref_init_type == "specify":
        p = init_data["pressure"]
        tdry = init_data["temperature"]
        alt = init_data.variables["z"][:]
        qv = init_data.variables["vapor_mixing_ratio"]
        Ref.specify(alt, p, tdry, qv)

    else:
        UtilitiesParallel.print_root("Unrecognized init type, exiting.")
        sys.exit()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")
    qc = ScalarState.get_field("qc")

    zl = ModelGrid.z_local

    init_z = init_data.variables["z"][:]

    raw_qv = init_data.variables["vapor_mixing_ratio"][:]
    raw_u = init_data.variables["u"][:]
    raw_v = init_data.variables["v"][:]

    init_var_from_sounding(raw_u, init_z, zl, u)
    init_var_from_sounding(raw_v, init_z, zl, v)
    init_var_from_sounding(raw_qv, init_z, zl, qv)

    u -= Ref.u0
    v -= Ref.v0

    try:
        raw_clwc = init_data.variables["cloud_water_content"][:]
        init_qc = True
        UtilitiesParallel.print_root("\t \t Initialization of qc is true")
    except:
        init_qc = False
        qc.fill(0.0)
        UtilitiesParallel.print_root("\t \t Initialization of qc is false")

    if init_qc:
        init_var_from_sounding(raw_clwc, init_z, zl, qc)
        shape = qc.shape
        if micro_scheme == "sbm" and sbm_init_type == "assume_distribution":
            UtilitiesParallel.print_root(
                "\t \t Initializing cloud liquid to bins with assumptions that number of bins = 33!"
            )
            nbins = 33
            # Assuming a width parameter of the initial distribution, this could be added as a namelist parameter
            sig1 = 1.2

            # ff1i1= ScalarState.get_field('ff1i1')
            ff_list = []
            for ibin in range(nbins):
                ff_list.append(ScalarState.get_field("ff1i" + str(np.int(ibin + 1))))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        qc_sum = 0.0
                        if qc[i, j, k] > 0.0:

                            f, xl = get_lognormal_dist(
                                nbins, qc[i, j, k], sbm_init_nc, sig1
                            )

                            for ibin in range(nbins):

                                ff_list[ibin][i, j, k] = (
                                    f[ibin] * 1e6 * xl[ibin] / Ref.rho0[k]
                                )  # /col* col
                                qc_sum += ff_list[ibin][i, j, k]
                        # qc[i,j,k] = qc[i,j,k]/Ref.rho0[k]
                        qc[i, j, k] = qc_sum
            UtilitiesParallel.print_root("Max on Rank of Bins")
            for ff in ff_list:
                max_on_rank = np.amax(ff)
                UtilitiesParallel.print_root(str(max_on_rank))

        elif micro_scheme == "sbm" and sbm_init_type == "all_vapor":
            UtilitiesParallel.print_root(
                "\t \t SBM initialization with cloud water dumped into vapor."
            )
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        qv[i, j, k] += qc[i, j, k] / Ref.rho0[k]
                        qc[i, j, k] = 0.0
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        qc[i, j, k] = qc[i, j, k] / Ref.rho0[k]

    try:
        raw_temperature = init_data.variables["temperature"][:]
        init_var_from_sounding(raw_temperature, init_z, zl, s)
        # hardwire for now, could make inputs in namelist or data file
        pert_amp = 0.1
        pert_max_height = 200.0
        shape = s.shape
        perts = np.random.uniform(
            pert_amp * -1.0, pert_amp, (shape[0], shape[1], shape[2])
        )
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    t = s[i, j, k]
                    if zl[k] < pert_max_height:
                        t += perts[i, j, k]
                    s[i, j, k] = MoistThermo.s(zl[k], t, qc[i, j, k], 0.0)
    except:
        raw_theta = init_data.variables["potential_temperature"][:]
        init_var_from_sounding(raw_theta, init_z, zl, s)
        # hardwire for now, could make inputs in namelist or data file
        pert_amp = 0.1
        pert_max_height = 200.0
        shape = s.shape
        perts = np.random.uniform(
            pert_amp * -1.0, pert_amp, (shape[0], shape[1], shape[2])
        )
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    t = s[i, j, k] * Ref.exner[k]
                    if zl[k] < pert_max_height:
                        t += perts[i, j, k]
                    s[i, j, k] = MoistThermo.s(zl[k], t, qc[i, j, k], 0.0)

    return


def init_var_from_sounding(profile_data, profile_z, grid_z, var3d):
    var3d.fill(0.0)
    grid_profile = interpolate.interp1d(
        profile_z, profile_data, fill_value="extrapolate", assume_sorted=True
    )(grid_z)
    var3d += grid_profile[np.newaxis, np.newaxis, :]
    return


def get_lognormal_dist(nbins, qc, nc_m3, sig1):
    # SMALLEST BIN SIZE IN SBM FOR 33 BINS
    xl0 = 3.35e-8 * 1e-6
    xl = np.zeros(nbins, dtype=np.double)
    xl[0] = xl0
    for i in np.arange(1, nbins):
        xl[i] = 2 * xl[i - 1]
    rl = np.zeros(nbins)
    rhow = 1000.0
    for i in np.arange(nbins):
        rl[i] = (0.75 * xl[i] / np.pi / rhow) ** (1.0 / 3.0)
    rl_cm = rl * 100.0
    ccncon1 = nc_m3 * 1e-6
    mass_mean_kg = qc / (nc_m3)
    radius_mean_m = (0.75 * mass_mean_kg / np.pi / rhow) ** (1.0 / 3.0)
    radius_mean1 = radius_mean_m * 100 * 0.95
    f = np.zeros(nbins)
    arg11 = ccncon1 / (np.sqrt(2.0 * np.pi) * np.log(sig1))
    dNbydlogR_norm1 = 0.0
    for kr in np.arange(nbins - 1, -1, -1):
        arg12 = (np.log(rl_cm[kr] / radius_mean1)) ** 2.0
        arg13 = 2.0 * ((np.log(sig1)) ** 2.0)
        dNbydlogR_norm1 = arg11 * np.exp(-arg12 / arg13) * (np.log(2.0) / 3.0)
        f[kr] = dNbydlogR_norm1
    return f, xl


"""
CK: Here I am starting with the simplest case and assuming the start time of the forcing
files is the same as the simulation start time. This can easily be revisited, and made 
more sophisticated when/if needed

Assume heat fluxes are given
"""


class SurfaceTestbed(Surface.SurfaceBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        VelocityState,
        ScalarState,
        DiagnosticState,
        TimeSteppingController,
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

        self._TimeSteppingController = TimeSteppingController

        file = namelist["testbed"]["input_filepath"]
        data = nc.Dataset(file, "r")
        surface_data = data.groups["surface"]
        self._forcing_times = surface_data.variables["times"][:]
        self._forcing_shf = surface_data.variables["sensible_heat_flux"][:]
        self._forcing_lhf = surface_data.variables["latent_heat_flux"][:]

        """different testbed cases have different availability of surface temperature information. Here trying to
        distinguish when a skin temperature value is directly available, in contrast to backing a temperature out from
        surface LW flux data (as done for the ENA varanal forcing). This would be an area for future clean up"""
        try:
            self._forcing_skintemp = surface_data.variables["skin_temperature"][:]
        except:
            self._forcing_skintemp = surface_data.variables["surface_temperature"][:]
            UtilitiesParallel.print_root(
                "\t \t surface skin temp inferred from LW radiative fluxes"
            )
        """allow for multiplicative increase/decrease of friction velocity values without having to recreate the input file."""
        try:
            ustar_factor = namelist["testbed"]["ustar_factor"]
        except:
            ustar_factor = 1.0
            UtilitiesParallel.print_root(
                "\t \t Using ustar directly as given in the forcing file"
            )

        self._forcing_ustar = (
            surface_data.variables["friction_velocity"][:] * ustar_factor
        )
        # Read off other variables needed for radiation..?

        data.close()

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc)
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)

        # Use these to store the fluxes for output
        self._shf = self._forcing_shf[0]
        self._lhf = self._forcing_lhf[0]
        self._ustar = self._forcing_ustar[0]

        self._z0 = 0.0

        self._Timers.add_timer("SurfaceTestbed_update")
        return

    def io_initialize(self, rt_grp):

        timeseries_grp = rt_grp["timeseries"]

        # Add thermodynamic fluxes
        timeseries_grp.createVariable("shf", np.double, dimensions=("time",))
        timeseries_grp.createVariable("lhf", np.double, dimensions=("time",))
        timeseries_grp.createVariable("ustar", np.double, dimensions=("time",))
        timeseries_grp.createVariable("z0", np.double, dimensions=("time",))
        return

    def io_update(self, rt_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            timeseries_grp = rt_grp["timeseries"]

            timeseries_grp["shf"][-1] = self._shf
            timeseries_grp["lhf"][-1] = self._lhf

            timeseries_grp["ustar"][-1] = self._ustar
            timeseries_grp["z0"][-1] = self._z0

        return

    def update(self):
        current_time = self._TimeSteppingController.time

        # Interpolate to the current time
        shf_interp = interpolate.interp1d(
            self._forcing_times,
            self._forcing_shf,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        lhf_interp = interpolate.interp1d(
            self._forcing_times,
            self._forcing_lhf,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        ustar_interp = interpolate.interp1d(
            self._forcing_times,
            self._forcing_ustar,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        self.T_surface = interpolate.interp1d(
            self._forcing_times,
            self._forcing_skintemp,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        # Get grid & reference profile info
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        exner_edge = self._Ref.exner_edge

        # Get fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        # Get tendencies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]

        # Compute the surface stress & apply it
        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )
        # OPTION 1-- u* from the input file
        # -----------------------------------------------------------
        self._ustar_sfc[:, :] = ustar_interp
        self._ustar = ustar_interp

        # --------------------------------------------------------
        # OPTION 2-- z0 from windspeed (this is rough, should be improved), then get u*
        # Using the mean windspeed rather than pointwise, should also be interpolated to 10m
        # Expression from ARPS based on anderson 1993, we also used this in pycles
        # wspd_local = np.sum(self._windspeed_sfc[nh[0]:-nh[0], nh[1]:-nh[1]])/(self._Grid.n[0] * self._Grid.n[1])
        # wspd_mean = UtilitiesParallel.ScalarAllReduce(wspd_local)
        # self._z0 =self._compute_z0(self._Grid.dx[2]/2.0,wspd_mean)
        # self._bflx_sfc[:,:] = (parameters.G * self._Ref.alpha0[nh[2]] * parameters.ICPD/self.T_surface
        #                         * (shf_interp + (parameters.EPSVI -1.0) * parameters.CPD * self.T_surface * lhf_interp / parameters.LV))
        # Surface_impl.compute_ustar_sfc(self._windspeed_sfc, self._bflx_sfc, self._z0, self._Grid.dx[2]/2.0, self._ustar_sfc)
        # ustar_local = np.sum(self._ustar_sfc[nh[0]:-nh[0], nh[1]:-nh[1]])/(self._Grid.n[0] * self._Grid.n[1])
        # self._ustar = UtilitiesParallel.ScalarAllReduce(ustar_local)

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
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, self._taux_sfc, ut
        )
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, self._tauy_sfc, vt
        )

        # Apply the heat fluxes
        s_flx_sf = (
            np.zeros_like(self._taux_sfc)
            + shf_interp * alpha0_edge[nh[2] - 1] / parameters.CPD
        )
        qv_flx_sf = (
            np.zeros_like(self._taux_sfc)
            + lhf_interp * alpha0_edge[nh[2] - 1] / parameters.LV
        )
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, s_flx_sf, st
        )
        Surface_impl.surface_flux_application(
            dxi2, nh, alpha0, alpha0_edge, qv_flx_sf, qvt
        )

        # Store the surface fluxes for output
        self._shf = shf_interp
        self._lhf = lhf_interp
        return

    def _compute_z0(self, z1, wspd):
        kappa = 0.4
        c1 = (0.4 + 0.079 * wspd) / 1000.0
        return z1 * np.exp(-kappa / np.sqrt(c1))


"""
Note regarding set-up of original LASSO cases (e.g. HISCALE tested case)
These simulations use the initial sounding of horizontal winds as the geostrophic winds.
Here, I require the specification of geostrophic winds to be made explicitly in the 
large-scale forcing group of input file 
"""


class ForcingTestbed(Forcing.ForcingBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        VelocityState,
        ScalarState,
        DiagnosticState,
        TimeSteppingController,
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
        self._TimeSteppingController = TimeSteppingController

        file = namelist["testbed"]["input_filepath"]
        self._momentum_forcing_method = namelist["testbed"]["momentum_forcing"]
        # Options: relaxation, mean_relaxation, geostrophic, none
        self._subsidence_forcing_method = namelist["testbed"]["subsidence_forcing"]
        # Options: vertical tendency, subsidence, fixed_subsidence
        # Vertical tendency: provide time-height profiles for qv and temperature (not theta)
        # Subsidence: provide time-height profile of subsidence velocity and tendencies are computed
        # Fixed_subsidence: provide a large scale divergence D and get subsidence velocity as -D * z (constant in time)
        if self._subsidence_forcing_method == "fixed_subsidence":
            _ls_divergence = namelist["testbed"]["largescale_divergence"]

        data = nc.Dataset(file, "r")
        forcing_data = data.groups["forcing"]
        lat = forcing_data.variables["latitude"][0]
        self._f = 2.0 * parameters.OMEGA * np.sin(lat * np.pi / 180.0)
        zl = self._Grid.z_local

        # Read in the data, we want to
        forcing_z = forcing_data.variables["z"][:]
        self._forcing_times = forcing_data.variables["times"][:]
        ntimes = np.shape(self._forcing_times)[0]
        if self._momentum_forcing_method == "geostrophic":
            raw_ug = forcing_data.variables["u_geostrophic"][:, :]
            raw_vg = forcing_data.variables["v_geostrophic"][:, :]
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            raw_ur = forcing_data.variables["u_relaxation"][:, :]
            raw_vr = forcing_data.variables["v_relaxation"][:, :]

        raw_adv_qt = forcing_data.variables["qt_advection"][:, :]
        # If both theta and temperature advection are provided, take temperature
        # Otherwise, take theta
        try:
            raw_adv_temperature = forcing_data.variables["temperature_advection"][:, :]
            self._use_temperature_advection = True
        except:
            raw_adv_theta = forcing_data.variables["theta_advection"][:, :]
            self._use_temperature_advection = False

        # Get the subsidence/vertical advection tendency ifnromation
        # Assuming that vertical advection tendencies are provided for temperature, not theta
        # as this is what varanal provides
        if self._subsidence_forcing_method == "fixed_subsidence":
            raw_subsidence = (
                np.ones_like(raw_adv_qt)
                * np.tile(forcing_z[np.newaxis, :], (ntimes, 1))
                * _ls_divergence
            )
        else:
            raw_subsidence = forcing_data.variables["subsidence"][:, :]
        if self._subsidence_forcing_method == "vertical_tendency":
            raw_vtend_qt = forcing_data.variables["qt_vertical_tendency"][:, :]
            # temperature, not theta
            raw_vtend_temperature = forcing_data.variables[
                "temperature_vertical_tendency"
            ][:, :]

        data.close()

        if self._momentum_forcing_method == "geostrophic":
            self._ug = interpolate.interp1d(
                forcing_z, raw_ug, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
            self._vg = interpolate.interp1d(
                forcing_z, raw_vg, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            self._ur = interpolate.interp1d(
                forcing_z, raw_ur, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)
            self._vr = interpolate.interp1d(
                forcing_z, raw_vr, axis=1, fill_value="extrapolate", assume_sorted=True
            )(zl)

        self._subsidence = interpolate.interp1d(
            forcing_z,
            raw_subsidence,
            axis=1,
            fill_value="extrapolate",
            assume_sorted=True,
        )(zl)
        self._adv_qt = interpolate.interp1d(
            forcing_z, raw_adv_qt, axis=1, fill_value="extrapolate", assume_sorted=True
        )(zl)
        if self._use_temperature_advection:
            self._adv_temperature = interpolate.interp1d(
                forcing_z,
                raw_adv_temperature,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)
        else:
            self._adv_theta = interpolate.interp1d(
                forcing_z,
                raw_adv_theta,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)
        if self._subsidence_forcing_method == "vertical_tendency":
            self._vtend_temperature = interpolate.interp1d(
                forcing_z,
                raw_vtend_temperature,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)

            self._vtend_qt = interpolate.interp1d(
                forcing_z,
                raw_vtend_qt,
                axis=1,
                fill_value="extrapolate",
                assume_sorted=True,
            )(zl)

        z_top = self._Grid.l[2]
        # Performing relaxation nudging over the same depth as damping, this is an assumption to revisit
        _depth = namelist["damping"]["depth"]
        znudge = z_top - _depth
        # Assume nudging timescale of 1 hour, again this is an assumption that could be revisited
        self._compute_relaxation_coefficient(znudge, 3600.0)

        self._Timers.add_timer("ForcingTestBed_update")

        return

    def update(self):

        self._Timers.start_timer("ForcingTestBed_update")

        current_time = self._TimeSteppingController.time

        # interpolate in time
        if self._momentum_forcing_method == "geostrophic":
            current_ug = interpolate.interp1d(
                self._forcing_times,
                self._ug,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vg = interpolate.interp1d(
                self._forcing_times,
                self._vg,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        if (
            self._momentum_forcing_method == "relaxation"
            or self._momentum_forcing_method == "mean_relaxation"
        ):
            current_ur = interpolate.interp1d(
                self._forcing_times,
                self._ur,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vr = interpolate.interp1d(
                self._forcing_times,
                self._vr,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        current_subsidence = interpolate.interp1d(
            self._forcing_times,
            self._subsidence,
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        current_adv_qt = interpolate.interp1d(
            self._forcing_times,
            self._adv_qt,
            axis=0,
            fill_value="extrapolate",
            assume_sorted=True,
        )(current_time)
        if self._use_temperature_advection:
            current_adv_temperature = interpolate.interp1d(
                self._forcing_times,
                self._adv_temperature,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        else:
            current_adv_theta = interpolate.interp1d(
                self._forcing_times,
                self._adv_theta,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
        if self._subsidence_forcing_method == "vertical_tendency":
            current_vtend_temperature = interpolate.interp1d(
                self._forcing_times,
                self._vtend_temperature,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)
            current_vtend_qt = interpolate.interp1d(
                self._forcing_times,
                self._vtend_qt,
                axis=0,
                fill_value="extrapolate",
                assume_sorted=True,
            )(current_time)

        exner = self._Ref.exner

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        if self._use_temperature_advection:
            st += (current_adv_temperature)[np.newaxis, np.newaxis, :]
        else:
            st += (current_adv_theta * exner)[np.newaxis, np.newaxis, :]
        qvt += (current_adv_qt)[np.newaxis, np.newaxis, :]
        if self._momentum_forcing_method == "geostrophic":
            Forcing_impl.large_scale_pgf(
                current_ug,
                current_vg,
                self._f,
                u,
                v,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
            )
        if self._momentum_forcing_method == "relaxation":
            Forcing_impl.relax_velocities(
                current_ur,
                current_vr,
                u,
                v,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
                self._relaxation_coefficient,
            )
        if self._momentum_forcing_method == "mean_relaxation":
            umean = self._VelocityState.mean("u")
            vmean = self._VelocityState.mean("v")
            Forcing_impl.relax_mean_velocities(
                current_ur,
                current_vr,
                umean,
                vmean,
                self._Ref.u0,
                self._Ref.v0,
                ut,
                vt,
                self._relaxation_coefficient,
            )

        if self._subsidence_forcing_method == "vertical_tendency":
            st += (
                current_vtend_temperature
                + current_subsidence * parameters.G * parameters.ICPD
            )[np.newaxis, np.newaxis, :]
            qvt += (current_vtend_qt)[np.newaxis, np.newaxis, :]
        else:
            Forcing_impl.apply_subsidence(current_subsidence, self._Grid.dxi[2], s, st)
            Forcing_impl.apply_subsidence(
                current_subsidence, self._Grid.dxi[2], qv, qvt
            )

        self._Timers.end_timer("ForcingTestBed_update")

        return

    def _compute_relaxation_coefficient(self, znudge, timescale):

        self._relaxation_coefficient = np.zeros(self._Grid.ngrid[2], dtype=np.double)

        z = self._Grid.z_global

        for k in range(self._Grid.ngrid[2]):
            self._relaxation_coefficient[k] = 1.0 / timescale
            if z[k] < znudge:
                self._relaxation_coefficient[k] *= (
                    np.sin((np.pi / 2.0) * (1.0 - (znudge - z[k]) / znudge)) ** 2.0
                )

        return
