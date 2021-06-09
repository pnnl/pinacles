import numpy as np
import pinacles.ThermodynamicsDry_impl as DryThermo
import pinacles.ThermodynamicsMoist_impl as MoistThermo
import netCDF4 as nc
from scipy import interpolate
from pinacles import UtilitiesParallel

CASENAMES = [
    "colliding_blocks",
    "sullivan_and_patton",
    "stable_bubble",
    "bomex",
    "rico",
    "atex",
    "testbed",
]


def colliding_blocks(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    # Integrate the reference profile.
    Ref.set_surface()
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    s = ScalarState.get_field("s")

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    u.fill(0.0)
    v.fill(0.0)

    shape = s.shape
    for i in range(shape[0]):
        x = xl[i] - (np.max(xg) - np.min(xg)) / 2.0
        for j in range(shape[1]):
            y = yl[j] - (np.max(yg) - np.min(yg)) / 2.0
            for k in range(shape[2]):
                if x > -225 and x <= -125 and y >= -50 and y <= 50:
                    s[i, j, k] = 25.0
                    u[i, j, k] = 2.5
                if x >= 125 and x < 225 and y >= -100 and y <= 100:
                    s[i, j, k] = -25.0
                    u[i, j, k] = -2.5

    return


def sullivan_and_patton(namelist, ModelGrid, Ref, ScalarState, VelocityState):

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

    # Wind is uniform initiall
    u.fill(5.0)
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
                    t = 308.0 + (zl[k] - 1074.0) * 0.0034
                    t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i, j, k]
                s[i, j, k] = DryThermo.s(zl[k], t)

    return


def bomex(namelist, ModelGrid, Ref, ScalarState, VelocityState):

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

    perts = np.random.uniform(-0.01, 0.01, (shape[0], shape[1], shape[2])) * 10.0
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


def atex(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    # Integrate the reference profile.
    Ref.set_surface(Psfc=1.0154e5, Tsfc=295.750, u0=-8.0, v0=-1.0)
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

    # Wind is uniform initially
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape
    temp = np.empty(shape[2], dtype=np.double)
    perts = np.random.uniform(-0.01, 0.01, (shape[0], shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]
                if z <= 150.0:
                    t = 295.75
                    qv[i, j, k] = 13.0 + z * (12.50 - 13.0) / 150.0
                    u[i, j, k] = max(-11.0 + z * (-10.55 - -11.00) / 150.0, -8.0)
                    v[i, j, k] = -2.0 + z * (-1.90 - -2.0) / 150.0
                elif z > 150.0 and z <= 700.0:
                    dz = 700.0 - 150.0
                    t = 295.75
                    qv[i, j, k] = 12.50
                    u[i, j, k] = max(-10.55 + (z - 150.0) * (-8.90 - -10.55) / dz, -8.0)
                    v[i, j, k] = -1.90 + (z - 150.0) * (-1.10 - -1.90) / dz
                elif z > 700.0 and z <= 750.0:
                    dz = 750.0 - 700.0
                    t = 295.75 + (z - 700.0) * (296.125 - 295.75) / dz
                    qv[i, j, k] = 12.50 + (z - 700.0) * (11.50 - 12.50) / dz
                    u[i, j, k] = max(-8.90 + (z - 700.0) * (-8.75 - -8.90) / dz, -8.0)
                    v[i, j, k] = -1.10 + (z - 700.0) * (-1.00 - -1.10) / dz
                elif z > 750.0 and z <= 1400.0:
                    dz = 1400.0 - 750.0
                    t = 296.125 + (z - 750.0) * (297.75 - 296.125) / dz
                    qv[i, j, k] = 11.50 + (z - 750.0) * (10.25 - 11.50) / dz
                    u[i, j, k] = max(-8.75 + (z - 750.0) * (-6.80 - -8.75) / dz, -8.0)
                    v[i, j, k] = -1.00 + (z - 750.0) * (-0.14 - -1.00) / dz
                elif z > 1400.0 and z <= 1650.0:
                    dz = 1650.0 - 1400.0
                    t = 297.75 + (z - 1400.0) * (306.75 - 297.75) / dz
                    qv[i, j, k] = 10.25 + (z - 1400.0) * (4.50 - 10.25) / dz
                    u[i, j, k] = max(-6.80 + (z - 1400.0) * (-5.75 - -6.80) / dz, -8.0)
                    v[i, j, k] = -0.14 + (z - 1400.0) * (0.18 - -0.14) / dz
                elif z > 1650.0:
                    dz = 4000.0 - 1650.0
                    t = 306.75 + (z - 1650.0) * (314.975 - 306.75) / dz
                    qv[i, j, k] = 4.50
                    u[i, j, k] = max(-5.75 + (z - 1650.0) * (1.00 - -5.75) / dz, -8.0)
                    v[i, j, k] = 0.18 + (z - 1650.0) * (2.75 - 0.18) / dz
                temp[k] = qv[i, j, k]
                t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i, j, k]
                s[i, j, k] = DryThermo.s(zl[k], t)

    u -= Ref.u0
    v -= Ref.v0

    # u.fill(0.0)
    qv /= 1000.0

    return


def rico(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    # Integrate the reference profile.
    Ref.set_surface(Tsfc=299.8, Psfc=1.0154e5, u0=-9.9, v0=-3.8)
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
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]

                if z <= 740.0:
                    t = 297.7
                else:
                    t = 297.9 + (317.0 - 297.9) / (4000.0 - 740.0) * (z - 740.0)

                if z <= 740.0:
                    q = 16.0 + (13.8 - 16.0) / 740.0 * z
                elif z > 740.0 and z <= 3260.0:
                    q = 13.8 + (2.4 - 13.8) / (3260.0 - 740.0) * (z - 740.0)
                else:
                    q = 2.4 + (1.8 - 2.4) / (4000.0 - 3260.0) * (z - 3260.0)

                q /= 1000.0

                t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i, j, k]
                s[i, j, k] = DryThermo.s(z, t)
                qv[i, j, k] = q
                u[i, j, k] = -9.9 + 2.0e-3 * z
                v[i, j, k] = -3.8
    u -= Ref.u0
    v -= Ref.v0

    # u.fill(0.0)

    return


def stable_bubble(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    # Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0)
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

    # Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    dista = np.zeros_like(u)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt(
                    (xl[i] / 1000.0 - 25.6) ** 2.0
                    + ((zl[k] / 1000.0 - 3.0) / 2.0) ** 2.0
                )

                t = 300.0
                if dist <= 1.0:
                    t -= 7.5

                t *= exner[k]

                s[i, j, k] = DryThermo.s(zl[k], t)
                # dist = min(dist, 1.0)
                # t = (300.0 ) - 15.0*( np.cos(np.pi * dist) + 1.0) /2.0
                # dista[i,j,k] = dist

    return


def testbed(namelist, ModelGrid, Ref, ScalarState, VelocityState):
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
    Ref.integrate()

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


def factory(namelist):
    assert namelist["meta"]["casename"] in CASENAMES

    if namelist["meta"]["casename"] == "colliding_blocks":
        return colliding_blocks
    elif namelist["meta"]["casename"] == "sullivan_and_patton":
        return sullivan_and_patton
    elif namelist["meta"]["casename"] == "stable_bubble":
        return stable_bubble
    elif namelist["meta"]["casename"] == "bomex":
        return bomex
    elif namelist["meta"]["casename"] == "rico":
        return rico
    elif namelist["meta"]["casename"] == "atex":
        return atex
    elif namelist["meta"]["casename"] == "testbed":
        return testbed


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    return
