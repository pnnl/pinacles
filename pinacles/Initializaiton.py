import numpy as np
import pinacles.ThermodynamicsDry_impl as DryThermo
import netCDF4 as nc
from scipy import interpolate

CASENAMES = ['colliding_blocks',
            'sullivan_and_patton',
            'stable_bubble',
            'bomex',
            'rico',
            'atex',
            'testbed']

def colliding_blocks(namelist, ModelGrid, Ref, ScalarState, VelocityState):


    #Integrate the reference profile.
    Ref.set_surface()
    Ref.integrate()


    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global


    u.fill(0.0)
    v.fill(0.0)

    shape = s.shape
    for i in range(shape[0]):
        x = xl[i] - (np.max(xg) - np.min(xg))/2.0
        for j in range(shape[1]):
            y = yl[j] - (np.max(yg) - np.min(yg))/2.0
            for k in range(shape[2]):
                if x > -225 and x <= -125 and y >= -50 and y <= 50:
                    s[i,j,k] = 25.0
                    u[i,j,k] = 2.5
                if x >= 125 and x < 225  and y >= -100 and y <= 100:
                    s[i,j,k] = -25.0
                    u[i,j,k] = -2.5

    return

def sullivan_and_patton(namelist, ModelGrid, Ref, ScalarState, VelocityState):


    #Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0, u0=0.0, v0=0.0)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    p1 = ScalarState.get_field('p1')
    
    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    #Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)
    p1.fill(0.0)

    u -= Ref.u0
    v -= Ref.v0

    shape = s.shape
    perts = np.random.uniform(-0.001, 0.001,(shape[0],shape[1],shape[2]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                if zl[k] < 974.0:
                    t = 300.0
                    t *=exner[k]
                elif 974.0 <= zl[k] and zl[k] < 1074.0:
                    t = 300.0 + (zl[k] - 974.0) * 0.08
                    t *= exner[k]
                    p1[i,j,k] = 1.0
                else:
                    t = 308.0 + (zl[k] - 1074.0) * 0.0034
                    t *= exner[k]
                    p1[i,j,k] = 1.0
                if zl[k] < 200.0:
                    t += perts[i,j,k]
                s[i,j,k] = DryThermo.s(zl[k], t)

    return

def bomex(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    #Integrate the reference profile.
    Ref.set_surface(Psfc=1015e2, Tsfc=300.4, u0=-8.75, v0=0.0)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    qv = ScalarState.get_field('qv')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner


    #Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    perts = np.random.uniform(-0.01, 0.01,(shape[0],shape[1],shape[2])) * 10.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            u700 = 0
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]
                if z < 520.0:
                    t = 298.7 
                    qv[i,j,k] = 17.0 + z * (16.3-17.0)/520.0
                elif z >= 520.0 and z <= 1480.0: 
                    t = 298.7 + (z - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
                    qv[i,j,k] = 16.3 + (z - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0)
                elif z > 1480.0 and z <= 2000:
                    t = 302.4 + (z - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
                    qv[i,j,k] =  10.7 + (z - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0)
                elif z > 2000.0:
                    t = 308.2 + (z - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)
                    qv[i,j,k] = 4.2 + (z- 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0)

                t *= exner[k]
                if zl[k] < 400.0:
                    t += perts[i,j,k]
                s[i,j,k] = DryThermo.s(zl[k], t)

                if z <= 700.0: 
                    u[i,j,k] = -8.75
                else: 
                    u[i,j,k] = -8.75 + (z- 700.0) * 1.8e-3

    u -= Ref.u0
    v -= Ref.v0

    #u.fill(0.0)
    qv /= 1000.0

    return

def atex(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    #Integrate the reference profile.
    Ref.set_surface(Psfc=1.0154e5, Tsfc=295.750, u0=-8.0, v0=-1.0)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    qv = ScalarState.get_field('qv')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    #Wind is uniform initially
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape
    temp = np.empty(shape[2], dtype=np.double)
    perts = np.random.uniform(-0.01, 0.01,(shape[0],shape[1],shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]
                if z <= 150.0:
                    t = 295.75
                    qv[i,j,k] = 13.0 + z * (12.50 - 13.0)/150.0
                    u[i, j, k] = max(-11.0 + z * (-10.55 -- 11.00)/150.0,-8.0)
                    v[i, j, k] = -2.0 + z *(-1.90 - -2.0) / 150.0
                elif z > 150.0 and z <= 700.0:
                    dz = (700.0-150.0)
                    t = 295.75
                    qv[i,j,k] = 12.50
                    u[i,j,k] = max(-10.55 + (z - 150.0)* (-8.90 -- 10.55)/dz,-8.0)
                    v[i,j,k] = -1.90 + (z - 150.0) *(-1.10 - -1.90)/dz
                elif z > 700.0 and z <= 750.0:
                    dz = (750.0 - 700.0)
                    t = 295.75 + (z - 700.0) * (296.125 - 295.75)/dz
                    qv[i,j,k] = 12.50 + (z - 700.0)*(11.50 - 12.50)/dz
                    u[i,j,k] = max(-8.90 + (z - 700.0)* (-8.75 -- 8.90)/dz,-8.0)
                    v[i,j,k] = -1.10 + (z - 700.0) *(-1.00 -- 1.10)/dz
                elif z > 750.0 and z <= 1400.0:
                    dz = (1400.0 - 750.0)
                    t = 296.125 + (z - 750.0)  * (297.75 - 296.125)/dz
                    qv[i,j,k] = 11.50 + (z - 750.0)*(10.25 - 11.50)/dz
                    u[i,j,k] = max(-8.75 + (z - 750.0)* (-6.80 -- 8.75)/dz,-8.0)
                    v[i,j,k] = -1.00 + (z - 750.0) *(-0.14 -- 1.00)/dz
                elif z > 1400.0 and z <= 1650.0:
                    dz = 1650.0 - 1400.0
                    t = 297.75 + (z - 1400.0) * (306.75 - 297.75)/dz
                    qv[i,j,k] =  10.25 + (z - 1400.0) * (4.50 - 10.25)/dz
                    u[i,j,k] = max(-6.80 + (z - 1400.0)* (-5.75--6.80)/dz,-8.0)
                    v[i,j,k] = -0.14 + (z - 1400.0) *(0.18 -- 0.14)/dz
                elif z > 1650.0:
                    dz = 4000.0 - 1650.0
                    t = 306.75 + (z - 1650.0) * (314.975 - 306.75)/dz
                    qv[i,j,k] =  4.50
                    u[i,j,k] = max(-5.75 + (z - 1650.0)* (1.00 -- 5.75)/dz,-8.0)
                    v[i,j,k] = 0.18 + (z - 1650.0) *(2.75 - 0.18)/dz
                temp[k] = qv[i,j,k]
                t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i,j,k]
                s[i,j,k] = DryThermo.s(zl[k], t)

    u -= Ref.u0
    v -= Ref.v0

    #u.fill(0.0)
    qv /= 1000.0

    return


def rico(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    #Integrate the reference profile.
    Ref.set_surface(Tsfc=299.8, Psfc=1.0154e5, u0=-9.9, v0=-3.8)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    qv = ScalarState.get_field('qv')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner


    #Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    perts = np.random.uniform(-0.01, 0.01,(shape[0],shape[1],shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                z = zl[k]

                if z <= 740.0:
                    t = 297.7
                else:
                    t = 297.9 + (317.0-297.9)/(4000.0-740.0)*(z - 740.0)

                if z <= 740.0:
                    q =  16.0 + (13.8 - 16.0)/740.0 * z
                elif z > 740.0 and z <= 3260.:
                    q = 13.8 + (2.4 - 13.8)/(3260.0-740.0) * (z - 740.0)
                else:
                    q = 2.4 + (1.8-2.4)/(4000.0-3260.0)*(z - 3260.0)

                q/=1000.0

                t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i,j,k]
                s[i,j,k] = DryThermo.s(z, t)
                qv[i,j,k] = q
                u[i,j,k] =  -9.9 + 2.0e-3 * z
                v[i,j,k] =  -3.8
    u -= Ref.u0
    v -= Ref.v0

    #u.fill(0.0)




    return

def stable_bubble(namelist, ModelGrid, Ref, ScalarState, VelocityState):


    #Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    zl = ModelGrid.z_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

    exner = Ref.exner

    #Wind is uniform initiall
    u.fill(0.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    print(zl)
    
    dista = np.zeros_like(u)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist  = np.sqrt((xl[i]/1000.0 - 25.6)**2.0 + ((zl[k]/1000.0 - 3.0)/2.0)**2.0)

                t = 300.0 
                if dist <= 1.0:
                    t -= 7.5

                t *= exner[k]

                s[i,j,k] = DryThermo.s(zl[k], t)
                #dist = min(dist, 1.0)
                #t = (300.0 ) - 15.0*( np.cos(np.pi * dist) + 1.0) /2.0
                #dista[i,j,k] = dist

    print(np.amax(s), np.amin(s))
    #import pylab as plt
    #plt.contourf(dista[:,4,:].T)
    #plt.show()

    #import sys; sys.exit()
    return

def testbed(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    file = namelist['testbed']['input_filepath']
    data = nc.Dataset(file, 'r')
    init_data = data.groups['initialization']
    psfc = init_data.variables['surface_pressure'][0] * 100.0 # Convert from hPa to Pa
    tsfc = init_data.variables['surface_temperature'][0]
    u0 = init_data.variables['reference_u0'][0]
    v0 = init_data.variables['reference_v0'][0]
    
    Ref.set_surface(Psfc=psfc, Tsfc=tsfc, u0=u0, v0=v0)
    Ref.integrate()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    qv = ScalarState.get_field('qv')
    zl = ModelGrid.z_local

    init_z = init_data.variables['z'][:]
    raw_theta = init_data.variables['potential_temperature'][:]
    raw_qv = init_data.variables['vapor_mixing_ratio'][:]/1000.0
    raw_u = init_data.variables['u'][:]
    raw_v = init_data.variables['v'][:]
    
    init_var_from_sounding(raw_u, init_z, zl, u)
    init_var_from_sounding(raw_v, init_z, zl, v)
    init_var_from_sounding(raw_qv, init_z, zl, qv)
    init_var_from_sounding(raw_theta, init_z, zl, s)

    u -= Ref.u0
    v -= Ref.v0


    # hardwire for now, could make inputs in namelist or data file
    pert_amp = 0.1
    pert_max_height = 200.0
    shape = s.shape
    perts = np.random.uniform(pert_amp*-1.0, pert_amp,(shape[0],shape[1],shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = s[i,j,k] * Ref.exner[k]
                if zl[k] < pert_max_height:
                    t += perts[i,j,k]
                s[i,j,k] = DryThermo.s(zl[k], t)
    
    return
       

def init_var_from_sounding(profile_data, profile_z,  grid_z, var3d):
    var3d.fill(0.0)
    grid_profile = interpolate.interp1d(profile_z, profile_data, fill_value='extrapolate',assume_sorted=True)(grid_z)
    var3d += grid_profile[np.newaxis, np.newaxis, :]
    return 
    

    # Integrate the reference profile


def factory(namelist):
    assert(namelist['meta']['casename'] in CASENAMES)

    if namelist['meta']['casename'] == 'colliding_blocks':
        return colliding_blocks
    elif namelist['meta']['casename'] == 'sullivan_and_patton':
        return sullivan_and_patton
    elif namelist['meta']['casename'] == 'stable_bubble':
        return stable_bubble
    elif namelist['meta']['casename'] == 'bomex':
        return bomex
    elif namelist['meta']['casename'] == 'rico':
        return rico
    elif namelist['meta']['casename'] == 'atex':
        return atex
    elif namelist['meta']['casename'] == 'testbed':
        return testbed


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    return