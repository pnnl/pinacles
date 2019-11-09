import numpy as np
import Columbia.ThermodynamicsDry_impl as DryThermo

CASENAMES = ['colliding_blocks', 'sullivan_and_patton']

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
    u.fill(1.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    perts = np.random.randn(shape[0],shape[1],shape[2]) * 0.001

    tp  = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                t = 0.0
                if zl[k] < 974.0:
                    t = 300.0
                    t*=exner[k]
                elif 974.0 <= zl[k] and zl[k] < 1074.0:
                    t = 300.0 + (zl[k] - 974.0) * 0.08
                    t *= exner[k]
                else:
                    t = 308.0 + (zl[k] - 1074.0) * 0.003
                    t *= exner[k]
                if zl[k] < 200.0:
                    t += perts[i,j,k]
                    #u[i,j,k] += perts[i,j,k]
                s[i,j,k] = DryThermo.s(zl[k], t)

    return

def factory(namelist):
    assert(namelist['meta']['casename'] in CASENAMES)

    if namelist['meta']['casename'] == 'colliding_blocks':
        return colliding_blocks
    elif namelist['meta']['casename'] == 'sullivan_and_patton':
        return sullivan_and_patton

def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    return