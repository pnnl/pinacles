import numpy as np

CASENAMES = ['colliding_blocks']

def colliding_blocks(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')

    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global

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

def factory(namelist):
    assert(namelist['meta']['casename'] in CASENAMES)

    #if namelist['meta']['casename'] is 'colliding_blocks':
    return colliding_blocks

def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    return