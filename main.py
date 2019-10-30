import numba 
import json
import argparse
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping, ReferenceState
from Columbia import MomentumAdvection
from Columbia import PressureSolver
from scipy.ndimage import laplace
from mpi4py import MPI
import numpy as np
import time
import pylab as plt

def main(namelist):
    TerminalIO.start_message()

    t0 = time.time()
    ModelGrid = Grid.RegularCartesian(namelist)
    ScalarState = Containers.ModelState(ModelGrid, prognostic=True)
    VelocityState = Containers.ModelState(ModelGrid, prognostic=True)
    DiagnosticState = Containers.ModelState(ModelGrid)
    ScalarTimeStepping = TimeStepping.factory(namelist, ModelGrid, ScalarState)
    VelocityTimeStepping = TimeStepping.factory(namelist, ModelGrid, VelocityState)


    # Add velocity variables
    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w', bcs='value zero')

    for i in range(1):
        ScalarState.add_variable(str(i))

    # Set up the reference state class
    Ref =  ReferenceState.factory(namelist, ModelGrid)
    Ref.set_surface()
    Ref.integrate()

    # Set up the thermodynamics class
    Thermo = Thermodynamics.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState, DiagnosticState)

    # In the futhre the microphyics should be initialized here


    #Setup the scalar advection calss
    ScalarAdv = ScalarAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    MomAdv = MomentumAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    #Setup the pressure solver
    PSolver = PressureSolver.factory(namelist, ModelGrid, Ref, VelocityState, DiagnosticState)

    # Allocate all of the big parallel arrays needed for the container classes
    ScalarState.allocate()
    VelocityState.allocate()
    DiagnosticState.allocate()

    ScalarTimeStepping.initialize()
    VelocityTimeStepping.initialize()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    ut = VelocityState.get_tend('u')
    #u[35:65,20:50,:] = -2.5
    #u[35:65,50:80,:] =  2.5

    #laplace(u,u)


    #u[35:66,45:55,4:6] = 5.0
    #u.fill(-2.5)
    #v.fill(2.5)
    xl = ModelGrid.x_local
    yl = ModelGrid.y_local
    shape = s.shape
    for i in range(shape[0]):
        x = xl[i] - (np.max(xl) - np.min(xl))/2.0
        for j in range(shape[1]):
            y = yl[j] - (np.max(yl) - np.min(yl))/2.0
            for k in range(shape[2]):
                if x > -225 and x <= -125 and y >= -50 and y <= 50: 
                    s[i,j,k] = 25.0  
                    u[i,j,k] = 2.5
                if x >= 125 and x < 225  and y >= -100 and y <= 100: 
                    s[i,j,k] = -25.0 
                    u[i,j,k] = -2.5
 

    import pylab as plt
    plt.figure(12)
    plt.contourf(s[:,:,5],50)#,vmin=0.0, vmax=25.0)
    plt.colorbar()
    plt.show()
    plt.close()

    #s[50+35:65,20:50,:]= 25.0
    #s[35:65,50:80,:] = -25.0
    #s[85:95,45:55,4:6] = -25.0
    #s[:,:,:] = np.arange(s.shape[1], dtype=np.double)[:,np.newaxis,np.newaxis]
    t1 = time.time()
    #import pylab as plt
    #plt.figure(12)
    #plt.contourf(np.diff(s[:,:,5],axis=0),150)
    #plt.colorbar()
#    plt.savefig('./figs/' + str(1000000 + i) + '.png')
#    plt.close() 
   # plt.show() 

    print(t1 - t0)
    times = []
    PSolver.update()
    PSolver.update()
    ScalarState.boundary_exchange()
    VelocityState.boundary_exchange()
    ScalarState.update_all_bcs()
    VelocityState.update_all_bcs()
    for i in range(2000):
        print(i)
        t0 = time.time()
        for n in range(ScalarTimeStepping.n_rk_step):
            #Thermo.update()
            ScalarAdv.update()
            MomAdv.update()
            ScalarTimeStepping.update()
            VelocityTimeStepping.update()
            ScalarState.boundary_exchange()
            VelocityState.boundary_exchange()
            ScalarState.update_all_bcs()
            VelocityState.update_all_bcs()
            PSolver.update()
            print(np.amax(w))
        t1 = time.time()
        import pylab as plt
        plt.figure(12)
        plt.contourf(s[:,:,5],100,vmin=-25.1, vmax=25.1, cmap=plt.cm.seismic)
        plt.clim(-25.1, 25.1)
        plt.colorbar()
        plt.savefig('./figs/' + str(1000000 + i) + '.png')
        times.append(t1 - t0)
        plt.close() 
        print('Scalar Integral ', np.sum(s[ModelGrid.n_halo[0]:-ModelGrid.n_halo[0],
            ModelGrid.n_halo[1]:-ModelGrid.n_halo[1],
            ModelGrid.n_halo[2]:-ModelGrid.n_halo[2]]))
        print('W-max', VelocityState.mean('w'))

    print('Timing: ', np.min(times))


    TerminalIO.end_message()

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input for Columbia an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        #cProfile.run('main(namelist)_h')
        main(namelist)
