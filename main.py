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

    # Set up the reference state class
    Ref =  ReferenceState.factory(namelist, ModelGrid)


    # Add velocity variables
    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w', bcs='value zero')

    for i in range(1):
        ScalarState.add_variable(str(i))

    # Set up the thermodynamics class
    Thermo = Thermodynamics.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState, DiagnosticState)

    # In the future the microphyics should be initialized here

    #Setup the scalar advection calss
    ScalarAdv = ScalarAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    MomAdv = MomentumAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    #Setup the pressure solver
    PSolver = PressureSolver.factory(namelist, ModelGrid, Ref, VelocityState, DiagnosticState)

    # Allocate all of the big parallel arrays needed for the container classes
    ScalarState.allocate()
    VelocityState.allocate()
    DiagnosticState.allocate()

    #Integrate the reference profile.
    Ref.set_surface()
    Ref.integrate()


    PSolver.initialize()

    ScalarTimeStepping.initialize()
    VelocityTimeStepping.initialize()

    u = VelocityState.get_field('u')
    v = VelocityState.get_field('v')
    w = VelocityState.get_field('w')
    s = ScalarState.get_field('s')
    ut = VelocityState.get_tend('u')

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

    t1 = time.time()

    print(t1 - t0)
    times = []


    ScalarState.boundary_exchange()
    VelocityState.boundary_exchange()
    ScalarState.update_all_bcs()
    VelocityState.update_all_bcs()
    PSolver.update()

    for i in range(4000):
        #print(i)
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
        t1 = time.time()
        import pylab as plt
        s_slice = ScalarState.get_field_slice_z('s')
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('step: ', i, ' time: ', t1 - t0)
        if MPI.COMM_WORLD.Get_rank() == 0 and np.mod(i,5) == 0:
            plt.figure(12)
            levels = np.linspace(-27.1, 27.1, 100)
            plt.contourf(s_slice[:,:],100,levels=levels, cmap=plt.cm.seismic)
            plt.clim(-27.1, 27.1)
            #plt.colorbar()
            plt.savefig('./figs/' + str(1000000 + i) + '.png', dpi=300)
            times.append(t1 - t0)
            plt.close()
            print('Scalar Integral ', np.sum(s_slice))

            print('S-min max', np.amin(s), np.amax(s))

    print('Timing: ', np.min(times),)

    TerminalIO.end_message()

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input for Columbia an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        main(namelist)
