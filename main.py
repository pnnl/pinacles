import numba 
import json
import argparse
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping, ReferenceState
from Columbia import MomentumAdvection
from Columbia import PressureSolver
from mpi4py import MPI
import numpy as np
import time

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
    s = ScalarState.get_field('s')
    ut = VelocityState.get_tend('u')
    #u[45:55,55:65,4:6] = 1.0    
    u[85:95,45:55,4:6] = -5.0
    #s[45:55,55:65,4:6] = 25.0
    s[85:95,45:55,4:6] = -25.0
    t1 = time.time()
    print(t1 - t0)
    times = []
    PSolver.update()
    for i in range(1000):
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
            print(np.amax(u))
        t1 = time.time()
        import pylab as plt
        plt.figure(12)
        plt.contourf(u[:,:,5],50)
        plt.colorbar()
        plt.savefig('./figs/' + str(1000000 + i) + '.png')
        times.append(t1 - t0)
        plt.close() 

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
