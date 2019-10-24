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
    VelocityState.add_variable('w')

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
    u[5:10,5:10,5:10] = 1.0

    t1 = time.time()
    print(t1 - t0)
    times = []
    for i in range(20):
        print(i)
        t0 = time.time()
        for n in range(ScalarTimeStepping.n_rk_step):
            Thermo.update()
            ScalarAdv.update()
            MomAdv.update()
            ScalarTimeStepping.update()
            #VelocityTimeStepping.update()
            ScalarState.boundary_exchange()
            VelocityState.boundary_exchange()
            PSolver.update()
        t1 = time.time()
        times.append(t1 - t0)

    print(np.min(times))


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
