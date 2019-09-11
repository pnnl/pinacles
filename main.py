import json  
import argparse 
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping, ReferenceState
from mpi4py import MPI 
import numpy as numpy
import time

def main(namelist): 
    TerminalIO.start_message() 

    t0 = time.time() 
    ModelGrid = Grid.RegularCartesian(namelist)
    ScalarState = Containers.ModelState(ModelGrid, prognostic=True) 
    VelocityState = Containers.ModelState(ModelGrid, prognostic=True)
    DiagnosticState = Containers.ModelState(ModelGrid)
    
    # Add velocity variables 
    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w')

    for i in range(200): 
        ScalarState.add_variable(str(i))

    # Set up the thermodynamics class 
    Thermo = Thermodynamics.factory(namelist, ModelGrid, ScalarState, DiagnosticState)

    # In the futhre the microphyics should be initialized here 

    # Set up the reference state class 
    Ref =  ReferenceState.factory(namelist, ModelGrid, Thermo)
    Ref.set_surface()
    Ref.integrate()

    #Setup the scalar advection calss 
    ScalarAdv = ScalarAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    # Allocate all of the big parallel arrays needed for the container classes
    ScalarState.allocate()
    VelocityState.allocate()
    DiagnosticState.allocate()
    t1 = time.time() 
    print(t1 - t0)

    ScalarState.boundary_exchange()
    VelocityState.boundary_exchange()

    for i in range(10): 
        t0 = time.time() 
        ScalarAdv.update()
        t1 = time.time()
        print(t1 - t0)


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
