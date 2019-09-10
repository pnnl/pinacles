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
    PrognosticState = Containers.ModelState(ModelGrid, prognostic=True) 
    DiagnosticState = Containers.ModelState(ModelGrid)
    Thermo = Thermodynamics.factory(namelist, ModelGrid, PrognosticState, DiagnosticState)
    Ref =  ReferenceState.factory(namelist, ModelGrid, Thermo)
    Ref.set_surface()
    Ref.integrate()
    PrognosticState.allocate()
    DiagnosticState.allocate()
    t1 = time.time() 
    print(t1 - t0)

    PrognosticState.boundary_exchange()

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