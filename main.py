import json  
import argparse 
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping
from mpi4py import MPI 
import cProfile
import numpy as n

def main(namelist): 
    TerminalIO.start_message() 

    ModelGrid = Grid.RegularCartesian(namelist)
    PrognosticState = Containers.ModelState(ModelGrid, prognostic=True) 
    DiagnosticState = Containers.ModelState(ModelGrid) 
    PrognosticState.add_variable('q_t')
    #SA = ScalarAdvection.ScalarAdvectionFactory(namelist)

    Thermo = Thermodynamics.factory(namelist, ModelGrid, PrognosticState, DiagnosticState)


    PrognosticState.allocate()
    DiagnosticState.allocate()


    PrognosticState.boundary_exchange()
    TestArr = ParallelArrays.GhostArray(ModelGrid, ndof=5)

    TestArr.set(MPI.COMM_WORLD.Get_rank()) 
    import time 
    for i in range(1): 
        t0 = time.time() 
        TestArr.boundary_exchange()
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