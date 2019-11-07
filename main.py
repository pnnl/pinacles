import numba
import json
import argparse
from Columbia import Initializaiton
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping, ReferenceState
from Columbia import MomentumAdvection
from Columbia import PressureSolver
from Columbia import Damping
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
    RayleighDamping = Damping.Rayleigh(namelist, ModelGrid)
    RayleighDamping.add_state(VelocityState)
    RayleighDamping.add_state(ScalarState)

    # Set up the reference state class
    Ref =  ReferenceState.factory(namelist, ModelGrid)

    # Add velocity variables
    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w', loc='z', bcs='value zero')

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

    ScalarTimeStepping.initialize()
    VelocityTimeStepping.initialize()


    Initializaiton.initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    PSolver.initialize() #Must be called after reference profile is integrated

    s = ScalarState.get_field('s')
    w = VelocityState.get_field('w')
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
            #Update Thermodynamics
            Thermo.update()

            #Update scalar advection
            ScalarAdv.update()
            MomAdv.update()

            #Do Damping
            RayleighDamping.update()

            #Do time stepping
            ScalarTimeStepping.update()
            VelocityTimeStepping.update()

            #Update boundary conditions
            ScalarState.boundary_exchange()
            VelocityState.boundary_exchange()
            ScalarState.update_all_bcs()
            VelocityState.update_all_bcs()

            #Call pressure solver
            PSolver.update()

        t1 = time.time()
        import pylab as plt
        s_slice = ScalarState.get_field_slice_z('s', indx=16)
        print('w mean', np.mean(w[3:-3, 3:-3,:], axis=(0,1)))
        b = DiagnosticState.get_field('T')
        theta = b / Ref.exner[np.newaxis, np.newaxis,:]
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('step: ', i, ' time: ', t1 - t0)
        if MPI.COMM_WORLD.Get_rank() == 0 and np.mod(i,5) == 0:
            plt.figure(12)
            #evels = np.linspace(299, 27.1, 100)
            levels = np.linspace(-2.0, 2.0, 100)
            plt.contourf(w[:,10,:].T,levels=levels, cmap=plt.cm.seismic)
            #plt.contourf(w[:,:,16], levels=levels, cmap=plt.cm.seismic)
            #plt.clim(-2.0, 2.0)
            plt.colorbar()
            plt.savefig('./figs/' + str(1000000 + i) + '.png', dpi=300)
            times.append(t1 - t0)
            plt.close()
            print('Scalar Integral ', np.sum(s_slice))

            print('S-min max', np.amin(w), np.amax(w))

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
