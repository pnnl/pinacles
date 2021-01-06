import numba
import json
import argparse
import uuid
import datetime
from pinacles import Initializaiton
from pinacles import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from pinacles import ScalarAdvectionFactory
from pinacles import ScalarAdvection, TimeStepping, ReferenceState
from pinacles import ScalarDiffusion, MomentumDiffusion
from pinacles import MomentumAdvection
from pinacles import PressureSolver
from pinacles import Damping
from pinacles import SurfaceFactory
from pinacles import ForcingFactory
from pinacles.Stats import Stats
from pinacles import DumpFields
from pinacles import MicrophysicsFactory
from pinacles import RadiationFactory
from pinacles import Kinematics
from pinacles import SGSFactory
from pinacles import DiagnosticsTurbulence
from pinacles import DiagnosticsClouds
from pinacles import TowersIO
from mpi4py import MPI
import numpy as np
import time
import pylab as plt

import os 
from termcolor import colored
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

def main(namelist):
    TerminalIO.start_message()


    t0 = time.time()
    ModelGrid = Grid.RegularCartesian(namelist)
    ScalarState = Containers.ModelState(ModelGrid, container_name='ScalarState', prognostic=True)
    VelocityState = Containers.ModelState(ModelGrid, container_name='VelocityState', prognostic=True)
    DiagnosticState = Containers.ModelState(ModelGrid, container_name='DiagnosticState')
    ScalarTimeStepping = TimeStepping.factory(namelist, ModelGrid, ScalarState)
    VelocityTimeStepping = TimeStepping.factory(namelist, ModelGrid, VelocityState)
    TimeSteppingController = TimeStepping.TimeSteppingController(namelist, ModelGrid, VelocityState)
    RayleighDamping = Damping.RayleighInitial(namelist, ModelGrid)
    RayleighDamping.add_state(VelocityState)
    RayleighDamping.add_state(ScalarState)



    TimeSteppingController.add_timestepper(ScalarTimeStepping)
    TimeSteppingController.add_timestepper(VelocityTimeStepping)
    #TimeSteppingController.add_timematch(60.0)
    # Set up the reference state class
    Ref =  ReferenceState.factory(namelist, ModelGrid)


    # Add velocity variables
    VelocityState.add_variable('u', long_name = 'u velocity component', units='m/s', latex_name = 'u')
    VelocityState.add_variable('v', long_name = 'v velocity component', units='m/s', latex_name = 'v')
    VelocityState.add_variable('w', long_name = 'w velocity component', units='m/s', latex_name = 'w', loc='z', bcs='value zero')

    # Set up the thermodynamics class
    Kine = Kinematics.Kinematics(ModelGrid, Ref, VelocityState, DiagnosticState)
    SGS = SGSFactory.factory(namelist, ModelGrid, Ref, VelocityState, DiagnosticState)
    Micro = MicrophysicsFactory.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController)
    Thermo = Thermodynamics.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState, DiagnosticState, Micro)


    # In the future the microphyics should be initialized here

    #Setup the scalar advection calss
    ScalarAdv = ScalarAdvectionFactory.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState, ScalarTimeStepping)
    MomAdv = MomentumAdvection.factory(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    ScalarDiff = ScalarDiffusion.ScalarDiffusion(namelist, ModelGrid, Ref, DiagnosticState, ScalarState)
    MomDiff = MomentumDiffusion.MomentumDiffusion(namelist, ModelGrid, Ref, DiagnosticState, Kine, VelocityState)


    #Setup the pressure solver
    PSolver = PressureSolver.factory(namelist, ModelGrid, Ref, VelocityState, DiagnosticState)

    # Allocate all of the big parallel arrays needed for the container classes
    Force = ForcingFactory.factory(namelist, ModelGrid, Ref, Micro, VelocityState, ScalarState, DiagnosticState, TimeSteppingController)
    Surf = SurfaceFactory.factory(namelist, ModelGrid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController)
    Rad = RadiationFactory.factory(namelist, ModelGrid, Ref, ScalarState, DiagnosticState, Surf, TimeSteppingController)


    ScalarState.allocate()
    VelocityState.allocate()
    DiagnosticState.allocate()


    ScalarTimeStepping.initialize()
    VelocityTimeStepping.initialize()

    Initializaiton.initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState)

    Rad.init_profiles()
    RayleighDamping.init_means()
    PSolver.initialize() #Must be called after reference profile is integrated

    #Setup Stats-IO
    StatsIO = Stats(namelist, ModelGrid, Ref, TimeSteppingController)

    IOTower= TowersIO.Towers(namelist, ModelGrid, TimeSteppingController)

    IOTower.add_state_container(VelocityState)
    IOTower.add_state_container(ScalarState)
    IOTower.add_state_container(DiagnosticState)
    IOTower.initialize()

    DiagClouds = DiagnosticsClouds.DiagnosticsClouds(ModelGrid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState)
    DiagTurbulence = DiagnosticsTurbulence.DiagnosticsTurbulence(ModelGrid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState)
    ScalarDiff.initialize_io_arrays()
    ScalarAdv.initialize_io_arrays()


    # Add diagnostics
    StatsIO.add_class(Surf)
    StatsIO.add_class(ScalarAdv)
    StatsIO.add_class(ScalarDiff)
    StatsIO.add_class(VelocityState)
    StatsIO.add_class(ScalarState)
    StatsIO.add_class(DiagnosticState)
    StatsIO.add_class(Micro)
    StatsIO.add_class(DiagTurbulence)
    StatsIO.add_class(DiagClouds)



    FieldsIO = DumpFields.DumpFields(namelist, ModelGrid, TimeSteppingController)
    FieldsIO.add_class(ScalarState)
    FieldsIO.add_class(VelocityState)
    FieldsIO.add_class(DiagnosticState)


    StatsIO.initialize()


    s = ScalarState.get_field('s')
    #qv = ScalarState.get_field('qv')
    u = VelocityState.get_field('u')
    t1 = time.time()

    print(t1 - t0)
    times = []

    ScalarState.boundary_exchange()
    VelocityState.boundary_exchange()
    ScalarState.update_all_bcs()
    VelocityState.update_all_bcs()
    PSolver.update()

    #Call stats for the first time

    #for i in range(4*3600*10):
    i = 0
    while TimeSteppingController.time <= TimeSteppingController.time_max:
        #print(i)
        t0 = time.time()
        for n in range(ScalarTimeStepping.n_rk_step):
            TimeSteppingController.adjust_timestep(n)

            #Update Thermodynamics
            Thermo.update()

            #Do StatsIO if it is time
            if n == 0:
                StatsIO.update()
                MPI.COMM_WORLD.barrier()
                IOTower.update()

            #Update the surface
            Surf.update()

            #Update the forcing
            Force.update()
            Rad.update(n)

            #Update Kinematics
            Kine.update()
            SGS.update()

            #Update scalar advection
            ScalarAdv.update()
            MomAdv.update()

            ScalarDiff.update()
            MomDiff.update()

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

            if n== 1:
                Thermo.update(apply_buoyancy=False)
                #We call the microphysics update at the end of the RK steps.
                Micro.update()
                ScalarState.boundary_exchange()
                ScalarState.update_all_bcs()

        i += 1


        t1 = time.time()
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(colored('\t Walltime: ', 'green'), colored(t1 -t0, 'green'), colored('\tModeltime/Walltime: ', 'green'), colored(TimeSteppingController._dt/(t1 - t0), 'green'))
        #s_slice = DiagnosticState.get_field_slice_z('T', indx=5)
        #s_slice = VelocityState.get_field_slice_z('w', indx=5)
        # b = DiagnosticState.get_field('T')
        # #theta = b / Ref.exner[np.newaxis, np.newaxis,:]
        xl = ModelGrid.x_local
        zl = ModelGrid.z_local
        if np.isclose((TimeSteppingController._time + TimeSteppingController._dt)%600.0,0.0):
            FieldsIO.update()
            if MPI.COMM_WORLD.Get_rank() == 0:
                pass 
        #     #print('step: ', i, ' time: ', t1 - t0)
        #         plt.figure(12)
        #     #evels = np.linspace(299, 27.1, 100)
                 #levels = np.linspace(-4,4, 100)
                 #levels = np.linspace(-5, 5,100)
                 #plt.contourf(s_slice,cmap=plt.cm.seismic, levels=levels) #,levels=levels, cmap=plt.cm.seismic)
                 #plt.contourf((s[3:-3,16,3:-3]) .T ,100,cmap=plt.cm.seismic) 
        #         plt.contourf(s_slice, 100) 
        #     #plt.contourf(w[:,:,16], levels=levels, cmap=plt.cm.seismic)
                 #plt.clim(-4,5)
                 #plt.colorbar()
                # plt.ylim(0.0*1000,4.0*1000)
                # plt.xlim(25.6*1000,40.0*1000)
        #         plt.savefig('./figs/' + str(1000000 + i) + '.png', dpi=300)
        #         times.append(t1 - t0)
        #         plt.close()
        #         print('Scalar Integral ', np.sum(s_slice))
        #         print('S-min max', np.amin(w), np.amax(w))

    print('Timing: ', np.min(times),)

    TerminalIO.end_message()

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input for pinacles an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    #Broadcast a uuid and wall time
    unique_id= None
    wall_time = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        unique_id = uuid.uuid4()
        wall_time = datetime.datetime.now()


    unique_id = MPI.COMM_WORLD.bcast(str(unique_id))
    wall_time = MPI.COMM_WORLD.bcast(str(wall_time))

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        namelist['meta']['unique_id'] = unique_id
        namelist['meta']['wall_time'] = wall_time
        main(namelist)
