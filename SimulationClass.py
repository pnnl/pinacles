import numba
import json
import argparse
from Columbia import Initializaiton
from Columbia import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from Columbia import ScalarAdvection, TimeStepping, ReferenceState
from Columbia import ScalarDiffusion, MomentumDiffusion
from Columbia import MomentumAdvection
from Columbia import PressureSolver
from Columbia import Damping
from Columbia import SurfaceFactory
from Columbia import ForcingFactory
from Columbia.Stats import Stats
from Columbia import DumpFields
from Columbia import MicrophysicsFactory
from Columbia import Kinematics
from Columbia import SGSFactory
from Columbia import Parallel
from mpi4py import MPI
import numpy as np
import time
import pylab as plt
import copy

import os 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

class Simulation: 
    def __init__(self, namelist, domain_number, LOCAL_COMM):
        self._namelist = copy.deepcopy(namelist)
        self._domain_number = domain_number
        self._Parallel = Parallel.Parallel(LOCAL_COMM)
    
        return
    
    def initialize(self):
        namelist = self._namelist

        self.ModelGrid = Grid.RegularCartesian(namelist, self._Parallel)

 
        self.ScalarState = Containers.ModelState(self.ModelGrid, self._Parallel, container_name='ScalarState', prognostic=True)
        self.VelocityState = Containers.ModelState(self.ModelGrid, self._Parallel, container_name='VelocityState', prognostic=True)
        self.DiagnosticState = Containers.ModelState(self.ModelGrid, self._Parallel, container_name='DiagnosticState')

        self.ScalarTimeStepping = TimeStepping.factory(namelist, self.ModelGrid, self._Parallel, self.ScalarState)
        self.VelocityTimeStepping = TimeStepping.factory(namelist, self.ModelGrid, self._Parallel, self.VelocityState)
        self.TimeSteppingController = TimeStepping.TimeSteppingController(namelist, self.ModelGrid, self._Parallel, self.VelocityState)
        self.RayleighDamping = Damping.RayleighInitial(namelist, self.ModelGrid)
        self.RayleighDamping.add_state(self.VelocityState)
        self.RayleighDamping.add_state(self.ScalarState)

        self.TimeSteppingController.add_timestepper(self.ScalarTimeStepping)
        self.TimeSteppingController.add_timestepper(self.VelocityTimeStepping)

        self.Ref =  ReferenceState.factory(namelist, self.ModelGrid)
       
        # Add velocity variables
        self.VelocityState.add_variable('u')
        self.VelocityState.add_variable('v')
        self.VelocityState.add_variable('w', loc='z', bcs='value zero')

        # Set up the thermodynamics class
        self.Kine = Kinematics.Kinematics(self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        self.SGS = SGSFactory.factory(namelist, self.ModelGrid, self._Parallel, self.Ref, self.VelocityState, self.DiagnosticState)
        self.Micro = MicrophysicsFactory.factory(namelist, self.ModelGrid, self._Parallel, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.TimeSteppingController) 
        self.Thermo = Thermodynamics.factory(namelist,self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.Micro)

        # In the future the microphyics should be initialized here

        #Setup the scalar advection calss
        self.ScalarAdv = ScalarAdvection.factory(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.ScalarTimeStepping)
        self.MomAdv = MomentumAdvection.factory(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState)

        self.ScalarDiff = ScalarDiffusion.ScalarDiffusion(namelist, self.ModelGrid, self.Ref, self.DiagnosticState, self.ScalarState)
        self.MomDiff = MomentumDiffusion.MomentumDiffusion(namelist, self.ModelGrid, self.Ref, self.DiagnosticState, self.Kine, self.VelocityState)


        #Setup the pressure solver
        self.PSolver = PressureSolver.factory(namelist, self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        #self.Micro = MicrophysicsFactory.factory(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.TimeSteppingController)

        # Allocate all of the big parallel arrays needed for the container classes
        self.ScalarState.allocate()
        self.VelocityState.allocate()
        self.DiagnosticState.allocate()

        self.ScalarTimeStepping.initialize()
        self.VelocityTimeStepping.initialize()

        Initializaiton.initialize(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState)

        self.RayleighDamping.init_means()
        self.Surf = SurfaceFactory.factory(namelist, self.ModelGrid, self.Ref, self.VelocityState, self.ScalarState, self.DiagnosticState)
        self.Force = ForcingFactory.factory(namelist, self.ModelGrid, self.Ref, self.VelocityState, self.ScalarState, self.DiagnosticState)
        self.PSolver.initialize() #Must be called after reference profile is integrated

        #Setup Stats-IO
        self.StatsIO = Stats(namelist, self.ModelGrid, self._Parallel, self.Ref, self.TimeSteppingController)

        self.StatsIO.add_class(self.VelocityState)
        self.StatsIO.add_class(self.ScalarState)
        self.StatsIO.add_class(self.DiagnosticState)
        self.StatsIO.add_class(self.Micro)


        self.FieldsIO = DumpFields.DumpFields(namelist, self.ModelGrid, self.TimeSteppingController, self._Parallel)
        self.FieldsIO.add_class(self.ScalarState)
        self.FieldsIO.add_class(self.VelocityState)
        self.FieldsIO.add_class(self.DiagnosticState)


        self.StatsIO.initialize()


        self.ScalarState.boundary_exchange()
        self.VelocityState.boundary_exchange()
        self.ScalarState.update_all_bcs()
        self.VelocityState.update_all_bcs()
        self.PSolver.update()


    def update(self, timestop, ls_forcing):
        timing = {}

        for key in ['micro','pressure','bc', 'time_int', 'MMF', 'Rayleigh', 'Diff', 'Adv', 'SGS', 'Forcing', 'surf', 'Thermo']:
            timing[key] = 0.0

        while self.TimeSteppingController.time < timestop:
            for n in range(self.ScalarTimeStepping.n_rk_step):
                self.TimeSteppingController.adjust_timestep(n)

                t0 = time.time() 
                #Update Thermodynamics
                self.Thermo.update()
                t1 = time.time()
                timing['Thermo'] += t1 - t0

                #Do StatsIO if it is time
                if n == 0:
                    self.StatsIO.update()
                    #self._Parallel self._Parallel 

                #Update microphysics
                #self.Micro.update()

                #Update the surface
                t0 = time.time() 
                self.Surf.update()
                t1 = time.time()
                timing['surf'] += t1 - t0

                #Update the forcing
                t0 = time.time() 
                self.Force.update()
                t1 = time.time()
                timing['Forcing'] += t1 - t0

                #Update Kinematics
                t0 = time.time() 
                self.Kine.update()
                self.SGS.update()
                t1 = time.time()
                timing['SGS'] += t1 - t0

                #Update scalar advection
                t0 = time.time() 
                self.ScalarAdv.update()
                self.MomAdv.update()
                t1 = time.time()
                timing['Adv'] += t1 - t0

                t0 = time.time() 
                self.ScalarDiff.update()
                self.MomDiff.update()
                t1 = time.time() 
                timing['Diff'] += t1 - t0

                #Do Damping
                t0 = time.time() 
                self.RayleighDamping.update()
                t1 = time.time() 
                timing['Rayleigh'] += t1 - t0

                t0 = time.time()
                #Apply large scale forcing
                for v in ls_forcing:
                    self.ScalarState.get_tend(v)[:,:,:] += ls_forcing[v][np.newaxis,np.newaxis,:] 
                t1 = time.time()
                timing['MMF'] += t1 - t0

                #Do time stepping
                t0 = time.time() 
                self.ScalarTimeStepping.update()
                self.VelocityTimeStepping.update()
                t1 = time.time() 
                timing['time_int'] += t1 - t0

                #Update boundary conditions
                
                self.ScalarState.boundary_exchange()
                self.VelocityState.boundary_exchange()
                self.ScalarState.update_all_bcs()
                self.VelocityState.update_all_bcs()
                timing['bc'] += t1 - t0

                #Call pressure solver
                t0 = time.time()
                self.PSolver.update()
                t1 = time.time()
                timing['pressure'] += t1 - t0
                
                t0 = time.time()
                if n== 1: 
                    self.Thermo.update(apply_buoyancy=False)
                    #We call the microphysics update at the end of the RK steps.
                    self.Micro.update()
                    self.ScalarState.boundary_exchange()
                    self.ScalarState.update_all_bcs()
                t1 = time.time()
                timing['micro'] += t1 - t0
        print(timing)
        return

