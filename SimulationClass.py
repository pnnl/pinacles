import numba
import json
import argparse
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
from pinacles import Kinematics
from pinacles import SGSFactory
from mpi4py import MPI
import numpy as np
import time
import pylab as plt
import copy
import uuid
import datetime

import os 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

class Simulation: 
    def __init__(self, namelist, domain_number):
        self._namelist = copy.deepcopy(namelist)
        self._domain_number = domain_number
    
    
    
    def initialize(self):
        namelist = self._namelist
        unique_id= None
        wall_time = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            unique_id = str(uuid.uuid4())
            wall_time = str(datetime.datetime.now())
        namelist['meta']['unique_id'] = unique_id
        namelist['meta']['wall_time'] = wall_time

        self.ModelGrid = Grid.RegularCartesian(namelist)
        self.ScalarState = Containers.ModelState(self.ModelGrid, container_name='ScalarState', prognostic=True)
        self.VelocityState = Containers.ModelState(self.ModelGrid, container_name='VelocityState', prognostic=True)
        self.DiagnosticState = Containers.ModelState(self.ModelGrid, container_name='DiagnosticState')
        self.ScalarTimeStepping = TimeStepping.factory(namelist, self.ModelGrid, self.ScalarState)
        self.VelocityTimeStepping = TimeStepping.factory(namelist, self.ModelGrid, self.VelocityState)
        self.TimeSteppingController = TimeStepping.TimeSteppingController(namelist, self.ModelGrid, self.VelocityState)
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
        self.SGS = SGSFactory.factory(namelist, self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        self.Micro = MicrophysicsFactory.factory(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.TimeSteppingController) 
        self.Thermo = Thermodynamics.factory(namelist,self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.Micro)

        # In the future the microphyics should be initialized here

        #Setup the scalar advection calss
        self.ScalarAdv = ScalarAdvectionFactory.factory(namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.ScalarTimeStepping)
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
        self.Surf = SurfaceFactory.factory(namelist, self.ModelGrid, self.Ref, self.VelocityState, self.ScalarState, self.DiagnosticState, self.TimeSteppingController)
        self.Force = ForcingFactory.factory(namelist, self.ModelGrid, self.Ref, self.Micro, self.VelocityState, self.ScalarState, self.DiagnosticState, self.TimeSteppingController)
        self.PSolver.initialize() #Must be called after reference profile is integrated

        #Setup Stats-IO
        self.StatsIO = Stats(namelist, self.ModelGrid, self.Ref, self.TimeSteppingController)

        self.ScalarDiff.initialize_io_arrays()
        self.ScalarAdv.initialize_io_arrays()

        self.StatsIO.add_class(self.VelocityState)
        self.StatsIO.add_class(self.ScalarState)
        self.StatsIO.add_class(self.DiagnosticState)
        self.StatsIO.add_class(self.Micro)


        self.FieldsIO = DumpFields.DumpFields(namelist, self.ModelGrid, self.TimeSteppingController)
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
        while self.TimeSteppingController.time < timestop:
            for n in range(self.ScalarTimeStepping.n_rk_step):
                self.TimeSteppingController.adjust_timestep(n)

                #Update Thermodynamics
                self.Thermo.update()

                #Do StatsIO if it is time
                if n == 0:
                    self.StatsIO.update()
                    MPI.COMM_WORLD.barrier()

                #Update microphysics
                #self.Micro.update()

                #Update the surface
                self.Surf.update()

                #Update the forcing
                self.Force.update()

                #Update Kinematics
                self.Kine.update()
                self.SGS.update()

                #Update scalar advection
                self.ScalarAdv.update()
                self.MomAdv.update()

                self.ScalarDiff.update()
                self.MomDiff.update()

                #Do Damping
                self.RayleighDamping.update()

                #Apply large scale forcing
                for v in ls_forcing:
                    self.ScalarState.get_tend(v)[:,:,:] += ls_forcing[v][np.newaxis,np.newaxis,:] 

                #Do time stepping
                self.ScalarTimeStepping.update()
                self.VelocityTimeStepping.update()

                #Update boundary conditions
                self.ScalarState.boundary_exchange()
                self.VelocityState.boundary_exchange()
                self.ScalarState.update_all_bcs()
                self.VelocityState.update_all_bcs()

                #Call pressure solver
                self.PSolver.update()
                if n== 1: 
                    self.Thermo.update(apply_buoyancy=False)
                    #We call the microphysics update at the end of the RK steps.
                    self.Micro.update()
                    self.ScalarState.boundary_exchange()
                    self.ScalarState.update_all_bcs()

        return


