
import numba
import json
import argparse
import uuid
import datetime
from pinacles import SimulationBase

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
from pinacles import Restart   
from mpi4py import MPI
import numpy as np
import time
import pylab as plt

import os 
from termcolor import colored
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"


class SimulationStandard(SimulationBase.SimulationBase):

    def __init__(self, namelist):
        
        self._namelist = namelist

        #Set-up the restart
        self.Restart = Restart.Restart(namelist)

        #Initialize differently if this is a restart simulation
        if not self.Restart.restart_simulation:
            self.initialize()
        else:
            self.initialize_restart()

        return

    def initialize(self):

        # Instantiate the model grid
        self.ModelGrid = Grid.RegularCartesian(self._namelist)

        # Instantiate variables for storing containers
        self.ScalarState = Containers.ModelState(self.ModelGrid, container_name='ScalarState', prognostic=True)
        self.VelocityState = Containers.ModelState(self.ModelGrid, container_name='VelocityState', prognostic=True)
        self.DiagnosticState = Containers.ModelState(self.ModelGrid, container_name='DiagnosticState')

        # Instantiate the time stepping
        self.ScalarTimeStepping = TimeStepping.factory(self._namelist, self.ModelGrid, self.ScalarState)
        self.VelocityTimeStepping = TimeStepping.factory(self._namelist, self.ModelGrid, self.VelocityState)
        self.TimeSteppingController = TimeStepping.TimeSteppingController(self._namelist, self.ModelGrid, self.VelocityState)
        
        self.RayleighDamping = Damping.RayleighInitial(self._namelist, self.ModelGrid)
        self.RayleighDamping.add_state(self.VelocityState)
        self.RayleighDamping.add_state(self.ScalarState)
            
        self.TimeSteppingController.add_timestepper(self.ScalarTimeStepping)
        self.TimeSteppingController.add_timestepper(self.VelocityTimeStepping)

        self.Ref = ReferenceState.factory(self._namelist, self.ModelGrid)

        # Add three dimensional velocity compoonents
        self.VelocityState.add_variable('u', long_name = 'u velocity component', units='m/s', latex_name = 'u')
        self.VelocityState.add_variable('v', long_name = 'v velocity component', units='m/s', latex_name = 'v')
        self.VelocityState.add_variable('w', long_name = 'w velocity component', units='m/s', latex_name = 'w', loc='z', bcs='value zero')       

        # Instantiate kinematics and the SGS model
        self.Kine = Kinematics.Kinematics(self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        self.SGS = SGSFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        
        # Instantiate microphysics and thermodynamics
        self.Micro = MicrophysicsFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.TimeSteppingController)        
        self.Thermo = Thermodynamics.factory(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.Micro)
        
        # Instantiate scalar advection
        self.ScalarAdv = ScalarAdvectionFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.ScalarTimeStepping)
        self.MomAdv = MomentumAdvection.factory(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState)
        
        # Instantiate scalar diffusion 
        self.ScalarDiff = ScalarDiffusion.ScalarDiffusion(self._namelist, self.ModelGrid, self.Ref, self.DiagnosticState, self.ScalarState)
        self.MomDiff = MomentumDiffusion.MomentumDiffusion(self._namelist, self.ModelGrid, self.Ref, self.DiagnosticState, self.Kine, self.VelocityState)
        
        # Instantiate the pressure solver
        self.PSolver = PressureSolver.factory(self._namelist, self.ModelGrid, self.Ref, self.VelocityState, self.DiagnosticState)
        
        # Instantiate forcing
        self.Force = ForcingFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.Micro, self.VelocityState, self.ScalarState, self.DiagnosticState, self.TimeSteppingController)

        # Instantiate surface
        self.Surf= SurfaceFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.VelocityState, self.ScalarState, self.DiagnosticState, self.TimeSteppingController)

        # Instantiate radiation
        self.Rad = RadiationFactory.factory(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.DiagnosticState, self.Surf, self.TimeSteppingController)       

        # Allocate memory for storage arrays in container classes. This should come after most classes are instantiated becuase the 
        # containter must know how much memory to allocate
        self.ScalarState.allocate()
        self.VelocityState.allocate()
        self.DiagnosticState.allocate()

        # Allocate and initialze memory in the time-stepping routines
        self.ScalarTimeStepping.initialize()
        self.VelocityTimeStepping.initialize()

        # Do case sepcific initalizations the initial profiles are integrated here
        Initializaiton.initialize(self._namelist, self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState)
        
        # Now that the initial profiles have been integrated, the pressure solver and be initialzied
        self.PSolver.initialize()

        # If necessary initalize Radiation initial profiles. 
        self.Rad.init_profiles() 

        # Initialize mean profiles for top of domain damping
        self.RayleighDamping.init_means()

        # Intialize statistical output
        self.StatsIO = Stats(self._namelist, self.ModelGrid, self.Ref, self.TimeSteppingController)

        # Instantiate optional TowerIO
        self.IOTower= TowersIO.Towers(self._namelist, self.ModelGrid, self.TimeSteppingController)
        # Add state container to TowerIO
        # Todo move this inside of TowerIO class instantiation
        for state in [self.VelocityState, self.ScalarState, self.DiagnosticState]:
            self.IOTower.add_state_container(state)
        self.IOTower.initialize()

        #Initialze statistical diagnostics for turbulence and clouds 
        self.DiagClouds = DiagnosticsClouds.DiagnosticsClouds(self.ModelGrid, self.Ref, self.Thermo, self.Micro, self.VelocityState, self.ScalarState, self.DiagnosticState)
        self.DiagTurbulence = DiagnosticsTurbulence.DiagnosticsTurbulence(self.ModelGrid, self.Ref, self.Thermo, self.Micro, self.VelocityState, self.ScalarState, self.DiagnosticState)
        
        # Initalize memory for outputting Advective and Diffusive Fluxes
        self.ScalarDiff.initialize_io_arrays()
        self.ScalarAdv.initialize_io_arrays()

        # Add all statistical io classes
        self.StatsIO.add_class(self.Surf)
        self.StatsIO.add_class(self.ScalarAdv)
        self.StatsIO.add_class(self.ScalarDiff)
        self.StatsIO.add_class(self.VelocityState)
        self.StatsIO.add_class(self.ScalarState)
        self.StatsIO.add_class(self.DiagnosticState)
        self.StatsIO.add_class(self.Micro)
        self.StatsIO.add_class(self.DiagTurbulence)
        self.StatsIO.add_class(self.DiagClouds)

        # Now iniitalzie the IO field
        self.StatsIO.initialize()

        # Now initialze for the output of 3D fields
        self.FieldsIO = DumpFields.DumpFields(self._namelist, self.ModelGrid, self.TimeSteppingController)
        # Add container classes that will dump 3D fields
        self.FieldsIO.add_class(self.ScalarState)
        self.FieldsIO.add_class(self.VelocityState)
        self.FieldsIO.add_class(self.DiagnosticState)

        # At this point the model is basically initalized, however we should also do boundary exchanges to insure 
        # the halo regions are set and the to a pressure solver to insure that the velocity field is initially satifies
        # the anelastic continuity equation
        for prog_state in [self.ScalarState, self.VelocityState]:
            prog_state.boundary_exchange()
            prog_state.update_all_bcs()

        # Update thermo this is mostly for IO at time 0
        self.Thermo.update(apply_buoyancy=False)
        
        self.PSolver.update()

        return

    def initialize_from_restart(self):


        return
    
    def update(self, integrate_by_dt = 0.0):

        """ This function integrates the model forward by integrate_by_dt seconds. """

        #Compute the startime and endtime for this integration 
        start_time = self.TimeSteppingController.time
        end_time = start_time + integrate_by_dt

        while self.TimeSteppingController.time < end_time:
            
            #  Start wall time for this time step
            t0 = time.perf_counter()

            # Loop over the Runge-Kutta steps 
            for n in range(self.ScalarTimeStepping.n_rk_step):
                
                #Adjust the timestep at the beginning of the step
                self.TimeSteppingController.adjust_timestep(n, end_time)

                #Update Thermodynamics
                self.Thermo.update()

                #Update the surface
                self.Surf.update()
                
                #Update the forcing
                self.Force.update()
                self.Rad.update(n)

                #Update Kinematics and SGS model
                self.Kine.update()
                self.SGS.update()

                #Update scalar and momentum advection
                self.ScalarAdv.update()
                self.MomAdv.update()

                #Update scalar and momentum diffusion
                self.ScalarDiff.update()
                self.MomDiff.update()

                #Do Damping
                self.RayleighDamping.update()

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


            self.TimeSteppingController._time += self.TimeSteppingController._dt

            # End wall time for this timestep
            t1 = time.perf_counter()
            MPI.COMM_WORLD.Barrier()

            if MPI.COMM_WORLD.Get_rank() == 0:
                print(colored('\t Walltime: ', 'green'), colored(t1 -t0, 'green'), 
                    colored('\tModeltime/Walltime: ', 'green'), 
                    colored(self.TimeSteppingController._dt/(t1 - t0), 'green'))
    
        return
