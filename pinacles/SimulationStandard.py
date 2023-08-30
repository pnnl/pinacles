import time

global_start_time = time.perf_counter()
import numba
import json
import argparse
import uuid
import datetime
from pinacles import SimulationBase

from pinacles import Initialization
from pinacles import TerminalIO, Grid, ParallelArrays, Containers, Thermodynamics
from pinacles import ScalarAdvectionFactory
from pinacles import ScalarAdvection, TimeStepping, ReferenceState
from pinacles import ScalarDiffusion, MomentumDiffusion
from pinacles import MomentumAdvection
from pinacles import PressureSolverFactory
from pinacles import Damping
from pinacles import SurfaceFactory
from pinacles import ForcingFactory
from pinacles.Stats import Stats
from pinacles import DumpFields
from pinacles import Fields2D
from pinacles import MicrophysicsFactory
from pinacles import RadiationFactory
from pinacles import Kinematics
from pinacles import SGSFactory
from pinacles import DiagnosticsTurbulence
from pinacles import DiagnosticsClouds
from pinacles import TowersIO
from pinacles import Plumes
from pinacles import PlatformSimulator
from pinacles import Restart
from pinacles import UtilitiesParallel
from pinacles import ParticlesFactory
from pinacles import Timers
from pinacles import LateralBCsFactory
from pinacles.ingest import Ingest
from pinacles import DiagnosticsCoarseGrain
from pinacles import DiagnosticsCase
from pinacles import SHOC
##from pinacles import reproducibility
from mpi4py import MPI
import numpy as np
import os
from termcolor import colored


class SimulationStandard(SimulationBase.SimulationBase):
    def __init__(
        self, namelist, llx=0.0, lly=0.0, llz=0.0, ParentNest=None, nest_num=0
    ):

        self.ParentNest = ParentNest

        self._ll_corner = (llx, lly, llz)

        assert type(nest_num) == int
        self._nest_num = nest_num

        # This is used to keep track of how long the model has been running.
        self.t_init = time.perf_counter()
        self.t_init = MPI.COMM_WORLD.bcast(self.t_init)
        self._walltime_restart_dumped = False

        self._namelist = namelist

        # Set up the restart, restart modifies the namelist, so this call should not be moved
        self.Restart = Restart.Restart(namelist)
        #Re = reproducibility.Reproducibility(namelist)

        # Initialize differently if this is a restart simulation
        if not self.Restart.restart_simulation:
            self.initialize(self.ParentNest)
        else:
            self.initialize_from_restart()

        return

    def initialize(self, ParentNest=None):

        # Instantiate the model grid
        self.ModelGrid = Grid.RegularCartesian(
            self._namelist,
            llx=self._ll_corner[0],
            lly=self._ll_corner[1],
            llz=self._ll_corner[2],
        )

        # Instantiate variables for storing containers
        self.ScalarState = Containers.ModelState(
            self._namelist,
            self.ModelGrid,
            container_name="ScalarState",
            prognostic=True,
        )
        self.VelocityState = Containers.ModelState(
            self._namelist,
            self.ModelGrid,
            container_name="VelocityState",
            prognostic=True,
        )

        self.DiagnosticState = Containers.ModelState(
            self._namelist, self.ModelGrid, container_name="DiagnosticState"
        )

        self.TimeSteppingController = TimeStepping.TimeSteppingController(
            self._namelist, self.ModelGrid, self.DiagnosticState, self.VelocityState
        )

        # Ingest data
        self.Ingest = Ingest.IngestFactory(
            self._namelist, self.ModelGrid, self.TimeSteppingController
        )

        self.Timers = Timers.Timer(self._namelist, self.TimeSteppingController)

        # Instantiate the time stepping
        self.ScalarTimeStepping = TimeStepping.factory(
            self._namelist, self.Timers, self.ModelGrid, self.ScalarState
        )
        self.VelocityTimeStepping = TimeStepping.factory(
            self._namelist, self.Timers, self.ModelGrid, self.VelocityState
        )

        # Instantiate Raleigh Damping
        self.RayleighDamping = Damping.Rayleigh(
            self._namelist, self.Timers, self.ModelGrid, self.DiagnosticState
        )
        self.RayleighDamping.add_state(self.VelocityState)
        self.RayleighDamping.add_state(self.ScalarState)

        # Instantiate Time-stepping controller
        self.TimeSteppingController.add_timestepper(self.ScalarTimeStepping)
        self.TimeSteppingController.add_timestepper(self.VelocityTimeStepping)

        # Instantiate the reference state
        self.Ref = ReferenceState.factory(self._namelist, self.ModelGrid)

        # Add three-dimensional velocity components
        self.VelocityState.add_variable(
            "u", long_name="u velocity component", units="m/s", latex_name="u"
        )
        self.VelocityState.add_variable(
            "v", long_name="v velocity component", units="m/s", latex_name="v"
        )
        self.VelocityState.add_variable(
            "w",
            long_name="w velocity component",
            units="m/s",
            latex_name="w",
            loc="z",
            bcs="value zero",
        )

        # self.ScalarState.add_variable(
        #    "spray",
        #    long_name="spray concentration",
        #    units="kg/kg",
        #    latex_name="spray",
        #    flux_divergence='EMONO',
        #    limit=True
        # )

        # Instantiate kinematics and the SGS model
        self.Kine = Kinematics.Kinematics(
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.DiagnosticState,
        )
        self.SGS = SGSFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
        )

        # Instantiate microphysics and thermodynamics
        self.Micro = MicrophysicsFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )
        self.Thermo = Thermodynamics.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.Micro,
        )

        # Instantiate scalar diffusion
        self.ScalarDiff = ScalarDiffusion.ScalarDiffusion(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.DiagnosticState,
            self.ScalarState,
        )
        self.MomDiff = MomentumDiffusion.MomentumDiffusion(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.DiagnosticState,
            self.Kine,
            self.VelocityState,
        )

        # Instantiate the pressure solver
        self.PSolver = PressureSolverFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.DiagnosticState,
        )

        # Instantiate forcing
        self.Force = ForcingFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )

        # Instantiate particles
        self.Parts = ParticlesFactory.ParticlesFactory(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.TimeSteppingController,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        # Instantiate surface
        self.Surf = SurfaceFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
            self.TimeSteppingController,
            self.Ingest,
        )

        # Instantiate plumes if there are any
        self.Plumes = Plumes.Plumes(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.TimeSteppingController,
            self._nest_num,
        )

        # Instantiate radiation
        self.Rad = RadiationFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.Ingest,
            self.ScalarState,
            self.DiagnosticState,
            self.Surf,
            self.Micro,
            self.TimeSteppingController,
        )

        self.SHOC = SHOC.SHOC(self.ModelGrid, self.Ref, self.ScalarState, self.VelocityState, self.DiagnosticState, self.TimeSteppingController, self.Surf)

        

        self.PlatSim = PlatformSimulator.PlatformSimulators(
            self._namelist,
            self.TimeSteppingController,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
        )

        # Add classes to restart
        self.Restart.add_class_to_restart(self.ModelGrid)
        self.Restart.add_class_to_restart(self.ScalarState)
        self.Restart.add_class_to_restart(self.VelocityState)
        self.Restart.add_class_to_restart(self.DiagnosticState)
        self.Restart.add_class_to_restart(self.TimeSteppingController)
        self.Restart.add_class_to_restart(self.Force)
        self.Restart.add_class_to_restart(self.Surf)
        self.Restart.add_class_to_restart(self.Micro)
        self.Restart.add_class_to_restart(self.Rad)
        self.Restart.add_class_to_restart(self.Parts)

        # Allocate memory for storage arrays in container classes. This should come after most classes are instantiated becuase the
        # container must know how much memory to allocate

        # Instantiate scalar advection
        self.ScalarAdv = ScalarAdvectionFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.ScalarTimeStepping,
        )

        self.MomAdv = MomentumAdvection.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
        )

        self.ScalarState.allocate()
        self.VelocityState.allocate()
        self.DiagnosticState.allocate()

        # Allocate and initialize memory in the time-stepping routines
        self.ScalarTimeStepping.initialize()
        self.VelocityTimeStepping.initialize()

        if self.ParentNest is None:
            print("Nest None")
            self.LBC = LateralBCsFactory.LateralBCsFactory(
                self._namelist,
                self.ModelGrid,
                self.Ref,
                self.DiagnosticState,
                self.ScalarState,
                self.VelocityState,
                self.TimeSteppingController,
                self.Ingest,
            )
            self.LBCVel = LateralBCsFactory.LateralBCsFactory(
                self._namelist,
                self.ModelGrid,
                self.Ref,
                self.DiagnosticState,
                self.VelocityState,
                self.VelocityState,
                self.TimeSteppingController,
                self.Ingest,
            )
        else:
            self.LBC = LateralBCsFactory.LateralBCsFactory(
                self._namelist,
                self.ModelGrid,
                self.Ref,
                self.DiagnosticState,
                self.ScalarState,
                self.VelocityState,
                self.TimeSteppingController,
                self.Ingest,
                Parent=self.ParentNest,
                NestState=self.ParentNest.ScalarState,
            )
            self.LBCVel = LateralBCsFactory.LateralBCsFactory(
                self._namelist,
                self.ModelGrid,
                self.Ref,
                self.DiagnosticState,
                self.VelocityState,
                self.VelocityState,
                self.TimeSteppingController,
                self.Ingest,
                Parent=self.ParentNest,
                NestState=self.ParentNest.VelocityState,
            )

        if self.Ingest is not None:
            self.Ingest.initialize()

        # Do case sepcific initalizations the initial profiles are integrated here
        Initialization.initialize(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.Ingest,
        )

        # If necessary initialize Radiation initial profiles.
        self.Rad.init_profiles()
        self.SHOC.initialize()
        self.LBC.init_vars_on_boundary()
        self.LBCVel.init_vars_on_boundary()

        # import sys; sys.exit()
        self.Surf.initialize()

        # Initialize any work arrays for the microphysics package
        self.Micro.initialize()

        # Now that the initial profiles have been integrated, the pressure solver and be initialized
        self.PSolver.initialize()

        # Initialize mean profiles for top of domain damping
        # self.RayleighDamping.init_means()

        # Initialize statistical output
        self.StatsIO = Stats(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.TimeSteppingController,
        )
        self.Fields2d = Fields2D.factory(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )

        self.Fields2d.add_class(self.Micro)
        self.Fields2d.add_class(self.Thermo)
        self.Fields2d.add_class(self.Plumes)
        self.Fields2d.add_class(self.Rad)
        self.Fields2d.add_class(self.Surf)

        # Instantiate optional TowerIO
        self.IOTower = TowersIO.Towers(
             self._namelist,
             self.Timers,
             self.ModelGrid,
             self.TimeSteppingController,
             self.Surf,
             self.Rad,
             self.Micro
        )
        # Add state container to TowerIO
        # Todo move this inside of TowerIO class instantiation
        for state in [self.VelocityState, self.ScalarState, self.DiagnosticState]:
             self.IOTower.add_state_container(state)
        self.IOTower.initialize()

        self.PlatSim.initialize()

        # Initialize statistical diagnostics for turbulence and clouds
        self.DiagClouds = DiagnosticsClouds.DiagnosticsClouds(
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )
        self.DiagTurbulence = DiagnosticsTurbulence.DiagnosticsTurbulence(
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        self.DiagCase = DiagnosticsCase.DiagnosticsCase(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        # Initialize memory for outputting Advective and Diffusive Fluxes
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
        self.StatsIO.add_class(self.DiagCase)
        self.StatsIO.add_class(self.Rad)

        # Now initialize the IO field
        self.StatsIO.initialize()

        # Now initialize for the output of 3D fields
        self.FieldsIO = DumpFields.DumpFieldsFactory(
            self._namelist, self.Timers, self.ModelGrid, self.TimeSteppingController
        )

        # Add container classes that will dump 3D fields
        self.FieldsIO.add_class(self.ScalarState)
        self.FieldsIO.add_class(self.VelocityState)
        self.FieldsIO.add_class(self.DiagnosticState)

        # Set up coarse grainers
        # Set up coarse grainers
        self.CoarseGrain = DiagnosticsCoarseGrain.CoarseGrainFactory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.TimeSteppingController,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.Micro,
        )
        self.CoarseGrain.add_class(self.ScalarState)
        self.CoarseGrain.add_class(self.VelocityState)
        self.CoarseGrain.add_class(self.DiagnosticState)

        # At this point the model is basically initialized, however we should also do boundary exchanges to ensure
        # the halo regions are set and consistent with the pressure solver to ensure that the velocity field initially satisfies
        # the anelastic continuity equation

        self.ScalarState.boundary_exchange()
        self.VelocityState.update_all_bcs()
        self.ScalarState.update_all_bcs()

        for lbc in [self.LBC, self.LBCVel]:
            # lbc.set_vars_on_boundary_to_mean()
            lbc.set_vars_on_boundary(ParentNest=self.ParentNest)
       # self.LBC.inflow_pert(self.LBCVel)

        for prog_state in [self.ScalarState, self.VelocityState]:
            prog_state.boundary_exchange()
            prog_state.update_all_bcs()

        for lbc in [self.LBC, self.LBCVel]:
            lbc.update()

        u = self.VelocityState.get_field("u")
        v = self.VelocityState.get_field("v")
        s = self.ScalarState.get_field("s")
        qv = self.ScalarState.get_field("qv")

        # import pylab as plt
        # plt.subplot(311)
        # plt.plot(u[:,5,5],'.')
        # plt.plot(u[5,:,5],'.')
        # plt.subplot(312)
        # plt.plot(v[:,5,5],'.')
        # plt.plot(v[5,:,5],'.')
        # plt.subplot(313)
        # plt.plot(s[:,5,5], '.')
        # plt.plot(s[5,:,5], '.')
        # plt.show()

        # import pylab as plt
        # plt.plot(u[:,5,5])

        # Update thermo this is mostly for IO at time 0
        self.Thermo.update(apply_buoyancy=False)
        self.Rad.update(force=True)
        self.PSolver.update()
        self.Surf.update()

        # import pylab as plt
        # plt.subplot(311)
        # plt.plot(u[:,5,5],'.')
        # plt.subplot(312)
        # plt.plot(v[:,5,5],'.')
        # plt.subplot(313)
        # plt.plot(s[:,5,5], '.')
        # plt.show()

        # for prog_state in [self.ScalarState, self.VelocityState]:
        #    prog_state.boundary_exchange()
        #    prog_state.update_all_bcs()

        self.ScalarState.boundary_exchange()
        self.VelocityState.update_all_bcs()
        self.ScalarState.update_all_bcs()

        # for lbc in [self.LBC, self.LBCVel]:
        self.LBC.update()
        self.LBCVel.update(normal=False)

        for prog_state in [self.ScalarState, self.VelocityState]:
            prog_state.boundary_exchange()
            prog_state.update_all_bcs()

        # plt.plot(u[:,5,5])
        # plt.show()

        # Initialize timers
        self.Timers.add_timer("Restart")
        self.Timers.add_timer("ScalarLimiter")
        self.Timers.add_timer("BoundaryUpdate")
        self.Timers.add_timer("main")
        self.Timers.initialize()

        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     gather =  self.ModelGrid.CreateGather((0,64), (0,64))
        # else:
        #     gather =  self.ModelGrid.CreateGather((0,64), (0,64))#

        # u = self.VelocityState.get_field('u')
        # u[:,:,:] = 10.0
        # ug = gather.call(u)
        # t0 = time.time()
        # arr = gather.call(u)
        # t1 = time.time()
        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     print(t1 - t0)#

        # if MPI.COMM_WORLD.Get_rank() == 0:
        #     import pylab as plt
        #     plt.pcolor(ug[5,:,:].T)
        #     plt.colorbar()
        #     plt.title('ug')
        #     plt.show()
        # import sys; sys.exit()

        return

    def initialize_from_restart(self):

        # Announce that this is a restart simulation
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("This is a restarted simulation!")
            print("Simulation is being restarted from: ", self.Restart.infile)

        # Instantiate required classes, this sets up the classes that will be need by the simulations.
        # Much of the data in many of these classes will be overwritten by the restart.

        # Instantiate the model grid
        self.ModelGrid = Grid.RegularCartesian(self._namelist)

        # Instantiate variables for storing containers
        self.ScalarState = Containers.ModelState(
            self._namelist,
            self.ModelGrid,
            container_name="ScalarState",
            prognostic=True,
        )
        self.VelocityState = Containers.ModelState(
            self._namelist,
            self.ModelGrid,
            container_name="VelocityState",
            prognostic=True,
        )
        self.DiagnosticState = Containers.ModelState(
            self._namelist, self.ModelGrid, container_name="DiagnosticState"
        )

        self.TimeSteppingController = TimeStepping.TimeSteppingController(
            self._namelist, self.ModelGrid, self.DiagnosticState, self.VelocityState
        )

        self.Timers = Timers.Timer(self._namelist, self.TimeSteppingController)

        # Instantiate the time stepping
        self.ScalarTimeStepping = TimeStepping.factory(
            self._namelist, self.Timers, self.ModelGrid, self.ScalarState
        )
        self.VelocityTimeStepping = TimeStepping.factory(
            self._namelist, self.Timers, self.ModelGrid, self.VelocityState
        )

        self.RayleighDamping = Damping.Rayleigh(
            self._namelist, self.Timers, self.ModelGrid, self.DiagnosticState
        )
        self.RayleighDamping.add_state(self.VelocityState)
        self.RayleighDamping.add_state(self.ScalarState)

        self.TimeSteppingController.add_timestepper(self.ScalarTimeStepping)
        self.TimeSteppingController.add_timestepper(self.VelocityTimeStepping)

        self.Ref = ReferenceState.factory(self._namelist, self.ModelGrid)

        # Add three-dimensional velocity components
        self.VelocityState.add_variable(
            "u", long_name="u velocity component", units="m/s", latex_name="u"
        )
        self.VelocityState.add_variable(
            "v", long_name="v velocity component", units="m/s", latex_name="v"
        )
        self.VelocityState.add_variable(
            "w",
            long_name="w velocity component",
            units="m/s",
            latex_name="w",
            loc="z",
            bcs="value zero",
        )

        # self.ScalarState.add_variable(
        #    "spray",
        #    long_name="spray concentration",
        #    units="kg/kg",
        #    latex_name="spray",
        #    flux_divergence='EMONO',
        #    limit=True
        # )

        # Instantiate kinematics and the SGS model
        self.Kine = Kinematics.Kinematics(
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.DiagnosticState,
        )
        self.SGS = SGSFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
        )

        # Instantiate microphysics and thermodynamics
        self.Micro = MicrophysicsFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )
        self.Thermo = Thermodynamics.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.Micro,
        )

        # Instantiate scalar advection
        self.ScalarAdv = ScalarAdvectionFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.ScalarTimeStepping,
        )
        self.MomAdv = MomentumAdvection.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
        )

        # Instantiate scalar diffusion
        self.ScalarDiff = ScalarDiffusion.ScalarDiffusion(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.DiagnosticState,
            self.ScalarState,
        )

        self.MomDiff = MomentumDiffusion.MomentumDiffusion(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.DiagnosticState,
            self.Kine,
            self.VelocityState,
        )

        # Instantiate the pressure solver
        self.PSolver = PressureSolverFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.DiagnosticState,
        )

        # Instantiate forcing
        self.Force = ForcingFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )

        # Instantiate particles
        self.Parts = ParticlesFactory.ParticlesFactory(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.TimeSteppingController,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        # Instantiate surface
        self.Surf = SurfaceFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
            self.TimeSteppingController,
            self.Ingest,
        )

        # Instantiate plumes if there are any
        self.Plumes = Plumes.Plumes(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.TimeSteppingController,
        )

        # Instantiate radiation
        self.Rad = RadiationFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.Ingest,
            self.ScalarState,
            self.DiagnosticState,
            self.Surf,
            self.Micro,
            self.TimeSteppingController,
        )

        # Add classes to restart
        self.Restart.add_class_to_restart(self.ModelGrid)
        self.Restart.add_class_to_restart(self.ScalarState)
        self.Restart.add_class_to_restart(self.VelocityState)
        self.Restart.add_class_to_restart(self.DiagnosticState)
        self.Restart.add_class_to_restart(self.TimeSteppingController)
        self.Restart.add_class_to_restart(self.Force)
        self.Restart.add_class_to_restart(self.Surf)
        self.Restart.add_class_to_restart(self.Micro)
        self.Restart.add_class_to_restart(self.Rad)
        self.Restart.add_class_to_restart(self.Parts)

        # Allocate memory for storage arrays in container classes. This should come after most classes are instantiated becuase the
        # container must know how much memory to allocate

        self.ScalarAdv = ScalarAdvectionFactory.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.ScalarTimeStepping,
        )

        self.MomAdv = MomentumAdvection.factory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
        )

        self.ScalarState.allocate()
        self.VelocityState.allocate()
        self.DiagnosticState.allocate()

        # Allocate and initialize memory in the time-stepping routines
        self.ScalarTimeStepping.initialize()
        self.VelocityTimeStepping.initialize()

        # Do case specific initializations the initial profiles are integrated here
        Initialization.initialize(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
        )

        # Initialize any work arrays for the microphysics package
        self.Micro.initialize()

        # Now that the initial profiles have been integrated, the pressure solver and be initialized
        self.PSolver.initialize()

        # If necessary initialize Radiation initial profiles.
        self.Rad.init_profiles()

        # Initialize mean profiles for top of domain damping
        self.RayleighDamping.init_means()

        # Initialize statistical output
        self.StatsIO = Stats(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.Ref,
            self.TimeSteppingController,
        )

        # Instantiate optional TowerIO
        self.IOTower = TowersIO.Towers(
             self._namelist,
             self.Timers,
             self.ModelGrid,
             self.TimeSteppingController,
             self.Surf,
             self.Rad,
             self.Micro
         )
        # Add state container to TowerIO
        # Todo move this inside of TowerIO class instantiation
        for state in [self.VelocityState, self.ScalarState, self.DiagnosticState]:
             self.IOTower.add_state_container(state)
        self.IOTower.initialize()

        # Initialize statistical diagnostics for turbulence and clouds
        self.DiagClouds = DiagnosticsClouds.DiagnosticsClouds(
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )
        self.DiagTurbulence = DiagnosticsTurbulence.DiagnosticsTurbulence(
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        self.DiagCase = DiagnosticsCase.DiagnosticsCase(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.Thermo,
            self.Micro,
            self.VelocityState,
            self.ScalarState,
            self.DiagnosticState,
        )

        # Initialize memory for outputting Advective and Diffusive Fluxes
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
        self.StatsIO.add_class(self.Rad)
        self.StatsIO.add_class(self.DiagCase)

        # Now initialize the IO field
        self.StatsIO.initialize()

        self.Fields2d = Fields2D.factory(
            self._namelist,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.TimeSteppingController,
        )
        self.Fields2d.add_class(self.Micro)
        self.Fields2d.add_class(self.Thermo)
        self.Fields2d.add_class(self.Plumes)
        self.Fields2d.add_class(self.Surf)

        # Now initialize for the output of 3D fields
        self.FieldsIO = DumpFields.DumpFieldsFactory(
            self._namelist, self.Timers, self.ModelGrid, self.TimeSteppingController
        )

        # Add container classes that will dump 3D fields
        self.FieldsIO.add_class(self.ScalarState)
        self.FieldsIO.add_class(self.VelocityState)
        self.FieldsIO.add_class(self.DiagnosticState)

        # Set up coarse grainers
        self.CoarseGrain = DiagnosticsCoarseGrain.CoarseGrainFactory(
            self._namelist,
            self.Timers,
            self.ModelGrid,
            self.TimeSteppingController,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
            self.Micro,
        )
        self.CoarseGrain.add_class(self.ScalarState)
        self.CoarseGrain.add_class(self.VelocityState)
        self.CoarseGrain.add_class(self.DiagnosticState)

        # Now overwrite model state with restart
        self.Restart.restart()

        self.PlatSim = PlatformSimulator.PlatformSimulators(
            self._namelist,
            self.TimeSteppingController,
            self.ModelGrid,
            self.Ref,
            self.ScalarState,
            self.VelocityState,
            self.DiagnosticState,
        )

        self.PlatSim.initialize()

        # These boundary updates are probably not necessary, but just to be safe we will do them.
        # At this point the model is basically initialized, however we should also do boundary exchanges to ensure
        # the halo regions are set and consistent with the pressure solver to ensure that the velocity field initially satisfies
        # the anelastic continuity equation
        for prog_state in [self.ScalarState, self.VelocityState]:
            prog_state.boundary_exchange()
            prog_state.update_all_bcs()

        self.Thermo.update(apply_buoyancy=False)
        self.Rad.update(force=True)
        # self.PSolver.update()

        self.Timers.add_timer("Restart")
        self.Timers.add_timer("ScalarLimiter")
        self.Timers.add_timer("BoundaryUpdate")
        self.Timers.add_timer("main")
        self.Timers.initialize()

        return

    def update(self, ParentNest=None, ListOfSims=[], integrate_by_dt=0.0):

        u = self.VelocityState.get_field("u")
        v = self.VelocityState.get_field("v")
        s = self.ScalarState.get_field("s")

        """This function integrates the model forward by integrate_by_dt seconds."""
        # Compute the startime and endtime for this integration
        start_time = self.TimeSteppingController.time
        end_time = start_time + integrate_by_dt

        # Update boundaries for nest if this is Simulation is a nest
        # if ParentNest is not None:
        #    self.Nest.update_boundaries(ParentNest)

        while self.TimeSteppingController.time < end_time:
            self.Timers.start_timer("main")
            #  Start wall time for this time step
            t0 = time.perf_counter()

            # Loop over the Runge-Kutta steps
            for n in range(self.ScalarTimeStepping.n_rk_step):
                # Adjust the timestep at the beginning of the step
                self.TimeSteppingController.adjust_timestep(n, end_time)

                # Update Thermodynamics
                self.Thermo.update()

                # Update the surface
                self.Surf.update()

                # Update plumes if any
                self.Plumes.update()

                # Update the forcing
                self.Force.update()

                # Update Kinematics and SGS model
                self.Kine.update()
                self.SGS.update()

                # Update scalar and momentum advection
                self.ScalarAdv.update()
                self.MomAdv.update()

                # Update scalar and momentum diffusion
                self.ScalarDiff.update()
                self.MomDiff.update()

                #Call shoc update here
                self.SHOC.update()

                # Do Damping
                self.RayleighDamping.update()

                self.LBC.lateral_nudge()
                self.LBCVel.lateral_nudge()

                # if ParentNest is not None:
                #    self.Nest.update(ParentNest)

                # Do time stepping
                self.ScalarTimeStepping.update()
                self.VelocityTimeStepping.update()

                self.VelocityState.boundary_exchange()
                self.VelocityState.update_all_bcs()

                self.LBCVel.update()

                # Call pressure solver
                self.PSolver.update()

                for lbcs in [self.LBC, self.LBCVel]:
                    lbcs.set_vars_on_boundary(ParentNest=self.ParentNest)

                #self.LBC.inflow_pert(self.LBCVel)
                self.LBCVel.update(normal=False)

                self.Timers.start_timer("ScalarLimiter")
                self.ScalarState.apply_limiter()
                self.Timers.end_timer("ScalarLimiter")

                # Update boundary conditions
                self.Timers.start_timer("BoundaryUpdate")

                self.ScalarState.boundary_exchange()
                self.ScalarState.update_all_bcs()

                self.Timers.end_timer("BoundaryUpdate")

                if n == 1:
                    # We call the microphysics update at the end of the RK steps.
                    self.Thermo.update(apply_buoyancy=False)
                    self.Micro.update()
                    if not self.Rad.time_synced:
                        self.Rad.update()
                    self.Rad.update_apply_tend()
                    self.Parts.update()
                    self.Timers.start_timer("BoundaryUpdate")
                    self.ScalarState.boundary_exchange()
                    self.ScalarState.update_all_bcs()

                    self.Timers.end_timer("BoundaryUpdate")

                    # Get a consistant temperature for io
                    self.Thermo.update(apply_buoyancy=False)

            self.LBC.update()

            self.Timers.finish_timestep()
            self.TimeSteppingController._time += self.TimeSteppingController._dt

            # End wall time for
            #  this timestep
            MPI.COMM_WORLD.Barrier()
            t1 = time.perf_counter()
            self.Timers.end_timer("main")

            # Here we dump a restart file if we are getting close to a walltime limit
            if self.Restart.do_walltime_restart and not self._walltime_restart_dumped:
                self.walltime_restart()

            if MPI.COMM_WORLD.Get_rank() == 0:
                print(
                    colored("\t Walltime: ", "green"),
                    colored(t1 - t0, "green"),
                    colored("\tModeltime/Walltime: ", "green"),
                    colored(self.TimeSteppingController._dt / (t1 - t0), "green"),
                )

            # Here we use recursion to update all sub-nests
            if len(ListOfSims) - 1 > self._nest_num:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(
                        "Recursively calling update of Nest " + str(self._nest_num + 1),
                        " from Nest: ",
                        str(self._nest_num),
                    )
                ListOfSims[self._nest_num + 1].update(
                    integrate_by_dt=self.TimeSteppingController._dt,
                    ParentNest=ListOfSims[self._nest_num],
                    ListOfSims=ListOfSims,
                )

            if ParentNest is not None and self.LBCVel.two_way:
                self.LBC.update_parent(dt=self.TimeSteppingController.dt)
                ParentNest.ScalarState.boundary_exchange()
                ParentNest.ScalarState.update_all_bcs()

                self.LBCVel.update_parent(dt=self.TimeSteppingController.dt)
                ParentNest.VelocityState.boundary_exchange()
                ParentNest.VelocityState.update_all_bcs()
                ParentNest.LBCVel.update()
                # Call pressure solver
                ParentNest.PSolver.update()

                for lbcs in [ParentNest.LBC, ParentNest.LBCVel]:
                    lbcs.set_vars_on_boundary(ParentNest=ParentNest)
                #self.LBC.inflow_pert(self.LBCVel)

                ParentNest.LBCVel.update(normal=False)

        return

    def walltime_restart(self):
        t1 = time.perf_counter()
        time_from_start = np.array([t1 - self.t_init], dtype=np.double)
        MPI.COMM_WORLD.Bcast(time_from_start)

        if (
            time_from_start[0] >= self.Restart.walltime_restart
            and not self._walltime_restart_dumped
        ):
            UtilitiesParallel.print_root("\t \t Doing a walltime based restart!")
            self.Restart.dump_restart(self.TimeSteppingController.time)
            self._walltime_restart_dumped = True

        return
