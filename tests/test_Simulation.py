import pytest
import os
import copy
import numpy as np
from pinacles import SimulationBase, SimulationStandard


#Here we do some tests on the base class
@pytest.fixture
def base_mocks():

    base_mocks = []
    base_mocks.append(SimulationBase.SimulationBase({}))

    return base_mocks

def test_base_attributes(base_mocks):

    #est that the class has the correct methods
    assert all(hasattr(base, 'initialize') for base in base_mocks)
    assert all(hasattr(base, 'initialize_from_restart') for base in base_mocks)
    assert all(hasattr(base, 'update') for base in base_mocks)
    
    #Make sure the namelist is stored correctly
    assert all(base._namelist == {} for base in base_mocks)
    return


#Here we do some tests on the standard class
@pytest.fixture
def standard_mocks(tmpdir):

    base_mocks = []

    namelist = {}
    namelist['meta'] = {}
    namelist['meta']['simname'] = 'simtest_no_restart' 
    namelist['meta']['casename'] = 'sullivan_and_patton'
    namelist['meta']['output_directory'] = os.path.join(tmpdir, namelist['meta']['casename'])
    namelist['meta']['unique_id'] = 'not_unique'
    namelist['meta']['wall_time'] = '12:00:00'

    namelist['restart'] = {}
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['restart_simulation'] = False

    namelist['grid'] = {}
    namelist['grid']['n'] = [6, 6, 6] 
    namelist['grid']['n_halo'] = [3, 3, 3]
    namelist['grid']['l'] = [2000.0, 2000.0, 2000.0]

    namelist['time'] = {}
    namelist['time']['cfl'] = 0.6
    namelist['time']['time_max'] = 1800.0

    namelist['damping'] = {}
    namelist['damping']['vars'] = ['s', 'w']
    namelist['damping']['depth'] = 100.0
    namelist['damping']['timescale'] = 600.0

    namelist['momentum_advection'] = {}
    namelist['momentum_advection']['type'] = 'weno5'

    namelist['scalar_advection'] = {}
    namelist['scalar_advection']['type'] = 'weno5'

    namelist['stats'] = {}
    namelist['stats']['frequency'] = 60.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'base'

    namelist['Thermodynamics'] = {}
    namelist['Thermodynamics']['type'] = 'dry'

    namelist['sgs'] = {}
    namelist['sgs']['model'] = 'smagorinsky'

    # Generate simulations for dry cases
    base_mocks.append(SimulationStandard.SimulationStandard(copy.deepcopy(namelist)))

    # Generate simulations for moist cases
    for casename in ['bomex', 'rico']:

        namelist['Thermodynamics'] = {}
        namelist['Thermodynamics']['type'] = 'moist'

        namelist['meta']['casename'] = casename
        namelist['meta']['output_directory'] = os.path.join(tmpdir, namelist['meta']['casename'])
        
        for micro in ['kessler', 'p3']:
            namelist['microphysics']['scheme'] = micro

        base_mocks.append(SimulationStandard.SimulationStandard(copy.deepcopy(namelist)))

    return base_mocks


def test_simulation_standard_attributes(standard_mocks):

    #Test that the class has the correct methods
    assert all(hasattr(base, 'initialize') for base in standard_mocks)
    assert all(hasattr(base, 'initialize_from_restart') for base in standard_mocks)
    assert all(hasattr(base, 'update') for base in standard_mocks)
    
    return

def test_simulation_standard_files_created(standard_mocks):
    ''' Test to make sure that each simulation creates the correct output files 
        in the correct directory hieararchy.'''

    for sim in standard_mocks:

        out_dir = sim._namelist['meta']['output_directory']
        assert os.path.exists(out_dir)
        
        case_dir = os.path.join(out_dir, sim._namelist['meta']['simname'])
        assert os.path.exists(case_dir)

        assert os.path.exists(os.path.join(case_dir, 'input.json'))
        assert os.path.exists(os.path.join(case_dir, 'Fields'))
        assert os.path.exists(os.path.join(case_dir, 'stats.nc'))

    return

def test_simulation_standard_update(standard_mocks):

    #First test with the default dt=0.0 make sure nothing changes 
    for sim in standard_mocks:
        scalar_array = np.copy(sim.ScalarState._state_array.array)
        vel_array = np.copy(sim.VelocityState._state_array.array)

        # Call update with the default value of zero
        sim.update()

        assert np.array_equal(scalar_array, sim.ScalarState._state_array.array)
        assert np.array_equal(vel_array, sim.VelocityState._state_array.array)
        
    for sim in standard_mocks:

        # Make of copy of the fields before the update
        scalar_array = np.copy(sim.ScalarState._state_array.array)
        vel_array = np.copy(sim.VelocityState._state_array.array)

        # Call update without a default value
        sim.update(integrate_by_dt=1.0)

        # The fields should now have been updated in time.
        assert not np.array_equal(scalar_array, sim.ScalarState._state_array.array)
        assert not np.array_equal(vel_array, sim.VelocityState._state_array.array)


    return


