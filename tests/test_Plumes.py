import pytest
import os
import copy
import numpy as np
from pinacles import Plumes
from pinacles import SimulationStandard

#Here we do some tests on the standard class
@pytest.fixture
def standard_plume_mocks(tmpdir):

    base_mocks = []

    namelist = {}
    namelist['meta'] = {}
    namelist['meta']['simname'] = 'simtest_no_restart' 
    namelist['meta']['casename'] = 'sullivan_and_patton'
    namelist['meta']['output_directory'] = os.path.join(tmpdir, namelist['meta']['casename'])
    namelist['meta']['unique_id'] = 'not_unique'
    namelist['meta']['wall_time'] = '12:00:00'

    namelist['restart'] = {}
    namelist['restart']['frequency'] = 10.0
    namelist['restart']['restart_simulation'] = False

    namelist['grid'] = {}
    namelist['grid']['n'] = [10, 10, 10] 
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

    # 
    namelist['plumes'] = {}
    namelist['plumes']['locations'] = [[500.0, 500.0, 100.0],
                                        [1000.0, 1000.0, 500.0]]
    namelist['plumes']['starttimes'] = [0.0,
                                        20.0]
    namelist['plumes']['plume_flux'] = [1.0, 
                                  2.0]
    namelist['plumes']['qv_flux'] = [1e-05, 
                                     1e-5]
    namelist['plumes']['ql_flux'] = [0.0, 
                                     0.0]
    namelist['plumes']['heat_flux'] = [100.0, 
        100.0]


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

def test_plume_attributes(standard_plume_mocks):

    for sims in standard_plume_mocks:
        list_of_plumes = sims.Plumes._list_of_plumes

        for count, plume in enumerate(list_of_plumes):
            assert sims._namelist['plumes']['ql_flux'][count] == plume.plume_ql_flux
            assert sims._namelist['plumes']['qv_flux'][count] == plume.plume_qv_flux
            assert sims._namelist['plumes']['heat_flux'][count] == plume.plume_heat_flux
            assert sims._namelist['plumes']['plume_flux'][count] == plume.plume_flux
            assert count == plume.plume_number
            assert 'plume_' + str(count) == plume.scalar_name
    
    return

def test_added_scalar(standard_plume_mocks):

    for sims in standard_plume_mocks:
        list_of_plumes = sims.Plumes._list_of_plumes
        for count, plume in enumerate(list_of_plumes):

            scalar = sims.ScalarState.get_field(plume.scalar_name)
            assert np.all(scalar == 0.0)
            scalar = sims.ScalarState.get_tend(plume.scalar_name)
            assert np.all(scalar == 0.0)

    return


def test_update(standard_plume_mocks):

    dt = 10.0
    
    for sims in standard_plume_mocks:
        # Integrate forwared by dt
        sims.update(dt)
    

    return