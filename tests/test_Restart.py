import pytest
import pickle
import os
import copy
from pinacles import Restart


class MockSim:
    def __init__(self, namelist):
        self.namelist = namelist
        self.Restart = Restart.Restart(self.namelist)
        return

@pytest.fixture
def mock_sims(tmpdir):

    list_of_sims = []

    namelist = {}
    namelist['meta'] = {}
    namelist['meta']['output_directory'] = tmpdir
    namelist['meta']['simname'] = 'mock_sim' 

    namelist['restart'] = {}
    namelist['restart']['restart_simulation'] = False
    namelist['restart']['frequency'] = 600.0

    list_of_sims.append(MockSim(copy.deepcopy(namelist)))
    
    # Test restart simulation
    
    #First need to create a mock input file to test the read of
    mock_in_files = os.path.join(tmpdir, 'mock_restart_in_files')
    mock_in_files = os.path.join(mock_in_files, '0.0') # Time directory
    mock_in_files = os.path.join(mock_in_files, '0')   # Rank
    os.makedirs(mock_in_files)

    namelist['meta']['simname'] = 'mock_sim_restart' 
    namelist['restart']['restart_simulation'] = True
    namelist['restart']['infile'] = mock_in_files

    with open(os.path.join(mock_in_files, '0.pkl'), 'wb') as f:
        pickle.dump({'a':1}, f)
    
    list_of_sims.append(MockSim(copy.deepcopy(namelist)))

    return list_of_sims

def test_restart_attributes(tmpdir, mock_sims):

    # Test that all simulation have an output path
    assert all(hasattr(sim.Restart, '_path') for sim in mock_sims)


    # Test that required methods exist 
    assert all(hasattr(sim.Restart, 'dump') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'read') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'path') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'frequency') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'infile') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'restart_simulation') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'data_dict') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'purge_data_dict') for sim in mock_sims)
    assert all(hasattr(sim.Restart, '_namelist') for sim in mock_sims)

    # Test that the path property returns the correct value
    assert all(sim.Restart.path == sim.Restart._path for sim in mock_sims)

    #Make sure that the data_dict is a dictionary
    assert all(type(sim.Restart.data_dict) is type({}) for sim in mock_sims)
    
    #Test that the namelist is indeed the namelist
    assert all(sim.namelist == sim.Restart._namelist for sim in mock_sims)

    #Test that is restart is set correctly
    assert all(sim.namelist['restart']['restart_simulation'] == sim.Restart.restart_simulation for sim in mock_sims)

    for sim in mock_sims:
        test_nml = sim.namelist
        test_path = os.path.join(os.path.join(
            test_nml['meta']['output_directory'], 
            test_nml['meta']['simname']), 'Restart')

        assert test_path == sim.Restart._path

        # Test that the restart frequency is set
        assert test_nml['restart']['frequency'] == sim.Restart.frequency

    return

def test_directory_creation(mock_sims):

    assert all(os.path.exists(sim.Restart._path) for sim in mock_sims)
    
    return

def test_dump(mock_sims):
    for sim in mock_sims:
        sim.Restart.dump(0.0)
    
        # Check that the correct directory was created just test the 0 rank for now
        path = os.path.join(os.path.join(sim.Restart.path, str(0.0)), '0.pkl')
        assert os.path.exists(path)

        # Open file and make sure that at this point it is just an empty dictionary except for the namelist
        with open(path, 'rb') as f:
            d = pickle.load(f)   
            assert d['namelist'] == sim.namelist

        # After the dump is complete the data directory should now be empty
        assert sim.Restart.data_dict ==  {}

    return

def test_read(mock_sims):

    for sim in mock_sims:
        if sim.Restart.restart_simulation == True:
            
            # Read in mock restart file and make sure it is correct
            sim.Restart.read()
            assert sim.Restart.data_dict == {'a':1}

            # Make sure that the data dict is purged
            sim.Restart.purge_data_dict()
            assert sim.Restart.data_dict == {}

    return

