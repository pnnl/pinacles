import pytest
import pickle
import os
import copy
import numpy as np
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

    np.random.seed(0)
    with open(os.path.join(mock_in_files, '0.pkl'), 'wb') as f:
        pickle.dump({'a':1, 'b': 2.0, 'c':np.random.randn(30)}, f)
    
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
    assert all(hasattr(sim.Restart, 'restart') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'dump_restart') for sim in mock_sims)
    assert all(hasattr(sim.Restart, 'add_class_to_restart') for sim in mock_sims)
    
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
            # Reset the random seed so we get the same array as before
            np.random.seed(0)
            test_dict = {'a':1, 'b': 2.0, 'c':np.random.randn(30)}
            assert sim.Restart.data_dict['a'] == test_dict['a']
            assert sim.Restart.data_dict['b'] == test_dict['b']
            assert np.array_equal(sim.Restart.data_dict['c'], test_dict['c'])
            # Make sure that the data dict is purged
            sim.Restart.purge_data_dict()
            assert sim.Restart.data_dict == {}

    return


@pytest.fixture()
def mock_full_dump(tmpdir):
    # This test fixture is used to setup a simulation that dumps a restart file. 

    namelist = {}
    namelist['meta'] = {}
    namelist['meta']['output_directory'] = tmpdir
    namelist['meta']['simname'] = 'mock_sim_restart_full_functionality' 

    namelist['restart'] = {}
    namelist['restart']['restart_simulation'] = False
    namelist['restart']['frequency'] = 100.0

    return MockSim(namelist)

@pytest.fixture()
def mock_full_restart(tmpdir, mock_full_dump):
    # This test fixutre is used to setup a restart simulation, based on the simulation 
    # this is mocked in mock_full_dump.

    namelist = {}
    namelist['meta'] = {}
    namelist['meta']['output_directory'] = tmpdir
    namelist['meta']['simname'] = 'mock_sim_restart_full_functionality_restart' 

    namelist['restart'] = {}
    namelist['restart']['restart_simulation'] = True
    namelist['restart']['frequency'] = 100.0

    # We use the mock_full_dump test fixture above to create the restart_infiles
    # this means that mock_full_dump must be called before mock_full retstart. Note
    # this hasn't caused problems on systems we have tested on thus far, but may 
    # in the future, depending on the order in which pytest executes the fixtures. 
    # Worst case scenario this causes tests to fail. 
    restart_infiles = os.path.join(mock_full_dump.Restart._path, '10.0')
    namelist['restart']['infile'] = restart_infiles
    print('Restart Infiles', namelist['restart']['infile'] )

    return MockSim(namelist)


def test_full_functionality(mock_full_dump, mock_full_restart):

    class fake_class:
        ''' This is just a helper class used to make testing easier. The important point is that it has the methods
        restart and dump_restart that are required by any class that is to be restarted.
        '''

        def __init__(self):
            self.a = 20.0 
            self.b = 'a'
            self.c = np.random.randn(50)
            
            self.a_restart = None
            self.b_restart = None
            self.c_restart = None
            return

        def restart(self, restart_data_dict):

            self.a_restart = restart_data_dict['fake_class']['a']
            self.b_restart = restart_data_dict['fake_class']['b']
            self.c_restart = restart_data_dict['fake_class']['c']

            return

        def dump_restart(self, restart_data_dict):
            
            restart_data_dict['fake_class'] = {}
            restart_data_dict['fake_class']['a'] = self.a
            restart_data_dict['fake_class']['b'] = self.b
            restart_data_dict['fake_class']['c'] = self.c
            
            return

  
    fake_instance = fake_class()
    mock_full_dump.Restart.add_class_to_restart(fake_instance)

    # There should now be 1 class 
    assert mock_full_dump.Restart.n_classes == 1
    assert type(mock_full_dump.Restart._classes_to_restart) is type([])

    #Makes sure we can call dump restart
    mock_full_dump.Restart.dump_restart(str(10.0))
    assert os.path.exists(mock_full_dump.Restart.path)
    time_path = os.path.join(mock_full_dump.Restart._path, '10.0')
    assert os.path.exists(time_path)
    rank_path = os.path.join(time_path, '0.pkl')
    assert os.path.exists(time_path)


    mock_full_restart.Restart.add_class_to_restart(fake_instance)
    # Now try to read from the restart path
    mock_full_restart.Restart.restart()

    #Now test to make sure that the read was successful and returned values were correct
    assert fake_instance.a_restart == fake_instance.a    
    assert fake_instance.b_restart == fake_instance.b
    assert np.array_equal(fake_instance.c_restart, fake_instance.c)

    return
