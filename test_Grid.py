from Columbia import Grid
import numpy as np 
import json 


json_in = open('./tests/testing_inputs/domain_test.json', 'r')
namelist = json.load(json_in)

grid = Grid.RegularCartesian(namelist)

def test_namelist(): 

    #First make sure that the namelist is a dictionary 
    assert type(namelist) is type({})

    #Now make sure it has a member grid
    assert 'grid'  in namelist

    #Now make sure grid has all of the necessary variables 
    assert 'n' in namelist['grid']
    assert 'n_halo' in namelist['grid']
    assert 'l' in namelist['grid']
    
    #Now make sure each of these has the correct type and length
    assert type(namelist['grid']['n']) is type([])
    assert type(namelist['grid']['n_halo']) is type([])
    assert type(namelist['grid']['l']) is type([])

    assert len(namelist['grid']['n']) == 3 
    assert len(namelist['grid']['n_halo']) == 3 
    assert len(namelist['grid']['l']) == 3 

    return 

def test_RegularCartesianGrid_properties(): 

    assert np.all(namelist['grid']['n'] == grid.n) 
    assert np.all(namelist['grid']['n_halo'] == grid.n_halo)
    assert np.all(namelist['grid']['l'] == grid.l) 

    return 


