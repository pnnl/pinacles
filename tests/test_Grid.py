import numpy as np
import Columbia.Grid as Grid

namelist = {}
namelist['grid']={}
namelist['grid']['n'] =  (10, 10, 10)
namelist['grid']['n_halo'] = (3,3,3)
namelist['grid']['l']  = (1000.0, 1000.0, 1000.0)

def test_Grid(): 

    TestGrid = Grid.RegularCartesian(namelist)

    #Test that things are set correctly
    assert(np.all(TestGrid.n == [10,10,10]))


    return 