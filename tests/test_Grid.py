import numpy as np
import Columbia.Grid as Grid

namelist = {}
namelist['grid']={}
namelist['grid']['n'] =  (10, 10, 10)
namelist['grid']['n_halo'] = (3,3,3)
namelist['grid']['l']  = (1000.0, 1000.0, 1000.0)

def test_GridInit():

    TestGrid = Grid.RegularCartesian(namelist)

    #Test that things are set correctly
    assert(np.all(TestGrid.n == [10,10,10]))
    assert(np.all(TestGrid.n_halo == [3,3,3]))
    assert(np.all(TestGrid.l == [1000.0, 1000.0, 1000.0]))
    assert(np.all(TestGrid.dx == [100.0, 100.0, 100.0]))
    assert(np.all(TestGrid.dxi == [1.0/100.0, 1.0/100.0, 1.0/100.0]))
    assert(np.all(TestGrid.ngrid == [16, 16, 16]))
    assert(np.all(TestGrid.local_shape == [10, 10, 10]))
    assert(np.all(TestGrid.local_start == [0, 0, 0]))

    x = TestGrid.x_local
    y = TestGrid.y_local
    z = TestGrid.z_local


    return