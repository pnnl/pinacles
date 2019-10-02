import numpy as np
import Columbia.ParallelArrays as ParallelArrays
import Columbia.Grid as Grid

#First we setup a grid class that requries a mock namelist
namelist = {}
namelist['grid']={}
namelist['grid']['n'] =  (10, 10, 10)
namelist['grid']['n_halo'] = (3,3,3)
namelist['grid']['l']  = (1000.0, 1000.0, 1000.0)

TestGrid = Grid.RegularCartesian(namelist)

#Now create a parallel array with 2 dofs
TestArray = ParallelArrays.GhostArray(TestGrid, ndof=2)

def test_GhostArraySetZero():

    TestArray.array[:,:,:,:] = 1.0
    assert(np.all(TestArray.array[:,:,:,:] == 1.0))

    #Set all dofs to zero
    TestArray.zero()
    assert(np.all(TestArray.array == 0.0))

    #Check that we can set one DOF at a time to zero
    TestArray.array[:,:,:,:] = 1.0
    TestArray.zero(dof=0)
    assert(np.all(TestArray.array[:,:,:,0] == 0.0))
    assert(np.all(TestArray.array[:,:,:,1] == 1.0))

    TestArray.array[:,:,:,:] = 1.0
    TestArray.zero(dof=1)
    assert(np.all(TestArray.array[:,:,:,0] == 1.0))
    assert(np.all(TestArray.array[:,:,:,1] == 0.0))

    return

def test_GhostArraySetValue():
    TestArray.zero()
    assert(np.all(TestArray.array == 0.0))

    TestArray.set(10.0)
    assert(np.all(TestArray.array == 10.0))

    TestArray.zero()
    assert(np.all(TestArray.array == 0.0))

    TestArray.set(10.0, dof=0)
    assert(np.all(TestArray.array[:,:,:,0] == 10.0))
    assert(np.all(TestArray.array[:,:,:,1] == 0.0))

    TestArray.zero()
    assert(np.all(TestArray.array == 0.0))
    TestArray.set(10.0, dof=1)
    assert(np.all(TestArray.array[:,:,:,0] == 0.0))
    assert(np.all(TestArray.array[:,:,:,1] == 10.0))

    return