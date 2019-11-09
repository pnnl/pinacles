import numpy as np
import pytest
from Columbia.Containers import ModelState
from Columbia.Grid import RegularCartesian

def mock_grid(n=16, nhalo=3):
    namelist = {}
    namelist['grid'] = {}
    namelist['grid']['n'] = [n,n,n]
    namelist['grid']['n_halo'] = [nhalo, nhalo, nhalo]
    namelist['grid']['l'] = [1000.0, 1000.0, 1000.0]


    return namelist, RegularCartesian(namelist)

namelist, Grid = mock_grid()

def test_allocation():

    ProgState = ModelState(Grid, prognostic=True)
    ScalarState = ModelState(Grid, prognostic=True, identical_bcs=True)
    DiagState = ModelState(Grid, prognostic=False)

    #Let's add three fields to each state
    for i in range(3):
        ProgState.add_variable(str(i))
        ScalarState.add_variable(str(i))
        DiagState.add_variable(str(i))


    assert(ProgState.identical_bcs is False)
    assert(ScalarState.identical_bcs is True)

    # The states have not yet allocated so check that the
    # underlying ParallelArrays are none
    assert(ProgState.get_state_array is None)
    assert(ProgState.get_tend_array is None)
    assert(DiagState.get_state_array is None)
    assert(DiagState.get_tend_array is None)

    #Check that the number of variables is correct
    assert(ProgState.nvars == 3)
    assert(DiagState.nvars == 3)

    ProgState.allocate()
    ScalarState.allocate()
    DiagState.allocate()

    # After DiagState has allocated the tend array should
    # still be none
    assert(DiagState.get_tend_array is None)

    #Now check that get field retruns an array
    for i in range(3):
         field = ProgState.get_field(str(i))
         assert(np.shape(field) == tuple(Grid.ngrid))

         tend_field = ProgState.get_field(str(i))
         assert(np.shape(tend_field) == tuple(Grid.ngrid))

         diag_field = DiagState.get_field(str(i))
         assert(np.shape(diag_field) == tuple(Grid.ngrid))

    return

def test_update_bcs():

    ScalarState = ModelState(Grid, prognostic=True, identical_bcs=True)
    VelState = ModelState(Grid, prognostic=False)

    ScalarState.add_variable('q')
    ScalarState.add_variable('s')

    VelState.add_variable('u', bcs='gradient zero', loc=('xe, y, z'))
    VelState.add_variable('v', bcs='gradient zero', loc=('x', 'ye', 'z'))
    VelState.add_variable('w', bcs='value zero', loc=('x', 'y', 'ze'))

    ScalarState.allocate()
    VelState.allocate()

    ScalarState.update_all_bcs()

    s = ScalarState.get_field('q')
    q = ScalarState.get_field('s')

    s[:,:,:] = np.arange(s.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]
    q[:,:,:] = np.arange(q.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]*2.0

    ScalarState.update_all_bcs()
    nh2 = Grid.n_halo[2]

    #Test the lower boundary condition
    assert(np.all(s[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(q[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    #Test the upper boundary condition
    n = s.shape[2]
    assert(np.all(s[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(q[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    s[:,:,:] = np.arange(s.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]
    q[:,:,:] = np.arange(q.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]*2.0

    ScalarState._gradient_zero_bc('s')
    ScalarState._gradient_zero_bc('q')
    #Test the lower boundary condition
    assert(np.all(s[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(q[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    #Test the upper boundary condition
    n = s.shape[2]
    assert(np.all(s[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(q[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    u = VelState.get_field('u')
    v = VelState.get_field('v')
    w = VelState.get_field('w')

    u[:,:,:] = np.arange(u.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]
    v[:,:,:] = np.arange(v.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]*2.0
    w[:,:,:] = np.arange(w.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]
    VelState._zero_value_bc('w')
    assert(np.all(w[:,:,nh2-1] == 0.0))
    assert(np.all(w[:,:,-nh2-1] == 0.0))

    w[:,:,:] = np.arange(w.shape[2], dtype=np.double)[np.newaxis, np.newaxis,:]
    VelState.update_all_bcs()
    assert(np.all(w[:,:,nh2-1] == 0.0))
    assert(np.all(w[:,:,-nh2-1] == 0.0))

    #Test the lower boundary condition
    assert(np.all(u[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(v[:,:,:nh2] == np.arange(nh2, 2*nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    #Test the upper boundary condition
    n = s.shape[2]
    assert(np.all(u[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]))
    assert(np.all(v[:,:,-nh2:] == np.arange(n-2*nh2, n-nh2, dtype=np.double)[np.newaxis, np.newaxis, ::-1]*2.0))

    return