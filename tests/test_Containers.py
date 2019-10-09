import numpy as np
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
    DiagState = ModelState(Grid, prognostic=False)

    #Let's add three fields to each state
    for i in range(3): 
        ProgState.add_variable(str(i))    
        DiagState.add_variable(str(i))

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