import Columbia.Damping as Damping
from Columbia.Containers import ModelState
from Columbia.Grid import RegularCartesian


def mock_grid(n=16, nhalo=3): 
    namelist = {} 
    namelist['grid'] = {} 
    namelist['grid']['n'] = [n,n,n] 
    namelist['grid']['n_halo'] = [nhalo, nhalo, nhalo]
    namelist['grid']['l'] = [1000.0, 1000.0, 1000.0]

    namelist['Damping'] = {}
    namelist['Damping']['Vars'] = ['P1','P2']
    namelist['Damping']['depth'] = 500.0
    namelist['Damping']['timescale'] = 100.0

    return namelist, RegularCartesian(namelist)

namelist, Grid = mock_grid()
ProgState = ModelState(Grid, prognostic=True)
ScalarState = ModelState(Grid, prognostic=True, identical_bcs=True)
DiagState = ModelState(Grid, prognostic=False)

#Let's add three fields to each state
for i in range(3):
    ProgState.add_variable('P' + str(i))
    ScalarState.add_variable('S' + str(i))
    DiagState.add_variable('D' + str(i))

ProgState.allocate()
ScalarState.allocate()
DiagState.allocate()


def test_RayleighDamping():

    TestDamping = Damping.Rayleigh(namelist, Grid)

    for var in ['P1', 'P2']:
        assert(var in TestDamping.vars)

    #Check the damping depth
    assert(TestDamping.depth == 500.0)

    TestDamping.add_state(ProgState)
    TestDamping.add_state(ScalarState)

    #Check that the sates are added
    assert(ProgState in TestDamping._states)
    assert(ScalarState in TestDamping._states)
    assert(len(TestDamping._states) == 2)


    TestDamping.update()


    return
