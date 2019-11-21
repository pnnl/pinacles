import numpy as np
from Columbia import Surface_impl
from Columbia import SurfaceFactory
from Columbia.Containers import ModelState
from Columbia.Grid import RegularCartesian
from Columbia.ReferenceState import ReferenceDry

def test_compute_ustar():

    #Check that ustar monotonically increases regardless of the sign of the buoyancy flux
    bflux = [-0.005, 0.0, 0.005]
    for bf in bflux:
        windspeed = np.linspace(0.1, 30.0, 60)
        ustar = np.zeros_like(windspeed)

        count = 0
        for wspd in windspeed:
            ustar[count] = Surface_impl.compute_ustar(wspd, bf, 0.01, 5.0)
            count += 1

        #Check that ustart monotonically increases
        np.all(ustar[:-1] < ustar[1:])

        #TODO add more testing here

    return

def mock_grid(n=16, nhalo=3):
    namelist = {}
    namelist['grid'] = {}
    namelist['grid']['n'] = [n,n,n]
    namelist['grid']['n_halo'] = [nhalo, nhalo, nhalo]
    namelist['grid']['l'] = [1000.0, 1000.0, 1000.0]

    namelist['meta'] = {}
    namelist['meta']['casename'] = 'sullivan_and_patton'


    return namelist, RegularCartesian(namelist)

namelist, Grid = mock_grid()

def test_SullivanAndPatton():

    VelState = ModelState(Grid, prognostic=True)
    ScalarState = ModelState(Grid, prognostic=True, identical_bcs=True)
    DiagState = ModelState(Grid, prognostic=False)

    #Let's add three fields to each state
    for i in range(3):
        VelState.add_variable(str(i))
        ScalarState.add_variable(str(i))
        DiagState.add_variable(str(i))

    VelState.add_variable('u')
    VelState.add_variable('v')
    ScalarState.add_variable('s')

    VelState.allocate()
    ScalarState.allocate()
    DiagState.allocate()

    Ref = ReferenceDry(namelist, Grid)
    Ref.set_surface()
    Ref.integrate()

    Surf = SurfaceFactory.factory(namelist, Grid, Ref, VelState, ScalarState, DiagState)


    Surf.update()

    return