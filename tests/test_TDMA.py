import numpy as np 
import Columbia.Grid as Grid
import Columbia.Containers as Containers
import Columbia.ReferenceState as ReferenceState
import Columbia.TDMA as TDMA

def test_TDMA():

    a = np.array([0, -1, -1, -1], dtype=np.double)
    b = np.array([4, 4, 4, 4], dtype=np.double)
    c = np.array([-1, -1, -1, 0], dtype=np.double)

    rhs = np.empty((4,4,4), dtype=np.double)
    rhs[:,:,:] = np.array([21, 69, 34, 22], dtype=np.double)[np.newaxis, np.newaxis, :]

    TDMA.Thomas(rhs, a, b, c)

    a_test = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4] ], dtype=np.double)
    b_test = np.array([21, 69, 34, 22], dtype=np.double)

    test_soln = np.linalg.solve(a_test, b_test)
    assert(np.allclose(rhs[:,:,:], test_soln[np.newaxis, np.newaxis, :]))

    return

def build_mocks(n=[16, 16, 100]): 
    namelist = {}
    namelist['grid'] = {} 
    namelist['grid']['n'] = n
    namelist['grid']['l'] = [1000.0, 1000.0, 1000.0]
    namelist['grid']['n_halo'] = [3, 3, 3]

    ModelGrid = Grid.RegularCartesian(namelist)
    ScalarState = Containers.ModelState(ModelGrid, prognostic=True)
    VelocityState = Containers.ModelState(ModelGrid, prognostic=True)
    DiagnosticState = Containers.ModelState(ModelGrid)   
    Ref = ReferenceState.ReferenceDry(namelist, ModelGrid) 
    Ref.set_surface()
    Ref.integrate()

    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w')

    DiagnosticState.add_variable('divergence')
    ScalarState.add_variable('dynamic_pressure')
    ScalarState.add_variable('h')

    #Allocate memory 
    VelocityState.allocate() 
    DiagnosticState.allocate()
    ScalarState.allocate()

    return namelist, Ref, ModelGrid, ScalarState, VelocityState, DiagnosticState


def test_modified_wavenumbers(): 
    n = 16
    namelist, Ref, ModelGrid, ScalarState, VelocityState, DiagnosticState = build_mocks([n,n,n]) 
    
    #Since we are just checking the wave numbers, no need to init any fields
    PresTest = TDMA.PressureTDMA(ModelGrid, Ref)

    #Check that the sizes are correct 
    assert(np.shape(PresTest._kx2) == (16,))
    assert(np.shape(PresTest._ky2) == (16,))

    #The test domain is symmetric so check that kx2 equals ky2
    assert(np.all(PresTest._kx2 == PresTest._ky2))

    #Check that we have removed the odd ball
    assert(PresTest._kx2[0] == 0.0)
    assert(PresTest._ky2[0] == 0.0)

    #Check symmetry not included first and last element
    for i in range(1,int(n/2)-1): 
        assert(PresTest._kx2[i] == PresTest._kx2[-i])
        assert(PresTest._ky2[i] == PresTest._ky2[-i])

    return 