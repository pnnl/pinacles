import Columbia.TDMA as TDMA
import numpy as np 

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