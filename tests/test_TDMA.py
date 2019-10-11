import Columbia.TDMA as TDMA
import numpy as np 

def test_TDMA():

    a = np.array([0, -1, -1, -1], dtype=np.double)
    b = np.array([4, 4, 4, 4], dtype=np.double)
    c = np.array([-1, -1, -1, 0], dtype=np.double)

    rhs = np.empty((4,4,4), dtype=np.double)
    rhs[:,:,:] = np.array([5, 5, 10, 23], dtype=np.double)[np.newaxis, np.newaxis, :]

    TDMA.Thomas(rhs, a, b, c)


    return