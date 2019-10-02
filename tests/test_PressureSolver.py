import numpy as np
import Columbia.PressureSolver_impl as PresSolve

def test_divergence():

    shape = (10, 10, 10)
    dx = (100.0, 100.0, 100.0)
    u = np.zeros(shape, order='F', dtype=np.double)
    v = np.zeros_like(u)
    w = np.zeros_like(v)

    div  = np.zeros_like(u)

    #Compute the divergence
    PresSolve.divergence(dx, u, v, w, div)


    return