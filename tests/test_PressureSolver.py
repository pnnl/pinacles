import numpy as np
import Columbia.PressureSolver_impl as Pres_impl

def test_divergence():

    shape = (10, 10, 10)
    dx = (100.0, 100.0, 100.0)
    u = np.zeros(shape, dtype=np.double)
    v = np.zeros_like(u)
    w = np.zeros_like(v)
    rho0 = np.ones(shape[0], dtype=np.double)
    rho0_edge = np.ones_like(rho0)

    div  = np.zeros_like(u)

    #Compute the divergence
    Pres_impl.divergence(dx, rho0, rho0_edge, 
        u, v, w, div)



    return