import numpy as np
import Columbia.MomentumAdvection_impl as Mom_impl

def test_uv_flux_divergence():

    shape = (10, 10, 10)
    fluxx = np.zeros(shape, order='F', dtype=np.double)
    fluxy = np.zeros_like(fluxx)
    fluxz = np.zeros_like(fluxy)
    uvt = np.zeros_like(fluxx)

    #First we setup a test where we give a known gradient u_flux
    x = np.linspace(0.0, 1000.0, shape[0], endpoint=False)
    y = np.linspace(0.0, 1000.0, shape[1], endpoint=False)
    z = np.linspace(0.0, 1000.0, shape[2], endpoint=False)
    alpha0_half = np.ones_like(z)

    dx = 1000.0/shape[0]
    dy = 1000.0/shape[1]
    dz = 1000.0/shape[2]


    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                fluxz[i,j,k] =  -2.0 * z[k]

    #Call the flux divergence
    Mom_impl.uv_flux_div(1.0/dx, 1.0/dy, 1.0/dz, alpha0_half, fluxx, fluxy, fluxz, uvt)

    assert(np.all(uvt[1:-1,1:-1,1:-1] == 2.0))

    #Reset the x fluxes
    fluxz[:,:,:] = 0.0

    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                fluxx[i,j,k] =  -3.0 * x[i]

    #Call the flux divergence
    Mom_impl.uv_flux_div(1.0/dx, 1.0/dy, 1.0/dz, alpha0_half, fluxx, fluxy, fluxz, uvt)

    assert(np.all(uvt[1:-1,1:-1,1:-1] == 5.0))

    return



def test_w_flux_divergence():

    shape = (10, 10, 10)

    return