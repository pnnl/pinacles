from Columbia import Kinematics_impl

import numpy as np

def test_gradients():

    n = (12, 13, 14)


    # First check that a field with zero graident indeed has zero gradient

    u = np.ones(n, dtype=np.double)

    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dudz = np.zeros_like(u)

    dxi = (1.0/2.0, 1.0/4.0, 1.0/6.0)

    Kinematics_impl.u_gradients(dxi, u, dudx, dudy, dudz)

    assert(np.all(dudx == 0.0))
    assert(np.all(dudy == 0.0))
    assert(np.all(dudz == 0.0))


    #Now check that we exactly approximate the gradient of a linear function
    x = np.arange(n[0], dtype=np.double)*(1.0/dxi[0])
    y = np.arange(n[1], dtype=np.double)*(1.0/dxi[1])
    z = np.arange(n[2], dtype=np.double)*(1.0/dxi[2])

    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                u[i,j,k] = 2.0*x[i] -3.0*y[j] + 5.0*z[k]

    Kinematics_impl.u_gradients(dxi, u, dudx, dudy, dudz)
    assert(np.all(dudx[1:-1,1:-1,1:-1] == 2.0))
    assert(np.all(dudy[1:-1,1:-1,1:-1] == -3.0))
    assert(np.all(dudz[1:-1,1:-1,1:-1] == 5.0))


    #Now test the v graident function
    v = np.ones(n, dtype=np.double)
    dvdx = np.zeros_like(v)
    dvdy = np.zeros_like(v)
    dvdz = np.zeros_like(v)

    Kinematics_impl.v_gradients(dxi, v, dvdx, dvdy, dvdz)
    assert(np.all(dvdx == 0.0))
    assert(np.all(dvdy == 0.0))
    assert(np.all(dvdz == 0.0))

    #Now check that we exactly approximate the gradient of a linear function
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                v[i,j,k] = 2.0*x[i] -3.0*y[j] + 5.0*z[k]

    Kinematics_impl.v_gradients(dxi, v, dvdx, dvdy, dvdz)
    assert(np.all(dvdx[1:-1,1:-1,1:-1] == 2.0))
    assert(np.all(dvdy[1:-1,1:-1,1:-1] == -3.0))
    assert(np.all(dvdz[1:-1,1:-1,1:-1] == 5.0))


    #Now test the w gradient function
    w = np.ones(n, dtype=np.double)
    dwdx = np.zeros_like(w)
    dwdy = np.zeros_like(w)
    dwdz = np.zeros_like(w)
    Kinematics_impl.w_gradients(dxi, u, dwdx, dwdy, dwdz)

    #Now check that we exactly approximate the gradient of a linear function
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                w[i,j,k] = 2.0*x[i] -3.0*y[j] + 5.0*z[k]
    Kinematics_impl.w_gradients(dxi, w, dwdx, dwdy, dwdz)
    assert(np.all(dwdx[1:-1,1:-1,1:-1] == 2.0))
    assert(np.all(dwdy[1:-1,1:-1,1:-1] == -3.0))
    assert(np.all(dwdz[1:-1,1:-1,1:-1] == 5.0))

    return 