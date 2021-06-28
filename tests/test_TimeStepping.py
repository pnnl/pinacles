import numpy as np
from pinacles import TimeStepping_impl


def test_compute_local_cfl_max():

    nhalo = (3, 1, 2)
    dxi = (1.0 / 50.0, 1.0 / 20.0, 1.0 / 10.0)

    asize = (16, 17, 19)
    for float_type in [np.double, np.single]:
        u = np.zeros(asize, dtype=float_type)
        v = np.zeros_like(u)
        w = np.zeros_like(u)

        u.fill(1.0)
        v.fill(2.0)
        w.fill(3.0)

        cfl_max, umax, vmax, wmax = TimeStepping_impl.comput_local_cfl_max(
            nhalo, dxi, u, v, w
        )

        assert umax == 1.0
        assert vmax == 2.0
        assert wmax == 3.0

        cfl_test_val = 1.0 * dxi[0] + 2.0 * dxi[1] + 3.0 * dxi[2]
        assert cfl_test_val == cfl_max

    return


def test_compute_local_diff_num_max():

    dt = 5.0
    nhalo = (3, 3, 3)
    dxi = (1.0 / 50.0, 1.0 / 20.0, 1.0 / 10.0)
    asize = (16, 17, 19)

    for float_type in [np.double, np.single]:
        Km = np.zeros(asize, dtype=float_type)

        Km.fill(2.0)

        diff_num_max = TimeStepping_impl.compute_local_diff_num_max(nhalo, dxi, dt, Km)
        diff_num_max_test_val = (
            dt * 2.0 * ((dxi[0] * dxi[0]) + (dxi[1] * dxi[1]) + (dxi[2] * dxi[2]))
        )
        assert diff_num_max == diff_num_max_test_val

    return
