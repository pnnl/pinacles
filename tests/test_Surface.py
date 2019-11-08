import numpy as np
from Columbia import Surface_impl

def test_compute_ustar():


    #Check that ustart monotonically increases regardless of the sign of the buoyancy flux
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