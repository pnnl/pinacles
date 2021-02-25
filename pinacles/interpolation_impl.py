import numba 

@numba.njit(fastmath=True)
def centered_second(phi, phip1):
    return 0.5*(phi + phip1)

@numba.njit(fastmath=True)
def centered_fourth(phim1, phi, phip1, phip2):
    #return (7.0/12.0)*(phi + phip1 ) -(1.0/12.0)*(phim1 + phip2)
    return (9.0/16.0)*(phi + phip1 ) -(1.0/16.0)*(phim1 + phip2)

@numba.njit(fastmath=True)
def centered_sixth(phim2, phim1, phi, phip1, phip2, phip3):
    return (75.0/128.0)*(phi + phip1 ) -(25.0/256.0)*(phim1 + phip2) + (3.0/256.0)*(phim2 + phip3)

@numba.njit(fastmath=True)
def interp_weno3(phim1, phi, phip1):

    p0 = (-1.0/2.0) * phim1 + (3.0/2.0) * phi
    p1 = (1.0/2.0) * phi + (1.0/2.0) * phip1

    beta1 = (phip1 - phi) * (phip1 - phi)
    beta0 = (phi - phim1) * (phi - phim1)

    alpha0 = (1.0/3.0) /((beta0 + 1e-10) * (beta0 + 1.0e-10))
    alpha1 = (2.0/3.0)/((beta1 + 1e-10) * (beta1 + 1.0e-10))

    alpha_sum_inv = 1.0/(alpha0 + alpha1)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv


    return w0 * p0 + w1 * p1

    return

@numba.njit(fastmath=True)
def interp_weno5(phim2, phim1,  phi, phip1, phip2):

    # Interpolations
    #p0 = (1.0/3.0)*phim2 - (7.0/6.0)*phim1 + (11.0/6.0)*phi
    #p1 = (-1.0/6.0) * phim1 + (5.0/6.0)*phi + (1.0/3.0)*phip1
    #p2 = (1.0/3.0) * phi + (5.0/6.0)*phip1 - (1.0/6.0)*phip2
    p0 = 0.375 * phim2  -1.25* phim1 + 1.875 * phi
    p1 = -0.125*phim1  +  0.75*phi + 0.375 * phip1
    p2 =  0.375*phi  + 0.75 * phip1  -0.125 * phip2

    # Smoothness indicators
    beta0 = (13.0/12.0 * (phim2 - 2.0 * phim1 + phi)*(phim2 - 2.0 * phim1 + phi)
                        + 0.25 * (phim2 - 4.0 * phim1 + 3.0 * phi)*(phim2 - 4.0 * phim1 + 3.0 * phi))
    beta1 = (13.0/12.0 * (phim1 - 2.0 * phi + phip1)*(phim1 - 2.0 * phi + phip1)
                        + 0.25 * (phim1 - phip1)*(phim1 - phip1))
    beta2 = (13.0/12.0 * (phi - 2.0 * phip1 + phip2)*(phi - 2.0 * phip1 + phip2)
                        + 0.25 * (3.0 * phi - 4.0 * phip1 + phip2)*(3.0 * phi - 4.0 * phip1 + phip2))

    #Normalzized stencil weights
    alpha0 = 0.1/((beta0 + 1e-10) * (beta0 + 1e-10))
    alpha1 = 0.6/((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = 0.3/((beta2 + 1e-10) * (beta2 + 1e-10))

    alpha_sum_inv = 1.0/(alpha0 + alpha1 + alpha2)


    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2

@numba.njit(fastmath=True)
def interp_weno7( phim3,  phim2, phim1, phi, phip1,  phip2,  phip3):

   # p0 = (-1.0/4.0)*phim3 + (13.0/12.0) * phim2 + (-23.0/12.0) * phim1 + (25.0/12.0)*phi
   # p1 = (1.0/12.0)*phim2 + (-5.0/12.0)*phim1 + (13.0/12.0)*phi + (1.0/4.0)*phip1
   # p2 = (-1.0/12.0)*phim1 + (7.0/12.0)*phi +  (7.0/12.0)*phip1 + (-1.0/12.0)*phip2
   # p3 = (1.0/4.0)*phi + (13.0/12.0)*phip1 + (-5.0/12.0)*phip2 + (1.0/12.0)*phip3

    p0 = (-0.3125)*phim3 + (1.3125) * phim2 + (-2.1875) * phim1 + (2.1875)*phi
    p1 = (0.0625)*phim2 + (-0.3125)*phim1 + (0.9375)*phi + (0.3125)*phip1
    p2 = (-0.0625)*phim1 + (0.5625)*phi +  (0.5625)*phip1 + (-0.0625)*phip2
    p3 = (0.3125)*phi + (0.9375)*phip1 + (-0.3125)*phip2 + (0.0625)*phip3

    beta0 = (phim3*(547.0*phim3 - 3882.0*phim2 + 4642.0*phim1 - 1854.0*phi)
                         + phim2*(7043.0*phim2 - 17246.0*phim1 + 7042.0*phi)
                         + phim1*(11003.0*phim1 - 9402.0*phi)
                         + 2107.0*phi*phi)
    beta1 =(phim2*(267.0*phim2 - 1642.0*phim1 + 1602.0*phi - 494.0*phip1)
                        + phim1*(2843.0*phim1 - 5966.0*phi + 1922.0*phip1)
                        + phi*(3443.0*phi - 2522.0*phip1)
                        + 547.0*phip1*phip1)
    beta2 = (phim1*(547.0*phim1 - 2522.0*phi + 1922.0*phip1 - 494.0*phip2)
                         + phi*(3443.0*phi -5966.0*phip1 + 1602.0*phip2)
                         + phip1*(2843.0*phip1 - 1642.0*phip2)
                         + 267.0*phip2* phip2)
    beta3 = (phi*(2107.0*phi - 9402.0*phip1 + 7042.0*phip2 - 1854.0*phip3)
                         + phip1*(11003.0*phip1 - 17246.0*phip2 + 4642.0*phip3)
                         + phip2*(7043.0*phip2 - 3882.0*phip3)
                         + 547.0*phip3*phip3)

    alpha0 = (1.0/35.0)/((beta0 + 1e-8) * (beta0 + 1e-8))
    alpha1 = (12.0/35.0)/((beta1 + 1e-8) * (beta1 + 1e-8))
    alpha2 = (18.0/35.0)/((beta2 + 1e-8) * (beta2 + 1e-8))
    alpha3 = (4.0/35.0)/((beta3 + 1e-8) * (beta3 + 1e-8))



    alpha_sum_inv = 1.0/(alpha0 + alpha1 + alpha2 + alpha3)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv
    w3 = alpha3 * alpha_sum_inv


    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3