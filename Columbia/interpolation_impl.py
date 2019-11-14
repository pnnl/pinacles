import numba 

@numba.njit
def centered_second(phi, phip1):
    return 0.5*(phi + phip1)

@numba.njit
def centered_fourth(phim1, phi, phip1, phip2):
    return (9.0/16.0)*(phi + phip1 ) -(1.0/16.0)*(phim1 + phip2);

@numba.njit()
def interp_weno5(phim2, phim1,  phi, phip1, phip2):
    p0 = (1.0/3.0)*phim2 - (7.0/6.0)*phim1 + (11.0/6.0)*phi
    p1 = (-1.0/6.0) * phim1 + (5.0/6.0)*phi + (1.0/3.0)*phip1
    p2 = (1.0/3.0) * phi + (5.0/6.0)*phip1 - (1.0/6.0)*phip2

    beta2 = (13.0/12.0 * (phi - 2.0 * phip1 + phip2)*(phi - 2.0 * phip1 + phip2)
                        + 0.25 * (3.0 * phi - 4.0 * phip1 + phip2)*(3.0 * phi - 4.0 * phip1 + phip2))
    beta1 = (13.0/12.0 * (phim1 - 2.0 * phi + phip1)*(phim1 - 2.0 * phi + phip1)
                        + 0.25 * (phim1 - phip1)*(phim1 - phip1))
    beta0 = (13.0/12.0 * (phim2 - 2.0 * phim1 + phi)*(phim2 - 2.0 * phim1 + phi)
                        + 0.25 * (phim2 - 4.0 * phim1 + 3.0 * phi)*(phim2 - 4.0 * phim1 + 3.0 * phi))

    alpha0 = 0.1/((beta0 + 1e-10) * (beta0 + 1e-10))
    alpha1 = 0.6/((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = 0.3/((beta2 + 1e-10) * (beta2 + 1e-10))

    alpha_sum_inv = 1.0/(alpha0 + alpha1 + alpha2)
    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2