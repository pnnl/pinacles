import numpy as np
import numba
from Columbia import Surface_impl

GAMMA_M = 15.0
GAMMA_H = 9.0 
VKB = 0.35

@numba.njit
def psi_m_unstable(zeta, zeta0):
    x = (1.0 - GAMMA_M * zeta)**0.25
    x0 = (1.0 - GAMMA_H * zeta0)**0.25
    psi_m = 2.0 * np.log((1.0 + x)/(1.0 + x0)) + np.log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*np.atan(x)+2.0*np.atan(x0)

    return psi_m

@numba.njit
def psi_h_unstable(zeta, zeta0):

    y = np.sqrt(1.0 - GAMMA_H * zeta )
    y0 = np.sqrt(1.0 - GAMMA_H * zeta0 )
    psi_h = 2.0 * np.log((1.0 + y)/(1.0+y0))

    return psi_h

@numba.njit
def psi_m_stable(zeta, zeta0):
    psi_m = -beta_m * (zeta - zeta0)
    return psi_m

@numba.njit
def psi_h_stable(zeta, zeta0):
    psi_h = -beta_h * (zeta - zeta0)
    return psi_h

@numba.njit
def compute_ustar(windspeed, buoyancy_flux, z0, zb):
    logz = np.log(zb/z0)
    #use neutral condition as first guess
    ustar0 = windspeed * VKB/logz  
    if(np.abs(buoyancy_flux) > 1.0e-20):
        lmo = -ustar0 * ustar0 * ustar0/(buoyancy_flux * VKB)
        zeta = zb/lmo
        zeta0 = z0/lmo
        if(zeta >= 0.0):
            f0 = windspeed - ustar0/VKB*(logz - psi_m_stable(zeta,zeta0))
            ustar1 = windspeed*VKB/(logz - psi_m_stable(zeta,zeta0))
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * VKB)
            zeta = zb/lmo
            zeta0 = z0/lmo
            f1 = windspeed - ustar1/VKB*(logz - psi_m_stable(zeta,zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while(np.abs(delta_ustar) > 1e-10):
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * VKB)
                zeta = zb/lmo
                zeta0 = z0/lmo
                f1 = windspeed - ustar1/VKB*(logz - psi_m_stable(zeta,zeta0))
                delta_ustar = ustar1 -ustar0
        else:
            f0 = windspeed - ustar0/VKB*(logz - psi_m_unstable(zeta,zeta0))
            ustar1 = windspeed*VKB/(logz - psi_m_unstable(zeta,zeta0))
            lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * VKB)
            zeta = zb/lmo
            zeta0 = z0/lmo
            f1 = windspeed - ustar1/VKB*(logz - psi_m_unstable(zeta,zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while(np.abs(delta_ustar) > 1e-10):
                ustar_new = ustar1 - f1 * delta_ustar/(f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1/(buoyancy_flux * VKB)
                zeta = zb/lmo
                zeta0 = z0/lmo
                f1 = windspeed - ustar1/VKB*(logz - psi_m_unstable(zeta,zeta0))
                delta_ustar = ustar1 -ustar0
    else:
        ustar = ustar0
    return ustar
