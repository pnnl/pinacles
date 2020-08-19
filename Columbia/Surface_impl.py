import numpy as np
import numba
from Columbia import Surface_impl

GAMMA_M = 15.0
GAMMA_H = 9.0
VKB = 0.35
PR0 = 0.74
BETA_M = 4.7
BETA_H = BETA_M/PR0

@numba.njit
def psi_m_unstable(zeta, zeta0):
    x = (1.0 - GAMMA_M * zeta)**0.25
    x0 = (1.0 - GAMMA_M * zeta0)**0.25
    psi_m = 2.0 * np.log((1.0 + x)/(1.0 + x0)) + np.log((1.0 + x*x)/(1.0 + x0 * x0))-2.0*np.arctan(x)+2.0*np.arctan(x0)

    return psi_m

@numba.njit
def psi_h_unstable(zeta, zeta0):

    y = np.sqrt(1.0 - GAMMA_H * zeta )
    y0 = np.sqrt(1.0 - GAMMA_H * zeta0 )
    psi_h = 2.0 * np.log((1.0 + y)/(1.0+y0))

    return psi_h

@numba.njit
def psi_m_stable(zeta, zeta0):
    psi_m = -BETA_M * (zeta - zeta0)
    return psi_m

@numba.njit
def psi_h_stable(zeta, zeta0):
    psi_h = -BETA_H * (zeta - zeta0)
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

@numba.njit()
def compute_windspeed_sfc(u, v, u0, v0, gustiness, windspeed):

    shape = windspeed.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            ui = 0.5 * (u[i-1,j] + u[i,j]) + u0
            vi = 0.5 * (v[i,j-1] + v[i,j]) + v0
            spd = np.sqrt(ui*ui + vi*vi)
            windspeed[i,j] = max(spd, gustiness)

    return


@numba.njit()
def compute_ustar_sfc(windspeed_sfc, buoyancy_flux_sfc, z0, zb, ustar_sfc):

    shape = ustar_sfc.shape
    for i in range(1,shape[0]):
        for j in range(1,shape[1]):
            ustar_sfc[i,j] = compute_ustar(windspeed_sfc[i,j], buoyancy_flux_sfc[i,j], z0, zb)
    return


@numba.njit()
def tau_given_ustar(ustar_sfc, usfc, vsfc, u0, v0, windspeed_sfc, taux_sfc, tauy_sfc):
    shape = ustar_sfc.shape
    for i in range(1,shape[0]-1):
        for j in range(1, shape[1]-1):
            ustar_at_u = 0.5 * (ustar_sfc[i,j] + ustar_sfc[i+1,j])
            ustar_at_v = 0.5 * (ustar_sfc[i,j] + ustar_sfc[i,j+1])


            windspeed_at_u = 0.5 * (windspeed_sfc[i,j] +windspeed_sfc[i+1,j])
            windspeed_at_v = 0.5 * (windspeed_sfc[i,j] +windspeed_sfc[i,j+1])

            taux_sfc[i,j] = -(ustar_at_u * ustar_at_u)/windspeed_at_u * (usfc[i,j] + u0)
            tauy_sfc[i,j] = -(ustar_at_v * ustar_at_v)/windspeed_at_v * (vsfc[i,j] + v0)

    return

@numba.njit
def iles_surface_flux_application(hd, z_edge, dxi2, nh, alpha0, alpha0_edge, zmax, flux, tend):
    shape = tend.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):
            for k in range(1, shape[2]-1):
                #if z_edge[k] <= zmax:
                #    tend[i,j,k] -= (flux[i,j] * (np.exp(-z_edge[k]/hd)/alpha0_edge[k] - np.exp(-z_edge[k-1]/hd)/alpha0_edge[k-1])) * alpha0[k] * dxi2
                tend[i,j,nh[2]] -= -flux[i,j]/alpha0_edge[nh[2]-1]*dxi2*alpha0[nh[2]]
    return

@numba.njit
def momentum_bulk_aero(windspeed_sfc, cm, u, v, u0, v0, taux, tauy):
    shape = windspeed_sfc.shape
    for i in range(1, shape[0]-1):
        for j in range(1, shape[1]-1):

            windspeed_at_u = 0.5 * (windspeed_sfc[i,j] +windspeed_sfc[i+1,j])
            windspeed_at_v = 0.5 * (windspeed_sfc[i,j] +windspeed_sfc[i,j+1])

            taux[i,j] = -cm * windspeed_at_u  * (u[i,j] + u0)
            tauy[i,j] = -cm * windspeed_at_v  * (v[i,j] + v0)

    return