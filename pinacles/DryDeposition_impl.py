import numba
import numpy as np
from pinacles import parameters
from pinacles.Surface_impl import psi_h_stable, psi_h_unstable

# Gravitational settling velocity
@numba.njit
def v_gravitational(particle_density, particle_diameter, temperature, air_density, pressure):
    mu = viscosity(temperature)
    c = cunningham_factor(particle_diameter, temperature, air_density, pressure)
    vg = particle_density * particle_diameter * particle_diameter * parameters.G * c /(18.0*mu)

    return vg

# Sutherland equation
@numba.njit
def viscosity(temperature):
    b = 1.458e-6 # kg/m/s/K^{1/2}
    S = 110.4 # K

    return b * temperature**1.5/(temperature + S)

@numba.njit
def cunningham_factor(particle_diameter, temperature, air_density, pressure):
    # approximation for mean free path  (Jennings, J. Aerosol Science, 1988)
    u = 0.4987445 # empirical factors
    mfp = np.sqrt(np.pi * 0.125/air_density/pressure) * viscosity(temperature)/u
    c = 1.0 + 2.0 * mfp/particle_diameter * (1.257 + 0.4 *np.exp(-0.55 * particle_diameter/mfp))
    return c

@numba.njit
def r_aerodynamic(z,z0, psi_h,ustar):
    return (np.log(z/z0) - psi_h)/(parameters.VONKARMAN * ustar)

@numba.njit
def r_surface(particle_diameter, air_density, temperature, vg, ustar):
    # particle diffusivity (approx)
    D = 0.65e-11/(particle_diameter)
    nu = viscosity(temperature)/air_density
    gamma = 0.5 # empirical value
    E_b = (nu/D)**(-gamma) # nu/D = Sc, 0.5 = gamma, empirical value

    St = vg*ustar*ustar/nu # Stokes number for 
    alpha = 100.0
    beta = 2.0
    E_im = (St/(alpha + St))** beta
    return 1.0/3.0/ustar/(E_b + E_im)
# compute dry deposition velocity for a given particle size and density
@numba.njit
def compute_dry_deposition_velocity( dry_particle_diameter, T,rh, nh,z, rho0, p0,shf, lhf, ustar, z0, vdep):
    shape = T.shape
    R_aero = np.zeros_like(rho0)
    particle_density = 2165.0 #kg/m^3, assuming sea-salt
    C1 = 0.7674
    C2 = 3.079
    C3 = 2.573e-11
    C4 = -1.424
    for i in range(nh[0], shape[0]-nh[0]):
        for j in range(nh[1],shape[1]-nh[1]):
            # Aerodynamic resistance depends on stability condition 
            bflux = ((parameters.G/parameters.CPD/T[i,j,nh[2]] * shf[i,j] 
                       + (parameters.EPSVI -1.0) * lhf[i,j]/parameters.LV)/rho0[nh[2]])
            if bflux > 1e-5:    
                lmo = -ustar[i,j]**3/bflux/parameters.VONKARMAN
                zeta0 = z0[i,j]/lmo
                for k in range(nh[2],shape[2]-nh[2]+1):
                    zeta = z[k]/lmo
                    psi_h = psi_h_unstable(zeta, zeta0)
                    R_aero[k] = r_aerodynamic(z[k], z0[i,j], psi_h, ustar[i,j])
            
            elif bflux < 1e-5:
                lmo = -ustar[i,j]**3/bflux/parameters.VONKARMAN
                zeta0 = z0[i,j]/lmo
                for k in range(nh[2],shape[2]-nh[2]+1):
                    zeta = z[k]/lmo
                    psi_h = psi_h_stable(zeta, zeta0)
                    R_aero[k] = r_aerodynamic(z[k], z0[i,j], psi_h, ustar[i,j])
            else:
                for k in range(nh[2],shape[2]-nh[2]+1):
                    R_aero[k] = r_aerodynamic(z[k], z0[i,j], 0.0, ustar[i,j])
                   
            for k in range(nh[2],shape[2]-nh[2]+1):
                rd = dry_particle_diameter * 0.5
                particle_diameter = 2.0 * (C1*rd**C2/(C3*rd**C4- np.log(rh[i,j,k])) + rd**3.0 )
                vg = v_gravitational(particle_density, particle_diameter, T[i,j,k], rho0[k],p0[k])
                R_surf = r_surface(particle_diameter, rho0[k], T[i,j,k], vg, ustar[i,j])
                vdep[i,j,k] = vg + 1.0/(R_aero[k]+ R_surf)
    return







