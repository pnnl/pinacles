from functools import partial
import jax
import jax.numpy as jnp

@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    
    
    eddy_diffusivity = Diagnostics.get_field("eddy_diffusivity")
    phi = Scalars.array
    phi_t = Scalars.tend_array 
    
    dxi = Grid.dxi
    rho0 = Ref.rho0
    
    #Compute the fluxes
    #fluxx = (
    #                -0.5
    #                * (eddy_diffusivity[:-1,:,:] + eddy_diffusivity[1:, :,:])
    #                * (phi[:, 1:, 1:, 1:] - phi[:,:-1, 1:, 1:])
    #                * dxi[0]
    #                * rho0[jnp.newaxis,jnp.newaxis, jnp.newaxis, :]
    #            )

    #fluxy = (
    #            -0.5
    ##            * (eddy_diffusivity[:, :-1,:] + eddy_diffusivity[:, 1:,:])
    #            * (phi[:, 1:, 1:, 1:] - phi[:, 1:, :-1, 1:])
    #            * dxi[2]
    #            * rho0[jnp.newaxis,jnp.newaxis, jnp.newaxis, :]
    #        )

    
    
    
    #diff_tend = -(fluxx[:,1:, 2:, 2:] - fluxx[:,:-1,2:,2:])*dxi[0] / rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, 2:]
    #diff_tend -= (fluxx[:, 2:, 1:, 2:] - fluxx[:,2:,:-1,2:])*dxi[0] / rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, 2:]
    
    #phi_t = phi_t.at[:,2:,2:,2:].add(diff_tend)
    
    
    Scalars = Scalars.replace(tend_array = phi_t)
    
    return Scalars

