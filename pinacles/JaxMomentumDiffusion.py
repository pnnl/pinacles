from functools import partial
import jax
import jax.numpy as jnp
from pinacles import JaxKinematic


@jax.jit
def compute_u_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s11, s12, s13, ut):
    nh = Grid.n_halo
    dxi = Grid.dxi

    flux = jnp.empty_like(ut)

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -2.0
        * rho0[jnp.newaxis, jnp.newaxis, 1:-1]
        * eddy_viscosity[1:-1, 1:-1, 1:-1]
        * s11[1:-1, 1:-1, 1:-1]
    )

    ut = ut.at[2:-2, 2:-2, 2:-2].add(
        -(flux[3:-1, 2:-2, 2:-2] - flux[2:-2, 2:-2, 2:-2])
        * dxi[0]
        / rho0[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0[jnp.newaxis, jnp.newaxis, 1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[2:, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 2:, 1:-1]
            + eddy_viscosity[2:, 2:, 1:-1]
        )
        * s12[1:-1, 1:-1, 1:-1]
    )

    ut = ut.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[2:-2, 1:-3, 2:-2])
        * dxi[1]
        / rho0[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[2:, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 1:-1, 2:]
            + eddy_viscosity[2:, 1:-1, 2:]
        )
        * s13[1:-1, 1:-1, 1:-1]
    )

    # Deal with bottom and top boundaries
    flux = flux.at[1:-1, 1:-1, : nh[2]].set(0.0)
    flux = flux.at[1:-1, 1:-1, -nh[2] :].set(0.0)

    ut = ut.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[2:-2, 2:-2, 1:-3]) * dxi[2] / rho0[2:-2]
    )

    return ut


@jax.jit
def compute_v_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s12, s22, s23, vt):
    nh = Grid.n_halo
    dxi = Grid.dxi

    flux = jnp.empty_like(vt)

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0[jnp.newaxis, jnp.newaxis, 1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[2:, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 2:, 1:-1]
            + eddy_viscosity[2:, 2:, 1:-1]
        )
        * s12[1:-1, 1:-1, 1:-1]
    )

    vt = vt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[1:-3, 2:-2, 2:-2])
        * dxi[0]
        / rho0[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -2.0 * rho0[jnp.newaxis, jnp.newaxis, 1:-1] * s22[1:-1, 1:-1, 1:-1]
    )

    vt = vt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 3:-1, 2:-2] - flux[2:-2, 2:-2, 2:-2])
        * dxi[1]
        / rho0[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 2:, 1:-1]
            + eddy_viscosity[1:-1, 1:-1, 2:]
            + eddy_viscosity[1:-1, 2:, 2:]
        )
        * s23[1:-1, 1:-1, 1:-1]
    )

    # Deal with bottom and top boundaries
    flux = flux.at[1:-1, 1:-1, : nh[2]].set(0.0)
    flux = flux.at[1:-1, 1:-1, -nh[2] :].set(0.0)

    vt = vt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[2:-2, 2:-2, 1:-3])
        * dxi[2]
        / rho0[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    return vt


@jax.jit
def compute_w_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s13, s23, s33, wt):
    nh = Grid.n_halo
    dxi = Grid.dxi

    flux = jnp.empty_like(wt)

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[2:, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 1:-1, 2:]
            + eddy_viscosity[2:, 1:-1, 2:]
        )
        * s13[1:-1, 1:-1, 1:-1]
    )

    wt = wt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[1:-3, 2:-2, 2:-2])
        * dxi[0]
        / rho0_edge[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -0.5
        * rho0_edge[1:-1]
        * (
            eddy_viscosity[1:-1, 1:-1, 1:-1]
            + eddy_viscosity[1:-1, 2:, 1:-1]
            + eddy_viscosity[1:-1, 1:-1, 2:]
            + eddy_viscosity[1:-1, 2:, 2:]
        )
        * s23[1:-1, 1:-1, 1:-1]
    )

    wt = wt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 2:-2] - flux[2:-2, 1:-3, 2:-2])
        * dxi[1]
        / rho0_edge[jnp.newaxis, jnp.newaxis, 2:-2]
    )

    flux = flux.at[1:-1, 1:-1, 1:-1].set(
        -2.0 * rho0[2:] * eddy_viscosity[1:-1, 1:-1, 1:-1] * s33[1:-1, 1:-1, 1:-1]
    )

    # Deal with bottom and top boundaries
    flux = flux.at[1:-1, 1:-1, : nh[2]].set(0.0)
    flux = flux.at[1:-1, 1:-1, -nh[2] :].set(0.0)

    wt = wt.at[2:-2, 2:-2, 2:-2].add(
        -(flux[2:-2, 2:-2, 3:-1] - flux[2:-2, 2:-2, 3:-1]) * dxi[2] / rho0_edge[2:-2]
    )
    

    return wt


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    ut = Velocities.get_tend("u")
    vt = Velocities.get_tend("v")
    wt = Velocities.get_tend("w")

    eddy_viscosity = Diagnostics.get_field("eddy_viscosity")

    dx = Grid.dx
    dxi = Grid.dxi
    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge
    nh = Grid.n_halo

    # Compute velocity gradients
    dudx, dudy, dudz = JaxKinematic.u_gradients(dxi, u)
    dvdx, dvdy, dvdz = JaxKinematic.v_gradients(dxi, v)
    dwdx, dwdy, dwdz = JaxKinematic.w_gradients(dxi, w)

    # Comptue the strain rate tensor
    s11, s22, s33, s12, s13, s23 = JaxKinematic.strain_rate_tensor(
        dxi, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz
    )

    # Compute fluxes and flux-divergences
    ut = compute_u_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s11, s12, s13, ut)
    vt = compute_v_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s12, s22, s23, vt)
    wt = compute_w_fluxes(Grid, rho0, rho0_edge, eddy_viscosity, s13, s23, s33, wt)

    # Update the velocity tendencies in the container class
    Velocities = Velocities.set_tend("u", ut)
    Velocities = Velocities.set_tend("v", vt)
    Velocities = Velocities.set_tend("w", wt)

    return Velocities
