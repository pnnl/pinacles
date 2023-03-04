from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    eddy_diffusivity = Diagnostics.get_field("eddy_diffusivity")
    phi = Scalars.array
    phi_t = Scalars.tend_array

    nh = Grid.n_halo
    dxi = Grid.dxi
    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge

    # Compute thermodynamic quantities
    ist, jst, kst = nh[0] - 1, nh[1], nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]

    # Compute the fluxes
    fluxx = (
        -0.5
        * (
            eddy_diffusivity[jnp.newaxis, ist:iend, jst:jend, kst:kend]
            + eddy_diffusivity[jnp.newaxis, ist + 1 : iend + 1, jst:jend, kst:kend]
        )
        * (
            phi[:, ist + 1 : iend + 1, jst:jend, kst:kend]
            - phi[:, ist:iend, jst:jend, kst:kend]
        )
        * dxi[0]
        * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    # Compute thermodynamic quantities
    ist, jst, kst = nh[0], nh[1] - 1, nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]

    fluxy = (
        -0.5
        * (
            eddy_diffusivity[jnp.newaxis, ist:iend, jst:jend, kst:kend]
            + eddy_diffusivity[jnp.newaxis, ist:iend, jst + 1 : jend + 1, kst:kend]
        )
        * (
            phi[:, ist:iend, jst + 1 : jend + 1, kst:kend]
            - phi[:, ist:iend, jst:jend, kst:kend]
        )
        * dxi[1]
        * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    ist, jst, kst = nh[0], nh[1], nh[2] - 1
    iend, jend, kend = -nh[0], -nh[1], -nh[2]
    fluxz = (
        -0.5
        * (
            eddy_diffusivity[jnp.newaxis, ist:iend, jst:jend, kst:kend]
            + eddy_diffusivity[jnp.newaxis, ist:iend, jst:jend, kst + 1 : kend + 1]
        )
        * (
            phi[:, ist:iend, jst:jend, kst + 1 : kend + 1]
            - phi[:, ist:iend, jst:jend, kst:kend]
        )
        * dxi[2]
        * rho0_edge[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    ist, jst, kst = nh[0], nh[1], nh[2]

    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:kend].add(
        -(
            (fluxx[:, 1:, :, :] - fluxx[:, :-1, :, :])
            * dxi[0]
            * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
        )
    )

    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:kend].add(
        -(
            (fluxy[:, :, 1:, :] - fluxy[:, :, :-1, :])
            * dxi[1]
            * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
        )
    )

    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:kend].add(
        -(
            (fluxz[:, :, :, 1:] - fluxz[:, :, :, :-1])
            * dxi[2]
            * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
        )
    )

    Scalars = Scalars.replace(tend_array=phi_t)

    return Scalars
