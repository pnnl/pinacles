from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def interp_weno7(phim3, phim2, phim1, phi, phip1, phip2, phip3):
    p0 = (-0.3125) * phim3 + (1.3125) * phim2 + (-2.1875) * phim1 + (2.1875) * phi
    p1 = (0.0625) * phim2 + (-0.3125) * phim1 + (0.9375) * phi + (0.3125) * phip1
    p2 = (-0.0625) * phim1 + (0.5625) * phi + (0.5625) * phip1 + (-0.0625) * phip2
    p3 = (0.3125) * phi + (0.9375) * phip1 + (-0.3125) * phip2 + (0.0625) * phip3

    beta0 = (
        phim3 * phim3 * 6649.0 / 2880.0
        - phim3 * phim2 * 2623.0 / 160.0
        + phim3 * phim1 * 9449.0 / 480.0
        - phim3 * phi * 11389.0 / 1440.0
        + phim2 * phim2 * 28547.0 / 960.0
        - phim2 * phim1 * 35047.0 / 480.0
        + phim2 * phi * 14369.0 / 480.0
        + phim1 * phim1 * 44747.0 / 960.0
        - phim1 * phi * 6383.0 / 160.0
        + phi * phi * 25729.0 / 2880.0
    )
    beta1 = (
        phim2 * phim2 * 3169 / 2880.0
        - phim2 * phim1 * 3229 / 480.0
        + phim2 * phi * 3169 / 480.0
        - phim2 * phip1 * 2989 / 1440.0
        + phim1 * phim1 * 11147 / 960.0
        - phim1 * phi * 11767 / 480.0
        + phim1 * phip1 * 1283 / 160.0
        + phi * phi * 13667 / 960.0
        - phi * phip1 * 5069 / 480.0
        + phip1 * phip1 * 6649 / 2880.0
    )
    beta2 = (
        phim1 * phim1 * 6649.0 / 2880.0
        - phim1 * phi * 5069.0 / 480.0
        + phim1 * phip1 * 1283.0 / 160.0
        - phim1 * phip2 * 2989.0 / 1440.0
        + phi * phi * 13667.0 / 960.0
        - phi * phip1 * 11767.0 / 480.0
        + phi * phip2 * 3169.0 / 480.0
        + phip1 * phip1 * 11147.0 / 960.0
        - phip1 * phip2 * 3229.0 / 480.0
        + phip2 * phip2 * 3169.0 / 2880.0
    )
    beta3 = (
        phi * phi * 25729.0 / 2880.0
        - phi * phip1 * 6383.0 / 160.0
        + phi * phip2 * 14369.0 / 480.0
        - phi * phip3 * 11389.0 / 1440.0
        + phip1 * phip1 * 44747.0 / 960.0
        - phip1 * phip2 * 35047.0 / 480.0
        + phip1 * phip3 * 9449.0 / 480.0
        + phip2 * phip2 * 28547.0 / 960.0
        - phip2 * phip3 * 2623.0 / 160.0
        + phip3 * phip3 * 6649.0 / 2880.0
    )

    alpha0 = (1.0 / 64.0) / ((beta0 + 1e-8) * (beta0 + 1e-8))
    alpha1 = (21.0 / 64.0) / ((beta1 + 1e-8) * (beta1 + 1e-8))
    alpha2 = (35.0 / 64.0) / ((beta2 + 1e-8) * (beta2 + 1e-8))
    alpha3 = (7.0 / 64.0) / ((beta3 + 1e-8) * (beta3 + 1e-8))

    alpha_sum_inv = 1.0 / (alpha0 + alpha1 + alpha2 + alpha3)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv
    w3 = alpha3 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


def weno7_update(Grid, Ref, Scalars, Velocities, Diagnostics):
    nh = Grid.n_halo

    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge
    alpha0 = Ref.alpha0

    phi = Scalars.array
    phi_t = Scalars.tend_array

    dxi = Grid.dxi

    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    # Compute advection in the x-dimension
    ist, jst, kst = nh[0] - 1, nh[1], nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]

    flux_pos = interp_weno7(
        phi[:, ist - 3 : iend - 3, jst:jend, kst:kend],
        phi[:, ist - 2 : iend - 2, jst:jend, kst:kend],
        phi[:, ist - 1 : iend - 1, jst:jend, kst:kend],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist + 1 : iend + 1, jst:jend, kst:kend],
        phi[:, ist + 2 : iend + 2, jst:jend, kst:kend],
        phi[:, ist + 3 : iend + 3, jst:jend, kst:kend],
    )

    flux_neg = interp_weno7(
        phi[:, ist + 4 : iend + 4, jst:jend, kst:kend],
        phi[:, ist + 3 : iend + 3, jst:jend, kst:kend],
        phi[:, ist + 2 : iend + 2, jst:jend, kst:kend],
        phi[:, ist + 1 : iend + 1, jst:jend, kst:kend],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist - 1 : iend - 1, jst:jend, kst:kend],
        phi[:, ist - 2 : iend - 2, jst:jend, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (
        u[ist:iend, jst:jend, kst:kend] + jnp.abs(u[ist:iend, jst:jend, kst:kend])
    )  # 0 for u < 0
    neg_indicator = 0.5 * (
        u[ist:iend, jst:jend, kst:kend] - jnp.abs(u[ist:iend, jst:jend, kst:kend])
    )  # Zero for u > 0

    flux = (
        flux_pos * pos_indicator[jnp.newaxis, :, :, :]
        + flux_neg * neg_indicator[jnp.newaxis, :, :, :]
    ) * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]

    # Add the tendency
    ist, jst, kst = nh[0], nh[1], nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]
    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:jend].add(
        (flux[:, 1:, :, :] - flux[:, :-1, :, :])
        * dxi[0]
        * alpha0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    # Compute advection in the y-direction
    ist, jst, kst = nh[0], nh[1] - 1, nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]

    flux_pos = interp_weno7(
        phi[:, ist:iend, jst - 3 : jend - 3, kst:kend],
        phi[:, ist:iend, jst - 2 : jend - 2, kst:kend],
        phi[:, ist:iend, jst - 1 : jend - 1, kst:kend],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist:iend, jst + 1 : jend + 1, kst:kend],
        phi[:, ist:iend, jst + 2 : jend + 2, kst:kend],
        phi[:, ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_neg = interp_weno7(
        phi[:, ist:iend, jst + 4 : jend + 4, kst:kend],
        phi[:, ist:iend, jst + 3 : jend + 3, kst:kend],
        phi[:, ist:iend, jst + 2 : jend + 2, kst:kend],
        phi[:, ist:iend, jst + 1 : jend + 1, kst:kend],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist:iend, jst - 1 : jend - 1, kst:kend],
        phi[:, ist:iend, jst - 2 : jend - 2, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (
        v[ist:iend, jst:jend, kst:kend] + jnp.abs(v[ist:iend, jst:jend, kst:kend])
    )  # 0 for v < 0
    neg_indicator = 0.5 * (
        v[ist:iend, jst:jend, kst:kend] - jnp.abs(v[ist:iend, jst:jend, kst:kend])
    )  # Zero for v > 0

    flux = (
        flux_pos * pos_indicator[jnp.newaxis, :, :, :]
        + flux_neg * neg_indicator[jnp.newaxis, :, :, :]
    ) * rho0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]

    # Add the tendency
    ist, jst, kst = nh[0], nh[1], nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]
    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:jend].add(
        (flux[:, :, 1:, :] - flux[:, :, :-1, :])
        * dxi[1]
        * alpha0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    # Now do fluxes in the z-direction
    # Compute advection in the y-direction
    ist, jst, kst = nh[0], nh[1], nh[2] - 1
    iend, jend, kend = -nh[0], -nh[1], -nh[2]

    flux_pos = interp_weno7(
        phi[:, ist:iend, jst:jend, kst - 3 : kend - 3],
        phi[:, ist:iend, jst:jend, kst - 2 : kend - 2],
        phi[:, ist:iend, jst:jend, kst - 1 : kend - 1],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist:iend, jst:jend, kst + 1 : kend + 1],
        phi[:, ist:iend, jst:jend, kst + 2 : kend + 2],
        phi[:, ist:iend, jst:jend, kst + 3 : kend + 3],
    )

    flux_neg = interp_weno7(
        phi[:, ist:iend, jst:jend, kst + 4 : kend + 4],
        phi[:, ist:iend, jst:jend, kst + 3 : kend + 3],
        phi[:, ist:iend, jst:jend, kst + 2 : kend + 2],
        phi[:, ist:iend, jst:jend, kst + 1 : kend + 1],
        phi[:, ist:iend, jst:jend, kst:kend],
        phi[:, ist:iend, jst:jend, kst - 1 : kend - 1],
        phi[:, ist:iend, jst:jend, kst - 1 : kend - 1],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (
        w[ist:iend, jst:jend, kst:kend] + jnp.abs(w[ist:iend, jst:jend, kst:kend])
    )  # 0 for v < 0
    neg_indicator = 0.5 * (
        w[ist:iend, jst:jend, kst:kend] - jnp.abs(w[ist:iend, jst:jend, kst:kend])
    )  # Zero for v > 0

    flux = (
        flux_pos * pos_indicator[jnp.newaxis, :, :, :]
        + flux_neg * neg_indicator[jnp.newaxis, :, :, :]
    ) * rho0_edge[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]


    # Add the tendency
    ist, jst, kst = nh[0], nh[1], nh[2]
    iend, jend, kend = -nh[0], -nh[1], -nh[2]
    phi_t = phi_t.at[:, ist:iend, jst:jend, kst:jend].add(
        (flux[:, :, :, 1:] - flux[:, :, :, :-1])
        * dxi[2]
        * alpha0[jnp.newaxis, jnp.newaxis, jnp.newaxis, kst:kend]
    )

    Scalars = Scalars.replace(tend_array=phi_t)

    return Scalars


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    
    Scalars = weno7_update(Grid, Ref, Scalars, Velocities, Diagnostics)

    return Scalars
