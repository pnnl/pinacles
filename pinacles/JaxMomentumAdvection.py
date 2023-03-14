from functools import partial
import jax
import jax.numpy as jnp


def comp_bkz2(bk, tau, eps, p):
    return 1.0 + (tau / (bk + eps)) ** p


def interp_weno7(phim3, phim2, phim1, phi, phip1, phip2, phip3):
    # p0 = (-1.0/4.0)*phim3 + (13.0/12.0) * phim2 + (-23.0/12.0) * phim1 + (25.0/12.0)*phi
    # p1 = (1.0/12.0)*phim2 + (-5.0/12.0)*phim1 + (13.0/12.0)*phi + (1.0/4.0)*phip1
    # p2 = (-1.0/12.0)*phim1 + (7.0/12.0)*phi +  (7.0/12.0)*phip1 + (-1.0/12.0)*phip2
    # p3 = (1.0/4.0)*phi + (13.0/12.0)*phip1 + (-5.0/12.0)*phip2 + (1.0/12.0)*phip3

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

    tau5 = jnp.abs(beta0 - beta1 - beta2 + beta3)
    bk0 = comp_bkz2(beta0, tau5, 1e-40, 2.0)
    bk1 = comp_bkz2(beta1, tau5, 1e-40, 2.0)
    bk2 = comp_bkz2(beta2, tau5, 1e-40, 2.0)
    bk3 = comp_bkz2(beta3, tau5, 1e-40, 2.0)

    alpha0 = (1.0 / 64.0) * bk0  # ((beta0 + 1e-8) * (beta0 + 1e-8))
    alpha1 = (21.0 / 64.0) * bk1  # ((beta1 + 1e-8) * (beta1 + 1e-8))
    alpha2 = (35.0 / 64.0) * bk2  # ((beta2 + 1e-8) * (beta2 + 1e-8))
    alpha3 = (7.0 / 64.0) * bk3  # ((beta3 + 1e-8) * (beta3 + 1e-8))

    alpha_sum_inv = 1.0 / (alpha0 + alpha1 + alpha2 + alpha3)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv
    w3 = alpha3 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


@jax.jit
def centered_sixth(phim2, phim1, phi, phip1, phip2, phip3):
    return (
        (75.0 / 128.0) * (phi + phip1)
        - (25.0 / 256.0) * (phim1 + phip2)
        + (3.0 / 256.0) * (phim2 + phip3)
    )


@jax.jit
def advection_of_u(Grid, Ref, Velocities):
    nh = Grid.n_halo
    dxi = Grid.dxi

    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge
    alpha0 = Ref.alpha0

    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    ut = Velocities.get_tend("u")

    flux = jnp.zeros_like(ut)
    shape = u.shape

    ist, jst, kst = 3, 3, 3
    iend, jend, kend = shape[0] - 4, shape[1] - 4, shape[2] - 4

    # Interpolate the velocity to the flux location
    up = centered_sixth(
        u[ist - 2 : iend - 2, jst:jend, kst:kend],
        u[ist - 1 : iend - 1, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist + 1 : iend + 1, jst:jend, kst:kend],
        u[ist + 2 : iend + 2, jst:jend, kst:kend],
        u[ist + 3 : iend + 3, jst:jend, kst:kend],
    )

    # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist - 3 : iend - 3, jst:jend, kst:kend],
        u[ist - 2 : iend - 2, jst:jend, kst:kend],
        u[ist - 1 : iend - 1, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist + 1 : iend + 1, jst:jend, kst:kend],
        u[ist + 2 : iend + 2, jst:jend, kst:kend],
        u[ist + 3 : iend + 3, jst:jend, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist + 4 : iend + 4, jst:jend, kst:kend],
        u[ist + 3 : iend + 3, jst:jend, kst:kend],
        u[ist + 2 : iend + 2, jst:jend, kst:kend],
        u[ist + 1 : iend + 1, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist - 1 : iend - 1, jst:jend, kst:kend],
        u[ist - 2 : iend - 2, jst:jend, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (up + jnp.abs(up))  # 0 for u < 0
    neg_indicator = 0.5 * (up - jnp.abs(up))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    ut = ut.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[0:-2, 1:-1, 1:-1])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[0]
    )

    # Rest fluxes back to zero
    #flux = flux.at[...].set(0.0)

    vp = centered_sixth(
        v[ist - 2 : iend - 2, jst:jend, kst:kend],
        v[ist - 1 : iend - 1, jst:jend, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist + 1 : iend + 1, jst:jend, kst:kend],
        v[ist + 1 : iend + 1, jst:jend, kst:kend],
        v[ist + 2 : iend + 2, jst:jend, kst:kend],
    )

    # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist:iend, jst - 3 : jend - 3, kst:kend],
        u[ist:iend, jst - 2 : jend - 2, kst:kend],
        u[ist:iend, jst - 1 : jend - 1, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst + 1 : jend + 1, kst:kend],
        u[ist:iend, jst + 2 : jend + 2, kst:kend],
        u[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist:iend, jst + 4 : jend + 4, kst:kend],
        u[ist:iend, jst + 3 : jend + 3, kst:kend],
        u[ist:iend, jst + 2 : jend + 2, kst:kend],
        u[ist:iend, jst + 1 : jend + 1, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst - 1 : jend - 1, kst:kend],
        u[ist:iend, jst - 2 : jend - 2, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (vp + jnp.abs(vp))  # 0 for u < 0
    neg_indicator = 0.5 * (vp - jnp.abs(vp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    ut = ut.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 0:-2, 1:-1])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[1]
    )

    # Reset fluxes to zero
    #flux = flux.at[...].set(0.0)

    wp = centered_sixth(
        w[ist - 2 : iend - 2, jst:jend, kst:kend],
        w[ist - 1 : iend - 1, jst:jend, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist + 1 : iend + 1, jst:jend, kst:kend],
        w[ist + 1 : iend + 1, jst:jend, kst:kend],
        w[ist + 2 : iend + 2, jst:jend, kst:kend],
    )

    flux_pos = rho0_edge[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist:iend, jst:jend, kst - 3 : kend - 3],
        u[ist:iend, jst:jend, kst - 2 : kend - 2],
        u[ist:iend, jst:jend, kst - 1 : kend - 1],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst + 1 : kend + 1],
        u[ist:iend, jst:jend, kst + 2 : kend + 2],
        u[ist:iend, jst:jend, kst + 3 : kend + 3],
    )

    flux_neg = rho0_edge[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        u[ist:iend, jst:jend, kst + 4 : kend + 4],
        u[ist:iend, jst:jend, kst + 3 : kend + 3],
        u[ist:iend, jst:jend, kst + 2 : kend + 2],
        u[ist:iend, jst:jend, kst + 1 : kend + 1],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst - 1 : kend - 1],
        u[ist:iend, jst:jend, kst - 2 : kend - 2],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (wp + jnp.abs(wp))  # 0 for u < 0
    neg_indicator = 0.5 * (wp - jnp.abs(wp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    ut = ut.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 1:-1, 0:-2])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[2]
    )

    Velocities = Velocities.set_tend("u", ut)

    return Velocities


@jax.jit
def advection_of_v(Grid, Ref, Velocities):
    nh = Grid.n_halo
    dxi = Grid.dxi

    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge
    alpha0 = Ref.alpha0

    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    vt = Velocities.get_tend("v")

    flux = jnp.zeros_like(vt)
    shape = u.shape

    ist, jst, kst = 3, 3, 3
    iend, jend, kend = shape[0] - 4, shape[1] - 4, shape[2] - 4

    # Interpolate the velocity to the flux location
    up = centered_sixth(
        u[ist:iend, jst - 2 : jend - 2, kst:kend],
        u[ist:iend, jst - 1 : jend - 1, kst:kend],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst + 1 : jend + 1, kst:kend],
        u[ist:iend, jst + 2 : jend + 2, kst:kend],
        u[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist - 3 : iend - 3, jst:jend, kst:kend],
        v[ist - 2 : iend - 2, jst:jend, kst:kend],
        v[ist - 1 : iend - 1, jst:jend, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist + 1 : iend + 1, jst:jend, kst:kend],
        v[ist + 2 : iend + 2, jst:jend, kst:kend],
        v[ist + 3 : iend + 3, jst:jend, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist + 4 : iend + 4, jst:jend, kst:kend],
        v[ist + 3 : iend + 3, jst:jend, kst:kend],
        v[ist + 2 : iend + 2, jst:jend, kst:kend],
        v[ist + 1 : iend + 1, jst:jend, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist - 1 : iend - 1, jst:jend, kst:kend],
        v[ist - 2 : iend - 2, jst:jend, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (up + jnp.abs(up))  # 0 for u < 0
    neg_indicator = 0.5 * (up - jnp.abs(up))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )
    # Now compute the flux divergence
    vt = vt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[0:-2, 1:-1, 1:-1])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[0]
    )

    # Reset fluxes to zero
    #flux = flux.at[...].set(0.0)

    vp = centered_sixth(
        v[ist:iend, jst - 2 : jend - 2, kst:kend],
        v[ist:iend, jst - 1 : jend - 1, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst + 1 : jend + 1, kst:kend],
        v[ist:iend, jst + 2 : jend + 2, kst:kend],
        v[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    # # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist:iend, jst - 3 : jend - 3, kst:kend],
        v[ist:iend, jst - 2 : jend - 2, kst:kend],
        v[ist:iend, jst - 1 : jend - 1, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst + 1 : jend + 1, kst:kend],
        v[ist:iend, jst + 2 : jend + 2, kst:kend],
        v[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist:iend, jst + 4 : jend + 4, kst:kend],
        v[ist:iend, jst + 3 : jend + 3, kst:kend],
        v[ist:iend, jst + 2 : jend + 2, kst:kend],
        v[ist:iend, jst + 1 : jend + 1, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst - 1 : jend - 1, kst:kend],
        v[ist:iend, jst - 2 : jend - 2, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (vp + jnp.abs(vp))  # 0 for u < 0
    neg_indicator = 0.5 * (vp - jnp.abs(vp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    vt = vt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 0:-2, 1:-1])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[1]
    )

    # Reset fluxes to zero
    #flux = flux.at[...].set(0.0)

    wp = centered_sixth(
        w[ist:iend, jst - 2 : jend - 2, kst:kend],
        w[ist:iend, jst - 1 : jend - 1, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst + 1 : jend + 1, kst:kend],
        w[ist:iend, jst + 2 : jend + 2, kst:kend],
        w[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_pos = rho0_edge[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist:iend, jst:jend, kst - 3 : kend - 3],
        v[ist:iend, jst:jend, kst - 2 : kend - 2],
        v[ist:iend, jst:jend, kst - 1 : kend - 1],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst:jend, kst + 1 : kend + 1],
        v[ist:iend, jst:jend, kst + 2 : kend + 2],
        v[ist:iend, jst:jend, kst + 3 : kend + 3],
    )

    flux_neg = rho0_edge[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        v[ist:iend, jst:jend, kst + 4 : kend + 4],
        v[ist:iend, jst:jend, kst + 3 : kend + 3],
        v[ist:iend, jst:jend, kst + 2 : kend + 2],
        v[ist:iend, jst:jend, kst + 1 : kend + 1],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst:jend, kst - 1 : kend - 1],
        v[ist:iend, jst:jend, kst - 2 : kend - 2],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (wp + jnp.abs(wp))  # 0 for u < 0
    neg_indicator = 0.5 * (wp - jnp.abs(wp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    vt = vt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 1:-1, 0:-2])
        * alpha0[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[2]
    )

    Velocities = Velocities.set_tend("v", vt)

    return Velocities


@jax.jit
def advection_of_w(Grid, Ref, Velocities):
    nh = Grid.n_halo
    dxi = Grid.dxi

    rho0 = Ref.rho0
    rho0_edge = Ref.rho0_edge
    alpha0 = Ref.alpha0
    alpha0_edge = Ref.alpha0_edge

    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    wt = Velocities.get_tend("w")

    flux = jnp.zeros_like(wt)
    shape = u.shape

    ist, jst, kst = 3, 3, 3
    iend, jend, kend = shape[0] - 4, shape[1] - 4, shape[2] - 4

    # Interpolate the velocity to the flux location
    up = centered_sixth(
        u[ist:iend, jst:jend, kst - 2 : kend - 2],
        u[ist:iend, jst:jend, kst - 1 : kend - 1],
        u[ist:iend, jst:jend, kst:kend],
        u[ist:iend, jst:jend, kst + 1 : kend + 1],
        u[ist:iend, jst:jend, kst + 2 : kend + 2],
        u[ist:iend, jst:jend, kst + 3 : kend + 3],
    )

    # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        w[ist - 3 : iend - 3, jst:jend, kst:kend],
        w[ist - 2 : iend - 2, jst:jend, kst:kend],
        w[ist - 1 : iend - 1, jst:jend, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist + 1 : iend + 1, jst:jend, kst:kend],
        w[ist + 2 : iend + 2, jst:jend, kst:kend],
        w[ist + 3 : iend + 3, jst:jend, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        w[ist + 4 : iend + 4, jst:jend, kst:kend],
        w[ist + 3 : iend + 3, jst:jend, kst:kend],
        w[ist + 2 : iend + 2, jst:jend, kst:kend],
        w[ist + 1 : iend + 1, jst:jend, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist - 1 : iend - 1, jst:jend, kst:kend],
        w[ist - 2 : iend - 2, jst:jend, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (up + jnp.abs(up))  # 0 for u < 0
    neg_indicator = 0.5 * (up - jnp.abs(up))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    wt = wt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[0:-2, 1:-1, 1:-1])
        * alpha0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[0]
    )

    # Reset fluxes to zero
    #flux = flux.at[...].set(0.0)

    vp = centered_sixth(
        v[ist:iend, jst - 2 : jend - 2, kst:kend],
        v[ist:iend, jst - 1 : jend - 1, kst:kend],
        v[ist:iend, jst:jend, kst:kend],
        v[ist:iend, jst + 1 : jend + 1, kst:kend],
        v[ist:iend, jst + 2 : jend + 2, kst:kend],
        v[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    # # Now compute the left and right fluxes

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        w[ist:iend, jst - 3 : jend - 3, kst:kend],
        w[ist:iend, jst - 2 : jend - 2, kst:kend],
        w[ist:iend, jst - 1 : jend - 1, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst + 1 : jend + 1, kst:kend],
        w[ist:iend, jst + 2 : jend + 2, kst:kend],
        w[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst:kend] * interp_weno7(
        w[ist:iend, jst + 4 : jend + 4, kst:kend],
        w[ist:iend, jst + 3 : jend + 3, kst:kend],
        w[ist:iend, jst + 2 : jend + 2, kst:kend],
        w[ist:iend, jst + 1 : jend + 1, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst - 1 : jend - 1, kst:kend],
        w[ist:iend, jst - 2 : jend - 2, kst:kend],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (vp + jnp.abs(vp))  # 0 for u < 0
    neg_indicator = 0.5 * (vp - jnp.abs(vp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    wt = wt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 0:-2, 1:-1])
        * alpha0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[1]
    )

    # Reset fluxes to zero
    #flux = flux.at[...].set(0.0)

    wp = centered_sixth(
        w[ist:iend, jst - 2 : jend - 2, kst:kend],
        w[ist:iend, jst - 1 : jend - 1, kst:kend],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst + 1 : jend + 1, kst:kend],
        w[ist:iend, jst + 2 : jend + 2, kst:kend],
        w[ist:iend, jst + 3 : jend + 3, kst:kend],
    )

    flux_pos = rho0[jnp.newaxis, jnp.newaxis, kst + 1 : kend + 1] * interp_weno7(
        w[ist:iend, jst:jend, kst - 3 : kend - 3],
        w[ist:iend, jst:jend, kst - 2 : kend - 2],
        w[ist:iend, jst:jend, kst - 1 : kend - 1],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst:jend, kst + 1 : kend + 1],
        w[ist:iend, jst:jend, kst + 2 : kend + 2],
        w[ist:iend, jst:jend, kst + 3 : kend + 3],
    )

    flux_neg = rho0[jnp.newaxis, jnp.newaxis, kst + 1 : kend + 1] * interp_weno7(
        w[ist:iend, jst:jend, kst + 4 : kend + 4],
        w[ist:iend, jst:jend, kst + 3 : kend + 3],
        w[ist:iend, jst:jend, kst + 2 : kend + 2],
        w[ist:iend, jst:jend, kst + 1 : kend + 1],
        w[ist:iend, jst:jend, kst:kend],
        w[ist:iend, jst:jend, kst - 1 : kend - 1],
        w[ist:iend, jst:jend, kst - 2 : kend - 2],
    )

    # Compute upwinded flux
    pos_indicator = 0.5 * (wp + jnp.abs(wp))  # 0 for u < 0
    neg_indicator = 0.5 * (wp - jnp.abs(wp))  # Zero for u > 0

    flux = flux.at[ist:iend, jst:jend, kst:kend].set(
        pos_indicator * flux_pos + neg_indicator * flux_neg
    )

    # Now compute the flux divergence
    wt = wt.at[1:-1, 1:-1, 1:-1].add(
        -(flux[1:-1, 1:-1, 1:-1] - flux[1:-1, 1:-1, 0:-2])
        * alpha0_edge[jnp.newaxis, jnp.newaxis, 1:-1]
        * dxi[2]
    )

    Velocities = Velocities.set_tend("w", wt)
    return Velocities


@jax.jit
def weno7_update(Grid, Ref, Scalars, Velocities, Diagnostics):
    Velocities = advection_of_u(Grid, Ref, Velocities)
    Velocities = advection_of_v(Grid, Ref, Velocities)
    Velocities = advection_of_w(Grid, Ref, Velocities)

    return Velocities


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    Velocities = weno7_update(Grid, Ref, Scalars, Velocities, Diagnostics)

    return Velocities
