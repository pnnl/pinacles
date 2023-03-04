from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    dxi = Grid.dxi
    nh = Grid.n_halo

    u = Velocities.get_field("u")
    v = Velocities.get_field("v")
    w = Velocities.get_field("w")

    strain_rate_mag = Diagnostics.get_field("strain_rate_mag")
    qcrit = Diagnostics.get_field("Q_criterion")
    vertical_vorticity = Diagnostics.get_field("vertical_vorticity")
    helicity = Diagnostics.get_field("helicity")

    ist, jst, kst = nh[0], nh[1], nh[2]
    ien, jen, ken = -nh[0], -nh[1], -nh[2]

    dudx = (u[1:, 1:, 1:] - u[:-1, 1:, 1:]) * dxi[0]
    dudy = (u[1:, 1:, 1:] - u[1:, :-1, 1:]) * dxi[1]
    dudz = (u[1:, 1:, 1:] - u[1:, 1:, :-1]) * dxi[2]

    dvdx = (v[1:, 1:, 1:] - v[:-1, 1:, 1:]) * dxi[0]
    dvdy = (v[1:, 1:, 1:] - v[1:, :-1, 1:]) * dxi[1]
    dvdz = (v[1:, 1:, 1:] - v[1:, 1:, :-1]) * dxi[2]

    dwdx = (w[1:, 1:, 1:] - w[:-1, 1:, 1:]) * dxi[0]
    dwdy = (w[1:, 1:, 1:] - w[1:, :-1, 1:]) * dxi[1]
    dwdz = (w[1:, 1:, 1:] - w[1:, 1:, :-1]) * dxi[2]

    s11 = dudx
    s22 = dvdy
    s33 = dwdz

    s12 = 0.5 * ((dvdx + dudy))
    s13 = 0.5 * ((dudz + dwdx))
    s23 = 0.5 * ((dvdz + dwdy))

    s12 = s12.at[1:, 1:, 1:].set(
        0.25
        * (s12[1:, 1:, 1:] + s12[:-1, 1:, 1:] + s12[1:, :-1, 1:] + s12[:-1, :-1, 1:])
    )
    s13 = s13.at[1:, 1:, 1:].set(
        0.25
        * (s13[1:, 1:, 1:] + s13[:-1, 1:, 1:] + s13[1:, 1:, :-1] + s13[:-1, 1:, :-1])
    )
    s23 = s23.at[1:, 1:, 1:].set(
        0.25
        * (s23[1:, 1:, 1:] + s23[1:, :-1, 1:] + s23[1:, 1:, :-1] + s23[1:, :-1, :-1])
    )

    srm = jnp.sqrt(
        2.0 * (s11 * s11 + s22 * s22 + s33 * s33)
        + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
    )
    strain_rate_mag = strain_rate_mag.at[1:, 1:, 1:].set(srm)

    Diagnostics = Diagnostics.set_field("strain_rate_mag", strain_rate_mag)

    dudy = 0.25 * (
        dudy[1:, 1:, 1:] + dudy[:-1, 1:, 1:] + dudy[1:, :-1, 1:] + dudy[:-1, :-1, 1:]
    )
    dudz = 0.25 * (
        dudz[1:, 1:, 1:]
        + dudz[:-1, 1:, 1:]
        + dudz[1:, 1:, :-1]
        + dudz[
            :-1,
            1:,
            :-1,
        ]
    )

    dvdx = 0.25 * (
        dvdx[1:, 1:, 1:] + dvdx[:-1, 1:, 1:] + dvdx[1:, :-1, 1:] + dvdx[:-1, :-1, 1:]
    )
    dvdz = 0.25 * (
        dvdz[1:, 1:, 1:]
        + dvdz[1:, :-1, 1:]
        + dvdz[1:, 1:, :-1]
        + dvdz[
            1:,
            :-1,
            :-1,
        ]
    )

    dwdx = 0.25 * (
        dwdx[1:, 1:, 1:]
        + dwdx[:-1, 1:, 1:]
        + dwdx[1:, 1:, :-1]
        + dwdx[
            :-1,
            1:,
            :-1,
        ]
    )
    dwdy = 0.25 * (
        dwdy[1:, 1:, 1:]
        + dwdy[1:, :-1, 1:]
        + dwdy[1:, 1:, :-1]
        + dwdy[
            1:,
            :-1,
            :-1,
        ]
    )

    q12 = (0.5 * (dudy - dvdx)) ** 2.0
    q13 = (0.5 * (dudz - dwdx)) ** 2.0
    q21 = (0.5 * (dvdx - dudy)) ** 2.0
    q23 = (0.5 * (dvdz - dwdy)) ** 2.0
    q31 = (0.5 * (dwdx - dudz)) ** 2.0
    q32 = (0.5 * (dwdy - dvdz)) ** 2.0

    s12 = (0.5 * (dvdx + dudy)) ** 2.0
    s21 = s12
    s13 = (0.5 * (dudz + dwdx)) ** 2.0
    s31 = s13
    s23 = (0.5 * (dvdz + dwdy)) ** 2.0
    s32 = s23

    s11 = s11 * s11
    s22 = s22 * s22
    s33 = s33 * s33

    # Compute the q-criterion
    qcrit = qcrit.at[2:, 2:, 2:].set(
        0.5
        * (
            jnp.sqrt(q12 + q13 + q21 + q23 + q31 + q32)
            - jnp.sqrt(
                s11[1:, 1:, 1:]
                + s22[1:, 1:, 1:]
                + s33[1:, 1:, 1:]
                + s12
                + s21
                + s13
                + s31
                + s23
                + s32
            )
        )
    )

    # Compute the vertical vorticity
    vertical_vorticity = vertical_vorticity.at[2:, 2:, 2:].set(dvdx - dudy)

    # Compute the helicity
    helicity = helicity.at[2:, 2:, 2:].set(
        0.5
        * (
            (dwdy - dvdz) * 0.5 * (u[2:, 2:, 2:] + u[1:-1, 2:, 2:])
            + (dudz - dwdx) * 0.5 * (v[2:, 1:-1, 2:] + v[2:, 2:, 2:])
            + (dvdx - dudy) * 0.5 * (w[2:, 2:, 2:] + w[2:, 2:, 1:-1])
        )
    )

    Diagnostics = Diagnostics.set_field("helicity", helicity)
    Diagnostics = Diagnostics.set_field("vertical_vorticity", vertical_vorticity)
    Diagnostics = Diagnostics.set_field("Q_criterion", qcrit)

    return Diagnostics
