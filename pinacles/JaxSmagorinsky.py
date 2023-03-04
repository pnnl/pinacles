from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    dx = Grid.dx
    nh = Grid.n_halo
    filt_scale = (dx[0] * dx[1] * dx[2]) ** (1.0 / 3.0)
    cs = 0.21
    pr = 1.0
    pri = 1.0 / pr

    bvf = Diagnostics.get_field("bvf")
    eddy_diffusivity = Diagnostics.get_field("eddy_diffusivity")
    eddy_viscosity = Diagnostics.get_field("eddy_viscosity")
    strain_rate_mag = Diagnostics.get_field("strain_rate_mag")
    tke_sgs = Diagnostics.get_field("tke_sgs")

    ist, jst, kst = nh[0] - 1, nh[1] - 1, nh[2] - 1
    ien, jen, ken = -nh[0] + 1, -nh[1] + 1, -nh[2] + 1

    fb = jnp.ones_like(bvf[ist:ien, jst:jen, kst:ken])
    fb = jnp.where(
        jnp.logical_and(
            bvf[ist:ien, jst:jen, kst:ken] > 0,
            strain_rate_mag[ist:ien, jst:jen, kst:ken] > 1e-10,
        ),
        jnp.sqrt(
            jnp.maximum(
                jnp.array(0.0),
                1.0
                - bvf[ist:ien, jst:jen, kst:ken]
                / (
                    pr
                    * strain_rate_mag[ist:ien, jst:jen, kst:ken]
                    * strain_rate_mag[ist:ien, jst:jen, kst:ken]
                ),
            )
        ),
        fb,
    )

    eddy_viscosity = eddy_viscosity.at[ist:ien, jst:jen, kst:ken].set(
        (cs * filt_scale) ** 2.0 * (fb * strain_rate_mag[ist:ien, jst:jen, kst:ken])
    )
    eddy_diffusivity = eddy_diffusivity.at[ist:ien, jst:jen, kst:ken].set(
        eddy_viscosity[ist:ien, jst:jen, kst:ken] * pri
    )
    tke_sgs = tke_sgs.at[ist:ien, jst:jen, kst:ken].set(
        (eddy_viscosity[ist:ien, jst:jen, kst:ken] / (filt_scale * 0.1)) ** 2.0
    )

    Diagnostics = Diagnostics.set_field("eddy_viscosity", eddy_viscosity)
    Diagnostics = Diagnostics.set_field("eddy_diffusivity", eddy_diffusivity)
    Diagnostics = Diagnostics.set_field("tke_sgs", tke_sgs)

    return Diagnostics
