from functools import partial
import jax
import jax.numpy as jnp
from simple_pytree import Pytree, static_field


class Forcing(Pytree):
    f = static_field()
    ug = static_field()
    vg = static_field()

    def __init__(self, Forcing):
        self.f = Forcing._f
        self.ug = Forcing._ug
        self.vg = Forcing._vg

    @jax.jit
    def update(self, Grid, Ref, Scalars, Velocities, Diagnostics):
        u = Velocities.get_field("u")
        v = Velocities.get_field("v")

        ut = Velocities.get_tend("u")
        vt = Velocities.get_tend("v")

        nh = Grid.n_halo

        ist, jst, kst = nh[0], nh[1], nh[2]
        ind, jnd, knd = -nh[0], -nh[1], -nh[2]

        u_at_v =(0.25
            * (
                u[ist:ind, jst:jnd, kst:knd]
                + u[ist-1:ind-1, jst:jnd, kst:knd]
                + u[ist-1:ind-1, jst+1:jnd+1, kst:knd]
                + u[ist:ind, jst+1:jnd+1, kst:knd]
            )
        )

        v_at_u = (0.25
            * (
                v[ist:ind, jst:jnd, kst:knd]
                + v[ist+1:ind+1, jst:jnd, kst:knd]
                + v[ist+1:ind+1, jst-1:jnd-1, kst:knd]
                + v[ist:ind, jst-1:jnd-1, kst:knd]
            ))
    

        ut = ut.at[ist:ind, jst:jnd, kst:knd].add(
            -self.f * (self.vg[jnp.newaxis, jnp.newaxis, kst:knd] - v_at_u)
        )
        vt = vt.at[ist:ind, jst:jnd, kst:knd].add(
            self.f * (self.ug[jnp.newaxis, jnp.newaxis, kst:knd] - u_at_v)
        )

        Velocities = Velocities.set_tend("u", ut)
        Velocities = Velocities.set_tend("v", vt)

        return Velocities
