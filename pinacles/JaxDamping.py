from functools import partial
import jax
import jax.numpy as jnp
from simple_pytree import Pytree, static_field


class Damping(Pytree):
    depth = static_field()
    timescale = static_field()
    time_scale_profile_edge = static_field()

    def __init__(self, Damping):
        self.depth = Damping._depth
        self.timescale = Damping._timescale
        self.time_scale_profile_edge = jax.device_put(
            jnp.array(Damping._timescale_profile_edge)
        )

    def update_jax(self, Grid, Ref, Scalars, Velocities, Diagnostics):
        w = Velocities.get_field("w")
        wt = Velocities.get_tend("w")

        nh = Grid.n_halo

        ist, jst, kst = nh[0], nh[1], nh[2]
        ind, jnd, knd = -nh[0], -nh[1], -nh[2]
        
        # Only relaxing w to zero
        wt = wt.at[ist:ind, jst:jnd, kst:knd].add(
            self.time_scale_profile_edge[jnp.newaxis, jnp.newaxis, kst:knd] * -w[ist:ind, jst:jnd, kst:knd]
        )

        # Push wt back to the container
        Velocities = Velocities.set_tend("w", wt)

        return Velocities
