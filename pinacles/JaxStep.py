import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
import time
from simple_pytree import Pytree, static_field
import collections
from pinacles import ThermodynamicsDry
from pinacles import JaxKinematic
from pinacles import JaxSmagorinsky
from pinacles import JaxScalarDiffusion
from pinacles import JaxScalarAdvection
from pinacles import JaxMomentumAdvection


config.update("jax_enable_x64", True)
from functools import partial


class JaxGrid(Pytree):
    z_global = static_field()
    n_halo = static_field()
    dx = static_field()
    dxi = static_field()

    def __init__(self, Grid):
        self.z_global = jax.device_put(jnp.asarray(Grid.z_global))
        self.n_halo = Grid.n_halo
        self.dx = Grid.dx
        self.dxi = Grid.dxi


class JaxRef(Pytree):
    p0 = static_field()
    alpha0 = static_field()
    exner = static_field()
    T0 = static_field()
    rho0 = static_field()
    rho0_edge = static_field()

    def __init__(self, Ref):
        self.p0 = jax.device_put(jnp.asarray(Ref.p0))
        self.T0 = jax.device_put(jnp.asarray(Ref.T0))
        self.alpha0 = jax.device_put(jnp.asarray(Ref.alpha0))
        self.alpha0_edge = jax.device_put(jnp.asarray(Ref.alpha0_edge))
        self.exner = jax.device_put(jnp.asarray(Ref.exner))
        self.rho0 = jax.device_put(jnp.asarray(Ref.rho0))
        self.rho0_edge = jax.device_put(jnp.asarray(Ref.rho0_edge))


class JaxPrognostic(Pytree, mutable=True):
    dofs = static_field()
    nvars = static_field()

    def __init__(self, PrognosticState):
        self.dofs = PrognosticState.dofs
        self.array = jnp.array(PrognosticState._state_array.array, copy=False)
        self.tend_array = jnp.array(PrognosticState._tend_array.array, copy=False)
        self.nvars = self.tend_array.shape[0]
        return

    @partial(jax.jit, static_argnums=(0, 1), inline=True)
    def get_field(self, var):
        return self.array[self.dofs[var], :, :, :]

    @partial(jax.jit, static_argnums=(1), inline=True)
    def set_field(self, var, data):
        return self.replace(array=self.array.at[self.dofs[var], :, :, :].set(data))

    @partial(jax.jit, static_argnums=(1), inline=True)
    def set_tend(self, var, data):
        return self.replace(
            tend_array=self.tend_array.at[self.dofs[var], :, :, :].set(data)
        )

    @partial(jax.jit, static_argnums=(0, 1), inline=True)
    def get_tend(self, var):
        return self.tend_array[self.dofs[var], :, :, :]

    def update_arrays(self, PrognosticState):
        self.array = jnp.array(PrognosticState._state_array.array, copy=False)
        self.tend_array = jnp.array(PrognosticState._tend_array.array, copy=False)


class JaxDiagnostic(Pytree, mutable=True):
    dofs = static_field()

    def __init__(self, PrognosticState):
        self.dofs = PrognosticState.dofs
        self.array = jnp.array(PrognosticState._state_array.array, copy=False)
        return

    @partial(jax.jit, static_argnums=(0, 1), inline=True)
    def get_field(self, var):
        return self.array[self.dofs[var], :, :, :]

    @partial(jax.jit, static_argnums=(1), inline=True)
    def set_field(self, var, data):
        return self.replace(array=self.array.at[self.dofs[var], :, :, :].set(data))

    def update_arrays(self, PrognosticState):
        self.array = jnp.array(PrognosticState._state_array.array, copy=False)


class JaxStep:
    def __init__(self, Grid, Ref, Thermo, VelocityState, ScalarState, DiagnosticState):
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        self._grid_j = {
            "z": jax.device_put(jnp.asarray(self._Grid.z_global)),
        }

        self._JaxGrid = JaxGrid(Grid)
        self._JaxRef = JaxRef(Ref)

        self._JaxScalars = JaxPrognostic(ScalarState)
        self._JaxVelocities = JaxPrognostic(VelocityState)
        self._JaxDiagnostics = JaxDiagnostic(DiagnosticState)

    @partial(jax.jit, static_argnums=(0,))
    def jax_update(self, Grid, Ref, Scalars, Velocities, Diagnostics):
        Velocities, Diagnostics = ThermodynamicsDry.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        Diagnostics = JaxKinematic.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        Diagnostics = JaxSmagorinsky.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        Scalars = JaxScalarDiffusion.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        Scalars = JaxScalarAdvection.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        Velocities = JaxMomentumAdvection.update_jax(
            Grid, Ref, Scalars, Velocities, Diagnostics
        )

        return Scalars, Velocities, Diagnostics

    def update(self):
        # Here we copy to the deviced
        tic1 = time.perf_counter()

        self._JaxVelocities.update_arrays(self._VelocityState)
        self._JaxScalars.update_arrays(self._ScalarState)
        self._JaxDiagnostics.update_arrays(self._DiagnosticState)

        ticjax = time.perf_counter()
        JaxScalars, JaxVelocities, JaxDiagnostics = self.jax_update(
            self._JaxGrid,
            self._JaxRef,
            self._JaxScalars,
            self._JaxVelocities,
            self._JaxDiagnostics,
        )
        (jax.device_put(0.0) + 0).block_until_ready()
        tocjax = time.perf_counter()

        print("Jax ticotoc", tocjax - ticjax)
        # Copy arrays back to PINACLES containers
        pc = [self._ScalarState, self._VelocityState, self._DiagnosticState]
        jc = [JaxScalars, JaxVelocities, JaxDiagnostics]
        for p, j in zip(pc, jc):
            if hasattr(p, "_state_array"):
                p._state_array.array[:] = j.array
            if hasattr(p, "_tend_array") and p._tend_array is not None:
                p._tend_array.array[:] = j.tend_array

        return
