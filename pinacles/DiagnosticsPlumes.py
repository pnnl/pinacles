import numpy as np
from mpi4py import MPI
import numba
from pinacles import UtilitiesParallel


class DiagnosticsPlumes:
    def __init__(
        self, Grid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState, Plumes
    ):

        self._name = "DiagnosticsPlumes"
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._Micro = Micro
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Plumes = Plumes

        return

    def io_initialize(self, this_grp):

        if "plume_0" not in self._ScalarState.names:
            return

        # Get aliases to the timeseries and profiles groups
        timeseries_grp = this_grp["timeseries"]
        profiles_grp = this_grp["profiles"]
        
        for plume_i in self._Plumes._list_of_plumes:
        
            v = profiles_grp.createVariable(
                plume_i._scalar_name + "_frac",
                np.double,
                dimensions=(
                    "time",
                    "z",
                ),
            )
            v.long_name = "Plume fraction"
            v.units = ""
            v.standard_name = "Plume"

            v = profiles_grp.createVariable(
                "u_" + plume_i._scalar_name,
                np.double,
                dimensions=(
                    "time",
                    "z",
                ),
            )
            v.long_name = "u velocity plume conditional mean"
            v.units = "m/s"
            v.standard_name = "u_{plume}"

            v = profiles_grp.createVariable(
                "v_" + plume_i._scalar_name,
                np.double,
                dimensions=(
                    "time",
                    "z",
                ),
            )
            v.long_name = "v velocity plume conditional mean"
            v.units = "m/s"
            v.standard_name = "v_{plume}"

            v = profiles_grp.createVariable(
                "w_" + plume_i._scalar_name,
                np.double,
                dimensions=(
                    "time",
                    "z",
                ),
            )
            v.long_name = "w velocity plume conditional mean"
            v.units = "m/s"
            v.standard_name = "w_{plume}"

            for container in [self._ScalarState, self._DiagnosticState]:
                for var in container.names:
                    if "ff" in var:
                        continue

                    for stype in ["plume"]:
                        v = profiles_grp.createVariable(
                            var + "_" + plume_i._scalar_name,
                            np.double,
                            dimensions=(
                                "time",
                                "z",
                            ),
                        )
                        v.long_name = container.get_long_name(var) + " " + plume_i._scalar_name
                        v.units = container.get_units(var)
                        v.standard_name = (
                            container.get_standard_name(var) + "_{" + plume_i._scalar_name + "}"
                        )

        return

    @staticmethod
    @numba.njit()
    def _compute_plume_conditional_velocities(
        n_halo,
        plume_sc,
        u,
        v,
        w,
        plume_count,
        u_plume,
        v_plume,
        w_plume,
    ):

        shape = plume_sc.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):

                    # Plume stats
                    if plume_sc[i, j, k] > 1e0:
                        # Get a cell centered velocity
                        uc = 0.5 * (u[i, j, k] + u[i - 1, j, k])
                        vc = 0.5 * (v[i, j, k] + v[i, j - 1, k])
                        wc = 0.5 * (w[i, j, k] + w[i, j, k - 1])

                        plume_count[k] += 1.0
                        u_plume[k] += uc
                        v_plume[k] += vc
                        w_plume[k] += wc

        return

    @staticmethod
    @numba.njit()
    def _compute_plume_conditional_scalars(
        n_halo, plume_sc, phi, plume_count, phi_plume
    ):

        shape = plume_sc.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):
                    # Plume stats
                    if plume_sc[i, j, k] > 1e0:

                        plume_count[k] += 1.0
                        phi_plume[k] += phi[i, j, k]

        return

    def _update_plume_conditional_velocities(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")
        plume_0 = self._ScalarState.get_field("plume_0")
        
        for plume_i in self._Plumes._list_of_plumes:
        
            plume_count = np.zeros((plume_0.shape[2],), dtype=np.double, order="C")

            u_plume = np.zeros_like(plume_count)
            v_plume = np.zeros_like(plume_count)
            w_plume = np.zeros_like(plume_count)
            
            plume_sc = self._ScalarState.get_field(plume_i._scalar_name)
            
            self._compute_plume_conditional_velocities(
                n_halo,
                plume_sc,
                u,
                v,
                w,
                plume_count,
                u_plume,
                v_plume,
                w_plume,
            )

            plume_count = UtilitiesParallel.ScalarAllReduce(plume_count)
            plume_frac = plume_count / npts
            plume_points = plume_frac > 0
            for var in [u_plume, v_plume, w_plume]:
                var[:] = UtilitiesParallel.ScalarAllReduce(var)
                var[plume_points] = var[plume_points] / plume_count[plume_points]

            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp["profiles"]
                profiles_grp[plume_i._scalar_name + "_frac"][-1, :] = plume_frac[n_halo[2] : -n_halo[2]]

                profiles_grp["u_" + plume_i._scalar_name][-1, :] = u_plume[n_halo[2] : -n_halo[2]]
                profiles_grp["v_" + plume_i._scalar_name][-1, :] = v_plume[n_halo[2] : -n_halo[2]]
                profiles_grp["w_" + plume_i._scalar_name][-1, :] = w_plume[n_halo[2] : -n_halo[2]]

        return

    def _update_plume_conditional_scalars(self, this_grp):
        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        plume_0 = self._ScalarState.get_field("plume_0")
        w = self._VelocityState.get_field("w")

        plume_count = np.empty((plume_0.shape[2],), dtype=np.double, order="C")
        phi_plume = np.empty_like(plume_count)

        for container in [self._ScalarState, self._DiagnosticState]:

            for var in container.names:

                if "ff" in var:
                    # Skip SBM bin fields.
                    continue

                phi = container.get_field(var)

                for plume_i in self._Plumes._list_of_plumes:

                    plume_count.fill(0.0)
                    phi_plume.fill(0.0)

                    plume_sc = self._ScalarState.get_field(plume_i._scalar_name)
    
                    self._compute_plume_conditional_scalars(
                        n_halo, plume_sc, phi, plume_count, phi_plume
                    )

                    # Todo precompute these they shouldn't change by variable
                    plume_count = UtilitiesParallel.ScalarAllReduce(plume_count)

                    plume_points = plume_count > 0

                    phi_plume[:] = UtilitiesParallel.ScalarAllReduce(phi_plume)
                    phi_plume[plume_points] = (
                        phi_plume[plume_points] / plume_count[plume_points]
                    )

                    MPI.COMM_WORLD.barrier()

                    if my_rank == 0:
                        profiles_grp = this_grp["profiles"]
                        profiles_grp[var + "_" + plume_i._scalar_name][-1, :] = phi_plume[
                            n_halo[2] : -n_halo[2]
                        ]

        return

    def io_update(self, this_grp):

        if "plume_0" not in self._ScalarState.names:
            # No plumes in this simulation, so we can skip this output.
            return
        
        self._update_plume_conditional_velocities(this_grp)
        self._update_plume_conditional_scalars(this_grp)

        return

    @property
    def name(self):
        return self._name
