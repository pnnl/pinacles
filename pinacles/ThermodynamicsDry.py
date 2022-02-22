import numpy as np
import numba
from mpi4py import MPI
from pinacles import Thermodynamics, ThermodynamicsDry_impl
from pinacles import parameters


class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase):
    def __init__(self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(
            self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, None
        )

        ScalarState.add_variable(
            "qv",
            long_name="Water vapor mixing ratio",
            latex_name="q_v",
            units="kg kg^{-1}",
        )

        DiagnosticState.add_variable(
            "bvf",
            long_name="Brunt–Väisälä frequency squared",
            latex_name="N^2",
            units="s^-2",
        )

        DiagnosticState.add_variable(
            "thetav",
            long_name="Virtual Potential Temperature",
            latex_name="\theta_v",
            units="K",
        )

        self._Timers.add_timer("ThermoDynamicsDry_update")

        return

    def update(self, apply_buoyancy=True):

        self._Timers.start_timer("ThermoDynamicsDry_update")

        n_halo = self._Grid.n_halo
        z = self._Grid.z_global
        dz = self._Grid.dx[2]
        dxi = self._Grid.dxi
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        T0 = self._Ref.T0
        exner = self._Ref.exner
        tref = self._Ref.T0

        theta_ref = T0 / exner

        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")
        T = self._DiagnosticState.get_field("T")
        alpha = self._DiagnosticState.get_field("alpha")
        buoyancy = self._DiagnosticState.get_field("buoyancy")
        thetav = self._DiagnosticState.get_field("thetav")
        bvf = self._DiagnosticState.get_field("bvf")
        w_t = self._VelocityState.get_tend("w")
        buoyancy_gradient_mag = self._DiagnosticState.get_field("buoyancy_gradient_mag")

        ThermodynamicsDry_impl.eos(z, p0, alpha0, s, qv, T, tref, alpha, buoyancy)
        ThermodynamicsDry_impl.compute_bvf(
            n_halo, theta_ref, exner, T, qv, dz, thetav, bvf
        )

        if apply_buoyancy:
            ThermodynamicsDry_impl.apply_buoyancy(buoyancy, w_t)

        self._DiagnosticState.remove_mean("buoyancy")

        self.compute_buoyancy_gradient(dxi, buoyancy, buoyancy_gradient_mag)

        self._Timers.end_timer("ThermoDynamicsDry_update")

        return

    def io_fields2d_update(self, fx):

        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo

        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        # Output Temperature
        if fx is not None:
            t = fx.create_dataset(
                        "T",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                t.dims[i].attach_scale(fx[d])
                

        T = self._DiagnosticState.get_field("T")
        send_buffer[start[0] : end[0], start[1] : end[1]] = T[
            nh[0] : -nh[0], nh[1] : -nh[1], nh[2]
        ]
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            t[:, :] = recv_buffer

        return

    @staticmethod
    @numba.njit()
    def compute_thetali(exner, T, thetali):
        shape = T.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    thetali[i, j, k] = T[i, j, k] / exner[k]
        return

    def get_thetali(self):
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field("T")
        thetali = np.empty_like(T)
        self.compute_thetali(exner, T, thetali)
        return thetali

    def get_qt(self):
        # Todo this gets a copy. So modifying it does nothing!
        qv = self._ScalarState.get_field("qv")
        return np.copy(qv)
