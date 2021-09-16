from pinacles import Thermodynamics, ThermodynamicsMoist_impl
from pinacles import parameters
import numpy as np
import numba


class ThermodynamicsMoist(Thermodynamics.ThermodynamicsBase):
    def __init__(
        self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro
    ):
        Thermodynamics.ThermodynamicsBase.__init__(
            self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro
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

        self._Timers.add_timer("ThermoDynamicsMoist_update")

        return

    def update(self, apply_buoyancy=True):

        self._Timers.start_timer("ThermoDynamicsMoist_update")

        n_halo = self._Grid.n_halo
        z = self._Grid.z_global
        dz = self._Grid.dx[2]
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        tref = self._Ref.T0
        exner = self._Ref.exner
        theta_ref = self._Ref.T0 / self._Ref.exner

        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")
        ql = self._Micro.get_qc()
        qi = self._Micro.get_qi()

        T = self._DiagnosticState.get_field("T")
        thetav = self._DiagnosticState.get_field("thetav")
        alpha = self._DiagnosticState.get_field("alpha")
        buoyancy = self._DiagnosticState.get_field("buoyancy")
        bvf = self._DiagnosticState.get_field("bvf")
        w_t = self._VelocityState.get_tend("w")

        ThermodynamicsMoist_impl.eos_sam(
            z, p0, alpha0, s, qv, ql, qi, T, tref, alpha, buoyancy
        )

        # Compute the buoyancy frequency
        ThermodynamicsMoist_impl.compute_bvf(
            n_halo, theta_ref, exner, T, qv, ql, qi, dz, thetav, bvf
        )

        if apply_buoyancy:
            ThermodynamicsMoist_impl.apply_buoyancy(buoyancy, w_t)

        self._Timers.end_timer("ThermoDynamicsMoist_update")
        return

    @staticmethod
    @numba.njit()
    def compute_thetali(exner, T, thetali, qc):
        shape = T.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    thetali[i, j, k] = (
                        T[i, j, k]
                        / exner[k]
                        * (
                            1.0
                            - parameters.LV
                            * qc[i, j, k]
                            / (parameters.CPD * T[i, j, k])
                        )
                    )
        return

    def get_thetali(self):
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field("T")
        qc = self._Micro.get_qc()
        thetali = np.empty_like(T)
        self.compute_thetali(exner, T, thetali, qc)

        return thetali

    def get_qt(self):
        # Todo optimize

        qv = self._ScalarState.get_field("qv")
        qc = self._Micro.get_qc()
        qi = self._Micro.get_qi()

        return qv + qc + qi
