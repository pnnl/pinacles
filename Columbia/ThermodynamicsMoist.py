from Columbia import Thermodynamics, ThermodynamicsMoist_impl
from Columbia import parameters
import numpy as np

class ThermodynamicsMoist(Thermodynamics.ThermodynamicsBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState)

        return

    def update(self, apply_buoyancy=True):

        z = self._Grid.z_global
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        exner = self._Ref.exner


        s = self._ScalarState.get_field('s')
        qc = self._ScalarState.get_field('qc')
        qr = self._ScalarState.get_field('qr')
        ql = np.add(qc, qr)
        qi = np.zeros_like(ql)

        T = self._DiagnosticState.get_field('T')
        alpha = self._DiagnosticState.get_field('alpha')
        buoyancy = self._DiagnosticState.get_field('buoyancy')
        w_t = self._VelocityState.get_tend('w')

        ThermodynamicsMoist_impl.eos(z, p0, alpha0, s, ql, qi, T, alpha, buoyancy)
        if apply_buoyancy:
            ThermodynamicsMoist_impl.apply_buoyancy(buoyancy, w_t)    

        return

    