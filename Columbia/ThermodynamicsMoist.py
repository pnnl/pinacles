from Columbia import Thermodynamics, ThermodynamicsMoist_impl
from Columbia import parameters
import numpy as np

class ThermodynamicsMoist(Thermodynamics.ThermodynamicsBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, Micro)

        DiagnosticState.add_variable('bvf')
        DiagnosticState.add_variable('thetav')

        return

    def update(self, apply_buoyancy=True):

        z = self._Grid.z_global
        dz = self._Grid.dx[2]
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        tref = self._Ref.T0
        exner = self._Ref.exner
        theta_ref = self._Ref.T0 /self._Ref.exner

        s = self._ScalarState.get_field('s')
        #qc = self._ScalarState.get_field('qc')
        #qr = self._ScalarState.get_field('qr')
        #ql = np.add(qc, qr)
        qv = self._ScalarState.get_field('qv')
        ql = self._Micro.get_qc()
        qi = np.zeros_like(ql)

        T = self._DiagnosticState.get_field('T')
        thetav = self._DiagnosticState.get_field('thetav')
        alpha = self._DiagnosticState.get_field('alpha')
        buoyancy = self._DiagnosticState.get_field('buoyancy')
        bvf = self._DiagnosticState.get_field('bvf')
        w_t = self._VelocityState.get_tend('w')

        #ThermodynamicsMoist_impl.eos(z, p0, alpha0, s, ql, qi, T, alpha, buoyancy)
        ThermodynamicsMoist_impl.eos_sam(z, p0, alpha0, s, qv, ql, qi, T, tref,  alpha, buoyancy)
        
        #Compute the buoyancy frequency
        ThermodynamicsMoist_impl.compute_bvf(theta_ref, exner, T, qv, ql, dz, thetav, bvf)



        if apply_buoyancy:
            ThermodynamicsMoist_impl.apply_buoyancy(buoyancy, w_t)

        return

    