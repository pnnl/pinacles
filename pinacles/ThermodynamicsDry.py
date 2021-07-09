from pinacles import Thermodynamics, ThermodynamicsDry_impl
from pinacles import parameters
import numpy as np
import numba

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase):
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, None)

        ScalarState.add_variable('qv', 
            long_name = 'Water vapor mixing ratio',
            latex_name = 'q_v',
            units='kg kg^{-1}')

        DiagnosticState.add_variable('bvf', 
            long_name = 'Brunt–Väisälä frequency squared', 
            latex_name = 'N^2',
            units='s^-2')

        DiagnosticState.add_variable('thetav',
            long_name = 'Virtual Potential Temperature',
            latex_name = '\theta_v',
            units = 'K')

        return

    def update(self, apply_buoyancy=True):

        n_halo = self._Grid.n_halo
        z = self._Grid.z_global
        dz = self._Grid.dx[2]
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        T0 = self._Ref.T0
        exner = self._Ref.exner
        tref = self._Ref.T0

        theta_ref = T0/exner

        s = self._ScalarState.get_field('s')
        qv = self._ScalarState.get_field('qv')
        T = self._DiagnosticState.get_field('T')
        alpha = self._DiagnosticState.get_field('alpha')
        buoyancy = self._DiagnosticState.get_field('buoyancy')
        thetav = self._DiagnosticState.get_field('thetav')
        bvf = self._DiagnosticState.get_field('bvf')
        w_t = self._VelocityState.get_tend('w')

        ThermodynamicsDry_impl.eos_sam(z, p0, alpha0, s, qv, T, tref, alpha, buoyancy)
        ThermodynamicsDry_impl.compute_bvf(n_halo,theta_ref, exner, T, qv, dz, thetav, bvf)

        if apply_buoyancy:
            ThermodynamicsDry_impl.apply_buoyancy(buoyancy, w_t)

        return

    @staticmethod
    @numba.njit()
    def compute_thetali(exner, T, thetali):
        shape = T.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    thetali[i,j,k] = T[i,j,k]/exner[k]
        return

    def get_thetali(self):
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field('T')
        thetali = np.empty_like(T)
        self.compute_thetali(exner, T, thetali)
        return thetali

    def get_qt(self):
        #Todo this gets a copy. So modifying it does nothing!
        qv = self._ScalarState.get_field('qv')
        return np.copy(qv)