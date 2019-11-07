from Columbia import Thermodynamics, ThermodynamicsDry_impl
from Columbia import parameters
from Columbia import parameters
import numpy as np

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase): 
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState)

        return  

    def update(self): 

        z = self._Grid.z_global
        p0 = self._Ref.p0 
        alpha0 = self._Ref.alpha0
        T0 = self._Ref.T0
        exner = self._Ref.exner
        
        TH0 = T0/exner

        s = self._ScalarState.get_field('s')
        T = self._DiagnosticState.get_field('T')
        alpha = self._DiagnosticState.get_field('alpha')
        buoyancy = self._DiagnosticState.get_field('buoyancy')
        w_t = self._VelocityState.get_tend('w')

        ThermodynamicsDry_impl.eos(z, p0, alpha0, s, T, alpha, buoyancy)
        buoyancy[:,:,:] = parameters.G * (T/exner[np.newaxis, np.newaxis,:] - TH0[np.newaxis, np.newaxis,:])/TH0[np.newaxis, np.newaxis,:]
        ThermodynamicsDry_impl.apply_buoyancy(buoyancy, w_t)

        return 

