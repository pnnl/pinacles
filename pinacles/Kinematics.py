import numpy as np
import pylab as plt

from pinacles import Kinematics_impl


class Kinematics:
    def __init__(self, Grid, Ref, VelocityState, DiagnosticState):


        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        DiagnosticState.add_variable('strain_rate_mag', long_name = 'Magnitude of strain rate tensor', latex_name = '|S_{i,j}|', units='s^{-1}')

        nl = self._Grid.ngrid_local

        #Gradients of u
        self._dudx = np.zeros((nl[0], nl[1], nl[2]), dtype=np.double)
        self._dudy = np.zeros_like(self._dudx)
        self._dudz = np.zeros_like(self._dudx)

        #Gradients of V
        self._dvdx = np.zeros_like(self._dudx)
        self._dvdy = np.zeros_like(self._dudx)
        self._dvdz = np.zeros_like(self._dudx)

        #Gradients of W
        self._dwdx = np.zeros_like(self._dudx)
        self._dwdy = np.zeros_like(self._dudx)
        self._dwdz = np.zeros_like(self._dudx)


        return

    def update(self):

        #Get the velocity components
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        strain_rate_mag = self._DiagnosticState.get_field('strain_rate_mag')
        
        #Get grid spacing
        dxi = self._Grid.dxi


        #Compute the gradients
        Kinematics_impl.u_gradients(dxi, u, self._dudx, self._dudy, self._dudz)
        Kinematics_impl.v_gradients(dxi, v, self._dvdx, self._dvdy, self._dvdz)
        Kinematics_impl.w_gradients(dxi, w, self._dwdx, self._dwdy, self._dwdz)


        #Compute the strain rate mag
        Kinematics_impl.strain_rate_max(self._dudx, self._dudy, self._dudz,
            self._dvdx, self._dvdy, self._dvdz,
            self._dwdx, self._dwdy, self._dwdz,
            strain_rate_mag)

        #import pylab as plt
        #plt.contourf(self._dudy[:,4,:].T + self._dvdx[:,4,:].T)
        #plt.show()
        #eturn





