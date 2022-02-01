import numpy as np

from pinacles import Kinematics_impl


class Kinematics:
    def __init__(self, Timers, Grid, Ref, VelocityState, DiagnosticState):

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState

        DiagnosticState.add_variable(
            "strain_rate_mag",
            long_name="Magnitude of strain rate tensor",
            latex_name="|S_{i,j}|",
            units="s^{-1}",
        )

        DiagnosticState.add_variable(
            "Q_criterion", long_name="Q criterion", latex_name="Q", units="m^2 s^-2"
        )

        DiagnosticState.add_variable(
            "vertical_vorticity",
            long_name="Vertical Component of Vorticity",
            latex_name="Q",
            units="s^{-1}",
        )

        DiagnosticState.add_variable(
            "helicity", long_name="helicity", latex_name="helicity", units="m s^{-2}"
        )

        nl = self._Grid.ngrid_local

        # Gradients of u
        self._dudx = np.zeros((nl[0], nl[1], nl[2]), dtype=np.double)
        self._dudy = np.zeros_like(self._dudx)
        self._dudz = np.zeros_like(self._dudx)

        # Gradients of V
        self._dvdx = np.zeros_like(self._dudx)
        self._dvdy = np.zeros_like(self._dudx)
        self._dvdz = np.zeros_like(self._dudx)

        # Gradients of W
        self._dwdx = np.zeros_like(self._dudx)
        self._dwdy = np.zeros_like(self._dudx)
        self._dwdz = np.zeros_like(self._dudx)

        self._s11 = np.zeros_like(self._dudx)
        self._s22 = np.zeros_like(self._dudx)
        self._s33 = np.zeros_like(self._dudx)

        self._s12 = np.zeros_like(self._dudx)
        self._s13 = np.zeros_like(self._dudx)
        self._s23 = np.zeros_like(self._dudx)

        self._Timers.add_timer("Kinematics_update")

        return

    def update(self):

        self._Timers.start_timer("Kinematics_update")
        # Get the velocity components
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")

        strain_rate_mag = self._DiagnosticState.get_field("strain_rate_mag")
        Q_mag = self._DiagnosticState.get_field("Q_criterion")
        vertical_vorticity = self._DiagnosticState.get_field("vertical_vorticity")
        helicity = self._DiagnosticState.get_field("helicity")

        # Get grid spacing
        dxi = self._Grid.dxi

        # Compute the gradients
        Kinematics_impl.u_gradients(dxi, u, self._dudx, self._dudy, self._dudz)
        Kinematics_impl.v_gradients(dxi, v, self._dvdx, self._dvdy, self._dvdz)
        Kinematics_impl.w_gradients(dxi, w, self._dwdx, self._dwdy, self._dwdz)

        # Compute the strain rate mag
        Kinematics_impl.strain_rate(
            self._dudx,
            self._dudy,
            self._dudz,
            self._dvdx,
            self._dvdy,
            self._dvdz,
            self._dwdx,
            self._dwdy,
            self._dwdz,
            self._s11,
            self._s22,
            self._s33,
            self._s12,
            self._s13,
            self._s23,
            strain_rate_mag,
        )

        Kinematics_impl.q_criterion(
            u,
            v,
            w,
            self._dudx,
            self._dudy,
            self._dudz,
            self._dvdx,
            self._dvdy,
            self._dvdz,
            self._dwdx,
            self._dwdy,
            self._dwdz,
            Q_mag,
            vertical_vorticity,
            helicity,
        )

        self._Timers.end_timer("Kinematics_update")

        return
