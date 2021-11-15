from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
)


from pinacles import ThermodynamicsMoist_impl
from pinacles import UtilitiesParallel
from pinacles.WRFUtil import (
    to_wrf_order,
    wrf_tend_to_our_tend,
    wrf_theta_tend_to_our_tend,
    to_our_order,
)
from pinacles import parameters
from mpi4py import MPI
import numpy as np
import numba

@numba.njit(fastmath=True)
def compute_sat(temp, pressure):
    ep2 = 287.0 / 461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    es = 1000.0 * svp1 * np.exp(svp2 * (temp - svpt0) / (temp - svp3))
    qvs = ep2 * es / (pressure - es)

    return es, qvs

@numba.njit(fastmath=True)
def sa(z, p, s_in, qv_in, ql_in):

    T_1 = ThermodynamicsMoist_impl.T(z, s_in, ql_in, 0.0)
    
    es, qvs = compute_sat(T_1, p)
    
    qt_in = qv_in + ql_in
    if qt_in <= qvs:
        return T_1, qt_in, 0.0
        
    sigma_1 = qt_in - qvs

    s_1 = ThermodynamicsMoist_impl.s(z, T_1, sigma_1, 0.0)
    f_1 = s_in - s_1
    T_2 = ThermodynamicsMoist_impl.T(z, s_in, sigma_1, 0.0)
    delta_T = np.abs(T_2 - T_1)
    
    qv_star_2 = 0
    sigma_2 = -1.0
    while (delta_T >= 1e-4 or sigma_2 < 0.0):
        pv_star_2, qv_star_2 = compute_sat(T_2, p)
        sigma_2 = qt_in - qv_star_2 
        s_2 = ThermodynamicsMoist_impl.s(z, T_2, sigma_2, 0.0)
        f_2 = s_in - s_2
        T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
        T_1 = T_2
        T_2 = T_n
        f_1 = f_2 
        delta_T = np.abs(T_2 - T_1)
    
    qc = max(qt_in - qv_star_2, 0.0)
    qv = qt_in -  qc

    return T_1, qv, qc


@numba.njit
def compute_rh(qv, temp, pressure):
    return qv / compute_qvs(temp, pressure)

class MicroSA(MicrophysicsBase):

    def __init__(
        self,
        Timers,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):
    
        MicrophysicsBase.__init__(
            self,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )

        self._ScalarState.add_variable(
            "qv",
            long_name="water vapor mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_v",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qc",
            long_name="cloud water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_c",
            limit=True,
        )
        self._ScalarState.add_variable(
            "qr",
            long_name="rain water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{r}",
            limit=True,
        )


        self._Timers.add_timer("MicroSA_update")
        return 

    @staticmethod
    @numba.njit()
    def compute_sa(z, p, s, qv, ql, T):

        shape = qv.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    T[i,j,k], qv[i,j,k], ql[i,j,k] = sa(z[k], p[k], s[i,j,k], qv[i,j,k], ql[i,j,k]) 

        return



    def update(self):

        self._Timers.start_timer("MicroSA_update")
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")
        qc = self._ScalarState.get_field("qc")
        qr = self._ScalarState.get_field("qr")
        
        p0 = self._Ref.p0
        z = self._Grid.z_local

        self.compute_sa(z, p0, s, qv, qc, T)

        self._Timers.end_timer("MicroSA_update")

        return

    def io_initialize(self, nc_grp):


        return

    def io_update(self, nc_grp):


        return

    def io_fields2d_update(self, nc_grp):

        return

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    def get_reffc(self):
        qc = self._ScalarState.get_field("qc")
        reff = np.zeros_like(qc)
        reff[qc > 0.0] = 10 * 1.0e-6
        return reff

    def get_qi(self):
        qc = self._ScalarState.get_field("qc")
        return np.zeros_like(qc)

    def get_reffi(self):
        qc = self._ScalarState.get_field("qc")
        return np.zeros_like(qc)
