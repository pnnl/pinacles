import numpy as np
import numba
from mpi4py import MPI

from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
)
from pinacles import ThermodynamicsMoist_impl
from pinacles import UtilitiesParallel


@numba.njit(fastmath=True)
def compute_sat(temp, pressure):
    ep2 = 287.0 / 461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    _es = 1000.0 * svp1 * np.exp(svp2 * (temp - svpt0) / (temp - svp3))
    qvs = ep2 * _es / (pressure - _es)

    return _es, qvs


@numba.njit(fastmath=True)
def compute_qvs(temp, pressure):
    ep2 = 287.0 / 461.6
    svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    _es = 1000.0 * svp1 * np.exp(svp2 * (temp - svpt0) / (temp - svp3))
    qvs = ep2 * _es / (pressure - _es)

    return qvs


@numba.njit(fastmath=True)
def sa(z, p, s_in, qv_in, ql_in):

    T_1 = ThermodynamicsMoist_impl.T(z, s_in, ql_in, 0.0)

    qvs = compute_qvs(T_1, p)

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
    while delta_T >= 1e-4 or sigma_2 < 0.0:
        qv_star_2 = compute_qvs(T_2, p)
        sigma_2 = qt_in - qv_star_2
        s_2 = ThermodynamicsMoist_impl.s(z, T_2, sigma_2, 0.0)
        f_2 = s_in - s_2
        T_n = T_2 - f_2 * (T_2 - T_1) / (f_2 - f_1)
        T_1 = T_2
        T_2 = T_n
        f_1 = f_2
        delta_T = np.abs(T_2 - T_1)

    qc = max(qt_in - qv_star_2, 0.0)
    qv = qt_in - qc

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

    @staticmethod
    @numba.njit()
    def compute_sa(_z, _p, _s, _qv, _ql, _T):

        shape = _qv.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    _T[i, j, k], _qv[i, j, k], _ql[i, j, k] = sa(
                        _z[k], _p[k], _s[i, j, k], _qv[i, j, k], _ql[i, j, k]
                    )

    def update(self):

        self._Timers.start_timer("MicroSA_update")
        _T = self._DiagnosticState.get_field("T")
        _s = self._ScalarState.get_field("s")
        _qv = self._ScalarState.get_field("qv")
        _qc = self._ScalarState.get_field("qc")
        _qr = self._ScalarState.get_field("qr")

        _p0 = self._Ref.p0
        _z = self._Grid.z_local

        self.compute_sa(_z, _p0, _s, _qv, _qc, _T)

        self._Timers.end_timer("MicroSA_update")

    def io_initialize(self, nc_grp):

        timeseries_grp = nc_grp["timeseries"]
        profiles_grp = nc_grp["profiles"]

        # Cloud fractions
        _v = timeseries_grp.createVariable("CF", np.double, dimensions=("time",))
        _v.long_name = "Cloud Fraction"
        _v.standard_name = "CF"
        _v.units = ""

        # Liquid water path
        _v = timeseries_grp.createVariable("LWP", np.double, dimensions=("time",))
        _v.long_name = "Liquid Water Path"
        _v.standard_name = "LWP"
        _v.units = "kg/m^2"

        # Water vapor path
        _v = timeseries_grp.createVariable("VWP", np.double, dimensions=("time",))
        _v.long_name = "Water Vapor Path"
        _v.standard_name = "VWP"
        _v.units = "kg/m^2"

        # Now add cloud fraction and rain fraction profiles
        _v = profiles_grp.createVariable(
            "CF",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        _v.long_name = "Cloud Fraction"
        _v.standard_name = "CF"
        _v.units = ""

    def io_update(self, nc_grp):

        _my_rank = MPI.COMM_WORLD.Get_rank()

        _n_halo = self._Grid.n_halo
        _dz = self._Grid.dx[2]
        _rho = self._Ref.rho0
        _npts = self._Grid.n[0] * self._Grid.n[1]

        _qc = self._ScalarState.get_field("qc")
        _qv = self._ScalarState.get_field("qv")

        # First compute liquid water path
        _lwp = water_path(_n_halo, _dz, _npts, _rho, _qc)
        _lwp = UtilitiesParallel.ScalarAllReduce(_lwp)

        # Compute vapor water path
        _vwp = water_path(_n_halo, _dz, _npts, _rho, _qv)
        _vwp = UtilitiesParallel.ScalarAllReduce(_vwp)

        # Compute cloud fractions
        _cf = water_fraction(_n_halo, _npts, _qc, threshold=1e-5)
        _cf = UtilitiesParallel.ScalarAllReduce(_cf)

        _cf_prof = water_fraction_profile(_n_halo, _npts, _qc, threshold=1e-5)
        _cf_prof = UtilitiesParallel.ScalarAllReduce(_cf_prof)

        if _my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]
            profiles_grp = nc_grp["profiles"]

            timeseries_grp["CF"][-1] = _cf
            timeseries_grp["LWP"][-1] = _lwp
            timeseries_grp["VWP"][-1] = _vwp

            profiles_grp["CF"][-1, :] = _cf_prof[_n_halo[2] : -_n_halo[2]]

    def io_fields2d_update(self, nc_grp):

        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        # Compute and output the LWP
        if nc_grp is not None:
            lwp = nc_grp.createVariable(
                "LWP",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        _nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        _qc = self._ScalarState.get_field("qc")[_nh[0] : -_nh[0], _nh[1] : -_nh[1], :]
        lwp_compute = np.sum(
            _qc * rho0[np.newaxis, np.newaxis, 0] * self._Grid.dx[2], axis=2
        )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = lwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            lwp[:, :] = recv_buffer

        if nc_grp is not None:
            nc_grp.sync()

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    def get_reffc(self):
        _qc = self._ScalarState.get_field("qc")
        reff = np.zeros_like(_qc)
        reff[_qc > 0.0] = 10 * 1.0e-6
        return reff

    def get_qi(self):
        _qc = self._ScalarState.get_field("qc")
        return np.zeros_like(_qc)

    def get_reffi(self):
        _qc = self._ScalarState.get_field("qc")
        return np.zeros_like(_qc)
