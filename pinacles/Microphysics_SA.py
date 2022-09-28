import numpy as np
import numba
from mpi4py import MPI

from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
    compute_cloud_base_top,
)
from pinacles import ThermodynamicsMoist_impl
from pinacles import UtilitiesParallel
from pinacles import parameters


@numba.njit(fastmath=True)
def sa(z, rho0, p, s_in, qv_in, ql_in):

    T_1 = ThermodynamicsMoist_impl.T(z, s_in, ql_in, 0.0)

    qvs = ThermodynamicsMoist_impl.compute_qvs(T_1, rho0, p)

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
    

    while delta_T >= 1e-4:
        qv_star_2 = ThermodynamicsMoist_impl.compute_qvs(T_2, rho0, p)
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

    T_1 = ThermodynamicsMoist_impl.T(z, s_in, qc, 0.0)

    return T_1, qv, qc


@numba.njit
def compute_rh(qv, rho0, temp, pressure):
    return qv / ThermodynamicsMoist_impl.compute_qvs(temp, rho0, pressure)


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
            flux_divergence="EMONO",
        )
        self._ScalarState.add_variable(
            "qc",
            long_name="cloud water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_c",
            limit=True,
            flux_divergence="EMONO",
            is_prognosed_liquid=True
        )
        self._ScalarState.add_variable(
            "qr",
            long_name="rain water mixing ratio",
            units="kg kg^{-1}",
            latex_name="q_{r}",
            limit=True,
            flux_divergence="EMONO",
            is_prognosed_liquid=True
        )

        self._Timers.add_timer("MicroSA_update")

    @staticmethod
    @numba.njit()
    def compute_sa(_z, _rho0, _p, _s, _qv, _ql, _T):

        shape = _qv.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    _T[i, j, k], _qv[i, j, k], _ql[i, j, k] = sa(
                        _z[k], _rho0[k], _p[k], _s[i, j, k], _qv[i, j, k], _ql[i, j, k]
                    )

    def update(self):

        self._Timers.start_timer("MicroSA_update")
        _T = self._DiagnosticState.get_field("T")
        _s = self._ScalarState.get_field("s")
        _qv = self._ScalarState.get_field("qv")
        _qc = self._ScalarState.get_field("qc")
        _qr = self._ScalarState.get_field("qr")

        _p0 = self._Ref.p0
        _rho0 = self._Ref.rho0
        _z = self._Grid.z_local

        self.compute_sa(_z, _rho0, _p0, _s, _qv, _qc, _T)

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

        # Now add cloud top and cloud base profiles
        _v = timeseries_grp.createVariable(
            "cloud_base",
            np.double,
            dimensions=("time"),
        )
        _v.long_name = "Cloud Base"
        _v.standard_name = "Cloud Base"
        _v.units = "m"

        _v = timeseries_grp.createVariable(
            "cloud_base_mean",
            np.double,
            dimensions=("time"),
        )
        _v.long_name = "Cloud Base Mean"
        _v.standard_name = "Cloud Base Mean"
        _v.units = "m"

        _v = timeseries_grp.createVariable(
            "cloud_top",
            np.double,
            dimensions=("time"),
        )
        _v.long_name = "Cloud Top"
        _v.standard_name = "Cloud Top"
        _v.units = "m"

        _v = timeseries_grp.createVariable(
            "cloud_top_mean",
            np.double,
            dimensions=("time"),
        )
        _v.long_name = "Cloud Top Mean"
        _v.standard_name = "Cloud Top Mean"
        _v.units = "m"

    def io_update(self, nc_grp):

        _my_rank = MPI.COMM_WORLD.Get_rank()

        _n_halo = self._Grid.n_halo
        _dz = self._Grid.dx[2]
        _z = self._Grid.z_global
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
        _cf = water_fraction(_n_halo, _npts, _qc)
        _cf = UtilitiesParallel.ScalarAllReduce(_cf)

        # Compute cloud profile
        _cf_prof = water_fraction_profile(_n_halo, _npts, _qc)
        _cf_prof = UtilitiesParallel.ScalarAllReduce(_cf_prof)

        # Compute cloud base
        _cloud_base, _cloud_top, _base_mean, _top_mean, count = compute_cloud_base_top(
            _n_halo, _z, _qc
        )

        _cloud_base = UtilitiesParallel.ScalarAllReduce(
            np.amin(_cloud_base), op=MPI.MIN
        )
        _cloud_top = UtilitiesParallel.ScalarAllReduce(np.amax(_cloud_top), op=MPI.MAX)

        count = UtilitiesParallel.ScalarAllReduce(count)
        if count >= 1:
            _base_mean = UtilitiesParallel.ScalarAllReduce(_base_mean) / count
            _top_mean = UtilitiesParallel.ScalarAllReduce(_top_mean) / count

        if _cloud_base > 1e8:
            _cloud_base = -1.0

        if _my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]
            profiles_grp = nc_grp["profiles"]

            timeseries_grp["CF"][-1] = _cf
            timeseries_grp["LWP"][-1] = _lwp
            timeseries_grp["VWP"][-1] = _vwp
            timeseries_grp["cloud_base"][-1] = _cloud_base
            timeseries_grp["cloud_top"][-1] = _cloud_top
            timeseries_grp["cloud_base_mean"][-1] = _base_mean
            timeseries_grp["cloud_top_mean"][-1] = _top_mean

            profiles_grp["CF"][-1, :] = _cf_prof[_n_halo[2] : -_n_halo[2]]

    def io_fields2d_update(self, fx):

        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        # Compute and output the LWP
        if fx is not None:
            lwp = fx.create_dataset(
                "LWP",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                lwp.dims[i].attach_scale(fx[d])

        _nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        _qc = self._ScalarState.get_field("qc")[_nh[0] : -_nh[0], _nh[1] : -_nh[1], :]
        lwp_compute = np.sum(
            _qc * rho0[np.newaxis, np.newaxis, 0] * self._Grid.dx[2], axis=2
        )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = lwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        if fx is not None:
            lwp[:, :] = recv_buffer

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
