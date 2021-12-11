from pinacles import parameters
from pinacles import UtilitiesParallel
import time
import numba
import numpy as np
from mpi4py import MPI
from scipy import interpolate
import netCDF4 as nc
import sys


class RadiationDycoms:
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        ScalarState,
        DiagnosticState,
        Micro,
        TimeSteppingController,
    ):

        self._name = "RadiationDycoms"
        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Micro = Micro
        self._TimeSteppingController = TimeSteppingController

        self._DiagnosticState.add_variable("dTdt_rad")

        self.zi_mean = 0.0
        self.zi_min = 0.0
        self.zi_max = 0.0

        try:
            self.time_synced = namelist["radiation"]["time_synced"]
        except:
            self.time_synced = False

        try:
            self._radiation_frequency = namelist["radiation"]["update_frequency"]
        except:
            if self.time_synced:
                sys.exit(
                    "EXITING: for time syncing of DYCOMS, a radiation update frequency must be specified in the namelist"
                )
            else:
                self._radiation_frequency = 0.0

        self.frequency = self._radiation_frequency  # This is used for time syncing

        self.time_elapsed = parameters.LARGE

        self._restart_attributes = ["time_elapsed"]

        self._Timers.add_timer("RadiationDycoms")
        return

    def init_profiles(self):
        # Nothing to do here
        return

    def update(self, force=False):
        self._Timers.start_timer("RadiationDycoms")

        self.time_elapsed += self._TimeSteppingController.dt
        dTdt_rad = self._DiagnosticState.get_field("dTdt_rad")
        s = self._ScalarState.get_field("s")
        if (
            (self.time_elapsed > self._radiation_frequency and not self.time_synced)
            or (
                self.time_synced
                and np.allclose(
                    self._TimeSteppingController._time % self.frequency, 0.0
                )
            )
            or force
        ):
            self.time_elapsed = 0.0
            # heating_rate_lw = self._DiagnosticState.get_field('heating_rate_lw')

            qc = self._Micro.get_qc()
            qv = self._ScalarState.get_field("qv")
            rho = self._Ref._rho0

            nh = self._Grid.n_halo

            z = self._Grid.z_global
            z_edge = self._Grid.z_edge_global
            dt = self._TimeSteppingController.dt
            self.zi_mean, self.zi_min, self.zi_max = dycoms_rad_calc(
                nh, self._Grid.dxi[2], z, z_edge, rho, qc, qv, dTdt_rad
            )

        self._Timers.end_timer("RadiationDycoms")
        return

    def update_apply_tend(self):

        s = self._ScalarState.get_field("s")
        dTdt_rad = self._DiagnosticState.get_field("dTdt_rad")
        dt = self._TimeSteppingController.dt
        s[:, :, :] += dTdt_rad[:, :, :] * dt

        return

    def io_initialize(self, nc_grp):
        # add zi to the output?
        timeseries_grp = nc_grp["timeseries"]
        v = timeseries_grp.createVariable("zi_mean", np.double, dimensions=("time",))
        v.long_name = "Mean DYCOMS inversion"
        v.standard_name = "zi_mean"
        v.units = "m"

        v = timeseries_grp.createVariable("zi_min", np.double, dimensions=("time",))
        v.long_name = "Minimum DYCOMS inversion"
        v.standard_name = "zi_min"
        v.units = "m"

        v = timeseries_grp.createVariable("zi_max", np.double, dimensions=("time",))
        v.long_name = "Maximum DYCOMS inversion"
        v.standard_name = "zi_max"
        v.units = "m"

        return

    def io_update(self, nc_grp):
        my_rank = MPI.COMM_WORLD.Get_rank()

        npts = self._Grid.n[0] * self._Grid.n[1]
        zi_mean = UtilitiesParallel.ScalarAllReduce(self.zi_mean / npts)
        zi_min = UtilitiesParallel.ScalarAllReduce(self.zi_min, op=MPI.MIN)
        zi_max = UtilitiesParallel.ScalarAllReduce(self.zi_max, op=MPI.MAX)

        if my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]

            timeseries_grp["zi_mean"][-1] = zi_mean
            timeseries_grp["zi_max"][-1] = zi_max
            timeseries_grp["zi_min"][-1] = zi_min

        return

    def io_fields2d_update(self, nc_grp):
        return

    def restart(self, data_dict, **kwargs):
        return

    def dump_restart(self, data_dict):
        # Loop through all attributes storing them
        key = "Radiation"
        data_dict[key] = {}
        for item in self._restart_attributes:
            data_dict[key][item] = self.__dict__[item]
        return

    @property
    def name(self):
        return self._name


@numba.njit()
def dycoms_rad_calc(nh, dzi, z, z_edge, rho, qc, qv, dT):
    F0 = 70.0  # W m^-2
    F1 = 22.0  # W m^-2
    kappa = 85.0  # m^2 kg^-1
    a = 1.0  # K m^{-1/3}
    rho_zi = 1.12  # kg m^{-3} (air density at initial inversion zi = 795)
    qt_zi = 8.0e-3  # kg kg^-1 (total water threshold for diagnosing zi)
    D = 3.75e-6  # s^{-1} dive
    shape = qc.shape
    lw_flux = np.zeros(shape[2], dtype=np.double)
    qtop = np.zeros_like(lw_flux)
    qbot = np.zeros_like(lw_flux)
    kmin = nh[2] - 1
    kmax = shape[2] - nh[2]
    zi_mean = 0.0
    zi_max = -9999.0
    zi_min = 9999.0
    for i in range(nh[0], shape[0] - nh[0]):
        for j in range(nh[1], shape[1] - nh[1]):
            qt_index = qc[i, j, kmin] + qv[i, j, kmin]
            for k in range(kmin + 1, kmax):
                index = k
                qt_indexm1 = qt_index
                qt_index = qc[i, j, k] + qv[i, j, k]
                if qt_index < qt_zi:
                    break
            zi = (qt_zi - qt_indexm1) / (qt_index - qt_indexm1) / dzi + z[index - 1]
            zi_mean += zi
            if zi > zi_max:
                zi_max = zi
            elif zi < zi_min:
                zi_min = zi

            for k in range(kmax - 1, kmin - 1, -1):
                qtop[k] = qtop[k + 1] + qc[i, j, k + 1] * rho[k + 1] / dzi * kappa
            for k in range(kmin, kmax + 1):
                qbot[k] = qbot[k - 1] + qc[i, j, k] * rho[k] / dzi * kappa

                lw_flux[k] = F0 * np.exp(-qtop[k]) + F1 * np.exp(-qbot[k])
                if z_edge[k] > zi:
                    cbrt_z = (z_edge[k] - zi) ** (1.0 / 3.0)
                    lw_flux[k] += (
                        a
                        * rho_zi
                        * D
                        * parameters.CPD
                        * (0.25 * cbrt_z ** 4.0 + zi * cbrt_z)
                    )

                dT[i, j, k] = (
                    -(lw_flux[k] - lw_flux[k - 1]) * dzi / parameters.CPD / rho[k]
                )

    return zi_mean, zi_min, zi_max
