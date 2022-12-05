import numba
import numpy as np


@numba.njit()
def water_path(n_halo, dz, npts, rho, q):
    path = 0.0
    shape = q.shape
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                path += q[i, j, k] * rho[k] * dz
    return path / npts

@numba.njit()
def pseudo_albedo(n_halo, dz, npts, rho, q, re):
    path = 0.0
    shape = q.shape
    g = 0.86
    
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                tau = 1.5 * q[i, j, k] * rho[k] * dz / (1e6 * re)
                path += (1.0-g) * tau  / (2.0 + (1 - g) * tau)
    return path / npts


@numba.njit()
def water_path_lasso(n_halo, dz, rho, qc):
    shape = qc.shape
    npts = 0
    lwp = 0

    paths = np.zeros((shape[0], shape[1]), dtype=np.double)
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                if qc[i, j, k] > 1e-7:
                    paths[i, j] += qc[i, j, k] * rho[k] * dz

    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            if paths[i, j] * 1000.0 > 1.0:
                lwp += paths[i, j]
                npts += 1

    return lwp, npts


@numba.njit()
def water_fraction(n_halo, npts, q, threshold=1e-20):
    frac = 0.0
    shape = q.shape
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                if q[i, j, k] >= threshold:
                    frac += 1.0
                    break

    return frac / npts


@numba.njit()
def water_fraction_profile(n_halo, npts, q, threshold=1e-20):
    frac = 0.0
    shape = q.shape
    frac = np.zeros((shape[2],), dtype=np.double)
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                if q[i, j, k] >= threshold:
                    frac[k] += 1.0

    return frac / npts


@numba.njit()
def compute_cloud_base_top(n_halo, z, q, threshold=1e-20):
    base_mean = 0.0
    top_mean = 0.0
    count = 0
    
    shape = q.shape
    cloud_base = np.empty((shape[0], shape[1]), dtype=np.double)
    cloud_top =  np.empty((shape[0], shape[1]), dtype=np.double)
    
    cloud_base.fill(1e9)
    cloud_top.fill(-1)
    
    for i in range(n_halo[0], shape[0] - n_halo[0]):
        for j in range(n_halo[1], shape[1] - n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                if q[i,j,k] >= threshold:
                    cloud_base[i,j] = min(z[k], cloud_base[i,j])
                    cloud_top[i,j] = max(z[k], cloud_top[i,j])
            
            if cloud_top[i,j] >= 0.0:
                count += 1
                base_mean += cloud_base[i,j]
                top_mean += cloud_top[i,j]
            
    return cloud_base, cloud_top, base_mean, top_mean, count

class MicrophysicsBase:
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

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self.name = "MicroBase"

        return

    def initialize(self):
        return

    def update(self):
        return

    def io_initialize(self, nc_grp):
        return

    def io_update(self, nc_grp):
        return

    def io_fields2d_update(self, nc_grp):
        return
    
    def io_tower(self, rt_grp, i_indx, j_indx):
        return

    def io_tower_init(self, rt_grp):
       return

    def get_qc(self):
        return np.zeros((self._Grid._ngrid_local), dtype=np.double)

    def get_qcloud(self):
        return np.zeros((self._Grid._ngrid_local), dtype=np.double)

    def get_qi(self):
        return np.zeros((self._Grid._ngrid_local), dtype=np.double)

    def get_reffc(self):
        # make consistent with default effective radius values used by P3
        return np.ones((self._Grid._ngrid_local), dtype=np.double) * 10.0e-6

    def get_reffi(self):
        # make consistent with default effective radius values used by P3
        return np.ones((self._Grid._ngrid_local), dtype=np.double) * 25.0e-6

    def restart(self, data_dict, **kwargs):
        return

    def dump_restart(self, data_dict):
        return
