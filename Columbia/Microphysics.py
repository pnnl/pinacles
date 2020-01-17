import numba

@numba.njit()
def water_path(n_halo, dz, npts, rho, q):
    path = 0.0 
    shape = q.shape
    for i in range(n_halo[0], shape[0]-n_halo[0]):
        for j in range(n_halo[1], shape[1]-n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                path += q[i,j,k] * rho[k] * dz
    return path/npts


@numba.njit()
def water_fraction(n_halo, npts, q):
    frac = 0.0
    shape = q.shape
    for i in range(n_halo[0], shape[0]-n_halo[0]):
        for j in range(n_halo[1], shape[1]-n_halo[1]):
            for k in range(n_halo[2], shape[2] - n_halo[2]):
                if q[i,j,k] >= 1e-8:
                    frac += 1.0
                    break

    return frac/npts

class MicrophysicsBase:
    def __init__(self, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self.name = 'MicroBase'

        return

    def update(self):
        return

    def io_initialize(self, nc_grp):

        return

    def io_update(self, nc_grp):
        return

