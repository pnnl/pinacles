import numpy as np
import numba


@numba.njit(fastmath=True)
def Thomas(x, a, b, c):
    """ a generic Thomas algorithm tridiagonal solver. 
    
    Arguments:
        x {[type]} -- [description]
        a {[type]} -- [description]
        b {[type]} -- [description]
        c {[type]} -- [description]
    """
    shape = x.shape
    scratch = np.empty(shape[2], dtype=np.double)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Upward sweep
            scratch[0] = c[0] / b[0]
            x[i, j, 0] = x[i, j, 0] / b[0]
            for k in range(1, shape[2]):
                m = 1.0 / (b[k] - a[k] * scratch[k - 1])
                scratch[k] = c[k] * m
                x[i, j, k] = (x[i, j, k] - a[k] * x[i, j, k - 1]) * m
            # Downward sweep
            for k in range(shape[2] - 2, -1, -1):
                x[i, j, k] = x[i, j, k] - scratch[k] * x[i, j, k + 1]
    return


@numba.njit(fastmath=True)
def PressureThomas(
    n_halo, dxs, rho0, rho0_edge, kx2, ky2, x, a, c, wavenumber_substarts
):
    shape = x.shape
    scratch = np.empty(shape[2], dtype=np.double)
    b = np.empty(shape[2], dtype=np.double)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if wavenumber_substarts[0] + i != 0 or wavenumber_substarts[1] + j != 0:
                # For each i and j build the diagonal
                b[0] = rho0[n_halo[2]] * (kx2[i] + ky2[j]) - (rho0_edge[n_halo[2]]) / (
                    dxs[2] * dxs[2]
                )
                for k in range(1, shape[2] - 1):
                    b[k] = rho0[k + n_halo[2]] * (kx2[i] + ky2[j]) - (
                        rho0_edge[k + n_halo[2]] + rho0_edge[k + n_halo[2] - 1]
                    ) / (dxs[2] * dxs[2])
                k = shape[2] - 1
                b[k] = rho0[k + n_halo[2]] * (kx2[i] + ky2[j]) - (
                    rho0_edge[k + n_halo[2] - 1]
                ) / (dxs[2] * dxs[2])

                # Now begin the actual algorithm solve

                # Upward sweep
                scratch[0] = c[0] / b[0]
                x[i, j, 0] = x[i, j, 0] / b[0]
                for k in range(1, shape[2]):
                    m = 1.0 / (b[k] - a[k] * scratch[k - 1])
                    scratch[k] = c[k] * m
                    x[i, j, k] = (x[i, j, k] - a[k] * x[i, j, k - 1]) * m

                # Downward sweep
                for k in range(shape[2] - 2, -1, -1):
                    x[i, j, k] = x[i, j, k] - scratch[k] * x[i, j, k + 1]

    return


class PressureTDMA:
    def __init__(self, Grid, Ref, wavenumber_substarts, wavenumber_n):

        self._Grid = Grid
        self._Ref = Ref
        self._wavenumber_substarts = wavenumber_substarts
        self._wavenumber_n = wavenumber_n
        self._is_origin = False

        # Set up the diagonals for the solve
        self._a = None
        self._b = None
        self._c = None

        self._compute_modified_wavenumbers()
        self._set_upperlower_diagonals()

        if wavenumber_substarts[0] == 0 and wavenumber_substarts[1] == 0:
            self._is_origin = True

        return

    def _set_upperlower_diagonals(self):

        n_halo = self._Grid.n_halo
        nl = self._Grid.nl
        dxi = self._Grid.dxi

        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        self._a = np.zeros(self._Grid.n[2], dtype=np.double)
        self._c = np.zeros(self._Grid.n[2], dtype=np.double)

        # First set the lower boundary condition
        self._a[0] = 0.0
        self._c[0] = (0.5 * dxi[2] * dxi[2]) * rho0_edge[n_halo[2]]

        # Fill Matrix Values
        for k in range(1, nl[2] - 1):
            self._a[k] = (0.5 * dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2] - 1]
            self._c[k] = (0.5 * dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2]]

        # Now set surface boundary conditions
        k = nl[2] - 1
        self._a[k] = (0.5 * dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2] - 1]
        self._c[k] = 0.0

        return

    def _compute_modified_wavenumbers(self):
        n_h = self._Grid.n_halo
        dx = self._Grid.dx
        n = self._Grid.n

        self._kx2 = np.zeros(self._wavenumber_n[0], dtype=np.double)
        self._ky2 = np.zeros(self._wavenumber_n[1], dtype=np.double)

        # TODO the code below feels a bit like boilerplate

        for ii in range(self._wavenumber_n[0]):
            i = self._wavenumber_substarts[0] + ii
            if i <= n[0] / 2:
                xi = np.double(i)
            else:
                xi = np.double(i - n[0])
            self._kx2[ii] = (
                (2.0 * np.cos((2.0 * np.pi / n[0]) * xi) - 2.0) / dx[0] / dx[0]
            )

        for jj in range(self._wavenumber_n[1]):
            j = self._wavenumber_substarts[1] + jj
            if j <= n[1] / 2:
                yi = np.double(j)
            else:
                yi = np.double(j - n[1])
            self._ky2[jj] = (
                (2.0 * np.cos((2.0 * np.pi / n[1]) * yi) - 2.0) / dx[1] / dx[1]
            )

        # Remove the odd-ball
        if self._wavenumber_substarts[0] == 0:
            self._kx2[0] = 0.0
        if self._wavenumber_substarts[1] == 0:
            self._ky2[0] = 0.0

        return

    def solve(self, x):
        n_halo = self._Grid.n_halo
        dxs = self._Grid.dx
        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge
        PressureThomas(
            n_halo,
            dxs,
            rho0,
            rho0_edge,
            self._kx2,
            self._ky2,
            x,
            self._a,
            self._c,
            self._wavenumber_substarts,
        )

        return


class PressureNonPeriodicTDMA:
    def __init__(self, Grid, Ref, wavenumber_substarts, wavenumber_n):

        self._Grid = Grid
        self._Ref = Ref
        self._wavenumber_substarts = wavenumber_substarts
        self._wavenumber_n = wavenumber_n
        self._is_origin = False

        # Set up the diagonals for the solve
        self._a = None
        self._b = None
        self._c = None

        self._compute_modified_wavenumbers()
        self._set_upperlower_diagonals()

        if wavenumber_substarts[0] == 0 and wavenumber_substarts[1] == 0:
            self._is_origin = True

        print(wavenumber_substarts)
        return

    def _set_upperlower_diagonals(self):

        n_halo = self._Grid.n_halo
        nl = self._Grid.nl
        dxi = self._Grid.dxi

        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge

        self._a = np.zeros(self._Grid.n[2], dtype=np.double)
        self._c = np.zeros(self._Grid.n[2], dtype=np.double)

        # First set the lower boundary condition
        self._a[0] = 0.0
        self._c[0] = (dxi[2] * dxi[2]) * rho0_edge[n_halo[2]]

        # Fill Matrix Values
        for k in range(1, nl[2] - 1):
            self._a[k] = (dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2] - 1]
            self._c[k] = (dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2]]

        # Now set surface boundary conditions
        k = nl[2] - 1
        self._a[k] = (dxi[2] * dxi[2]) * rho0_edge[k + n_halo[2] - 1]
        self._c[k] = 0.0

        return

    def _compute_modified_wavenumbers(self):
        n_h = self._Grid.n_halo
        dx = self._Grid.dx
        n = self._Grid.n

        self._kx2 = np.zeros(self._wavenumber_n[0], dtype=np.double)
        self._ky2 = np.zeros(self._wavenumber_n[1], dtype=np.double)

        # TODO the code below feels a bit like boilerplate

        for ii in range(self._wavenumber_n[0]):
            i = self._wavenumber_substarts[0] + ii
            self._kx2[ii] = (
                -4 * (np.sin(np.pi / (2.0 * n[0]) * i) ** 2.0) / (dx[0] ** 2.0)
            )

        for jj in range(self._wavenumber_n[1]):
            j = self._wavenumber_substarts[1] + jj
            self._ky2[jj] = (
                -4 * (np.sin(np.pi / (2.0 * n[1]) * j) ** 2.0) / (dx[1] ** 2.0)
            )

        # Remove the odd-ball
        # if self._wavenumber_substarts[0] == 0:
        #    self._kx2[0] = 0.0
        # if self._wavenumber_substarts[1] == 0:
        #     self._ky2[0] = 0.0

        return

    def solve(self, x):
        n_halo = self._Grid.n_halo
        dxs = self._Grid.dx
        rho0 = self._Ref.rho0
        rho0_edge = self._Ref.rho0_edge
        PressureThomas(
            n_halo,
            dxs,
            rho0,
            rho0_edge,
            self._kx2,
            self._ky2,
            x,
            self._a,
            self._c,
            self._wavenumber_substarts,
        )
        return
