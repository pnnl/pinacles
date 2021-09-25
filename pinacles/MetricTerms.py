import numpy as np
import numba


class MetricTermsBase:
    def __init__(self, Grid, DiagnosticState):

        return

    def add_contravariant_velocities(self):

        return

    def update_contravariant_velocities(self):

        return


class MetricTerms(MetricTermsBase):
    def __init__(self, Grid, VelocityState, DiagnosticState):

        self._DiagnosticState = DiagnosticState
        self._VelocityState = VelocityState
        self._Grid = Grid

        self.dxx = None
        self.dyy = None
        self.dzz = None

        self.dxy = None
        self.dxz = None

        self.dyx = None
        self.dyz = None

        self.dzx = None
        self.dzy = None

        self._DiagnosticState.add_variable("Uc")
        self._DiagnosticState.add_variable("Vc")
        self._DiagnosticState.add_variable("Wc")

        self.compute_metric_terms()

        return

    def compute_metric_terms(self):

        shape = self._Grid.ngrid_local
        self.dxx = np.ones(shape, dtype=np.double)
        self.dyy = np.ones(shape, dtype=np.double)
        self.dzz = np.ones(shape, dtype=np.double)

        self.dxy = np.ones(shape, dtype=np.double)
        self.dxz = np.ones(shape, dtype=np.double)

        self.dyx = np.zeros(shape, dtype=np.double)
        self.dyz = np.zeros(shape, dtype=np.double)

        self.dzx = np.zeros(shape, dtype=np.double)
        self.dzy = np.zeros(shape, dtype=np.double)
        self.J = np.zeros(shape, dtype=np.double)


        self.dzz[:,:,:-1] = (self._Grid.z_local[1:] - self._Grid.z_local[:-1])[np.newaxis, np.newaxis,:]
        self.J = self.dxx + self.dyy + self.dzz


        print(self.dzz)


        return

    @staticmethod
    @numba.njit()
    def compute_Wc(dxx, dyy, dzx, dzy, dzz, u, v, w, Wc):

        shape = w.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    Wc[i, j, k] = (
                        w[i, j, k] / dzz[i, j, k]
                        - u[i, j, k] * dzx[i, j, k] / (dxx[i, j, k] * dzz[i, j, k])
                        - v[i, j, k] * dzy[i, j, k] / (dyy[i, j, k] * dzz[i, j, k])
                    )

        return

    def compute_contravariant_velocities(self):

        Uc = self._DiagnosticState.get_field("Uc")
        Vc = self._DiagnosticState.get_field("Vc")
        Wc = self._DiagnosticState.get_field("Wc")

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")


        # Actually compute the contravariant velocities
        np.multiply(u, self.dxx, out=Uc)
        np.multiply(v, self.dyy, out=Vc)
        self.compute_Wc(self.dxx, self.dyy, self.dzx, self.dzy, self.dzz, u, v, w, Wc)


        return
