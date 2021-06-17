import numpy as np
import mpi4py_fft
from mpi4py import MPI
from scipy.fft import dctn, idctn, fftn, ifftn


class dct_mpi4py:
    def __init__(self, n, subcomms):

        self._subcomms = subcomms  # Subcomms created by mpi4py (this typically comes form Grid class)
        self._n = n  # Total number of points to be transfomred

        # Setup z, y, x pencils
        self.p2 = mpi4py_fft.pencil.Pencil(self._subcomms, self._n, axis=2)  # z pencil
        self.p1 = self.p2.pencil(1)  # y-pencil
        self.p0 = self.p1.pencil(0)  # x-pencil

        self.transfer21 = self.p2.transfer(
            self.p1, np.single
        )  # transfer function z to y
        self.transfer10 = self.p1.transfer(
            self.p0, np.single
        )  # transfer funtion y to x

        self.a2 = np.zeros(self.p2.subshape, dtype=np.single)  # z pencil work array
        self.a1 = np.zeros(self.p1.subshape, dtype=np.single)  # y pencil work array
        self.a0 = np.zeros(self.p0.subshape, dtype=np.single)  # x pencil work array

        return

    def forward(self, data):

        # Copy data to work array
        # np.copyto(self.a2, data)

        # Transfer z-pencil to y-pencil
        self.transfer21.forward(data, self.a1)
        self.a1[:, :, :] = dctn(self.a1, axes=1, type=2)

        # Transfer y-pencil to x-pencil
        self.transfer10.forward(self.a1, self.a0)
        self.a0[:, :, :] = dctn(self.a0, axes=0, type=2)

        # Transfer back to z-pencil
        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)

        return self.a2

    def backward(self, data):

        # Copy data to work array
        # np.copyto(self.a2, data)

        # Transfer z-pencil to y-pencil
        self.transfer21.forward(data, self.a1)
        self.a1[:, :, :] = idctn(self.a1, axes=1, type=2)

        # Transfer y-pencil to x-pencil
        self.transfer10.forward(self.a1, self.a0)
        self.a0[:, :, :] = idctn(self.a0, axes=0, type=2)

        # Transfer back to z-pencil
        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)

        return self.a2

    @property
    def n(self):
        # Return total number of grid points
        return self._n


class fft_mpi4py:
    def __init__(self, n, subcomms):

        self._subcomms = subcomms  # Subcomms created by mpi4py (this typically comes form Grid class)
        self._n = n  # Total number of points to be transfomred

        # Setup z, y, x pencils
        self.p2 = mpi4py_fft.pencil.Pencil(self._subcomms, self._n, axis=2,)  # z pencil
        self.p1 = self.p2.pencil(1)  # y-pencil
        self.p0 = self.p1.pencil(0)  # x-pencil

        self.transfer21 = self.p2.transfer(
            self.p1, np.csingle
        )  # transfer function z to y
        self.transfer10 = self.p1.transfer(
            self.p0, np.csingle
        )  # transfer funtion y to x

        self.a2 = np.zeros(self.p2.subshape, dtype=np.csingle)  # z pencil work array
        self.a1 = np.zeros(self.p1.subshape, dtype=np.csingle)  # y pencil work array
        self.a0 = np.zeros(self.p0.subshape, dtype=np.csingle)  # x pencil work array

        return

    def forward(self, data):

        # Transfer z-pencil to y-pencil
        self.transfer21.forward(data, self.a1)
        fftn(self.a1, axes=1, overwrite_x=True)

        # Transfer y-pencil to x-pencil
        self.transfer10.forward(self.a1, self.a0)
        fftn(self.a0, axes=0, overwrite_x=True)

        # Transfer back to z-pencil
        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)

        return self.a2

    def backward(self, data):

        # Transfer z-pencil to y-pencil
        self.transfer21.forward(data, self.a1)
        ifftn(self.a1, axes=1, overwrite_x=True)

        # Transfer y-pencil to x-pencil
        self.transfer10.forward(self.a1, self.a0)
        ifftn(self.a0, axes=0, overwrite_x=True)

        # Transfer back to z-pencil
        self.transfer10.backward(self.a0, self.a1)
        self.transfer21.backward(self.a1, self.a2)

        return self.a2

    @property
    def n(self):
        # Return total number of grid points
        return self._n
