import numpy as np


class Types:
    def __init__(self, namelist):

        self._float = np.float64
        self._complex = np.complex128
        self._float_pressure = self._float
        self._complex_pressure = self._complex

        if "precision" in namelist:
            if namelist["precision"] == "mixed_pressure_only":
                self._float_pressure = np.float32
                self._complex_pressure = np.complex64

            if namelist["precision"] == "mixed":
                self._float = np.float32
                self._complex = np.complex64
                self._float_pressure = self._float
                self._complex_pressure = self._complex

        return

    @property
    def float(self):
        return self._float

    @property
    def complex(self):
        return self._complex

    @property
    def complex_pressure(self):
        return self._complex_pressure

    @property
    def float_pressure(self):
        return self._float_pressure
