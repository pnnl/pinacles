import numpy as np


class LambertConformal:
    def __init__(self, earth_radius, lat_1, lat_2, lat_0, lon_0):
        self.R = earth_radius
        self.phi_1 = lat_1 * np.pi / 180.0
        self.phi_2 = lat_2 * np.pi / 180.0
        self.phi_0 = lat_0 * np.pi / 180.0
        self.lam_0 = lon_0 * np.pi / 180.0

        self.n = None
        self.F = None
        self.rho0 = None

        self._compute_n()
        self._compute_F()
        self._compute_rho0()
        return

    def _compute_n(self):
        self.n = np.log(np.cos(self.phi_1) / np.cos(self.phi_2)) / np.log(
            np.tan(np.pi / 4.0 + self.phi_2 / 2.0)
            / np.tan(np.pi / 4.0 + self.phi_1 / 2.0)
        )
        print("n", self.n)
        return

    def _compute_F(self):
        self.F = (
            np.cos(self.phi_1)
            * (np.tan(np.pi / 4.0 + self.phi_1 / 2.0) ** self.n)
            / self.n
        )
        print("F", self.F)
        return

    def _compute_rho0(self):
        self.rho0 = self.R * self.F / (np.tan(np.pi / 4.0 + self.phi_0 / 2.0) ** self.n)
        print("rho0", self.rho0)
        return

    def compute_xy(self, phi, lam):
        phi_rad = phi * np.pi / 180.0
        lam_rad = lam * np.pi / 180.0
        rho = (self.R * self.F) / (np.tan(np.pi / 4.0 + phi_rad / 2.0) ** self.n)
        print("rho0", rho)
        theta = self.n * (lam_rad - self.lam_0)
        print("theta", theta)

        x = rho * np.sin(theta)
        y = self.rho0 - rho * np.cos(theta)
        return x, y

    def compute_kh(self, phi, lam):
        # Compute the map factors
        phi_rad = phi * np.pi / 180.0
        lam_rad = lam * np.pi / 180.0
        k = (np.cos(self.phi_1) * np.tan(np.pi / 4.0 + self.phi_1 / 2.0) ** self.n) / (
            np.cos(phi_rad) * np.tan(np.pi / 4.0 + phi_rad / 2.0) ** self.n
        )
        h = k
        return k, h

    def compute_latlon(self, x, y):

        rho = np.sign(self.n) * np.sqrt((x * x + (self.rho0 - y) ** 2.0))

        if self.n >= 0:
            theta = np.arctan(x / (self.rho0 - y))
        else:
            theta = np.arctan(-x / (-self.rho0 + y))

        lam = theta * 180.0 / np.pi / self.n + self.lam_0 * 180.0 / np.pi

        phi = 2.0 * np.arctan((self.R * self.F / rho) ** (1.0 / self.n)) - np.pi / 2.0

        return phi * 180.0 / np.pi, lam
