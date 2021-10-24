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

        # If center latitude is in southern hemisphere
        if self.phi_0 < 0.0:
            self.n = - self.n

        return

    def _compute_F(self):
        self.F = (
            np.cos(self.phi_1)
            * (np.tan(np.pi / 4.0 + self.phi_1 / 2.0) ** self.n)
            / self.n
        )
        return

    def _compute_rho0(self):
        self.rho0 = self.R * self.F / (np.tan(np.pi / 4.0 + self.phi_0 / 2.0) ** self.n)
        
        # If center latitude is in southern hemisphere
        if self.phi_0 < 0.0:
            self.rho0 = - self.rho0
        return

    def compute_xy(self, phi, lam):
        phi_rad = phi * np.pi / 180.0
        lam_rad = lam * np.pi / 180.0
        rho = (self.R * self.F) / (np.tan(np.pi / 4.0 + phi_rad / 2.0) ** self.n)
        theta = self.n * (lam_rad - self.lam_0)

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

        lam = theta / self.n + self.lam_0 

        phi = 2.0 * np.arctan((self.R * self.F / rho) ** (1.0 / self.n)) - np.pi / 2.0

        return lam * 180.0 / np.pi, phi * 180.0 / np.pi

    def compute_alpha(self, lon):

        diff = self.lam_0 - lon
        diff[diff > 180] = diff[diff > 180.0] - 360.0
        diff[diff < -180] = diff[diff < -180] + 360.0

        if self.phi_0 < 0.0:
            alpha = diff * self.n * np.pi/180.0 * -1
        else:
            alpha = diff * self.n * np.pi/180.0

        return alpha

    def rotate_wind(self, lon, u, v):

        shape = u.shape
        alpha = self.compute_alpha(lon)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        if len(shape) == 2:
            urot = v * sin_alpha + u * cos_alpha
            vrot = v * cos_alpha - u * sin_alpha
        elif len(shape) == 3:
            try:
                urot = v * sin_alpha[:,:,np.newaxis] + u * cos_alpha[:,:,np.newaxis]
                vrot = v * cos_alpha[:,:,np.newaxis] - u * sin_alpha[:,:,np.newaxis]
            except:
                urot = v * sin_alpha[np.newaxis,:,:] + u * cos_alpha[np.newaxis,:,:]
                vrot = v * cos_alpha[np.newaxis,:,:] - u * sin_alpha[np.newaxis,:,:]          

        return urot, vrot
