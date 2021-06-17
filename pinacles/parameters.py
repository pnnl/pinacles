import numpy as np


# Store model parameter here
G = np.single(9.80665)  #  Gravitational Acceleration
P00 = np.single(1.0e5)  #  Set the reference pressure (used in Exner function)
CPD = np.single(1004.0)  #  Specific heat of dry air at constant pressure
ICPD = np.single(1.0 / CPD)  # Specific heat of dry air at constant pressure
CL = np.single(4218.0)  # Specific heat of liquid water
CI = np.single(2106.0)  # Specific ehat of ice
RD = np.single(287.1)  #  Dry air gas constant
RV = np.single(461.5)  #  Water vapor gas constant
KAPPA = np.single(RD / CPD)
EPSV = np.single(RD / RV)
EPSVI = np.single(1.0 / EPSV)
OMEGA = np.single(7.2921151467064e-5)  # Earth's Rotation Rate
LV = np.single(2.5014e6)
LS = np.single(2.8440e6)
LF = np.single(0.336e6)
LARGE = np.single(1e11)  # A very large number
