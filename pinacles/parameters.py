import numpy as np

# Store model parameter here
G = np.double(9.80665)  #  Gravitational Acceleration
P00 = np.double(1.0e5)  #  Set the reference pressure (used in Exner function)
CPD = np.double(1004.0)  #  Specific heat of dry air at constant pressure
ICPD = np.double(1.0 / CPD)  # Specific heat of dry air at constant pressure
CL = np.double(4218.0)  # Specific heat of liquid water
CI = np.double(2106.0)  # Specific ehat of ice
RD = np.double(287.1)  #  Dry air gas constant
RV = np.double(461.5)  #  Water vapor gas constant
KAPPA = np.double(RD / CPD)
EPSV = np.double(RD / RV)
EPSVI = np.double(1.0 / EPSV)
OMEGA = np.double(7.2921151467064e-5)  # Earth's Rotation Rate
LV = np.double(2.5014e6)
LS = np.double(2.8440e6)
LF = np.double(0.336e6)
LARGE = 1e11  # A very large number
