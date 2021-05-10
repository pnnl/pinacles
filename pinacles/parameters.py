#Store model parameter here
G = 9.80665   #  Gravitational Acceleration
P00 = 1.0e5   #  Set the reference pressure (used in Exner function)
CPD = 1004.0  #  Specific heat of dry air at constant pressure
ICPD = 1.0/CPD # Specific heat of dry air at constant pressure
CL =  4218.0  # Specific heat of liquid water
CI = 2106.0   # Specific ehat of ice
RD = 287.1    #  Dry air gas constant
RV = 461.5    #  Water vapor gas constant
KAPPA = RD/CPD
EPSV = RD/RV
EPSVI = 1.0/EPSV
OMEGA = 7.2921151467064e-5  #Earth's Rotation Rate
LV = 2.5014e6
LS = 2.8440e6
LF = 0.336e6 
LARGE = 1e11  # A very large number
VONKARMAN = 0.4