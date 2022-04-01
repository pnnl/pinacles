import numpy as np
import os
from mpi4py import MPI


import time
from pinacles import UtilitiesParallel


def DumpFieldsFactory(namelist, Timers, Grid, TimeSteppingController):
    assert "fields" in namelist
    if "io_type" not in namelist["fields"]:
        from pinacles.DumpFieldsNetCDF import DumpFields
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "NETCDF":
        from pinacles.DumpFieldsNetCDF import DumpFields
        return DumpFields(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5":
        from pinacles.DumpFieldsHDF5 import DumpFields_hdf
        return DumpFields_hdf(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5_GATHER":
        from pinacles.DumpFieldsGatherHDF5 import DumpFields_hdf
        return DumpFields_hdf(namelist, Timers, Grid, TimeSteppingController)
    elif namelist["fields"]["io_type"].upper() == "HDF5_GATHER_PERPROC":
        from pinacles.DumpFieldsGatherHDF5PerProc import DumpFields_hdf_perproc
        return DumpFields_hdf_perproc(namelist, Timers, Grid, TimeSteppingController)


    return
