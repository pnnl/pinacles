import numpy as np
import pinacles.ThermodynamicsDry_impl as DryThermo
import pinacles.ThermodynamicsMoist_impl as MoistThermo
from pinacles.CaseReal import InitializeReanalysis
import netCDF4 as nc
from scipy import interpolate
from pinacles import UtilitiesParallel
from mpi4py import MPI
from pinacles import UtilitiesParallel
import pinacles.CaseSullivanAndPatton as CaseSullivanAndPatton
import pinacles.CaseStableBubble as CaseStableBubble
import pinacles.CaseBOMEX as CaseBomex
import pinacles.CaseDYCOMS as CaseDycoms
import pinacles.CaseRICO as CaseRico
import pinacles.CaseATEX as CaseATEX
import pinacles.CaseTestbed as CaseTestbed

CASENAMES = [
    "sullivan_and_patton",
    "stable_bubble",
    "bomex",
    "dycoms",
    "rico",
    "atex",
    "testbed",
    "real",
]


def real(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest, nest):

    init_class = InitializeReanalysis(
        namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest, nest
    )
    init_class.initialize()

    return


def factory(namelist):
    assert namelist["meta"]["casename"] in CASENAMES

    if namelist["meta"]["casename"] == "sullivan_and_patton":
        return CaseSullivanAndPatton.initialize
    elif namelist["meta"]["casename"] == "stable_bubble":
        return CaseStableBubble.initialize
    elif namelist["meta"]["casename"] == "bomex":
        return CaseBomex.initialize
    elif namelist["meta"]["casename"] == "dycoms":
        return CaseDycoms.initialize
    elif namelist["meta"]["casename"] == "rico":
        return CaseRico.initialize
    elif namelist["meta"]["casename"] == "atex":
        return CaseATEX.initialize
    elif namelist["meta"]["casename"] == "testbed":
        return CaseTestbed.initialize
    elif namelist["meta"]["casename"] == "real":
        return real
    else:
        UtilitiesParallel.print_root(
            "Caanot find initialization for: ", namelist["meta"]["casename"]
        )


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest=None, nest=None):
    init_function = factory(namelist)

    try:
        UtilitiesParallel.print_root('\t Initializing without ingest option')
        init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    except:
        UtilitiesParallel.print_root('\t Initializing with ingest option')
        init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest, nest)
    
    return
