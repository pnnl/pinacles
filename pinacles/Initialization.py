import numpy as np
from pinacles import CaseStableBubble
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
    "dycoms_rotated",
    "rico",
    "atex",
    "testbed",
]


def factory(namelist):
    assert namelist["meta"]["casename"] in CASENAMES

    if namelist["meta"]["casename"] == "sullivan_and_patton":
        return CaseSullivanAndPatton.initialize
    elif namelist["meta"]["casename"] == "stable_bubble":
        return CaseStableBubble.initialize
    elif namelist["meta"]["casename"] == "bomex":
        return CaseBomex.initialize
    elif (namelist["meta"]["casename"] == "dycoms" or namelist["meta"]["casename"] == "dycoms_rotated"):
        return CaseDycoms.initialize
    elif namelist["meta"]["casename"] == "rico":
        return CaseRico.initialize
    elif namelist["meta"]["casename"] == "atex":
        return CaseATEX.initialize
    elif namelist["meta"]["casename"] == "testbed":
        return CaseTestbed.initialize
    else:
        UtilitiesParallel.print_root(
            "Caanot find initialization for: ", namelist["meta"]["casename"]
        )


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    return
