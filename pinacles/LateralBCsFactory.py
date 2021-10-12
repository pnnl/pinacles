from pinacles import UtilitiesParallel
from pinacles.LateralBCs import LateralBCsDummy
from pinacles.LateralBCsMean import LateralBCsMean
from pinacles.LateralBCsRecycle import LateralBCsRecycle
from pinacles.LateralBCsNest import LateralBCsNest
from pinacles.CaseReal import LateralBCsReanalysis

def LateralBCsFactory(namelist, Grid, State, VelocityState, TimeSteppingController, Ingest, **kwargs):
    try:
        lbc = namelist["lbc"]
    except:
        return LateralBCsDummy()

    if lbc["type"].lower() == "periodic":
        return LateralBCsDummy()
    elif lbc["type"].lower() == "open":
        try:
            boundary_treatment = lbc["open_boundary_treatment"]
            if boundary_treatment.lower() == "mean":
                lbc_class = LateralBCsMean(Grid, State, VelocityState)
                UtilitiesParallel.print_root("Using mean boundary treatment.")
            if boundary_treatment.lower() == "recycle":
                lbc_class = LateralBCsRecycle(namelist, Grid, State, VelocityState)
                UtilitiesParallel.print_root("Using recycle boundary conditions.")
            if boundary_treatment.lower() == "nest":
                lbc_class = LateralBCsNest(namelist, Grid, State, VelocityState, NestState=kwargs['NestState'])
                UtilitiesParallel.print_root("Using nested boundary conditions")
            if boundary_treatment.lower() == 'reanalysis':
                lbc_class = LateralBCsReanalysis(namelist, Grid, State, VelocityState, TimeSteppingController, Ingest)
                UtilitiesParallel.print_root("Using reanalysis boundary conditions")
        except:
            UtilitiesParallel.print_root("Usinge mean boundary treatment.")

        return lbc_class