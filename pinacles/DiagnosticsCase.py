import numpy as np
import mpi4py as MPI
import numba
import pinacles.CaseSullivanAndPatton as CSP
import pinacles.CaseBOMEX as CB
class DiagnosticsCase:
    
    def __init__(
        self, namelist, Grid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState
    ):
        
        self._name = "DiagnosticsCase"
        self._namelist = namelist
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._Micro = Micro
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        
        self.case_diag_factory()
        
        return 
    
    
    def case_diag_factory(self):
        
        self.diag_class = None
        
        casename = self._namelist["meta"]["casename"]
        if casename == "sullivan_and_patton":
            self.diag_class = CSP.SullivanAndPattonDiagnostics(self._Grid, self._Ref, self._Thermo, self._Micro, self._VelocityState, self._ScalarState, self._DiagnosticState)
        if casename == "bomex":
            self.diag_class = CB.BomexDiagnostics(self._Grid, self._Ref, self._Thermo, self._Micro, self._VelocityState, self._ScalarState, self._DiagnosticState)
        
        return
    
    
    def io_initialize(self, this_grp):
        
        if self.diag_class is None:
            return

        self.diag_class.io_initialize(this_grp)
        return
    
    def io_update(self, this_grp):
    
        if self.diag_class is None:
            return
        
        self.diag_class.io_update(this_grp)
        
        
        return
    
    @property
    def name(self):
        return self._name
    