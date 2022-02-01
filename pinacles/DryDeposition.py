import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import Surface
from pinacles import UtilitiesParallel
from pinacles import parameters
from pinacles import DryDeposition_impl

class DryDeposition:
    def __init__(self, namelist, Grid, Ref, ScalarState,  DiagnosticState, Surface):
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Surface = Surface

        try:
            self._apply_deposition = namelist['dry_deposition']['apply_deposition']
        except:
            self._apply_deposition = False
        
        
        
        if not self._apply_deposition:
            return
        
        try:
            self._dry_diameter = namelist['dry_deposition']['particle_diameter']
        except:
            self._dry_diameter = 100.0e-9 # default 100 Nanometers

        self._DiagnosticState.add_variable('v_deposition')
        self._DiagnosticState.add_variable('rh')

        return
    
    def update(self):
        if not self._apply_deposition:
            return
        T = self._DiagnosticState.get_field('T')
        rh = self._DiagnosticState.get_field('rh')
        vdep = self._DiagnosticState.get_field('v_deposition')
        qv = self._ScalarState.get_field('qv')
        nh = self._Grid.n_halo
        z = self._Grid.z_global
        shape = T.shape

        shf2d=np.ones((shape[0],shape[1]),dtype=np.double) * self._Surface._shf
        lhf2d=np.ones_like(shf2d) * self._Surface._lhf
        ustar2d= np.ones_like(shf2d) * self._Surface._ustar

       
        z02d =np.ones_like(shf2d) * self._Surface._z0

        DryDeposition_impl.compute_rh_3d(nh, self._Ref._P0, qv, T, rh)
   
       
        DryDeposition_impl.compute_dry_deposition_velocity(self._dry_diameter, T, rh, nh, z,
                                        self._Ref._rho0, self._Ref.p0, shf2d, lhf2d, ustar2d, z02d, vdep)
        #vdep is positive but directed downward
        for var in self._ScalarState._dofs:
            if   'plume' in  var:
                phi = self._ScalarState.get_field(var)
                phi_t = self._ScalarState.get_tend(var)
                DryDeposition_impl.add_deposition_tendency(vdep, phi,phi_t, nh, self._Grid.dxi[2])
        
        return
        
        