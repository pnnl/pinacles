import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import Surface
from pinacles import UtilitiesParallel
from pinacles import parameters
from pinacles import DryDeposition_impl

class DryDeposition:
    def __init__(self, namelist, Grid, Ref, ScalarState,  DiagnosticState, Surface, TimeSteppingController):
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._Surface = Surface
        self._TimeSteppingController = TimeSteppingController

        self.name ='DryDeposition'
        try:
            self._apply_deposition = namelist['dry_deposition']['apply_deposition']
        except:
            self._apply_deposition = False
        
        # Testing
        self._apply_deposition = True
        
        if not self._apply_deposition:
            return
        
        try:
            self._dry_diameter = namelist['dry_deposition']['particle_diameter']
        except:
            self._dry_diameter = 100.0e-9 # default 100 Nanometers

        self._DiagnosticState.add_variable('v_deposition')
        self._DiagnosticState.add_variable('rh')


        # Store these fields for IO
        ng_local = self._Grid.ngrid_local
        self._surface_flux = np.zeros((ng_local[0], ng_local[1]), dtype=np.double)
        self._surface_accum = np.zeros_like(self._surface_flux)

        # Class data that will be dumped for restart
        self._restart_attributes = ['_surface_flux', '_surface_accum']

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

        DryDeposition_impl.compute_rh(nh, self._Ref._P0, qv, T, rh)

        DryDeposition_impl.compute_dry_deposition_velocity(self._dry_diameter, T, rh, nh, z,
                                        self._Ref._rho0, self._Ref._P0, shf2d, lhf2d, ustar2d, z02d, vdep)
        #vdep is positive but directed downward
        for var in self._ScalarState._dofs:
            if   'plume' in  var:
                phi = self._ScalarState.get_field(var)
                phi_t = self._ScalarState.get_tend(var)
                
                
                DryDeposition_impl.compute_dry_deposition_sedimentation(nh, vdep, self._Grid.dxi, 
                    self._Ref.rho0, phi, phi_t, self._surface_flux)                

                self._surface_flux  *= self._TimeSteppingController.dt
                self._surface_accum += self._surface_flux

    def io_fields2d_update(self, nc_grp):
        
        nh = self._Grid.n_halo
        sed =  nc_grp.createVariable('dry_sed_flux', np.double, dimensions=('X', 'Y',))       
        sed[:,:] = self._surface_flux[nh[0]:-nh[0], nh[1]:-nh[1]]

        nh = self._Grid.n_halo
        sed_accum =  nc_grp.createVariable('dry_sed_accum', np.double, dimensions=('X', 'Y',))       
        sed_accum[:,:] = self._surface_accum[nh[0]:-nh[0], nh[1]:-nh[1]]

        nc_grp.sync()
        return

    def restart(self, data_dict):

        key = self.name
        
        for att in self._restart_attributes:
            self.__dict__[att] = data_dict[key][att]

        return

    def dump_restart(self, data_dict):

        # Get the name of this particualr container and create a dictionary for it in the 
        # restart data dict. 
    
        key = self.name
        data_dict[key] = {}

        # Loop over the restart_attributes and add it to the data_dict
        for att in self._restart_attributes:
            data_dict[key][att] = self.__dict__[att]

        return
                