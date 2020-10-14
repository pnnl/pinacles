from pinacles import ParallelArrays
import numpy as np
from mpi4py import MPI

class ModelState:
    def __init__(self, Grid, container_name, prognostic=False, identical_bcs=False):

        self._Grid = Grid          #The grid to use for this ModelState container
        self._prognostic = prognostic  #Is prognostic, if True we will allocate a tendency array

        self._state_array = None   #This will store present values of the model state
        self._tend_array = None    #If prognostic this will store the values of the tend array
        self._dofs = {}            #This maps variable name to the GhostArray dof where it stored
        self._long_names = {}      #Store long names for the variables
        self._latex_names = {}     #Store latex names, this is handy for plotting
        self._units = {}           #Store the units, this is also hand for plotting
        self._nvars = 0            #The number of 3D field stored in this model state
        self._bcs = {}
        self._loc = {}
        self._identical_bcs = identical_bcs


        self.name = container_name

        return


    def get_long_name(self, name):
        return self._long_names[name]

    def get_standard_name(self, name):
        return self._latex_names[name]

    def get_units(self, name):
        return self._units[name]

    @property
    def nvars(self):
        return self._nvars

    @property
    def get_state_array(self): 
        #TODO this is a property so we should remove the get in the function name
        return self._state_array

    @property
    def get_tend_array(self):
        #TODO this is a property so we should remove the get in the function name
        return self._tend_array

    @property
    def identical_bcs(self):
        return self._identical_bcs

    def add_variable(self, name, long_name='None', latex_name='None', units='None', bcs='gradient zero', loc='c'):

        #Do some correctness checks and warn for some behavior
        assert(bcs in  ['gradient zero', 'value zero'])


        #TODO add error handling here. For example what happens if memory has alread been allocated for this container.
        self._dofs[name] = self._nvars
        self._long_names[name] = long_name
        self._latex_names[name] = latex_name
        self._units[name] = units
        self._bcs[name] = bcs
        self._loc[name] = loc

        #Increment the bumber of variables
        self._nvars += 1

        return

    def allocate(self):
        #Todo add error handling here, for example check to see if memory is already allocated.

        #Allocate tendency array
        self._state_array = ParallelArrays.GhostArray(self._Grid, ndof=self._nvars)
        self._state_array.zero()

        #Only allocate tendency array if this is a container for prognostic variables
        if self._prognostic:
            self._tend_array = ParallelArrays.GhostArray(self._Grid, ndof=self._nvars)
            self._tend_array.zero()
        return

    def boundary_exchange(self):
        #Call boundary exchange on the _state_array (Ghost Array)
        self._state_array.boundary_exchange()
        return

    def update_bcs(self, name):

        bc = self._bcs[name]
        if bc == 'gradient zero':
            self._gradient_zero_bc(name)
        elif bc == 'value zero':
            self._zero_value_bc(name)
        else:
            assert(bc in  ['gradient zero', 'value zero'])

        return

    def update_all_bcs(self):
        #TODO add other BC types. For now only assume everything is cell center and gradient zero
        #TODO PERFORMANCE. May want to use numba here.

        if self._identical_bcs:
            nh2 = self._Grid.n_halo[2]
            #First set the bottom boundary
            self._state_array.array[:,:,:,:nh2] = self._state_array.array[:,:,:,nh2:2*nh2][:,:,:,::-1]

            #Second set the top boundary
            self._state_array.array[:,:,:,-nh2:] = self._state_array.array[:,:,:,-2*nh2:-nh2][:,:,:,::-1]

        else:
            for name in self._dofs.keys():
                if self._bcs[name] == 'gradient zero':
                    self._gradient_zero_bc(name)
                else:
                    self._zero_value_bc(name)
        return

    def _gradient_zero_bc(self, name):
        #Todo add other BCS.
        #TODO PERFORMANCE. May want to use number here.
        nh2 = self._Grid.n_halo[2]
        field = self.get_field(name)

        #First set the bottom boundary
        field[:,:,:nh2] = field[:,:,nh2:2*nh2][:,:,::-1]

        #Second set the top boundary
        field[:,:,-nh2:]= field[:,:,-2*nh2:-nh2][:,:,::-1]

        return

    def _zero_value_bc(self, name):
        #Todo add other BCS
        #TODO PERFORMANCE. May want to use number here.
        nh2 = self._Grid.n_halo[2]
        field = self.get_field(name)

        #First set the bottom boundary
        field[:,:,:nh2-1] = 0.0 #-field[:,:,nh2:2*nh2-1][:,:,::-1]
        field[:,:,nh2-1] = 0.0

        #Second set the top boundary
        #field[:,:,-(nh2-1):] = -field[:,:,(-2*nh2+1):-nh2][:,:,::-1] #-field[:,:,(-2*nh2+1):(-2*nh2+1)+(nh2-1)][:,:,::-1]

        field[:,:,-nh2:]= 0.0 #-field[:,:,-2*nh2-1:-nh2-1][:,:,::-1]
        field[:,:,-nh2-1] = 0.0

        #print(field[3,3,:])


        return

    def get_field(self, name):
        #Return a contiguious memory slice of _state_array containing the values of name
        dof = self._dofs[name]
        return self._state_array.array[dof,:,:,:]

    def get_tend(self,name):
        #Return a contiguous memory slice of _tend_array containing the tendencies of name
        #TODO add error handling for this case.
        dof = self._dofs[name]
        return self._tend_array.array[dof,:,:,:]

    def remove_mean(self, name):
        #This removes the mean from a field
        dof = self._dofs[name]
        self._state_array.remove_mean(dof)
        return

    def mean(self, name, pow=1.0):
        dof = self._dofs[name]
        return self._state_array.mean(dof, pow=pow)

    def max_prof(self, name):
        dof = self._dofs[name]
        return self._state_array.max_prof(dof)

    def max(self, name):
        dof = self._dofs[name]
        return self._state_array.max(dof)

    def min_prof(self, name):
        dof = self._dofs[name]
        return self._state_array.min_prof(dof)

    def min(self, name):
        dof = self._dofs[name]
        return self._state_array.min(dof)

    def get_field_slice_z(self, name, indx=0):

        ls = self._Grid.local_start
        nl = self._Grid.nl
        nh = self._Grid.n_halo
        n = self._Grid.n

        local_data = self.get_field(name)[nh[0]:-nh[0],nh[1]:-nh[1],indx]
        local_copy_of_global = np.zeros((n[0], n[1]), dtype=np.double)

        local_copy_of_global[
            ls[0]:ls[0]+nl[0],
            ls[1]:ls[1]+nl[1]] = local_data

        recv_buf = np.empty_like(local_copy_of_global)

        MPI.COMM_WORLD.Allreduce(local_copy_of_global, recv_buf, op=MPI.SUM)

        return recv_buf


    def get_loc(self, var):
        return self._loc[var]

    @property
    def names(self):
        return self._dofs.keys()

    @property
    def state_array(self):
        return self._state_array.array[:,:,:,:]

    @property
    def tend_array(self):
        return self._tend_array.array[:,:,:,:]

    @property
    def stats_io_init(self):

        return

    def io_initialize(self, nc_grp):

        timeseries_grp = nc_grp['timeseries']
        profiles_grp = nc_grp['profiles']

        #Now loop over variables creating profiles for each
        for var in self._dofs:
            if  not 'ff' in  var:  #Avoid SBM Bins
                if self._loc[var] != 'z':
                    profiles_grp.createVariable(var, np.double, dimensions=('time', 'z',))
                    profiles_grp.createVariable(var + '_squared', np.double, dimensions=('time', 'z',))
                    profiles_grp.createVariable(var + '_min', np.double, dimensions=('time', 'z',))
                    profiles_grp.createVariable(var + '_max', np.double, dimensions=('time', 'z',))
                else:
                    profiles_grp.createVariable(var, np.double, dimensions=('time', 'z_edge',))
                    profiles_grp.createVariable(var + '_squared', np.double, dimensions=('time', 'z_edge',))
                    profiles_grp.createVariable(var + '_min', np.double, dimensions=('time', 'z_edge'))
                    profiles_grp.createVariable(var + '_max', np.double, dimensions=('time', 'z_edge'))

        #Now loop over variables createing domain max/min timeseries for each
        for var in self._dofs:
            if  not 'ff' in  var: #Avoid SBM Bins
                timeseries_grp.createVariable(var + '_max', np.double, dimensions=('time',))
                timeseries_grp.createVariable(var + '_min', np.double, dimensions=('time',))

        return


    def io_update(self, nc_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()
        nh = self._Grid.n_halo

        #Loop over variables and write  profiles
        for var in self._dofs:
            if  not 'ff' in  var:  #Avoid SBM Bins
                if self._loc[var] != 'z':
                    var_mean = self.mean(var)[nh[2]:-nh[2]]
                    var_mean_squared = self.mean(var, pow=2.0)[nh[2]:-nh[2]]
                    var_max = self.max_prof(var)[nh[2]:-nh[2]]
                    var_min = self.min_prof(var)[nh[2]:-nh[2]]
                else:
                    var_mean = self.mean(var)[nh[2]-1:-nh[2]]
                    var_mean_squared = self.mean(var, pow=2.0)[nh[2]-1:-nh[2]]
                    var_max = self.max_prof(var)[nh[2]-1:-nh[2]]
                    var_min = self.min_prof(var)[nh[2]-1:-nh[2]]

                #Only write from rank zero
                if my_rank == 0:
                    profiles_grp = nc_grp['profiles']

                    profiles_grp[var][-1,:] = var_mean
                    profiles_grp[var + '_squared'][-1,:] = var_mean_squared
                    profiles_grp[var + '_max'][-1,:] = var_max
                    profiles_grp[var + '_min'][-1,:] = var_min


        #Loop over variables and time series
        for var in self._dofs:
            if  not 'ff' in  var:  #Avoid SBM Bins
                var_max = self.max(var)
                var_min = self.min(var)

                #Only write from rank zero
                if my_rank == 0:
                    timeseries_grp = nc_grp['timeseries']
                    timeseries_grp[var + '_max'][-1] = var_max
                    timeseries_grp[var + '_min'][-1] = var_min

        return