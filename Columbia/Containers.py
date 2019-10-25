from Columbia import ParallelArrays
import numpy as np

class ModelState:
    def __init__(self, Grid, prognostic=False):

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


        return

    @property 
    def nvars(self): 
        return self._nvars

    @property
    def get_state_array(self): 
        return self._state_array

    @property 
    def get_tend_array(self): 
        return self._tend_array 

    def add_variable(self, name, long_name=None, latex_name=None, units=None, bcs='symmetric'):
        #TODO add error handling here. For example what happens if memory has alread been allocated for this container.
        self._dofs[name] = self._nvars
        self._long_names[name] = long_name
        self._latex_names[name] = latex_name
        self._units[name] = units
        self._bcs[name] = bcs

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
        n_halo = self._Grid.n_halo
        if bc == 'symmetric': 
            pass

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

    def mean(self, name): 
        dof = self._dofs[name]    
        return self._state_array.mean(dof)


    @property
    def names(self):
        return self._dofs.keys()

    @property
    def state_array(self):
        return self._state_array.array[:,:,:,:]

    @property
    def tend_array(self):
        return self._tend_array.array[:,:,:,:]
