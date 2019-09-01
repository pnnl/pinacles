from Columbia import ParallelArrays


class ModelState: 
    def __init__(self, Grid, prognostic=False): 

        self._Grid = Grid 
        self._prognostic = False

        self._state_array = None 
        self._tend_array = None 
        self._dofs = {}
        self._long_names = {}
        self._latex_names = {}
        self._units = {}
        self._nvars = 0
  

        return 

    def add_variable(self, name, long_name=None, latex_name=None, units=None): 

        self._dofs[name] = self._nvars
        self._long_names[name] = long_name
        self._latex_names[name] = latex_name 
        self._units[name] = units 

        self._nvars += 1 

        return 

    def allocate(self): 
        #Allocate tendency array 
        self._state_array = ParallelArrays.GhostArray(self._Grid, ndof=self._nvars)

        #Only allocate tendency array if this is a container for prognostic variables 
        if self._prognostic: 
            self._tend_array = ParallelArrays.GhostArray(self._Grid, ndof=self._nvars)

        return 

    def boundary_exchange(self):
        self._state_array.boundary_exchange()

        return

    