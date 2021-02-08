from pinacles import UtilitiesParallel
class Plume:

    def __init__(self, location, start_time, n, 
        Grid, Ref, ScalarState, TimeSteppingController):

        
        self._Grid = Grid
        self._Ref = Ref
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState


        # These are the plume properties
        self._location = location
        self._start_time = start_time
        self._plume_number = n
        self._scalar_name = 'plume_' + str(self._plume_number)

        self._plume_flux = 0.0
        self._plume_qv_flux = 0.0
        self._plume_heat_flux = 0.0
        self._plume_ql_flux = 0.0




        # Determine if plume is emitted on this rank
        self._plume_on_rank = self._Grid.point_on_rank(self._location[0], self._location[1], self._location[2])
        self._indicies = None
        if self._plume_on_rank:
            self._indicies = self._Grid.point_indicies(self._location[0], self._location[1], self._location[2])


        # Add a new scalar variable associated with this plume
        self._ScalarState.add_variable(self._scalar_name)

        return
        
    def update(self):

        dxs = self._Grid.dx

        grid_cell_volume = dxs[0] * dxs[1] * dxs[2]
        grid_cell_mass = grid_cell_volume * self._Ref.rho0[self._indicies[0]] 

        #Add the plume scalar flux
        plume_tend = self._ScalarState.get_tend(self._scalar_name)
        plume_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_flux/grid_cell_mass

        # Add the plume heat flux
        s_tend = self._ScalarState.get_tend('s')
        s_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_heat_flux/grid_cell_mass



        return
    
    @property
    def location(self):
        return self._location

    @property
    def start_time(self):
        return self._start_time

    @property
    def plume_number(self):
        return self._plume_number

    @property
    def scalar_name(self):
        return self._scalar_name

    @property
    def plume_flux(self):
        return self._plume_flux

    @property
    def plume_qv_flux(self):
        return self._plume_qv_flux

    @property
    def plume_ql_flux(self):
        return self._plume_ql_flux

    @property
    def plume_heat_flux(self):
        return self._plume_heat_flux

    def set_plume_flux(self, flux):
        self._plume_flux = flux

    def set_plume_qv_flux(self, flux):
        self._plume_qv_flux = flux

    def set_plume_ql_flux(self, flux):
        self._plume_ql_flux = flux

    def set_plume_heat_flux(self, flux):
        self._plume_heat_flux = flux

class Plumes:

    def __init__(self, namelist, Grid, Ref, ScalarState, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState
        self._locations = None
        self._startimes = None

        self._n = 0

        if 'plumes' in namelist:

            # Store the plume locations
            self._locations  = namelist['plumes']['locations']

            # Store when the plumes start
            self._startimes = namelist['plumes']['starttimes']

            # Store the scalar flux of the plumes
            self._plume_flux = namelist['plumes']['plume_flux']

            # Store the water vapor flux fo the plumes 
            self._plume_qv_flux = namelist['plumes']['qv_flux']

            # Store the sensible heat flux of the plumes
            self._plume_heat_flux = namelist['plumes']['heat_flux']

            # Store the liquid water flux of the plumes
            self._plume_ql_flux = namelist['plumes']['ql_flux']
        else: 
            # If plumes are not in namelist return since there is nothing 
            # to do.
            return

        assert len(self._locations) == len(self._startimes)
        self._n = len(self._startimes)

        # This is a list that will store one instance of the Plume class for each physical plume
        self._list_of_plumes = []

        # Initialize the plumes and update them and print to terminal
        UtilitiesParallel.print_root('Adding, ' + str(self._n) + ' plumes.' )
        
        count = 0
        for loc, start in zip(self._locations, self._startimes):

            UtilitiesParallel.print_root('\t Plume added at: ' + str(loc) + 'starting @' + str(start) + ' seconds.')
            
            # Add plume classes to the list of plumes
            self._list_of_plumes.append(Plume(loc, start, count,
                self._Grid, self._Ref, self._ScalarState, self._TimeSteppingController))
            count += 1 

        # Now set the plume fluxes for each plume
        for i, plume in enumerate(self._list_of_plumes):

            # The plume scalar flux    # Todo set units 
            plume.set_plume_flux(self._plume_flux[i])

            # The plume water vapor flux  # Todo set units
            plume.set_plume_qv_flux(self._plume_qv_flux[i])

            # The plume heat flux   # Todo set units
            plume.set_plume_heat_flux(self._plume_heat_flux[i])

        return

    def update(self):

        if self._n == 0:
            # If there ae no plumes, just return
            return

        # Iterate over the list of plumes and update them
        for plume_i in self._list_of_plumes:
            plume_i.update()

        return