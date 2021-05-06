from pinacles import UtilitiesParallel
from pinacles import parameters
import numpy as np
from mpi4py import MPI
import numba
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
        self._boundary_outflow = True

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
        self._ScalarState.add_variable(self._scalar_name, limit=True)

        return
        
    def update(self):

        if self._boundary_outflow and self._TimeSteppingController.time >=  self._start_time:
            plume_value = self._ScalarState.get_field(self._scalar_name)

            n_halo = self._Grid.n_halo

            x_local = self._Grid.x_local 
            x_global = self._Grid.x_global

            if np.amin(x_local) == np.amin(x_global):
                plume_value[:n_halo[0],:,:] = 0
            
            if np.max(x_local) == np.amax(x_global):
                plume_value[-n_halo[0]:,:,:] = 0

            y_local = self._Grid.y_local 
            y_global = self._Grid.y_global

            if np.amin(y_local) == np.amin(y_global):
                plume_value[:,:n_halo[1],:] = 0
            
            if np.max(y_local) == np.amax(y_global):
                plume_value[:,-n_halo[1]:,:] = 0


        # If it is not time to start the plume just return w/o doing anything
        if  self._TimeSteppingController.time  < self._start_time or not self._plume_on_rank :

            return

        dxs = self._Grid.dx

        grid_cell_volume = dxs[0] * dxs[1] * dxs[2]
        grid_cell_mass = grid_cell_volume * self._Ref.rho0[self._indicies[0]] 

        # Add the plume scalar flux
        plume_tend = self._ScalarState.get_tend(self._scalar_name)
        plume_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_flux/grid_cell_mass

        # Add the plume heat flux
        s_tend = self._ScalarState.get_tend('s')
        s_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_heat_flux/grid_cell_mass*parameters.ICPD

        # Add the plume liquid flux
        if 'qv' in self._ScalarState._dofs:
            qv_tend = self._ScalarState.get_tend('qv')
            qv_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_qv_flux/grid_cell_mass

        # Add the plume liquid flux
        if 'ql' in self._ScalarState._dofs:
            ql_tend = self._ScalarState.get_tend('ql')
            ql_tend[self._indicies[0], self._indicies[1], self._indicies[2]] = self._plume_ql_flux/grid_cell_mass


        return
    

    @property
    def boundary_outflow(self):
        return self._boundary_outflow

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
        return 

    def set_plume_qv_flux(self, flux):
        self._plume_qv_flux = flux
        return

    def set_plume_ql_flux(self, flux):
        self._plume_ql_flux = flux
        return

    def set_plume_heat_flux(self, flux):
        self._plume_heat_flux = flux
        return

    def set_boundary_outflow(self, boundary_outflow):
        self._boundary_outflow = boundary_outflow
        return

class Plumes:

    def __init__(self, namelist, Grid, Ref, ScalarState, TimeSteppingController):

        
        self.name = 'Plumes' # Class name used to make this classes output

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

            # Store the plume number flux
            self._plume_num_flux = namelist['plumes']['num_flux']

            # Store the water vapor flux fo the plumes 
            self._plume_qv_flux = namelist['plumes']['qv_flux']

            # Store the sensible heat flux of the plumes
            self._plume_heat_flux = namelist['plumes']['heat_flux']

            # Store the liquid water flux of the plumes
            self._plume_ql_flux = namelist['plumes']['ql_flux']

            # Store the boundary treatment
            self._boundary_outflow = namelist['plumes']['boundary_outflow']


        else: 
            # If plumes are not in namelist return since there is nothing 
            # to do.
            return


        # The critial number concentration used only for output
        # TODO make this namelist settable
        self._conc_crit = 100.0  # cm^-3

        assert len(self._locations) == len(self._startimes)
        self._n = len(self._startimes)

        # This is a list that will store one instance of the Plume class for each physical plume
        self._list_of_plumes = []

        # Initialize the plumes and update them and print to terminal
        UtilitiesParallel.print_root('Adding, ' + str(self._n) + ' plumes.' )
        
        count = 0
        for loc, start in zip(self._locations, self._startimes):

            UtilitiesParallel.print_root('\t Plume added at: ' + str(loc) + ' starting @ ' + str(start) + ' seconds.')
            
            # Add plume classes to the list of plumes
            self._list_of_plumes.append(Plume(loc, start, count,
                self._Grid, self._Ref, self._ScalarState, self._TimeSteppingController))
            count += 1 

        # Now set the plume fluxes for each plume and how the boundaries are treated
        for i, plume in enumerate(self._list_of_plumes):

            # The plume scalar flux    # Todo set units 
            plume.set_plume_flux(self._plume_flux[i])

            # The plume water vapor flux  # Todo set units
            plume.set_plume_qv_flux(self._plume_qv_flux[i])

            # The plume heat flux   # Todo set units
            plume.set_plume_heat_flux(self._plume_heat_flux[i])

            # set the treatment of the plume scalars on the boundary
            plume.set_boundary_outflow(self._boundary_outflow[i])

        return

    def update(self):

        if self._n == 0:
            # If there ae no plumes, just return
            return

        # Iterate over the list of plumes and update them
        for plume_i in self._list_of_plumes:
            plume_i.update()

        return

    def io_initialize(self, nc_grp):

        if self._n == 0:
            # No plumes so we can jump out here

            return

        timeseries_grp = nc_grp['timeseries']

        for pi, plume in enumerate(self._list_of_plumes):
            v = timeseries_grp.createVariable('Source2CBase_' + 'plume_' + str(pi), np.double, dimensions=('time',))
            v.long_name = 'Distance from emission point to first plume point at cloud base'
            v.standard_name = 'dist_cloud_base_' + 'plume_' + str(pi)
            v.units = 'm'

            v = timeseries_grp.createVariable('CriticalConcentrationHeight_' + 'plume_' + str(pi), np.double, dimensions=('time',))
            v.long_name = 'Maximum height excedeeing crtical concentration.'
            v.standard_name = 'crit_con_base_' + 'plume_' + str(pi)
            v.units = 'm'

        return 

    @staticmethod
    @numba.njit
    def dist_2_base(plume_x, plume_y, x_local, y_local, z_local, n_halo, rho0, conc_crit, emitted_number, qc, plume_var):

        shape = plume_var.shape
        min_dist = 99999999.9
        crit_height = 0.0
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):
                    plume_cc = plume_var[i,j,k] * rho0[k] / (100.0 * 100.0 * 100.0) * emitted_number
                    if qc[i,j,k] > 1e-5 and plume_cc > conc_crit:
                        dist = np.sqrt((x_local[i]-plume_x)**2.0 + (y_local[j] - plume_y)**2.0)
                        min_dist = min(dist, min_dist)

                    if plume_cc > conc_crit:
                        crit_height = max(crit_height, z_local[k])

        return (min_dist, crit_height) 

    def io_update(self, nc_grp):

        if self._n == 0:
            # No plumes so we can jump out here 

            return 

        my_rank = MPI.COMM_WORLD.Get_rank()
        n_halo = self._Grid.n_halo
        y_local = self._Grid.y_local
        x_local = self._Grid.x_local
        z_local = self._Grid.z_local
        rho0 = self._Ref.rho0

        # Loop over the number of plumes
        qc = self._ScalarState.get_field('qc')
        for pi, plume in enumerate(self._list_of_plumes):
            
            # Get the plume variable
            plume_var = self._ScalarState.get_field('plume_' + str(pi))
        


            dist, crit_con_height = self.dist_2_base(plume._location[0], plume._location[1], x_local, y_local ,z_local, 
                n_halo, rho0, self._conc_crit , self._plume_num_flux[pi], qc, plume_var)

            dist = UtilitiesParallel.ScalarAllReduce(dist, op=MPI.MIN)
            crit_con_height = UtilitiesParallel.ScalarAllReduce(crit_con_height, op=MPI.MIN)
            
            if my_rank == 0:
                timeseries_grp = nc_grp['timeseries']
                timeseries_grp['Source2CBase_' + 'plume_' + str(pi)][-1] = dist
                timeseries_grp['CriticalConcentrationHeight_' + 'plume_' + str(pi)][-1] = crit_con_height
        
        return

    @property
    def n(self):
        return self._n
