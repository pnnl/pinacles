import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import UtilitiesParallel
import os
import copy 


class Tower:
    def __init__(self, namelist, Grid, TimeSteppingController, Surface, Rad, Micro, loc=(10.0, 10.0), location_in_latlon=False):

        self._Grid = Grid
        self._Surface = Surface
        self._Rad = Rad
        self._Micro = Micro
        self._TimeSteppingController = TimeSteppingController
        self._containers = []
        self._out_file = None

        xbnds = self._Grid.x_range_local
        ybnds = self._Grid.y_range_local
        dx = self._Grid.dx


        if location_in_latlon:
            self.latlon_loc = copy.deepcopy(loc)
        x, y = self._Grid.latlon_to_xy(loc[0], loc[1])        
        loc = (x, y)


        # Get the tower indicies
        self._i_indx = int((loc[0] - xbnds[0]) // dx[0]) + self._Grid.n_halo[0] - 1
        self._j_indx = int((loc[1] - ybnds[0]) // dx[1]) + self._Grid.n_halo[1] - 1 

        try:
            self._frequency = namelist["towers"]["frequency"]
        except:
            self._frequency = 1e9

        self._tower_on_this_rank = False
        if (
            loc[0] > xbnds[0]
            and loc[0] <= xbnds[1]
            and loc[1] > ybnds[0]
            and loc[1] <= ybnds[1]
        ):
            self._tower_on_this_rank = True
        else:
            self._tower_on_this_rank = False
            return

        self._out_file = os.path.join(
            namelist["meta"]["output_directory"], namelist["meta"]["simname"]
        )
        if not location_in_latlon:
            self._out_file = os.path.join(
                self._out_file, "tower_" + str(loc[0]) + "_" + str(loc[1]) + ".nc"
            )
        else:
            self._out_file = os.path.join(
                self._out_file, "tower_" + str(self.latlon_loc[0]) + "_" + str(self.latlon_loc[1]) + ".nc"
            )
            

        return

    def add_state_container(self, container):
        if self._tower_on_this_rank:
            self._containers.append(container)
        return

    def initialize(self):

        if not self._tower_on_this_rank:
            return

        rt_grp = nc.Dataset(self._out_file, "w")

        nh = self._Grid.n_halo

        rt_grp.createDimension("z", self._Grid.n[2])
        vh = rt_grp.createVariable("z", np.double, dimensions=("z",))
        vh[:] = self._Grid.z_global[nh[2] : -nh[2]]

        rt_grp.createDimension("z_edge", self._Grid.n[2] + 1)
        vh = rt_grp.createVariable("z_edge", np.double, dimensions=("z_edge",))
        vh[:] = self._Grid.z_edge_global[nh[2] - 1 : -nh[2]]

        rt_grp.createDimension("time")
        rt_grp.createVariable(
            "time",
            np.double,
            dimensions=("time"),
        )

        for con in self._containers:
            for var in con._dofs.keys():
                if con._loc[var] != "z":
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z"))
                else:
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z_edge"))


        if hasattr(self._Surface, 'io_tower_init'):
            self._Surface.io_tower_init(rt_grp)

        if hasattr(self._Rad, 'io_tower_init'):
            self._Rad.io_tower_init(rt_grp)


        if hasattr(self._Rad, 'io_tower_init'):
            self._Micro.io_tower_init(rt_grp)

        rt_grp.close()
        return

    def update(self):

        if not self._tower_on_this_rank:
            return

        nh = self._Grid.n_halo

        rt_grp = nc.Dataset(self._out_file, "r+")

        time = rt_grp["time"]
        time[time.shape[0]] = self._TimeSteppingController._time

        for con in self._containers:
            for var in con._dofs.keys():
                if var == "u":
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx - 1, self._j_indx, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )

                elif var == "v":
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )

                else:
                    if con._loc[var] != "z":
                        phi = con.get_field(var)
                        rt_grp[var][-1, :] = phi[
                            self._i_indx, self._j_indx, nh[2] : -nh[2]
                        ]
                    else:
                        phi = con.get_field(var)
                        rt_grp[var][-1, :] = phi[
                            self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]
                        ]
        if hasattr(self._Surface, 'io_tower'):
            self._Surface.io_tower(rt_grp, self._i_indx, self._j_indx)
        if hasattr(self._Rad, 'io_tower'):
            self._Rad.io_tower(rt_grp, self._i_indx, self._j_indx)
        if hasattr(self._Micro, 'io_tower'):
            self._Micro.io_tower(rt_grp, self._i_indx, self._j_indx)


        rt_grp.close()

        return


class Towers:
    def __init__(self, namelist, Timers, Grid, TimeSteppingController, Surface, Rad, Micro):

        self._list_of_towers = []
        self._Timers = Timers
        self._Timers.add_timer("Towers")

        try:
            self._frequency = namelist["towers"]["frequency"]
        except:
            self._frequency = 1e9

        if "towers" not in namelist:
            return

        tower_locations = namelist["towers"]["location"]

        self.location_in_latlon = False
        if 'location_in_latlon' in namelist["towers"]:
            self.location_in_latlon = namelist["towers"]['location_in_latlon']
            UtilitiesParallel.print_root('\t \t Tower locations taken to be in lat-lon.')
        assert(type(self.location_in_latlon is type(bool)))
                                                
        for loc in tower_locations:
            self._list_of_towers.append(
                Tower(namelist, Grid, TimeSteppingController, Surface, Rad, Micro, loc=tuple(loc), location_in_latlon = self.location_in_latlon)
            )

        return

    def add_state_container(self, state_container):

        for tower in self._list_of_towers:
            tower.add_state_container(state_container)

        return

    def initialize(self):

        for tower in self._list_of_towers:
            tower.initialize()

        return

    def update(self):
        self._Timers.start_timer("Towers")

        for tower in self._list_of_towers:
            tower.update()

        self._Timers.end_timer("Towers")
        return

    @property
    def frequency(self):
        return self._frequency
