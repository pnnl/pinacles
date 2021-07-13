import numpy as np
import netCDF4 as nc
from mpi4py import MPI
import os


class Tower:
    def __init__(self, namelist, Grid, TimeSteppingController, loc=(10.0, 10.0)):

        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController
        self._containers = []
        self._out_file = None

        xbnds = self._Grid.x_range_local
        ybnds = self._Grid.y_range_local
        dx = self._Grid.dx

        # Get the tower indicies
        self._i_indx = int((loc[0] - xbnds[0]) // dx[0]) + self._Grid.n_halo[0]
        self._j_indx = int((loc[1] - ybnds[0]) // dx[1]) + self._Grid.n_halo[1]

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
        self._out_file = os.path.join(
            self._out_file, "tower_" + str(loc[0]) + "_" + str(loc[1]) + ".nc"
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
            "time", np.double, dimensions=("time"),
        )

        for con in self._containers:
            for var in con._dofs.keys():
                if con._loc[var] != "z":
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z"))
                else:
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z_edge"))

        rt_grp.close()
        return

    def update(self):

        print("Updating tower")

        # if  not  np.allclose(self._TimeSteppingController._time%self._frequency,0.0):
        #    return

        if not self._tower_on_this_rank:
            return

        nh = self._Grid.n_halo

        rt_grp = nc.Dataset(self._out_file, "r+")

        time = rt_grp["time"]
        time[time.shape[0]] = self._TimeSteppingController._time

        for con in self._containers:
            for var in con._dofs.keys():
                if con._loc[var] != "z":
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = phi[self._i_indx, self._j_indx, nh[2] : -nh[2]]
                else:
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = phi[
                        self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]
                    ]

        rt_grp.close()

        return


class Towers:
    def __init__(self, namelist, Timers, Grid, TimeSteppingController):

        self._list_of_towers = []
        self._Timers = Timers

        if "towers" not in namelist:
            return

        tower_locations = namelist["towers"]["location"]
        self._frequency = namelist["towers"]["frequency"]

        for loc in tower_locations:
            self._list_of_towers.append(
                Tower(namelist, Grid, TimeSteppingController, loc=tuple(loc))
            )

        self._Timers.add_timer("Towers")
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
