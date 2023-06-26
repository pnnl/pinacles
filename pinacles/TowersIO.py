import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from pinacles import UtilitiesParallel
import os
import copy 
import json


class Tower:
    def __init__(self, namelist, Grid, TimeSteppingController, Surface, Rad, Micro, loc=(10.0, 10.0), location_in_latlon=False):

        self._Grid = Grid
        self._Surface = Surface
        self._Rad = Rad
        self._Micro = Micro
        self._TimeSteppingController = TimeSteppingController
        self._containers = []
        self._accu = {}
        self._accuTime = 0.0
        self._out_file = None

        xbnds = self._Grid.x_range_local
        ybnds = self._Grid.y_range_local
        dx = self._Grid.dx


        if location_in_latlon:
            self.latlon_loc = copy.deepcopy(loc)

            x, y = self._Grid.latlon_to_xy(loc[1], loc[0])        
            loc = (x - self._Grid.MapProj.center_e, y - self._Grid.MapProj.center_n)

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

        rt_grp.createVariable(
            "accu_time",
            np.double,
            dimensions=("time"),
        )

        self._accuTime = 0.0

        for con in self._containers:
            for var in con._dofs.keys():
                if con._loc[var] != "z":
                    # create both snapshot and mean variables for the output file
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z"))
                    rt_grp.createVariable(
                        var + "_mean_tave", np.double, dimensions=("time", "z")
                    )
                    # create new np arrays to store the accumulated values
                    self._accu[var + "_mean_tave"] = np.zeros(
                        self._Grid.n[2], dtype=np.double
                    )
                else:
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z_edge"))
                    rt_grp.createVariable(
                        var + "_mean_tave", np.double, dimensions=("time", "z_edge")
                    )
                    self._accu[var + "_mean_tave"] = np.zeros(
                        self._Grid.n[2] + 1, dtype=np.double
                    )

                if var in ["u", "v"]:
                    rt_grp.createVariable(
                        var + "_squared_tave", np.double, dimensions=("time", "z")
                    )
                    self._accu[var + "_squared_tave"] = np.zeros(
                        self._Grid.n[2], dtype=np.double
                    )

                if var == "w":
                    rt_grp.createVariable(
                        var + "_squared_tave", np.double, dimensions=("time", "z_edge")
                    )
                    self._accu[var + "_squared_tave"] = np.zeros(
                        self._Grid.n[2] + 1, dtype=np.double
                    )
                    rt_grp.createVariable(
                        var + "_cubed_tave", np.double, dimensions=("time", "z_edge")
                    )
                    self._accu[var + "_cubed_tave"] = np.zeros(
                        self._Grid.n[2] + 1, dtype=np.double
                    )

        rt_grp.createVariable("uw_mean_tave", np.double, dimensions=("time", "z_edge"))
        self._accu["uw_mean_tave"] = np.zeros(self._Grid.n[2] + 1, dtype=np.double)
        rt_grp.createVariable("vw_mean_tave", np.double, dimensions=("time", "z_edge"))
        self._accu["vw_mean_tave"] = np.zeros(self._Grid.n[2] + 1, dtype=np.double)

        if hasattr(self._Surface, 'io_tower_init'):
            self._Surface.io_tower_init(rt_grp)

        if hasattr(self._Rad, 'io_tower_init'):
            self._Rad.io_tower_init(rt_grp)

        if hasattr(self._Micro, 'io_tower_init'):
            self._Micro.io_tower_init(rt_grp)

        rt_grp.close()

        return

    def accumulate(self):
        if not self._tower_on_this_rank:
            return

        weight = self._TimeSteppingController._dt / self._frequency

        nh = self._Grid.n_halo

        for con in self._containers:
            for var in con._dofs.keys():
                phi = con.get_field(var)
                if var == "u":
                    u_tower = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx - 1, self._j_indx, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                    self._accu[var + "_mean_tave"][:] += u_tower * weight
                    self._accu[var + "_squared_tave"][:] += (
                        np.power(u_tower, 2) * weight
                    )
                elif var == "uw_sgs":
                    self._accu[var + "_mean_tave"][:] += (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] -1 : -nh[2]],
                            phi[self._i_indx - 1, self._j_indx, nh[2] - 1: -nh[2]],
                        )
                        * 0.5
                        ) * weight
                elif var == "v":
                    v_tower = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                    self._accu[var + "_mean_tave"][:] += v_tower * weight
                    self._accu[var + "_squared_tave"][:] += (
                        np.power(v_tower, 2) * weight
                    )
                elif var == "vw_sgs":
                    self._accu[var + "_mean_tave"][:] += (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] - 1: -nh[2]],
                        )
                        * 0.5
                        ) * weight
                elif var == "w":
                    w_tower = phi[self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]]
                    self._accu[var + "_mean_tave"][:] += w_tower * weight
                    self._accu[var + "_squared_tave"][:] += (
                        np.power(w_tower, 2) * weight
                    )
                    self._accu[var + "_cubed_tave"][:] += np.power(w_tower, 3) * weight
                else:
                    if con._loc[var] != "z":
                        self._accu[var + "_mean_tave"][:] += (
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]] * weight
                        )
                    else:
                        self._accu[var + "_mean_tave"][:] += (
                            phi[self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]] * weight
                        )

        self._accu["uw_mean_tave"][1:-1] += (
            0.5 * w_tower[1:-1] * (u_tower[:-1] + u_tower[1:]) * weight
        )
        self._accu["vw_mean_tave"][1:-1] += (
            0.5 * w_tower[1:-1] * (v_tower[:-1] + v_tower[1:]) * weight
        )

        self._accuTime += self._TimeSteppingController._dt

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
                rt_grp[var + "_mean_tave"][-1, :] = self._accu[var + "_mean_tave"]
                self._accu[var + "_mean_tave"][:] = 0.0

                if var == "u":
                    rt_grp[var + "_squared_tave"][-1, :] = self._accu[
                        var + "_squared_tave"
                    ]
                    self._accu[var + "_squared_tave"][:] = 0.0
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx - 1, self._j_indx, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )

                elif var == "v":
                    rt_grp[var + "_squared_tave"][-1, :] = self._accu[
                        var + "_squared_tave"
                    ]
                    self._accu[var + "_squared_tave"][:] = 0.0
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                elif var == "w":
                    rt_grp[var + "_squared_tave"][-1, :] = self._accu[
                        var + "_squared_tave"
                    ]
                    self._accu[var + "_squared_tave"][:] = 0.0
                    rt_grp[var + "_cubed_tave"][-1, :] = self._accu[var + "_cubed_tave"]
                    self._accu[var + "_cubed_tave"][:] = 0.0
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = phi[
                        self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]
                    ]
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

        rt_grp["uw_mean_tave"][-1, :] = self._accu["uw_mean_tave"]
        self._accu["uw_mean_tave"][:] = 0.0
        rt_grp["vw_mean_tave"][-1, :] = self._accu["vw_mean_tave"]
        self._accu["vw_mean_tave"][:] = 0.0
        rt_grp["accu_time"][-1] = self._accuTime
        self._accuTime = 0.0

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

        if "location_file" in namelist["towers"]:
            with open(namelist["towers"]["location_file"], 'r') as f:
                tower_locations = json.load(f)
            UtilitiesParallel.print_root(f'\t \t {len(tower_locations)} tower locations read in from {namelist["towers"]["location_file"]}.')
        else:
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

    def accumulate(self):
        for tower in self._list_of_towers:
            tower.accumulate()

        return

    @property
    def frequency(self):
        return self._frequency
