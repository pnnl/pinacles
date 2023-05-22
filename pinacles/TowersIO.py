import numpy as np
import netCDF4 as nc
from mpi4py import MPI
import os


class Tower:
    def __init__(self, namelist, Grid, TimeSteppingController, loc=(10.0, 10.0)):

        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController
        self._containers = []
        self._accu = {}
        self._accuTime = 0.0
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
                    rt_grp.createVariable(var+"_mean", np.double, dimensions=("time", "z"))
                    # create new np arrays to store the accumulated values
                    self._accu[var+"_mean"] = np.zeros(self._Grid.n[2], dtype=np.double)
                else:
                    rt_grp.createVariable(var, np.double, dimensions=("time", "z_edge"))
                    rt_grp.createVariable(var+"_mean", np.double, dimensions=("time", "z_edge"))
                    self._accu[var+"_mean"] = np.zeros(self._Grid.n[2]+1, dtype=np.double)
                
                if var in ["u", "v"]:
                    rt_grp.createVariable(var+"_squared", np.double, dimensions=("time", "z"))
                    self._accu[var+"_squared"] = np.zeros(self._Grid.n[2], dtype=np.double)

                if var == "w":
                    rt_grp.createVariable(var+"_squared", np.double, dimensions=("time", "z_edge"))
                    self._accu[var+"_squared"] = np.zeros(self._Grid.n[2]+1, dtype=np.double)
                    rt_grp.createVariable(var+"_cubed", np.double, dimensions=("time", "z_edge"))
                    self._accu[var+"_cubed"] = np.zeros(self._Grid.n[2]+1, dtype=np.double)

        rt_grp.createVariable("uw_mean", np.double, dimensions=("time", "z"))
        self._accu["uw_mean"] = np.zeros(self._Grid.n[2], dtype=np.double)
        rt_grp.createVariable("vw_mean", np.double, dimensions=("time", "z"))
        self._accu["vw_mean"] = np.zeros(self._Grid.n[2], dtype=np.double)

        rt_grp.close()
        return

    def accumulate(self):

        if not self._tower_on_this_rank:
            return

        dt = self._TimeSteppingController._dt

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
                    self._accu[var+"_mean"][:] += u_tower * dt 
                    self._accu[var+"_squared"][:] += np.power(u_tower, 2) * dt
                elif var == "v":
                    v_tower = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                    self._accu[var+"_mean"][:] += v_tower * dt 
                    self._accu[var+"_squared"][:] += np.power(v_tower, 2) * dt
                elif var == "w":
                    w_tower = phi[
                        self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]
                    ]
                    self._accu[var+"_mean"][:] += w_tower * dt
                    self._accu[var+"_squared"][:] += np.power(w_tower, 2) * dt
                    self._accu[var+"_cubed"][:] += np.power(w_tower, 3) * dt
                else:
                    if con._loc[var] != "z":
                        self._accu[var+"_mean"][:] += phi[
                            self._i_indx, self._j_indx, nh[2] : -nh[2]
                        ] * dt
                    else:
                        self._accu[var+"_mean"][:] += phi[
                            self._i_indx, self._j_indx, nh[2] - 1 : -nh[2]
                        ] * dt

        self._accu["uw_mean"][:] += 0.5*(w_tower[1:]+w_tower[:-1])*u_tower*dt
        self._accu["vw_mean"][:] += 0.5*(w_tower[1:]+w_tower[:-1])*v_tower*dt

        self._accuTime += dt 

        return

    def update(self):

        if not np.allclose(self._TimeSteppingController._time % self._frequency, 0.0):
            return

        if not self._tower_on_this_rank:
            return

        nh = self._Grid.n_halo

        rt_grp = nc.Dataset(self._out_file, "r+")

        time = rt_grp["time"]
        time[time.shape[0]] = self._TimeSteppingController._time

        for con in self._containers:
            for var in con._dofs.keys():

                rt_grp[var+"_mean"][-1, :] = self._accu[var+"_mean"]/self._frequency
                self._accu[var+"_mean"][:] = 0.0

                if var == "u":
                    rt_grp[var+"_squared"][-1, :] = self._accu[var+"_squared"]/self._frequency
                    self._accu[var+"_squared"][:] = 0.0
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx - 1, self._j_indx, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                elif var == "v":
                    rt_grp[var+"_squared"][-1, :] = self._accu[var+"_squared"]/self._frequency
                    self._accu[var+"_squared"][:] = 0.0
                    phi = con.get_field(var)
                    rt_grp[var][-1, :] = (
                        np.add(
                            phi[self._i_indx, self._j_indx, nh[2] : -nh[2]],
                            phi[self._i_indx, self._j_indx - 1, nh[2] : -nh[2]],
                        )
                        * 0.5
                    )
                elif var == "w":
                    rt_grp[var+"_squared"][-1, :] = self._accu[var+"_squared"]/self._frequency
                    self._accu[var+"_squared"][:] = 0.0
                    rt_grp[var+"_cubed"][-1, :] = self._accu[var+"_cubed"]/self._frequency
                    self._accu[var+"_cubed"][:] = 0.0
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

        rt_grp["uw_mean"][-1, :] = self._accu["uw_mean"]/self._frequency
        self._accu["uw_mean"][:] = 0.0
        rt_grp["vw_mean"][-1, :] = self._accu["vw_mean"]/self._frequency
        self._accu["vw_mean"][:] = 0.0
        rt_grp["accu_time"][-1] = self._accuTime
        self._accuTime = 0.0

        rt_grp.close()


        return


class Towers:
    def __init__(self, namelist, Timers, Grid, TimeSteppingController):

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

        for loc in tower_locations:
            self._list_of_towers.append(
                Tower(namelist, Grid, TimeSteppingController, loc=tuple(loc))
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
