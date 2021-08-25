import numpy as np
from scipy import interpolate
import os
import mpi4py as mpi
from mpi4py import MPI
from pinacles import parameters
from pinacles import Surface, Surface_impl, Forcing_impl, Forcing
from pinacles import UtilitiesParallel
import xarray as xr
from pinacles.LateralBCs import LateralBCsBase
import numba


class SurfaceReanalysis(Surface.SurfaceBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        Surface.SurfaceBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        return

    def io_initialize(self, rt_grp):

        return

    def io_update(self, rt_grp):

        return

    def update(self):

        return


class ForcingReanalysis(Forcing.ForcingBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):
        Forcing.ForcingBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            VelocityState,
            ScalarState,
            DiagnosticState,
        )

        self.f_at_u = (
            2.0 * parameters.OMEGA * np.sin(np.pi / 180.0 * self._Grid.lat_local_edge_x)
        )
        self.f_at_v = (
            2.0 * parameters.OMEGA * np.sin(np.pi / 180.0 * self._Grid.lat_local_edge_y)
        )

        return

    def update(self):

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")

        self.coriolis_apparent_force(u, v, f_at_u, f_at_v, ut, vt)

        return

    @staticmethod
    @numba.njit()
    def coriolis_apparent_force(u, v, f_at_u, f_at_v, ut, vt):

        shape = u.shape
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                for k in range(1, shape[2] - 1):
                    u_at_v = 0.25 * (
                        u[i, j, k]
                        + u[i - 1, j, k]
                        + u[i - 1, j + 1, k]
                        + u[i, j + 1, k]
                    )
                    v_at_u = 0.25 * (
                        v[i, j, k]
                        + v[i + 1, j, k]
                        + v[i + 1, j - 1, k]
                        + v[i, j - 1, k]
                    )
                    ut[i, j, k] += f_at_u[i, j] * v_at_u
                    vt[i, j, k] -= f_at_v[i, j] * u_at_v

        return


class InitializeReanalysis:
    def __init__(self, namelist, Grid, Ref, ScalarState, VelocityState, Ingest):

        self._namelist = namelist
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._Ingest = Ingest

        assert "real_data" in namelist["meta"]
        self._real_data = namelist["meta"]["real_data"]
        assert os.path.exists(self._real_data)

        return

    def initialize(self):

        nhalo = self._Grid.n_halo

        # Compute reference profiles
        lon, lat, skin_T = self._Ingest.get_skin_T()

        print(np.amax(lon), np.amin(lon), np.amax(lat), np.amin(lat))

        print("Here")
        print(
            np.amax(self._Grid.lon_local),
            np.amin(self._Grid.lon_local),
            np.amax(self._Grid.lat_local),
            np.amin(self._Grid.lat_local),
        )

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_lat = (lon_grid.flatten(), lat_grid.flatten())

        print(np.shape(lon_lat[0]), skin_T.flatten().shape)

        TSKIN = interpolate.griddata(
            lon_lat,
            skin_T.flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="cubic",
        )

        lon, lat, slp = self._Ingest.get_slp()

        SLP = interpolate.griddata(
            lon_lat,
            slp.flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="cubic",
        )

        slp = MPI.COMM_WORLD.allreduce(
            np.sum(SLP[nhalo[0] : -nhalo[0], nhalo[1] : -nhalo[1]]), op=MPI.SUM
        )
        slp /= self._Grid.n[0] * self._Grid.n[1]

        TSKIN = MPI.COMM_WORLD.allreduce(
            np.sum(TSKIN[nhalo[0] : -nhalo[0], nhalo[1] : -nhalo[1]]), op=MPI.SUM
        )
        TSKIN /= self._Grid.n[0] * self._Grid.n[1]

        # Compute the reference state
        self._Ref.set_surface(Psfc=slp, Tsfc=TSKIN, u0=0.0, v0=0.0)
        self._Ref.integrate()

        # Now initialize T
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initialize temperature")

        T = self._Ingest.interp_T(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )
        s = self._ScalarState.get_field("s")
        s[:, :] = (
            T
            + self._Grid.z_local[np.newaxis, np.newaxis]
            * (parameters.G)
            / parameters.CPD
        )

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initizliaing specific humidity")

        qv = self._ScalarState.get_field("qv")
        qv[:, :, :] = self._Ingest.interp_qv(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        # Now initializing u
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initializaing u")
        u = self._VelocityState.get_field("u")
        u[:, :, :] = self._Ingest.interp_u(
            self._Grid.lon_local_edge_x, self._Grid.lat_local_edge_x, self._Grid.z_local
        )

        # Now initialize v
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initializaing v")
        v = self._VelocityState.get_field("v")
        v[:, :, :] = self._Ingest.interp_v(
            self._Grid.lon_local_edge_y, self._Grid.lat_local_edge_y, self._Grid.z_local
        )

        return


class LateralBCsReanalysis(LateralBCsBase):
    def __init__(
        self, namelist, Grid, State, VelocityState, TimeSteppingController, Ingest
    ):

        LateralBCsBase.__init__(self, Grid, State, VelocityState)

        self._Ingest = Ingest
        self._TimeSteppingController = TimeSteppingController

        self.time_previous = self._TimeSteppingController._time

        return

    def init_vars_on_boundary(self):

        self._previous_bdy = {}
        self._post_bdy = {}

        ng = self._Grid.ngrid_local

        for bdys in [self._previous_bdy, self._post_bdy]:
            for var_name in self._State._dofs:
                bdys[var_name] = {}

                bdys[var_name]["x_low"] = np.zeros((ng[1], ng[2]), dtype=np.double)
                bdys[var_name]["x_high"] = np.zeros((ng[1], ng[2]), dtype=np.double)
                bdys[var_name]["y_low"] = np.zeros((ng[0], ng[2]), dtype=np.double)
                bdys[var_name]["y_high"] = np.zeros((ng[0], ng[2]), dtype=np.double)

        self.initial_ingest()

        return super().init_vars_on_boundary()

    # Set initial values
    def initial_ingest(self):

        self.bdy_lats = {}
        self.bdy_lons = {}
        nh = self._Grid.n_halo

        # print(self._Grid._local_axes_edge[0][nh[0]-1])
        # print(self._Grid._local_axes_edge[1][-nh[1]-1])

        self.bdy_lats["x_low"] = self._Grid.lat_local_edge_x[nh[0] - 1, :].reshape(
            self._Grid.lat_local_edge_x[nh[0] - 1, :].shape[0], 1
        )
        self.bdy_lons["x_low"] = self._Grid.lon_local_edge_x[nh[0] - 1, :].reshape(
            self._Grid.lon_local_edge_x[nh[0] - 1, :].shape[0], 1
        )

        self.bdy_lats["x_high"] = self._Grid.lat_local_edge_y[-nh[0] - 1, :].reshape(
            self._Grid.lat_local_edge_x[-nh[0], :].shape[0], 1
        )
        self.bdy_lons["x_high"] = self._Grid.lon_local_edge_y[-nh[0] - 1, :].reshape(
            self._Grid.lon_local_edge_x[-nh[0], :].shape[0], 1
        )

        self.bdy_lats["y_low"] = self._Grid.lat_local_edge_y[:, nh[1] - 1].reshape(
            self._Grid.lat_local_edge_y[:, nh[1] - 1].shape[0], 1
        )
        self.bdy_lons["y_low"] = self._Grid.lon_local_edge_y[:, nh[1] - 1].reshape(
            self._Grid.lon_local_edge_y[:, nh[1] - 1].shape[0], 1
        )

        self.bdy_lats["y_high"] = self._Grid.lat_local_edge_y[:, -nh[1] - 1].reshape(
            self._Grid.lat_local_edge_y[:, -nh[1]].shape[0], 1
        )
        self.bdy_lons["y_high"] = self._Grid.lon_local_edge_y[:, -nh[1] - 1].reshape(
            self._Grid.lon_local_edge_y[:, -nh[1]].shape[0], 1
        )

        for bdy_data, shift in zip([self._previous_bdy, self._post_bdy], [0, 1]):
            for bdy in ["x_low", "x_high", "y_low", "y_high"]:
                for var in self._State._dofs:
                    if var == "u":
                        bdy_data[var][bdy][:, :] = self._Ingest.interp_u(
                            self.bdy_lons[bdy],
                            self.bdy_lats[bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()
                    elif var == "v":
                        bdy_data[var][bdy][:, :] = self._Ingest.interp_v(
                            self.bdy_lons[bdy],
                            self.bdy_lats[bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()
                    elif var == "s":
                        bdy_data[var][bdy][:, :] = (
                            self._Ingest.interp_T(
                                self.bdy_lons[bdy],
                                self.bdy_lats[bdy],
                                self._Grid.z_local,
                                shift=shift,
                            ).squeeze()
                            + self._Grid.z_local[np.newaxis, :]
                            * (parameters.G)
                            / parameters.CPD
                        )
                    elif var == "qv":
                        bdy_data[var][bdy][:, :] = self._Ingest.interp_qv(
                            self.bdy_lons[bdy],
                            self.bdy_lats[bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()
                    else:
                        bdy_data[var][bdy].fill(0.0)

        return

    def set_vars_on_boundary(self, **kwargs):
        nh = self._Grid.n_halo

        for var in self._State._dofs:
            #    # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var)

            if var != "w":

                x_low[:, :] = (
                    self._previous_bdy[var]["x_low"]
                    + (self._post_bdy[var]["x_low"] - self._previous_bdy[var]["x_low"])
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                x_high[:, :] = (
                    self._previous_bdy[var]["x_high"]
                    + (
                        self._post_bdy[var]["x_high"]
                        - self._previous_bdy[var]["x_high"]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                y_low[:, :] = (
                    self._previous_bdy[var]["y_low"]
                    + (self._post_bdy[var]["y_low"] - self._previous_bdy[var]["y_low"])
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                y_high[:, :] = (
                    self._previous_bdy[var]["y_high"]
                    + (
                        self._post_bdy[var]["y_high"]
                        - self._previous_bdy[var]["y_high"]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

            else:
                w = self._State.get_field("w")

                x_low[:, :] = w[nh[0], :, :]
                x_high[:, :] = w[-nh[0] - 1, :, :]
                y_low[:, :] = w[:, nh[1], :]
                y_high[:, :] = w[:, -nh[1] - 1, :]

        # print(self.time_previous)

        return
