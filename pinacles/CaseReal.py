from pkg_resources import PkgResourcesDeprecationWarning
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
from pinacles.WRF_Micro_Kessler import compute_qvs
import copy


class SurfaceReanalysis(Surface.SurfaceBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        VelocityState,
        ScalarState,
        DiagnosticState,
        Ingest,
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

        self._Ingest = Ingest

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._qvflx = np.zeros_like(self._windspeed_sfc)
        self._tflx = np.zeros_like(self._windspeed_sfc)
        self._lhf = np.zeros_like(self._windspeed_sfc)
        self._shf = np.zeros_like(self._windspeed_sfc)
        self._Ri = np.zeros_like(self._windspeed_sfc)
        self._N = np.zeros_like(self._windspeed_sfc)
        self._psi_m = np.zeros_like(self._windspeed_sfc)
        self._psi_h = np.zeros_like(self._windspeed_sfc)

        self._cm = np.zeros_like(self._windspeed_sfc)
        self._ch = np.zeros_like(self._windspeed_sfc)
        self._cq = np.zeros_like(self._windspeed_sfc)
        self._z0 = np.zeros_like(self._windspeed_sfc)
        self._ustar = np.zeros_like(self._windspeed_sfc)
        self._u10 = np.zeros_like(self._windspeed_sfc)
        self._v10 = np.zeros_like(self._windspeed_sfc)

        self._TSKIN = np.zeros_like(self._windspeed_sfc)
        self._TSKIN_pre = np.zeros_like(self._windspeed_sfc)
        self._TSKIN_post = np.zeros_like(self._windspeed_sfc)
        self.T_surface = np.zeros_like(self._windspeed_sfc)
        self._previous_ingest = 1

        return

    def initialize(self):

        for tskin, shift in zip([self._TSKIN_pre, self._TSKIN_post], [0, 1]):
            # Compute reference profiles
            lon, lat, skin_T = self._Ingest.get_skin_T(shift=shift)
            #lon_grid, lat_grid = np.meshgrid(lon, lat)
            lon_lat = (lon.flatten(), lat.flatten())

            tskin[:, :] = interpolate.griddata(
                lon_lat,
                skin_T.flatten(),
                (self._Grid.lon_local, self._Grid.lat_local),
                method="cubic",
            )

        self.T_surface[:, :] = self._TSKIN_pre[:, :]

        return super().initialize()

    def update_ingest(self):

        self._previous_ingest += 1
        self._TSKIN_pre = np.copy(self._TSKIN_post)

        lon, lat, skin_T = self._Ingest.get_skin_T(shift= self._previous_ingest)
        lon_lat = (lon.flatten(), lat.flatten())
        
        self._TSKIN_post =  interpolate.griddata(
                lon_lat,
                skin_T.flatten(),
                (self._Grid.lon_local, self._Grid.lat_local),
                method="cubic",
            )

        return

    def io_initialize(self, rt_grp):

        return

    def io_update(self, rt_grp):

        return

    def update(self):

        self._Timers.start_timer("SurfaceRICO_update")

        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge
        p0_edge = self._Ref.p0_edge
        exner = self._Ref.exner

        # Get Fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        qv = self._ScalarState.get_field("qv")
        s = self._ScalarState.get_field("s")

        T = self._DiagnosticState.get_field("T")
        # Get Tendnecies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        Ssfc = s[:, :, nh[2]]
        qvsfc = qv[:, :, nh[2]]
        Tsfc = T[:, :, nh[2]]

        self._TSKIN[:, :] = self._TSKIN_pre[:, :]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )

        self._windspeed_sfc *= np.random.uniform(
            0.5, 1.5, size=(self._windspeed_sfc.shape[0], self._windspeed_sfc.shape[1])
        )

        Surface_impl.compute_surface_layer_Ri(
            nh,
            z_edge[nh[2]] / 2.0,
            self._TSKIN,
            exner_edge[nh[2] - 1],
            p0_edge[nh[2] - 1],
            qvsfc,
            Tsfc,
            exner[nh[2]],
            qvsfc,
            self._windspeed_sfc,
            self._N,
            self._Ri,
        )

        self._qv0 = compute_qvs(self._TSKIN, self._Ref.Psfc)

        self._z0[self._z0 < 0.0002] = 0.0002


        # Surface_impl.compute_exchange_coefficients(
        #     self._Ri,
        #     z_edge[nh[2]] / 2.0,
        #     self._z0,
        #     self._cm,
        #     self._ch,
        #     self._psi_m,
        #     self._psi_h
        # )


        Surface_impl.compute_exchange_coefficients_charnock(
           self._Ri,
           z_edge[nh[2]] / 2.0,
           self._z0,
           self._windspeed_sfc,
           self._cm,
           self._ch,
           self._psi_m,
           self._psi_h
        )
        self._cq[:, :] = self._ch[:, :]

        self._tflx = -self._ch * self._windspeed_sfc * (Ssfc - self._TSKIN)
        self._qvflx = -self._cq * self._windspeed_sfc * (qvsfc - self._qv0)
        self._ustar = np.sqrt(self._cm**2.0 * self._windspeed_sfc**2.0)


        u10_star = np.zeros_like(self._ustar)
        v10_star = np.zeros_like(self._ustar)
        
        u10_star[:,:] = np.sqrt((self._cm[:,:]**2.0) * (usfc)**2.0)
        v10_star[:,:] = np.sqrt((self._cm[:,:]**2.0) * (vsfc)**2.0)

        self._u10[:,:] = u10_star/0.41 * (np.log(10.0/self._z0) - self._psi_m)
        self._v10[:,:] = v10_star/0.41 * (np.log(10.0/self._z0) - self._psi_m)

        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tflx, st
        )
        Surface_impl.iles_surface_flux_application(
            10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._qvflx, qvt
        )

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

        self.coriolis_apparent_force(u, v, self.f_at_u, self.f_at_v, ut, vt)

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

        lon, lat, skin_T = self._Ingest.get_skin_T()

        print(np.mean(lat), np.mean(lon))

        lon_lat = (lon.flatten(), lat.flatten())


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
            print('\t\t Surface Pressure: \t' + str(slp))
            print('\t\t SKIN Temperature: \t' + str(TSKIN))
            print("\t Initialize temperature")

        
        T = self._Ingest.interp_T(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )
        
        
        s = self._ScalarState.get_field("s")

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initizliaing specific humidity")



        qv = self._ScalarState.get_field("qv")
        qv[:, :, :] = self._Ingest.interp_qv(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        qv[:, :, :] = qv[:, :, :] #/ self._Ref.rho0[np.newaxis, np.newaxis, :]


        qc = self._ScalarState.get_field("qc")
        qc_interp = self._Ingest.interp_qc(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        #qc[:, :, :] = qc_interp[:, :, :]
        #qc[:, :, :] = qc[:, :, :] #/ self._Ref.rho0[np.newaxis, np.newaxis, :]


        try:
            qi = self._ScalarState.get_field("qi")
        except:
            qi = self._ScalarState.get_field("qi1")

        qi_interp = self._Ingest.interp_qi(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        #qi[:, :, :] = qi_interp[:, :, :]
        #qi[:, :, :] = qi[:, :, :] #/ self._Ref.rho0[np.newaxis, np.newaxis, :]

        s[:, :, :] = (
            T
            + (
                self._Grid.z_local[np.newaxis, np.newaxis, :] * (parameters.G)
                - parameters.LV * qc_interp
                - parameters.LS * qi_interp
            )
            / parameters.CPD
        )

        random = np.random.uniform(-0.1, 0.1, size=(s.shape[0], s.shape[1], 3))
        s[:, :, nhalo[2] : nhalo[2] + 3] += random

        # Remove condensate from qv
        qv[:, :, :] = qv[:, :, :] + qc_interp[:,:,:] + qi_interp[:,:,:]

        # qv[qv < 1e-9] = 1e-9



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


        #import pylab as plt
        #plt.pcolor(v[:,:,4].T, cmap=plt.cm.gist_ncar)
        #plt.show()

        # Now we need to rotate the wind field
        # u_at_v = self._Grid.upt_to_vpt(u)
        # v_at_u = self._Grid.vpt_to_upt(v)

        # urot, tmp = self._Grid.MapProj.rotate_wind(self._Grid.lon_local_edge_x, u, v_at_u)
        # tmp, vrot = self._Grid.MapProj.rotate_wind(self._Grid.lon_local_edge_y, u_at_v, v)

        # v[:,:,:] = vrot[:,:,:]
        # u[:,:,:] = urot[:,:,:]

        return


class LateralBCsReanalysis(LateralBCsBase):
    def __init__(
        self,
        namelist,
        Grid,
        Ref,
        DiagnosticState,
        State,
        VelocityState,
        TimeSteppingController,
        Ingest,
    ):

        LateralBCsBase.__init__(self, Grid, State, VelocityState)
        self._Ref = Ref
        self._DiagnosticState = DiagnosticState
        self._Ingest = Ingest
        self._TimeSteppingController = TimeSteppingController

        self.time_previous = self._TimeSteppingController._time

        self.nudge_width = 5
        self.ingest_freq = 1800.0

        return

    def init_vars_on_boundary(self):

        self._previous_bdy = {}
        self._post_bdy = {}

        ng = self._Grid.ngrid_local

        bdy_width = self.nudge_width + 1

        for bdys in [self._previous_bdy, self._post_bdy]:

            vars = list(self._State._dofs.keys())
            if "s" in vars:
                vars.append("T")

            for var_name in vars:
                bdys[var_name] = {}

                bdys[var_name]["x_low"] = np.zeros(
                    (bdy_width, ng[1], ng[2]), dtype=np.double
                )
                bdys[var_name]["x_high"] = np.zeros(
                    (bdy_width, ng[1], ng[2]), dtype=np.double
                )
                bdys[var_name]["y_low"] = np.zeros(
                    (ng[0], bdy_width, ng[2]), dtype=np.double
                )
                bdys[var_name]["y_high"] = np.zeros(
                    (ng[0], bdy_width, ng[2]), dtype=np.double
                )

        self.initial_ingest()

        return super().init_vars_on_boundary()

    # Set initial values
    def initial_ingest(self):

        self.presvious_shift = 1

        self.bdy_lats = {}
        self.bdy_lons = {}
        for v in ["u", "v", "scalar"]:
            self.bdy_lats[v] = {}
            self.bdy_lons[v] = {}

        nh = self._Grid.n_halo

        ##########################################################
        #
        #  Get latitude/longitude at u-point
        #
        ##########################################################

        # x_low (these are normal for u)
        start = self._Grid._ibl_edge[0] 
        end = start +  self.nudge_width + 1

        self.bdy_lats["u"]["x_low"] = self._Grid.lat_local_edge_x[start:end, :]
        self.bdy_lons["u"]["x_low"] = self._Grid.lon_local_edge_x[start:end, :]

        start =self._Grid._ibu_edge[0]-self.nudge_width #-nh[1] - self.nudge_width - 1
        end = self._Grid._ibu_edge[0] + 1

        self.bdy_lats["u"]["x_high"] = self._Grid.lat_local_edge_x[start:end, :]
        self.bdy_lons["u"]["x_high"] = self._Grid.lon_local_edge_x[start:end, :]

        # y_low and y_high (these are non-normal for u)
        start = nh[1] - 1
        end = start + (self.nudge_width + 1)
        self.bdy_lats["u"]["y_low"] = self._Grid.lat_local_edge_x[:, start:end]
        self.bdy_lons["u"]["y_low"] = self._Grid.lon_local_edge_x[:, start:end]

        start = -nh[1] - self.nudge_width + 1 - 1
        end = -nh[1] + 1

        self.bdy_lats["u"]["y_high"] = self._Grid.lat_local_edge_x[:, start:end]
        self.bdy_lons["u"]["y_high"] = self._Grid.lon_local_edge_x[:, start:end]

        ##########################################################
        #
        #  Get latitude/longitude at v-point
        #
        ##########################################################
        # x_low (these are normal for u)

        start = nh[0] - 1
        end = start + (self.nudge_width + 1)

        self.bdy_lats["v"]["x_low"] = self._Grid.lat_local_edge_y[start:end, :]
        self.bdy_lons["v"]["x_low"] = self._Grid.lon_local_edge_y[start:end, :]

        start = -nh[0] - self.nudge_width
        end = -nh[0] + 1
        # print('here', self._Grid._local_axes[0][start:end])
        self.bdy_lats["v"]["x_high"] = self._Grid.lat_local_edge_y[start:end, :]
        self.bdy_lons["v"]["x_high"] = self._Grid.lon_local_edge_y[start:end, :]

        # y_low and y_high (these are non-normal for u)
        start = self._Grid._ibl_edge[1] 
        end = start +  self.nudge_width + 1
        self.bdy_lats["v"]["y_low"] = self._Grid.lat_local_edge_y[:, start:end]
        self.bdy_lons["v"]["y_low"] = self._Grid.lon_local_edge_y[:, start:end]

        start =self._Grid._ibu_edge[1]-self.nudge_width #-nh[1] - self.nudge_width - 1
        end = self._Grid._ibu_edge[1] + 1
        self.bdy_lats["v"]["y_high"] = self._Grid.lat_local_edge_y[:, start:end]
        self.bdy_lons["v"]["y_high"] = self._Grid.lon_local_edge_y[:, start:end]

        ##################################################################
        #
        #   Get latitude/longitude at scalar-point
        #
        ##################################################################
        start = nh[0] - 1
        end = start + (self.nudge_width + 1)

        self.bdy_lats["scalar"]["x_low"] = self._Grid.lat_local[start:end, :]
        self.bdy_lons["scalar"]["x_low"] = self._Grid.lon_local[start:end, :]
        start = -nh[0] - self.nudge_width
        end = -nh[0] + 1

        self.bdy_lats["scalar"]["x_high"] = self._Grid.lat_local[start:end, :]
        self.bdy_lons["scalar"]["x_high"] = self._Grid.lon_local[start:end, :]

        start = nh[1] - 1
        end = start + (self.nudge_width + 1)
        self.bdy_lats["scalar"]["y_low"] = self._Grid.lat_local[:, start:end]
        self.bdy_lons["scalar"]["y_low"] = self._Grid.lon_local[:, start:end]

        start = -nh[1] - self.nudge_width
        end = -nh[1] + 1
        self.bdy_lats["scalar"]["y_high"] = self._Grid.lat_local[:, start:end]
        self.bdy_lons["scalar"]["y_high"] = self._Grid.lon_local[:, start:end]

        for bdy_data, shift in zip([self._previous_bdy, self._post_bdy], [0, 1]):
            for bdy in ["x_low", "x_high", "y_low", "y_high"]:
                for var in self._State._dofs:
                    MPI.COMM_WORLD.Barrier()
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print("Setting boundaries for: ", var, bdy)
                    if var == "u":
                        bdy_data[var][bdy][:, :] = self._Ingest.interp_u(
                            self.bdy_lons[var][bdy],
                            self.bdy_lats[var][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()
                    elif var == "v":
                        bdy_data[var][bdy][:, :] = self._Ingest.interp_v(
                            self.bdy_lons[var][bdy],
                            self.bdy_lats[var][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()
                    elif var == "s":

                        qc = self._Ingest.interp_qc(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        qi = self._Ingest.interp_qi(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        #qc[:, :] = 0.0  # qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        #qi[:, :] = 0.0  # qi[:,:]/self._Ref.rho0[np.newaxis,:]

                        T = self._Ingest.interp_T(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        bdy_data[var][bdy][:, :] = (
                            T
                            + (
                                self._Grid.z_local[np.newaxis, :] * (parameters.G)
                                - parameters.LV * qc 
                                - parameters.LS * qi 
                            )
                            / parameters.CPD
                        )

                        bdy_data["T"][bdy][:, :] = T

                        print(
                            np.amax(bdy_data[var][bdy][:, :]),
                            np.amin(bdy_data[var][bdy][:, :]),
                        )
                    elif var == "qv":

                        qc = self._Ingest.interp_qc(
                             self.bdy_lons["scalar"][bdy],
                             self.bdy_lats["scalar"][bdy],
                             self._Grid.z_local,
                             shift=shift,
                         ).squeeze()
                        
                        qi = self._Ingest.interp_qi(
                             self.bdy_lons["scalar"][bdy],
                             self.bdy_lats["scalar"][bdy],
                             self._Grid.z_local,
                             shift=shift,
                         ).squeeze()

                        qv = self._Ingest.interp_qv(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        # qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        # qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        qv[:, :] = qv[:, :] + qc[:,:] + qi[:,:]# / self._Ref.rho0[np.newaxis, :]
                        # qi[qi < 0.0] = 0.0

                        bdy_data[var][bdy][:, :] = qv

                    elif False:  # var == "qc":

                        qc = self._Ingest.interp_qc(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        qc[qc < 0.0] = 0.0

                        qc[:, :] = qc[:, :] / self._Ref.rho0[np.newaxis, :]

                        bdy_data[var][bdy][:, :] = qc
                        print(
                            np.amax(bdy_data[var][bdy][:, :]),
                            np.amin(bdy_data[var][bdy][:, :]),
                        )

                    elif False:  # var == "qi" or var == 'qi1':

                        qi = self._Ingest.interp_qi(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze()

                        qi[:, :] = qi[:, :] / self._Ref.rho0[np.newaxis, :]

                        qi[qi < 0.0] = 0.0

                        bdy_data[var][bdy][:, :] = qi
                        print(
                            np.amax(bdy_data[var][bdy][:, :]),
                            np.amin(bdy_data[var][bdy][:, :]),
                        )
                    else:
                        bdy_data[var][bdy].fill(0.0)

        return

    def update_ingest(self):

        self.presvious_shift += 1
        self._previous_bdy = copy.deepcopy(self._post_bdy)

        bdy_data = self._post_bdy
        shift = self.presvious_shift
        for bdy in ["x_low", "x_high", "y_low", "y_high"]:
            for var in self._State._dofs:
                MPI.COMM_WORLD.Barrier()
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("Setting boundaries for: ", var, bdy)
                if var == "u":
                    bdy_data[var][bdy][:, :] = self._Ingest.interp_u(
                        self.bdy_lons[var][bdy],
                        self.bdy_lats[var][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()
                elif var == "v":
                    bdy_data[var][bdy][:, :] = self._Ingest.interp_v(
                        self.bdy_lons[var][bdy],
                        self.bdy_lats[var][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()
                elif var == "s":

                    qc = self._Ingest.interp_qc(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qi = self._Ingest.interp_qi(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qc[:, :] = qc[:,:]#/self._Ref.rho0[np.newaxis,:]
                    qi[:, :] = qi[:,:]#/self._Ref.rho0[np.newaxis,:]

                    T = self._Ingest.interp_T(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    bdy_data[var][bdy][:, :] = (
                        T
                        + (
                            self._Grid.z_local[np.newaxis, :] * (parameters.G)
                            - parameters.LV * qc 
                            - parameters.LS * qi 
                        )
                        / parameters.CPD
                    )

                    bdy_data["T"][bdy][:, :] = T

                elif var == "qv":
                    qc = self._Ingest.interp_qc(
                     self.bdy_lons["scalar"][bdy],
                     self.bdy_lats["scalar"][bdy],
                     self._Grid.z_local,
                     shift=shift,
                     ).squeeze()
                    
                    qi = self._Ingest.interp_qi(
                         self.bdy_lons["scalar"][bdy],
                         self.bdy_lats["scalar"][bdy],
                         self._Grid.z_local,
                         shift=shift,
                     ).squeeze()
                    
                    qv = self._Ingest.interp_qv(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qc[:,:] = qc[:,:]#/self._Ref.rho0[np.newaxis,:]
                    qi[:,:] = qi[:,:]#/self._Ref.rho0[np.newaxis,:]
                    qv[:, :] = qv[:, :] #/ self._Ref.rho0[np.newaxis, :]

                    bdy_data[var][bdy][:, :] = qv + qi + qc

                elif False:  # var == "qc":

                    qc = self._Ingest.interp_qc(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qc[qc < 0.0] = 0.0

                    qc[:, :] = qc[:, :] #/ self._Ref.rho0[np.newaxis, :]

                    bdy_data[var][bdy][:, :] = qc


                elif False:  # var == "qi" or var == 'qi1':

                    qi = self._Ingest.interp_qi(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qi[:, :] = qi[:, :] #/ self._Ref.rho0[np.newaxis, :]

                    qi[qi < 0.0] = 0.0

                    bdy_data[var][bdy][:, :] = qi
                else:
                    bdy_data[var][bdy].fill(0.0)

        return

    def set_vars_on_boundary(self, **kwargs):

        if self.presvious_shift * self.ingest_freq<= self._TimeSteppingController.time:
            print("Updating boundary data: ", self._TimeSteppingController.time)
            self.update_ingest()
            self.time_previous = self._TimeSteppingController._time

        nh = self._Grid.n_halo
        for var in self._State._dofs:
            #    # Compute the domain mean of the variables
            x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var)

            if var != "w":

                x_low[:, :] = (
                    self._previous_bdy[var]["x_low"][0, :, :]
                    + (
                        self._post_bdy[var]["x_low"][0, :, :]
                        - self._previous_bdy[var]["x_low"][0, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                x_high[:, :] = (
                    self._previous_bdy[var]["x_high"][-1, :, :]
                    + (
                        self._post_bdy[var]["x_high"][-1, :, :]
                        - self._previous_bdy[var]["x_high"][-1, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                y_low[:, :] = (
                    self._previous_bdy[var]["y_low"][:, 0, :]
                    + (
                        self._post_bdy[var]["y_low"][:, 0, :]
                        - self._previous_bdy[var]["y_low"][:, 0, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                y_high[:, :] = (
                    self._previous_bdy[var]["y_high"][:, -1, :]
                    + (
                        self._post_bdy[var]["y_high"][:, -1, :]
                        - self._previous_bdy[var]["y_high"][:, -1, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

            else:
                w = self._State.get_field("w")

                x_low[:, :] = w[nh[0], :, :]
                x_high[:, :] = w[-nh[0] - 1, :, :]
                y_low[:, :] = w[:, nh[1], :]
                y_high[:, :] = w[:, -nh[1] - 1, :]

        return

    def nudge_half(self, weight):
        
        
        if self._Grid.low_rank[0] and self._Grid.low_rank[1]:
            weight[:self.nudge_width,:self.nudge_width]  *= 0.5 
        
        if self._Grid.low_rank[0] and self._Grid.high_rank[1]:
            weight[:self.nudge_width,-self.nudge_width:]  *= 0.5 
        
        if self._Grid.high_rank[0] and self._Grid.low_rank[1]:
            weight[-self.nudge_width:,:self.nudge_width]  *= 0.5 
        
        if self._Grid.high_rank[0] and self._Grid.high_rank[1]:
            weight[-self.nudge_width:,-self.nudge_width:]  *= 0.5 
        
        
        return weight


    def lateral_nudge(self):

        nudge_width = self.nudge_width
        #weight = (nudge_width - np.arange(nudge_width))/nudge_width / (10.0 * self._TimeSteppingController.dt)
        # weight =  1.0/(2.0 * self._TimeSteppingController.dt)  / (1.0 + np.arange(nudge_width))
        dx = self._Grid.dx
        assert(dx[0] == dx[1])
        
        weight = (1.0 - np.tanh(np.arange(self.nudge_width) / 2)) / (
        10.0
        )
        
        #weight = self.nudge_half(weight)
        
        #weight[self.nudge_width:-self.nudge_width,self.nudge_width:-self.nudge_width]  *= 0.5      
        
    
        weight_u =  (1.0 - np.tanh(self._Grid._edge_dist_u/dx[0] / 2)) / (
        10.0
        )
        
        weight_u = self.nudge_half(weight_u)
        
        #weight_u[self.nudge_width:-self.nudge_width,self.nudge_width:-self.nudge_width]  *= 0.5 
        
        weight_v =  (1.0 - np.tanh(self._Grid._edge_dist_v/dx[1] / 2)) / (
        10.0
        )
        
        weight_v = self.nudge_half(weight_v)
        
        # weight = 1.0/(100.0 * self._TimeSteppingController.dt) * np.arange(self.nudge_width,0,-1)/self.nudge_width

        # weight = (1.0 + np.cos(np.arange(self.nudge_width) * np.pi / self.nudge_width)/2.0) /(4.0 * self._TimeSteppingController.dt)

        for var in self._State._dofs:

            #    # Compute the domain mean of the variables
            # x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var)
            nh = self._Grid.n_halo
            if var == "u":

                u = self._State.get_field(var)
                ut = self._State.get_tend(var)

                # Nudge u on the low boundary for x

                u_nudge = (
                    self._previous_bdy[var]["x_low"][:, :, :]
                    + (
                        self._post_bdy[var]["x_low"][:, :, :]
                        - self._previous_bdy[var]["x_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                corner_low = nh[1] #+ self.nudge_width
                corner_high = -nh[1] #- self.nudge_width

                start = self._Grid._ibl_edge[0] + 1
                end = start +  self.nudge_width

                if self._Grid.low_rank[0]:
                    ut[start:end, corner_low:corner_high, :] -= (
                        u[start:end, corner_low:corner_high, :] - u_nudge[1:, corner_low:corner_high, :]
                    ) * weight_u[:self.nudge_width, :, np.newaxis]
                    

                    
                    

                u_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                start =self._Grid._ibu_edge[0]-self.nudge_width 
                end = self._Grid.ibu_edge[0] 
                
                if self._Grid.high_rank[0]:
                    ut[start:end, corner_low:corner_high, :] -= (
                        u[start:end, corner_low:corner_high, :] - u_nudge[:-1, corner_low:corner_high, :]
                    ) *  weight_u[-self.nudge_width-1:-1,:, np.newaxis]

                u_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )
                
                start = self._Grid._ibl[1]
                end = start + self.nudge_width
                if self._Grid.low_rank[1]:
                    ut[nh[0]:-nh[0], start:end, :] -= (
                        u[nh[0]:-nh[0], start:end, :] - u_nudge[nh[0]:-nh[0], 1:, :]
                    ) * weight_u[:, :self.nudge_width ,np.newaxis]#weight[np.newaxis, :, np.newaxis]

                u_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                end = self._Grid._ibu[1]
                start = end - self.nudge_width
                if self._Grid.high_rank[1]:
                    #if not self._Grid.high_rank[0]:
                    ut[nh[0]:-nh[0], start:end, :] -= (
                    u[nh[0]:-nh[0], start:end, :] - u_nudge[nh[0]:-nh[0], :-1, :]
                    ) * weight_u[:, -self.nudge_width: ,np.newaxis] #weight[np.newaxis, ::-1, np.newaxis]
                    #else:
                    #    ut[nh[0]:-nh[0], start:end, :] -= (
                    #    u[nh[0]:-nh[0], start:end, :] - u_nudge[nh[0]:-nh[0], :-1, :]
                    #    ) * weight_u[:, -self.nudge_width-1:-1 ,np.newaxis] #weight[np.newaxis, ::-1, np.newaxis]
            elif var == "v":
                v = self._State.get_field(var)
                vt = self._State.get_tend(var)

                v_nudge = (
                    self._previous_bdy[var]["x_low"][:, :, :]
                    + (
                        self._post_bdy[var]["x_low"][:, :, :]
                        - self._previous_bdy[var]["x_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                start = self._Grid._ibl[0]
                end = start + self.nudge_width
                
                corner_low = nh[0] #+ self.nudge_width
                corner_high = -nh[0]# - self.nudge_width
                   
                if self._Grid.low_rank[0]:
                    
                    vt[start:end, corner_low:corner_high, :] -= (
                        v[start:end, corner_low:corner_high, :] - v_nudge[:-1, corner_low:corner_high, :]
                    ) * weight_v[:self.nudge_width, :,np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                end = self._Grid._ibu[0]
                start = end - self.nudge_width
                
                corner_low = nh[0] #+ self.nudge_width
                corner_high = -nh[0] #- self.nudge_width
             
                if self._Grid.high_rank[0]:
                    vt[start:end, corner_low:corner_high, :] -= (
                        v[start:end, corner_low:corner_high, :] - v_nudge[:-1, corner_low:corner_high, :]
                    ) *  weight_v[-self.nudge_width:, :,np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )    

                start = self._Grid._ibl_edge[1] + 1
                end = start +  self.nudge_width
                if self._Grid.low_rank[1]:
                    vt[nh[0]:-nh[0], start:end, :] -= (
                        v[nh[0]:-nh[0], start:end, :] - v_nudge[nh[0]:-nh[0], 1:, :]
                    ) * weight_v[:,:self.nudge_width,np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                start =self._Grid._ibu_edge[1]-self.nudge_width #-nh[1] - self.nudge_width - 1
                end = self._Grid._ibu_edge[1] #-nh[1] - 1
                if self._Grid.high_rank[1]:
                    vt[nh[0]:-nh[0], start:end, :] -= (
                        v[nh[0]:-nh[0], start:end, :] - v_nudge[nh[0]:-nh[0], :-1, :]
                    ) * weight_v[:,-self.nudge_width-1:-1,np.newaxis]


            elif var == "s":
                s = self._State.get_field(var)
                st = self._State.get_tend(var)
                
                
                
                
                nz_pert = 5
                amp = 0.1
                if self._Grid.low_rank[0]: 
                    start = nh[0]           #     start = nh[0]
                    end = nh[0] + self.nudge_width
                    pert_shape = st[start:end, :, nh[2]:nh[2] + nz_pert].shape
                    s[start:end, :, nh[2]:nh[2] + nz_pert] += np.random.uniform(low=-1.0*amp, high=1.0*amp, size=pert_shape)
                
                if self._Grid.high_rank[0]:
                     start = -nh[0] - self.nudge_width
                     end = -nh[0]
                     pert_shape = st[start:end, :, nh[2]:nh[2] + nz_pert].shape
                     s[start:end, :, nh[2]:nh[2] + nz_pert] += np.random.uniform(low=-1.0*amp, high=1.0*amp, size=pert_shape)

                if self._Grid.low_rank[1]:
                    start = nh[1]
                    end = nh[1] + self.nudge_width    
                    pert_shape = st[:, start:end, nh[2]:nh[2] + nz_pert].shape
                    s[:, start:end, nh[2]:nh[2] + nz_pert] += np.random.uniform(low=-1.0*amp, high=1.0*amp, size=pert_shape)
                
                if self._Grid.high_rank[1]:
                    start = -nh[1] - self.nudge_width
                    end = -nh[1]
                    pert_shape = st[:, start:end, nh[2]:nh[2] + nz_pert].shape
                    s[:, start:end, nh[2]:nh[2] + nz_pert] += np.random.uniform(low=-1.0*amp, high=1.0*amp, size=pert_shape)
                
                
            # elif var == "w":

            #     w = self._State.get_field(var)
            #     wt = self._State.get_tend(var)

            #     # if self._TimeSteppingController.time < self.ingest_freq:
            #     #    wt[:,:,:] = -w[:,:,:] * 1/300.0

            #     start = nh[0]
            #     end = nh[0] + self.nudge_width
            #     wt[start:end, :, :] -= (
            #          w[start:end, :, :] - w[start:end, :, :] 
            #      ) * weight[:, np.newaxis, np.newaxis]

            #     start = -nh[0] - self.nudge_width
            #     end = -nh[0]
            #     wt[start:end, :, :] -= (
            #          w[start:end, :, :] - w[start : end, :, :] 
            #      ) * weight[::-1, np.newaxis, np.newaxis]

            #     start = nh[1]
            #     end = nh[1] + self.nudge_width
            #     wt[:, start:end, :] -= (
            #          w[:, start:end, :]
            #          - w[:, start: end, :] 
            #      ) * weight[np.newaxis, :, np.newaxis]

            #     start = nh[1]
            #     end = nh[1] + self.nudge_width
            #     wt[ : , start:end, :] -= (
            #          w[:, start:end, :]
            #          - w[:, start: end, :] 
            #      ) * weight[np.newaxis, ::-1, np.newaxis]

            # elif (
            #     var == "s" or var == "qv"
            # ):  # or var == "qc" or var=="qi" or var == 'qi1':
            #     pass 
                # phi = self._State.get_field(var)
                # phi_t = self._State.get_tend(var)
                # s_t = self._State.get_tend("s")

                # if var == "s":
                #     var = "T"
                #     phi = self._DiagnosticState.get_field("T")

                # # Needed for energy source term
                # L = 0.0
                # if var == "qc":
                #     L = parameters.LV
                # elif var == "qi" or var == "qi1":
                #     L = parameters.LS

                # phi_nudge = (
                #     self._previous_bdy[var]["x_low"][:, :, :]
                #     + (
                #         self._post_bdy[var]["x_low"][:, :, :]
                #         - self._previous_bdy[var]["x_low"][:, :, :]
                #     )
                #     * (self._TimeSteppingController._time - self.time_previous)
                #     / self.ingest_freq
                # )

                # start = nh[0]
                # end = nh[0] + self.nudge_width
                # if self._Grid.low_rank[0]:
                #     phi_t[start:end, :, :] -= (
                #         phi[start:end, :, :] - phi_nudge[1:, :, :]
                #     ) * weight[:, np.newaxis, np.newaxis]

                #     s_t[start:end, :, :] += (
                #         L
                #         * (phi[start:end, :, :] - phi_nudge[1:, :, :])
                #         * weight[:, np.newaxis, np.newaxis]
                #         / parameters.CPD
                #     )

                # phi_nudge = (
                #     self._previous_bdy[var]["x_high"][:, :, :]
                #     + (
                #         self._post_bdy[var]["x_high"][:, :, :]
                #         - self._previous_bdy[var]["x_high"][:, :, :]
                #     )
                #     * (self._TimeSteppingController._time - self.time_previous)
                #     / self.ingest_freq
                # )

                # start = -nh[0] - self.nudge_width
                # end = -nh[0]

                # if self._Grid.high_rank[0]:
                #     phi_t[start:end, :, :] -= (
                #         phi[start:end, :, :] - phi_nudge[:-1, :, :]
                #     ) * weight[::-1, np.newaxis, np.newaxis]

                #     s_t[start:end, :, :] += (
                #         L
                #         * (phi[start:end, :, :] - phi_nudge[:-1, :, :])
                #         * weight[::-1, np.newaxis, np.newaxis]
                #         / parameters.CPD
                #     )

                # phi_nudge = (
                #     self._previous_bdy[var]["y_low"][:, :, :]
                #     + (
                #         self._post_bdy[var]["y_low"][:, :, :]
                #         - self._previous_bdy[var]["y_low"][:, :, :]
                #     )
                #     * (self._TimeSteppingController._time - self.time_previous)
                #     / self.ingest_freq
                # )

                # start = nh[1]
                # end = nh[1] + self.nudge_width
                # if self._Grid.low_rank[1]:
                #     phi_t[:, start:end, :] -= (
                #         phi[:, start:end, :] - phi_nudge[:, 1:, :]
                #     ) * weight[np.newaxis, :, np.newaxis]

                #     s_t[:, start:end, :] += (
                #         L
                #         * (phi[:, start:end, :] - phi_nudge[:, 1:, :])
                #         * weight[np.newaxis, :, np.newaxis]
                #         / parameters.CPD
                #     )

                # phi_nudge = (
                #     self._previous_bdy[var]["y_high"][:, :, :]
                #     + (
                #         self._post_bdy[var]["y_high"][:, :, :]
                #         - self._previous_bdy[var]["y_high"][:, :, :]
                #     )
                #     * (self._TimeSteppingController._time - self.time_previous)
                #     / self.ingest_freq
                # )

                # start = -nh[1] - self.nudge_width
                # end = -nh[1]
                # if self._Grid.high_rank[1]:
                #     phi_t[:, start:end, :] -= (
                #         phi[:, start:end, :] - phi_nudge[:, :-1, :]
                #     ) * weight[np.newaxis, ::-1, np.newaxis]

                #     s_t[:, start:end, :] += (
                #         L
                #         * (phi[:, start:end, :] - phi_nudge[:, :-1, :])
                #         * weight[np.newaxis, ::-1, np.newaxis]
                #         / parameters.CPD
                #     )

        return
