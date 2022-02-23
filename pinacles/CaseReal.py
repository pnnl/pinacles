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
        self._Ri = np.zeros_like(self._windspeed_sfc)
        self._N = np.zeros_like(self._windspeed_sfc)

        self._cm = np.zeros_like(self._windspeed_sfc)
        self._ch = np.zeros_like(self._windspeed_sfc)
        self._cq = np.zeros_like(self._windspeed_sfc)

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

            lon = ((lon+180)%360)-180.0

            lon_grid, lat_grid = np.meshgrid(lon, lat)
            lon_lat = (lon_grid.flatten(), lat_grid.flatten())

            tskin[:, :] = interpolate.griddata(
                lon_lat,
                skin_T.flatten(),
                (self._Grid.lon_local, self._Grid.lat_local),
                method="cubic",
            )

        self.T_surface[:,:] = self._TSKIN_pre[:,:]

        return super().initialize()

    def update_ingest(self):

        self._previous_ingest += 1
        self._TSKIN_pre = np.copy(self._TSKIN_post)

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

        self._TSKIN[:,:] = self._TSKIN_pre[:,:]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )


        self._windspeed_sfc *= np.random.uniform(0.9, 1.1, size=(self._windspeed_sfc.shape[0], self._windspeed_sfc.shape[1]))

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

        Surface_impl.compute_exchange_coefficients_charnock(
            self._Ri,
            z_edge[nh[2]] / 2.0,
            np.zeros_like(qvsfc) + 0.0002,
            self._windspeed_sfc,
            self._cm,
            self._ch,
        )
        self._cq[:, :] = self._ch[:, :]

        self._tflx = -self._ch * self._windspeed_sfc * (Ssfc - self._TSKIN)
        self._qvflx = -self._cq * self._windspeed_sfc * (qvsfc - self._qv0)

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

        # Compute reference profiles
        lon, lat, skin_T = self._Ingest.get_skin_T()
        lon = ((lon+180)%360)-180.0

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_lat = (lon_grid.flatten(), lat_grid.flatten())

        TSKIN = interpolate.griddata(
            lon_lat,
            skin_T.flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="cubic",
        )


        lon, lat, slp = self._Ingest.get_slp()
        lon = ((lon+180)%360)-180.0
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

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Initizliaing specific humidity")

        qv = self._ScalarState.get_field("qv")
        qv[:, :, :] = self._Ingest.interp_qv(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )
        
        qv[:,:,:] = qv[:,:,:]/self._Ref.rho0[np.newaxis, np.newaxis, :]
        

        qc = self._ScalarState.get_field("qc")
        qc_interp = self._Ingest.interp_qc(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        qc[:,:,:] = qc_interp[:,:,:]
        qc[:,:,:] = qc[:,:,:]/self._Ref.rho0[np.newaxis, np.newaxis, :]
        
        
        try:
            qi= self._ScalarState.get_field("qi")
        except:
            qi= self._ScalarState.get_field("qi1")
        
        qi_interp = self._Ingest.interp_qi(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        qi[:,:,:] = qi_interp[:,:,:]
        qi[:,:,:] = qi[:,:,:]/self._Ref.rho0[np.newaxis, np.newaxis, :]

        s[:, :, :] = (
            T
            + (self._Grid.z_local[np.newaxis, np.newaxis,:]
            * (parameters.G)
            - parameters.LV * qc - parameters.LS * qi)
            / parameters.CPD
        )
        
        random = np.random.uniform(-0.1, 0.1, size=(s.shape[0], s.shape[1], 3))
        s[:,:,nhalo[2]:nhalo[2]+3]  += random 
        
        

        # Remove condensate from qv
        qv[:,:,:]  = qv[:,:,:] 



        #qv[qv < 1e-9] = 1e-9

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

        #Now we need to rotate the wind field
        #u_at_v = self._Grid.upt_to_vpt(u)
        #v_at_u = self._Grid.vpt_to_upt(v)
 
        #urot, tmp = self._Grid.MapProj.rotate_wind(self._Grid.lon_local_edge_x, u, v_at_u)
        #tmp, vrot = self._Grid.MapProj.rotate_wind(self._Grid.lon_local_edge_y, u_at_v, v)

        #v[:,:,:] = vrot[:,:,:]
        #u[:,:,:] = urot[:,:,:]




        return


class LateralBCsReanalysis(LateralBCsBase):
    def __init__(
        self, namelist, Grid, Ref, DiagnosticState, State, VelocityState, TimeSteppingController, Ingest
    ):

        LateralBCsBase.__init__(self, Grid, State, VelocityState)
        self._Ref = Ref
        self._DiagnosticState =  DiagnosticState
        self._Ingest = Ingest
        self._TimeSteppingController = TimeSteppingController

        self.time_previous = self._TimeSteppingController._time

        self.nudge_width = 5

        return

    def init_vars_on_boundary(self):

        self._previous_bdy = {}
        self._post_bdy = {}

        ng = self._Grid.ngrid_local

        bdy_width = self.nudge_width + 1

        for bdys in [self._previous_bdy, self._post_bdy]:
            
            vars = list(self._State._dofs.keys())
            if 's' in vars:
                vars.append('T')
            
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
        start = nh[0] - 1
        end = start + (self.nudge_width + 1)

        self.bdy_lats["u"]["x_low"] = self._Grid.lat_local_edge_x[start:end, :]
        self.bdy_lons["u"]["x_low"] = self._Grid.lon_local_edge_x[start:end, :]

        start = -nh[0] - self.nudge_width - 1
        end = -nh[0]

        self.bdy_lats["u"]["x_high"] = self._Grid.lat_local_edge_x[start:end, :]
        self.bdy_lons["u"]["x_high"] = self._Grid.lon_local_edge_x[start:end, :]

        # y_low and y_high (these are non-normal for u)
        start = nh[1] - 1
        end = start + (self.nudge_width + 1)
        self._Grid._local_axes[1][start:end]
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
        start = nh[1] - 1
        end = start + (self.nudge_width + 1)
        self.bdy_lats["v"]["y_low"] = self._Grid.lat_local_edge_y[:, start:end]
        self.bdy_lons["v"]["y_low"] = self._Grid.lon_local_edge_y[:, start:end]

        start = -nh[1] - self.nudge_width - 1
        end = -nh[1]
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
                        
                        qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        
                        
                        T = self._Ingest.interp_T(
                                self.bdy_lons["scalar"][bdy],
                                self.bdy_lats["scalar"][bdy],
                                self._Grid.z_local,
                                shift=shift,
                            ).squeeze()
                        
                        bdy_data[var][bdy][:, :] = (
                            T
                            +(self._Grid.z_local[np.newaxis, :]
                            * (parameters.G) - parameters.LV * qc - parameters.LS * qi)
                            / parameters.CPD
                        )
                    
                        bdy_data['T'][bdy][:, :] = T
                    
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    elif var == "qv":
                        
                        
                        #qc = self._Ingest.interp_qc(
                        #     self.bdy_lons["scalar"][bdy],
                        #     self.bdy_lats["scalar"][bdy],
                        #     self._Grid.z_local,
                        #     shift=shift,
                        # ).squeeze() 
                        
                        # qi = self._Ingest.interp_qi(
                        #     self.bdy_lons["scalar"][bdy],
                        #     self.bdy_lats["scalar"][bdy],
                        #     self._Grid.z_local,
                        #     shift=shift,
                        # ).squeeze() 
                        
                        
                        qv = self._Ingest.interp_qv(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        #qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        #qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        qv[:,:] = qv[:,:]/self._Ref.rho0[np.newaxis,:]
                        #qi[qi < 0.0] = 0.0
                        
                        bdy_data[var][bdy][:, :] =  qv
                        
                        
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    elif var == "qc":
                        
                        qc = self._Ingest.interp_qc(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        qc[qc < 0.0] = 0.0
                        
                        qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        
                        bdy_data[var][bdy][:, :] = qc     
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))        
                        
                    elif var == "qi" or var == 'qi1':
                        
                        qi = self._Ingest.interp_qi(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        
                        qi[qi < 0.0] = 0.0
                        
                        bdy_data[var][bdy][:, :] = qi
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    else:
                        bdy_data[var][bdy].fill(0.0)

        return

    def update_ingest(self):

        self.presvious_shift += 1
        self._previous_bdy = copy.deepcopy(self._post_bdy)

        bdy_data = self._post_bdy
        shift = self.presvious_shift
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
                        
                        qc[:,:] = 0.0 #qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        qi[:,:] = 0.0 #qi[:,:]/self._Ref.rho0[np.newaxis,:]
                    
                    
                        T = self._Ingest.interp_T(
                                self.bdy_lons["scalar"][bdy],
                                self.bdy_lats["scalar"][bdy],
                                self._Grid.z_local,
                                shift=shift,
                            ).squeeze()
                        
                        bdy_data[var][bdy][:, :] = (
                            T
                            +(self._Grid.z_local[np.newaxis, :]
                            * (parameters.G) - parameters.LV * qc - parameters.LS * qi)
                            / parameters.CPD
                        )
                        
                        bdy_data['T'][bdy][:, :] = T
                    
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    elif var == "qv":
                        
                        
                        # qc = self._Ingest.interp_qc(
                        #     self.bdy_lons["scalar"][bdy],
                        #     self.bdy_lats["scalar"][bdy],
                        #     self._Grid.z_local,
                        #     shift=shift,
                        # ).squeeze() 
                        
                        # qi = self._Ingest.interp_qi(
                        #     self.bdy_lons["scalar"][bdy],
                        #     self.bdy_lats["scalar"][bdy],
                        #     self._Grid.z_local,
                        #     shift=shift,
                        # ).squeeze() 
                        
                
                        qv = self._Ingest.interp_qv(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        #qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        #qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        qv[:,:] = qv[:,:]/self._Ref.rho0[np.newaxis,:]

                        bdy_data[var][bdy][:, :] =  qv
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    elif False: #var == "qc":
                        
                        qc = self._Ingest.interp_qc(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        qc[qc < 0.0] = 0.0
                        
                        qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        
                        bdy_data[var][bdy][:, :] = qc     
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))        
                        
                    elif False: #var == "qi" or var == 'qi1':
                        
                        qi = self._Ingest.interp_qi(
                            self.bdy_lons["scalar"][bdy],
                            self.bdy_lats["scalar"][bdy],
                            self._Grid.z_local,
                            shift=shift,
                        ).squeeze() 
                        
                        qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        
                        qi[qi < 0.0] = 0.0
                        
                        bdy_data[var][bdy][:, :] = qi
                        print(np.amax(bdy_data[var][bdy][:, :]), np.amin(bdy_data[var][bdy][:, :]))
                    else:
                        bdy_data[var][bdy].fill(0.0)

        return

    def set_vars_on_boundary(self, **kwargs):

        if self.presvious_shift * 3600 <= self._TimeSteppingController.time:
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
                    / 3600.0
                )

                x_high[:, :] = (
                    self._previous_bdy[var]["x_high"][-1, :, :]
                    + (
                        self._post_bdy[var]["x_high"][-1, :, :]
                        - self._previous_bdy[var]["x_high"][-1, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                y_low[:, :] = (
                    self._previous_bdy[var]["y_low"][:, 0, :]
                    + (
                        self._post_bdy[var]["y_low"][:, 0, :]
                        - self._previous_bdy[var]["y_low"][:, 0, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                y_high[:, :] = (
                    self._previous_bdy[var]["y_high"][:, -1, :]
                     + (
                         self._post_bdy[var]["y_high"][:,-1,:]
                         - self._previous_bdy[var]["y_high"][:,-1,:]
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

        return

    def lateral_nudge(self):

        nudge_width = self.nudge_width
        # weight =  1.0/(2.0 * self._TimeSteppingController.dt)  / (1.0 + np.arange(nudge_width))
        weight = (1.0 - np.tanh(np.arange(self.nudge_width) / 2)) / (
            10.0 * self._TimeSteppingController.dt
        )
        # weight = (1.0 + np.cos(np.arange(self.nudge_width) * np.pi / self.nudge_width)/2.0) /(4.0 * self._TimeSteppingController.dt)

        for var in self._State._dofs:

            #    # Compute the domain mean of the variables
            # x_low, x_high, y_low, y_high = self.get_vars_on_boundary(var)
            nh = self._Grid.n_halo
            if var == "u":

                u = self._State.get_field(var)
                ut = self._State.get_tend(var)

                # Nudge u on the low boundary for x
                start = nh[0]
                end = start + self.nudge_width

                u_nudge = (
                    self._previous_bdy[var]["x_low"][:, :, :]
                    + (
                        self._post_bdy[var]["x_low"][:, :, :]
                        - self._previous_bdy[var]["x_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                if self._Grid.low_rank[0]:
                    ut[start:end, :, :] -= (
                        u[start:end, :, :] - u_nudge[1:, :, :]
                    ) * weight[:, np.newaxis, np.newaxis]

                # print(self._previous_bdy['u']['x_high'][:-1,:,3] - u[start:end,:,3])

                start = -nh[0] - self.nudge_width - 1
                end = -nh[0] - 1

                u_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                if self._Grid.high_rank[0]:
                    ut[start:end, :, :] -= (
                        u[start:end, :, :] - u_nudge[:-1, :, :]
                    ) * weight[::-1, np.newaxis, np.newaxis]

                start = nh[1]
                end = nh[1] + self.nudge_width

                u_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                if self._Grid.low_rank[1]:
                    ut[:, start:end, :] -= (
                        u[:, start:end, :] - u_nudge[:, 1:, :]
                    ) * weight[np.newaxis, :, np.newaxis]

                u_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = -nh[1] - self.nudge_width
                end = -nh[1]
                if self._Grid.high_rank[1]:
                    ut[:, start:end, :] -= (
                        u[:, start:end, :] - u_nudge[:, :-1, :]
                    ) * weight[np.newaxis, ::-1, np.newaxis]

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
                    / 3600.0
                )

                start = nh[0]
                end = nh[0] + self.nudge_width
                if self._Grid.low_rank[0]:
                    vt[start:end, :, :] -= (
                        v[start:end, :, :] - v_nudge[1:, :, :]
                    ) * weight[:, np.newaxis, np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = -nh[0] - self.nudge_width
                end = -nh[0]

                if self._Grid.high_rank[0]:
                    vt[start:end, :, :] -= (
                        v[start:end, :, :] - v_nudge[:-1, :, :]
                    ) * weight[::-1, np.newaxis, np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = nh[1]
                end = start + self.nudge_width
                if self._Grid.low_rank[1]:
                    vt[:, start:end, :] -= (
                        v[:, start:end, :] - v_nudge[:, 1:, :]
                    ) * weight[np.newaxis, :, np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = -nh[1] - self.nudge_width - 1
                end = -nh[1] - 1
                if self._Grid.high_rank[1]:
                    vt[:, start:end, :] -= (
                        v[:, start:end, :] - v_nudge[:, :-1, :]
                    ) * weight[np.newaxis, ::-1, np.newaxis]

            elif var == "w":

                w = self._State.get_field(var)
                wt = self._State.get_tend(var)

                #if self._TimeSteppingController.time < 3600.0:
                #    wt[:,:,:] = -w[:,:,:] * 1/300.0 


                # start = nh[0]
                # end = nh[0] + self.nudge_width
                # wt[start:end, :, :] -= (
                #     w[start:end, :, :] - w[start + 1 : end + 1, :, :]
                # ) * weight[:, np.newaxis, np.newaxis]

                # start = -nh[0] - self.nudge_width
                # end = -nh[0]
                # wt[start:end, :, :] -= (
                #     w[start:end, :, :] - w[start - 1 : end - 1, :, :]
                # ) * weight[::-1, np.newaxis, np.newaxis]

                # start = nh[1]
                # end = nh[1] + self.nudge_width
                # wt[2 * nh[1] : -2 * nh[1], start:end, :] -= (
                #     w[2 * nh[1] : -2 * nh[1], start:end, :]
                #     - w[2 * nh[1] : -2 * nh[1], start + 1 : end + 1, :]
                # ) * weight[np.newaxis, :, np.newaxis]

                # start = -nh[1] - self.nudge_width
                # end = -nh[1]
                # wt[2 * nh[1] : -2 * nh[1], start:end, :] -= (
                #     w[2 * nh[1] : -2 * nh[1], start:end, :]
                #     - w[2 * nh[1] : -2 * nh[1], start - 1 : end - 1, :]
                # ) * weight[np.newaxis, ::-1, np.newaxis]

            elif var == "s" or var == "qv":# or var == "qc" or var=="qi" or var == 'qi1':

                phi = self._State.get_field(var)
                phi_t = self._State.get_tend(var)
                s_t = self._State.get_tend('s')


                if var == 's':
                    var = 'T'
                    phi = self._DiagnosticState.get_field('T')



                # Needed for energy source term
                L = 0.0 
                if var == "qc":
                    L =  parameters.LV
                elif var == "qi" or var == "qi1":
                    L = parameters.LS



                phi_nudge = (
                    self._previous_bdy[var]["x_low"][:, :, :]
                    + (
                        self._post_bdy[var]["x_low"][:, :, :]
                        - self._previous_bdy[var]["x_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = nh[0]
                end = nh[0] + self.nudge_width
                if self._Grid.low_rank[0]:
                    phi_t[start:end, :, :] -= (
                        phi[start:end, :, :] - phi_nudge[1:, :, :]
                    ) * weight[:, np.newaxis, np.newaxis]
                    
                    s_t[start:end, :, :] += L * (
                        phi[start:end, :, :] - phi_nudge[1:, :, :]
                    ) * weight[:, np.newaxis, np.newaxis]/parameters.CPD


                phi_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = -nh[0] - self.nudge_width
                end = -nh[0]

                if self._Grid.high_rank[0]:
                    phi_t[start:end, :, :] -= (
                        phi[start:end, :, :] - phi_nudge[:-1, :, :]
                    ) * weight[::-1, np.newaxis, np.newaxis]
                    
                    s_t[start:end, :, :] += L * (
                        phi[start:end, :, :] - phi_nudge[:-1, :, :]
                    ) * weight[::-1, np.newaxis, np.newaxis]/parameters.CPD
                    
                    

                phi_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = nh[1]
                end = nh[1] + self.nudge_width
                if self._Grid.low_rank[1]:
                    phi_t[:, start:end, :] -= (
                        phi[:, start:end, :] - phi_nudge[:, 1:, :]
                    ) * weight[np.newaxis, :, np.newaxis]
                    
                    s_t[:, start:end, :] += L *  (
                        phi[:, start:end, :] - phi_nudge[:, 1:, :]
                    ) * weight[np.newaxis, :, np.newaxis]/parameters.CPD

                phi_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / 3600.0
                )

                start = -nh[1] - self.nudge_width
                end = -nh[1]
                if self._Grid.high_rank[1]:
                    phi_t[:, start:end, :] -= (
                        phi[:, start:end, :] - phi_nudge[:, :-1, :]
                    ) * weight[np.newaxis, ::-1, np.newaxis]
                        
                    s_t[:, start:end, :] += L * (
                        phi[:, start:end, :] - phi_nudge[:, :-1, :]
                    ) * weight[np.newaxis, ::-1, np.newaxis]/parameters.CPD

        return
