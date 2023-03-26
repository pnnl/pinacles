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
        TimeSteppingController,
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
        self._TimeSteppingController = TimeSteppingController

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
        self._windspeed4 = np.zeros_like(self._windspeed_sfc)
        self._windspeed10 = np.zeros_like(self._windspeed_sfc)
        self._u10 = np.zeros_like(self._windspeed_sfc)
        self._v10 = np.zeros_like(self._windspeed_sfc)
        self._deltas_sfc = np.zeros_like(self._windspeed_sfc)
        self._deltaqv_sfc = np.zeros_like(self._windspeed_sfc)

        self._u4 = np.zeros_like(self._windspeed_sfc)
        self._v4 = np.zeros_like(self._windspeed_sfc)
        self._T4 = np.zeros_like(self._windspeed_sfc)
        self._qv4 = np.zeros_like(self._windspeed_sfc)

        self._TSKIN = np.zeros_like(self._windspeed_sfc)
        self._TSKIN_pre = np.zeros_like(self._windspeed_sfc)
        self._TSKIN_post = np.zeros_like(self._windspeed_sfc)
        self.T_surface = np.zeros_like(self._windspeed_sfc)
        self._previous_ingest = 1

        return

    def initialize(self):
        self._previous_shift = 1
        self.ingest_freq = 3600.0
        self.time_previous = self._TimeSteppingController._time
        for tskin, shift in zip([self._TSKIN_pre, self._TSKIN_post], [0, 1]):
            # Compute reference profiles
            lon, lat, skin_T = self._Ingest.get_skin_T(shift=shift)
            # lon_grid, lat_grid = np.meshgrid(lon, lat)
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

        self._previous_shift += 1
        self._TSKIN_pre = np.copy(self._TSKIN_post)
        self.time_previous = self._TimeSteppingController._time
        lon, lat, skin_T = self._Ingest.get_skin_T(shift=self._previous_shift)
        lon_lat = (lon.flatten(), lat.flatten())

        self._TSKIN_post = interpolate.griddata(
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

    def io_fields2d_update(self, fx):
        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)
        nh = self._Grid.n_halo
        z = self._Grid.z_global

        k80_below = np.argmin(np.abs(z - 80.0))
        if z[k80_below] > 80.0:
            k80_below -= 1
        k80_above = k80_below + 1

        dz = z[k80_above] - z[k80_below]

        rho0_edge = self._Ref.rho0_edge[nh[0] - 1]

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        # Output the latent heat flux
        if fx is not None:
            lhf = fx.create_dataset(
                "LHF",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                lhf.dims[i].attach_scale(fx[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = (
            rho0_edge * parameters.LV * self._qvflx[nh[0] : -nh[0], nh[1] : -nh[1]]
        )

        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            lhf[:, :] = recv_buffer

        # Output the sensible heat flux
        if fx is not None:
            shf = fx.create_dataset(
                "SHF",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                shf.dims[i].attach_scale(fx[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = (
            rho0_edge * parameters.CPD * self._tflx[nh[0] : -nh[0], nh[1] : -nh[1]]
        )

        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            shf[:, :] = recv_buffer

        out_list = []
        out_list.append({"name": "Tskin", "data": self._TSKIN})
        out_list.append({"name": "windspeed4", "data": self._windspeed4})
        out_list.append({"name": "windspeed10", "data": self._windspeed10})
        out_list.append({"name": "u4", "data": self._u4})
        out_list.append({"name": "v4", "data": self._v4})
        out_list.append({"name": "u10", "data": self._u10})
        out_list.append({"name": "v10", "data": self._v10})
        out_list.append({"name": "taux", "data": self._taux_sfc})
        out_list.append({"name": "tauy", "data": self._tauy_sfc})
        out_list.append({"name": "z0", "data": self._z0})
        out_list.append({"name": "T4", "data": self._T4})
        out_list.append({"name": "qv4", "data": self._qv4})
        out_list.append({"name": "cm", "data": self._cm})
        out_list.append({"name": "cq", "data": self._cq})
        out_list.append({"name": "deltaS_sfc", "data": self._deltas_sfc})
        out_list.append({"name": "deltaqv_sfc", "data": self._deltaqv_sfc})
        # Output the sensible heat flux
        for out_var in out_list:
            if fx is not None:
                vh = fx.create_dataset(
                    out_var["name"],
                    (1, self._Grid.n[0], self._Grid.n[1]),
                    dtype=np.double,
                )

                for i, d in enumerate(["time", "X", "Y"]):
                    vh.dims[i].attach_scale(fx[d])

            send_buffer[start[0] : end[0], start[1] : end[1]] = out_var["data"][
                nh[0] : -nh[0], nh[1] : -nh[1]
            ]

            MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
            if fx is not None:
                vh[:, :] = recv_buffer

        # Output the sensible heat flux
        # if fx is not None:
        #    u4 = fx.create_dataset(
        #        "u_4m",
        #        (1, self._Grid.n[0], self._Grid.n[1]),
        #        dtype=np.double,
        #    )#
        #
        #    for i, d in enumerate(["time", "X", "Y"]):
        #        u4.dims[i].attach_scale(fx[d])

        # send_buffer[start[0] : end[0], start[1] : end[1]] =  rho0_edge * parameters.CPD * self._tflx[nh[0]:-nh[0], nh[1]:-nh[1]]
        #
        # MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        # if fx is not None:
        #     shf[:, :] = recv_buffer

        u80 = (
            u[:, :, k80_below]
            + (80.0 - z[k80_below]) * (u[:, :, k80_above] - u[:, :, k80_below]) / dz
        )
        u80[1:, :] = 0.5 * (u80[1:, :] + u80[:-1, :])
        v80 = (
            v[:, :, k80_below]
            + (80.0 - z[k80_below]) * (v[:, :, k80_above] - v[:, :, k80_below]) / dz
        )
        v80[:, 1:] = 0.5 * (v80[:, 1:] + v80[:, :-1])

        # Output the sensible heat flux
        if fx is not None:
            U80 = fx.create_dataset(
                "u80",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                U80.dims[i].attach_scale(fx[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = u80[
            nh[0] : -nh[0], nh[1] : -nh[1]
        ]

        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            U80[:, :] = recv_buffer

        # Output the sensible heat flux
        if fx is not None:
            V80 = fx.create_dataset(
                "v80",
                (1, self._Grid.n[0], self._Grid.n[1]),
                dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                V80.dims[i].attach_scale(fx[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = v80[
            nh[0] : -nh[0], nh[1] : -nh[1]
        ]

        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            V80[:, :] = recv_buffer

        return

    def io_tower_init(self, rt_grp):

        vars = [
            "LHF",
            "SHF",
            "u4",
            "v4",
            "u10",
            "v10",
            "Tskin",
            "windspeed4",
            "windspeed10",
            "taux",
            "tauy",
            "z0",
            "T4", 
            "qv4",
            "cm", 
            "cq",
            "deltaS_sfc", 
            "deltaqv_sfc"
        ]
        for v in vars:
            rt_grp.createVariable(v, np.double, dimensions=("time"))

    def io_tower(self, rt_grp, i_indx, j_indx):
        out_list = []
        out_list.append({"name": "Tskin", "data": self._TSKIN})
        out_list.append({"name": "windspeed4", "data": self._windspeed4})
        out_list.append({"name": "windspeed10", "data": self._windspeed10})
        out_list.append({"name": "u4", "data": self._u4})
        out_list.append({"name": "v4", "data": self._v4})
        out_list.append({"name": "u10", "data": self._u10})
        out_list.append({"name": "v10", "data": self._v10})
        out_list.append({"name": "taux", "data": self._taux_sfc})
        out_list.append({"name": "tauy", "data": self._tauy_sfc})
        out_list.append({"name": "z0", "data": self._z0})
        out_list.append({"name": "T4", "data": self._T4})
        out_list.append({"name": "qv4", "data": self._qv4})
        out_list.append({"name": "LHF", "data": self._lhf})
        out_list.append({"name": "SHF", "data": self._shf})
        out_list.append({"name": "cm", "data": self._cm})
        out_list.append({"name": "cq", "data": self._cq})
        out_list.append({"name": "deltaS_sfc", "data": self._deltas_sfc})
        out_list.append({"name": "deltaqv_sfc", "data": self._deltaqv_sfc})
        
        for out_var in out_list:
            rt_grp[out_var['name']][-1] = out_var['data'][i_indx, j_indx]

        return

    def update(self):

        self._Timers.start_timer("SurfaceRICO_update")

        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global
        zsfc = self._Grid.z_local[nh[2]]


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
        N2 = self._DiagnosticState.get_field("bvf")
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
        N2sfc = N2[:, :, nh[2]]

        self._N[:, :] = N2sfc

        if self._previous_shift * self.ingest_freq <= self._TimeSteppingController.time:
            print("Updating boundary data: ", self._TimeSteppingController.time)
            self.update_ingest()
            self.time_previous = self._TimeSteppingController._time

        self._TSKIN[:, :] = (
            self._TSKIN_pre[:, :]
            + (self._TSKIN_post[:, :] - self._TSKIN_pre[:, :])
            * (self._TimeSteppingController._time - self.time_previous)
            / self.ingest_freq
        )
        self.T_surface[:, :] = self._TSKIN[:, :]

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )

        # self._windspeed_sfc *= np.random.uniform(
        #    0.5, 1.5, size=(self._windspeed_sfc.shape[0], self._windspeed_sfc.shape[1])
        # )

        Surface_impl.compute_surface_layer_Ri_N2_passed(
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
            self._psi_h,
        )
        self._cq[:, :] = self._ch[:, :]


        self._deltas_sfc = ((Tsfc + parameters.G*parameters.ICPD * zsfc) - self._TSKIN)
        self._deltaqv_sfc = (qvsfc - self._qv0)
        self._tflx = -self._ch * self._windspeed_sfc * self._deltas_sfc
        self._qvflx = -self._cq * self._windspeed_sfc * self._deltaqv_sfc


        self._lhf = self._qvflx * parameters.LV * rho0_edge[nh[0] - 1]
        self._shf = self._tflx * parameters.CPD * rho0_edge[nh[0] - 1]

        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        self._ustar = (self._taux_sfc ** 2.0 + self._tauy_sfc ** 2.0) ** (1.0 / 4.0)

        self._windspeed4 = self._ustar / 0.35 * (np.log(4.0 / self._z0) - self._psi_m)
        self._windspeed10 = self._ustar / 0.35 * (np.log(10.0 / self._z0) - self._psi_m)

        self._u10[:, :] = usfc / self._windspeed_sfc * self._windspeed10
        self._v10[:, :] = vsfc / self._windspeed_sfc * self._windspeed10

        self._u4[:, :] = usfc / self._windspeed_sfc * self._windspeed4
        self._v4[:, :] = vsfc / self._windspeed_sfc * self._windspeed4

        self._T4 = self._TSKIN - (self._tflx/self._ustar/0.35) * (np.log(4.0 / self._z0) - self._psi_h) - (parameters.G * 4.0)*parameters.ICPD
        self._qv4 = self._qv0 - (self._qvflx/self._ustar/0.35) * (np.log(4.0 / self._z0) - self._psi_h) 

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
        
        self.f =  (
            2.0 * parameters.OMEGA * np.sin(np.pi / 180.0 * self._Grid.lat_local)
        )

        self.e =  (
            2.0 * parameters.OMEGA * np.cos(np.pi / 180.0 * self._Grid.lat_local)
        )


        self.alpha = self._Grid.MapProj.compute_alpha(self._Grid.lat_local, self._Grid.lon_local)
        
        print(self.alpha)
        
        return

    def update(self):

        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w  = self._VelocityState.get_field("w")

        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        wt = self._VelocityState.get_tend("w")

        self.coriolis_apparent_force(u, v, w, self.e, self.f, self.alpha, ut, vt, wt)

        return

    @staticmethod
    @numba.njit()
    def coriolis_apparent_force(u, v, w, e, f, alpha, ut, vt, wt):

        shape = u.shape
        for i in range(1, shape[0] - 1):
            for j in range(1, shape[1] - 1):
                for k in range(1, shape[2] - 1):
                    ut_ = 0.0
                    vt_ = 0.0
                    wt_ = 0.0 
                    for s in range(1):                    
                        uc = (u[i + s - 1, j, k] + u[i + s, j, k] ) * 0.5
                        vc = (v[i , j+ s - 1, k] + v[i , j+ s, k] ) * 0.5
                        wc = (w[i , j, k+ s - 1] + w[i, j, k + s] ) * 0.5
                    
                         
                        ut_ += (f[i,j] * vc) - (uc/6370000.0 + e[i,j] * np.cos(alpha[i,j]))*wc
                        vt_ += -(f[i,j] * uc) + (vc/6370000.0 - e[i,j] * np.sin(alpha[i,j]))*wc                     
                        wt_ += e[i,j] * (uc * np.cos(alpha[i,j]) - vc*np.sin(alpha[i,j])) + (1/6370000.0) * (uc*uc + vc * vc)

                    ut[i, j, k] += 0.5 * ut_ 
                    vt[i, j, k] += 0.5 * vt_ 
                    wt[i ,j, k] += 0.5 *  wt_
                    
                    

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

        lon_lat = (lon.flatten(), lat.flatten())

        print(np.amax(self._Grid.lon_local), np.amin(self._Grid.lon_local))
        print(np.amax(lon), np.amin(lon))
        
        print(np.amax(self._Grid.lat_local), np.amin(self._Grid.lat_local))
        print(np.amax(lat), np.amin(lat))
        #import sys; sys.exit()

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
            print("\t\t Surface Pressure: \t" + str(slp))
            print("\t\t SKIN Temperature: \t" + str(TSKIN))
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

        qv[:, :, :] = qv[:, :, :]  # / self._Ref.rho0[np.newaxis, np.newaxis, :]

        qc = self._ScalarState.get_field("qc")
        qc_interp = self._Ingest.interp_qc(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        # qc[:, :, :] = qc_interp[:, :, :]
        # qc[:, :, :] = qc[:, :, :] #/ self._Ref.rho0[np.newaxis, np.newaxis, :]

        try:
            qi = self._ScalarState.get_field("qi")
        except:
            qi = self._ScalarState.get_field("qi1")

        qi_interp = self._Ingest.interp_qi(
            self._Grid.lon_local, self._Grid.lat_local, self._Grid.z_local
        )

        # qi[:, :, :] = qi_interp[:, :, :]
        # qi[:, :, :] = qi[:, :, :] #/ self._Ref.rho0[np.newaxis, np.newaxis, :]

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
        qv[:, :, :] = qv[:, :, :] + qc_interp[:, :, :] + qi_interp[:, :, :]

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

        self.nudge_width = 3 
        self.ingest_freq = 3600.0

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
        end = start + self.nudge_width + 1

        self.bdy_lats["u"]["x_low"] = self._Grid.lat_local_edge_x[start:end, :]
        self.bdy_lons["u"]["x_low"] = self._Grid.lon_local_edge_x[start:end, :]

        start = (
            self._Grid._ibu_edge[0] - self.nudge_width
        )  # -nh[1] - self.nudge_width - 1
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
        end = start + self.nudge_width + 1
        self.bdy_lats["v"]["y_low"] = self._Grid.lat_local_edge_y[:, start:end]
        self.bdy_lons["v"]["y_low"] = self._Grid.lon_local_edge_y[:, start:end]

        start = (
            self._Grid._ibu_edge[1] - self.nudge_width
        )  # -nh[1] - self.nudge_width - 1
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

                        # qc[:, :] = 0.0  # qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        # qi[:, :] = 0.0  # qi[:,:]/self._Ref.rho0[np.newaxis,:]

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

                        # qc[:,:] = qc[:,:]/self._Ref.rho0[np.newaxis,:]
                        # qi[:,:] = qi[:,:]/self._Ref.rho0[np.newaxis,:]
                        qv[:, :] = (
                            qv[:, :] + qc[:, :] + qi[:, :]
                        )  # / self._Ref.rho0[np.newaxis, :]
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

                    qc[:, :] = qc[:, :]  # /self._Ref.rho0[np.newaxis,:]
                    qi[:, :] = qi[:, :]  # /self._Ref.rho0[np.newaxis,:]

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

                    qc[:, :] = qc[:, :]  # /self._Ref.rho0[np.newaxis,:]
                    qi[:, :] = qi[:, :]  # /self._Ref.rho0[np.newaxis,:]
                    qv[:, :] = qv[:, :]  # / self._Ref.rho0[np.newaxis, :]

                    bdy_data[var][bdy][:, :] = qv + qi + qc

                elif False:  # var == "qc":

                    qc = self._Ingest.interp_qc(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qc[qc < 0.0] = 0.0

                    qc[:, :] = qc[:, :]  # / self._Ref.rho0[np.newaxis, :]

                    bdy_data[var][bdy][:, :] = qc

                elif False:  # var == "qi" or var == 'qi1':

                    qi = self._Ingest.interp_qi(
                        self.bdy_lons["scalar"][bdy],
                        self.bdy_lats["scalar"][bdy],
                        self._Grid.z_local,
                        shift=shift,
                    ).squeeze()

                    qi[:, :] = qi[:, :]  # / self._Ref.rho0[np.newaxis, :]

                    qi[qi < 0.0] = 0.0

                    bdy_data[var][bdy][:, :] = qi
                else:
                    bdy_data[var][bdy].fill(0.0)

        return

    def set_vars_on_boundary(self, **kwargs):

        if self.presvious_shift * self.ingest_freq <= self._TimeSteppingController.time:
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
            weight[: self.nudge_width, : self.nudge_width] *= 0.5

        if self._Grid.low_rank[0] and self._Grid.high_rank[1]:
            weight[: self.nudge_width, -self.nudge_width :] *= 0.5

        if self._Grid.high_rank[0] and self._Grid.low_rank[1]:
            weight[-self.nudge_width :, : self.nudge_width] *= 0.5

        if self._Grid.high_rank[0] and self._Grid.high_rank[1]:
            weight[-self.nudge_width :, -self.nudge_width :] *= 0.5

        return weight

    def lateral_nudge(self):

        nudge_width = self.nudge_width
        # weight = (nudge_width - np.arange(nudge_width))/nudge_width / (10.0 * self._TimeSteppingController.dt)
        # weight =  1.0/(2.0 * self._TimeSteppingController.dt)  / (1.0 + np.arange(nudge_width))
        dx = self._Grid.dx
        assert dx[0] == dx[1]

        weight = (1.0 - np.tanh(np.arange(self.nudge_width) / 2)) / (100.0)

        # weight = self.nudge_half(weight)

        # weight[self.nudge_width:-self.nudge_width,self.nudge_width:-self.nudge_width]  *= 0.5

        weight_u = (1.0 - np.tanh(self._Grid._edge_dist_u / dx[0] / 2)) / (100.0)

        weight_u = self.nudge_half(weight_u)

        # weight_u[self.nudge_width:-self.nudge_width,self.nudge_width:-self.nudge_width]  *= 0.5

        weight_v = (1.0 - np.tanh(self._Grid._edge_dist_v / dx[1] / 2)) / (100.0)

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

                corner_low = nh[1]  # + self.nudge_width
                corner_high = -nh[1]  # - self.nudge_width

                start = self._Grid._ibl_edge[0] + 1
                end = start + self.nudge_width

                if self._Grid.low_rank[0]:
                    ut[start:end, corner_low:corner_high, :] -= (
                        u[start:end, corner_low:corner_high, :]
                        - u_nudge[1:, corner_low:corner_high, :]
                    ) * weight_u[: self.nudge_width, :, np.newaxis]

                u_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                start = self._Grid._ibu_edge[0] - self.nudge_width
                end = self._Grid.ibu_edge[0]

                if self._Grid.high_rank[0]:
                    ut[start:end, corner_low:corner_high, :] -= (
                        u[start:end, corner_low:corner_high, :]
                        - u_nudge[:-1, corner_low:corner_high, :]
                    ) * weight_u[-self.nudge_width:, :, np.newaxis]

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
                    ut[nh[0] : -nh[0], start:end, :] -= (
                        u[nh[0] : -nh[0], start:end, :] - u_nudge[nh[0] : -nh[0], 1:, :]
                    ) * weight_u[
                        :, : self.nudge_width, np.newaxis
                    ]  # weight[np.newaxis, :, np.newaxis]

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
                    # if not self._Grid.high_rank[0]:
                    ut[nh[0] : -nh[0], start:end, :] -= (
                        u[nh[0] : -nh[0], start:end, :]
                        - u_nudge[nh[0] : -nh[0], :-1, :]
                    ) * weight_u[
                        :, -self.nudge_width :, np.newaxis
                    ]  # weight[np.newaxis, ::-1, np.newaxis]
                    # else:
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

                corner_low = nh[0]  # + self.nudge_width
                corner_high = -nh[0]  # - self.nudge_width

                if self._Grid.low_rank[0]:

                    vt[start:end, corner_low:corner_high, :] -= (
                        v[start:end, corner_low:corner_high, :]
                        - v_nudge[:-1, corner_low:corner_high, :]
                    ) * weight_v[: self.nudge_width, :, np.newaxis]

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

                corner_low = nh[0]  # + self.nudge_width
                corner_high = -nh[0]  # - self.nudge_width

                if self._Grid.high_rank[0]:
                    vt[start:end, corner_low:corner_high, :] -= (
                        v[start:end, corner_low:corner_high, :]
                        - v_nudge[:-1, corner_low:corner_high, :]
                    ) * weight_v[-self.nudge_width :, :, np.newaxis]

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
                end = start + self.nudge_width
                if self._Grid.low_rank[1]:
                    vt[nh[0] : -nh[0], start:end, :] -= (
                        v[nh[0] : -nh[0], start:end, :] - v_nudge[nh[0] : -nh[0], 1:, :]
                    ) * weight_v[:, : self.nudge_width, np.newaxis]

                v_nudge = (
                    self._previous_bdy[var]["y_high"][:, :, :]
                    + (
                        self._post_bdy[var]["y_high"][:, :, :]
                        - self._previous_bdy[var]["y_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                start = (
                    self._Grid._ibu_edge[1] - self.nudge_width
                )  # -nh[1] - self.nudge_width - 1
                end = self._Grid._ibu_edge[1]  # -nh[1] - 1
                if self._Grid.high_rank[1]:
                    vt[nh[0] : -nh[0], start:end, :] -= (
                        v[nh[0] : -nh[0], start:end, :]
                        - v_nudge[nh[0] : -nh[0], :-1, :]
                    ) * weight_v[:, -self.nudge_width:, np.newaxis]
                    
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

            elif var == 's' or var == 'qv': #(var == "s" or var == "qv" and False
            #):  # or var == "qc" or var=="qi" or var == 'qi1':
            
                phi = self._State.get_field(var)
                phi_t = self._State.get_tend(var)
                # s_t = self._State.get_tend("s")

                # if var == "s":
                #    var = "T"
                #   phi = self._DiagnosticState.get_field("T")

                # Needed for energy source term
                # L = 0.0
                # if var == "qc":
                #     L = parameters.LV
                # elif var == "qi" or var == "qi1":
                #     L = parameters.LS

                phi_nudge = (
                    self._previous_bdy[var]["x_low"][:, :, :]
                    + (
                        self._post_bdy[var]["x_low"][:, :, :]
                        - self._previous_bdy[var]["x_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                # start = nh[0]
                # end = nh[0] + self.nudge_width
                start = self._Grid._ibl[0]
                end = start + self.nudge_width

                if self._Grid.low_rank[0]:
                    phi_t[start:end, :, :] -= (
                        phi[start:end, :, :] - phi_nudge[1:, :, :]
                    ) * weight[:, np.newaxis, np.newaxis]

                phi_nudge = (
                    self._previous_bdy[var]["x_high"][:, :, :]
                    + (
                        self._post_bdy[var]["x_high"][:, :, :]
                        - self._previous_bdy[var]["x_high"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                # start = -nh[0] - self.nudge_width
                # end = -nh[0]

                end = self._Grid._ibu[0]
                start = end - self.nudge_width
                if self._Grid.high_rank[0]:
                    phi_t[start:end, :, :] -= (
                        phi[start:end, :, :] - phi_nudge[:-1, :, :]
                    ) * weight[::-1, np.newaxis, np.newaxis]

                phi_nudge = (
                    self._previous_bdy[var]["y_low"][:, :, :]
                    + (
                        self._post_bdy[var]["y_low"][:, :, :]
                        - self._previous_bdy[var]["y_low"][:, :, :]
                    )
                    * (self._TimeSteppingController._time - self.time_previous)
                    / self.ingest_freq
                )

                # start = nh[1]
                # end = nh[1] + self.nudge_width
                start = self._Grid._ibl[1]
                end = start + self.nudge_width
                if self._Grid.low_rank[1]:
                    phi_t[nh[0] : -nh[0], start:end, :] -= (
                        phi[nh[0] : -nh[0], start:end, :]
                        - phi_nudge[nh[0] : -nh[0], 1:, :]
                    ) * weight[np.newaxis, :, np.newaxis]

                phi_nudge = (
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
                # start = -nh[1] - self.nudge_width
                # end = -nh[1]
                if self._Grid.high_rank[1]:
                    phi_t[nh[0] : -nh[0], start:end, :] -= (
                        phi[nh[0] : -nh[0], start:end, :]
                        - phi_nudge[nh[0] : -nh[0], :-1, :]
                    ) * weight[np.newaxis, ::-1, np.newaxis]

            if var == "s":

                s = self._State.get_field(var)
                u = self._VelocityState.get_field('u')
                v = self._VelocityState.get_field('v')
                nz_pert = 1
                amp = 0.1
                Ek =0.26
                
                if self._Grid.low_rank[0]:
                    start = nh[0]  #     start = nh[0]
                    end = nh[0] + 3 #self.nudge_width
                    
                    #start = self._Grid._ibl[0] + self.nudge_width
                    #end = start + 3
                    
                    speed = u[start:end, :, nh[2] : nh[2] + nz_pert] ** 2.0 + v[start:end, :, nh[2] : nh[2] + nz_pert]**2.0
                    s_p = (speed)/(1250.0 * Ek)
                    
                    pert_shape = s[start:end, :, nh[2] : nh[2] + nz_pert].shape
                    s[start:end, :, nh[2] : nh[2] + nz_pert] += np.random.uniform(
                        low=-1.0 , high=1.0 , size=pert_shape
                    )    * s_p

                if self._Grid.high_rank[0]:
                    start = -nh[0] - 3#self.nudge_width
                    end = -nh[0]

                    #start = self._Grid._ibl[0] - 3
                    #end = start + 3
                        
                    speed = u[start:end, :, nh[2] : nh[2] + nz_pert] ** 2.0 + v[start:end, :, nh[2] : nh[2] + nz_pert]**2.0
                    s_p = (speed)/(1250.0 * Ek)
                    
                    pert_shape = s[start:end, :, nh[2] : nh[2] + nz_pert].shape
                    s[start:end, :, nh[2] : nh[2] + nz_pert] += np.random.uniform(
                        low=-1.0, high=1.0, size=pert_shape
                    ) * s_p

                if self._Grid.low_rank[1]:
                    start = nh[1]
                    end = nh[1] + 3 #self.nudge_width
 
                   # start = self._Grid._ibl[1] + self.nudge_width
                   # end = start + 3
 
                    pert_shape = s[:, start:end, nh[2] : nh[2] + nz_pert].shape
        
                    speed = u[:, start:end, nh[2] : nh[2]+ nz_pert] ** 2.0 + v[:, start:end, nh[2] : nh[2]+ nz_pert]**2.0
                    s_p = (speed)/(1250.0 * Ek)
                                
                    
                    s[:, start:end, nh[2] : nh[2] + nz_pert] += np.random.uniform(
                        low=-1.0, high=1.0 , size=pert_shape
                    ) * s_p

                if self._Grid.high_rank[1]:
                    start = -nh[1] - 3 #self.nudge_width
                    end = -nh[1]
                    
                    #start = self._Grid._ibl[1] - 3
                    #end = start + 3                   
                    
                    pert_shape = s[:, start:end, nh[2] : nh[2] + nz_pert].shape
                    
                    speed = u[:, start:end, nh[2] : nh[2]+ nz_pert] ** 2.0 + v[:, start:end, nh[2] : nh[2]+ nz_pert]**2.0
                    s_p = (speed)/(1250.0 * Ek)
                    
                    s[:, start:end, nh[2] : nh[2] + nz_pert] += np.random.uniform(
                        low=-1.0 , high=1.0 , size=pert_shape
                    )  * s_p

        return
