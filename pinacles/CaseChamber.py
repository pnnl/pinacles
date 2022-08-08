import numpy as np
import numba
from mpi4py import MPI
from pinacles import Surface, Surface_impl
from pinacles import Forcing, Forcing_impl
from pinacles import parameters
import pinacles.ThermodynamicsMoist_impl as MoistThermo
import pinacles.UtilitiesParallel as UtilitiesParallel


def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState):

    UtilitiesParallel.print_root("Initializing Stable-bubble Case")

    #  Optionally set a random seed as specified in the namelist
    try:
        rank = MPI.Get_rank()
        np.random.seed(namelist["meta"]["random_seed"] + rank)
    except:
        pass

    # Integrate the reference profile.
    Ref.set_surface(Tsfc=300.0)
    Ref.integrate()

    u = VelocityState.get_field("u")
    v = VelocityState.get_field("v")
    w = VelocityState.get_field("w")
    qv = ScalarState.get_field("qv")
    s = ScalarState.get_field("s")

    xl = ModelGrid.local_axes[0]
    xl_e = ModelGrid.local_axes_edge[0]
    yl = ModelGrid.local_axes[1]
    yl_e = ModelGrid.local_axes_edge[1]
    zl = ModelGrid.local_axes[2]
    xg = ModelGrid.x_global
    yg = ModelGrid.y_global
    nh = ModelGrid.n_halo
    
    lx = ModelGrid.l[0]
    ly = ModelGrid.l[1]
    

    exner = Ref.exner

    # Wind is uniform initiall
    u.fill(2.0)
    v.fill(0.0)
    w.fill(0.0)

    shape = s.shape

    s.fill(293.0)
    pert = np.random.uniform(low=-0.25, high=0.25, size=s.shape)
    pert_mean = np.mean(np.mean(pert[nh[0]:-nh[0], nh[1]:-nh[1], :],axis=0),axis=0)
    pert = pert - pert_mean[np.newaxis,np.newaxis,:]
    #s += np.random.uniform(low=-0.5, high=0.5, size=s.shape)
    
    
    
    zpert_i = np.min(np.argmin(np.abs(zl - 7.5))) 
    
    pert[:,:,zpert_i:] = 0
    s += pert

    shape = s.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                T = MoistThermo.T(zl[k], s[i,j,k], 0.0, 0.0)
                qs = MoistThermo.compute_qvs(T, Ref.rho0[k], Ref.p0[k]) 
                qv[i,j,k] = qs * 0.4
                
                
    #u_prof = (2.0)/10.0 * zl
    
    vamp = np.sin(32.0 * xl/lx * np.pi)
    uamp = np.sin(8.0 * yl/ly * np.pi)
    
    
    #u[:,:,:] = u_prof[np.newaxis, np.newaxis, :]

    v[:,:,:zpert_i] += vamp[:,np.newaxis,np.newaxis]
    u[:,:,:zpert_i] += uamp[np.newaxis,:,np.newaxis]

    


    #v[:,:,:8] = np.random.uniform(low=-1e-1, high=1e-1, size=v[:,:,:8].shape)
    #print( np.random.uniform(low=-1e-1, high=1e-1, size=v[:,:,:8].shape))



class ForcingChamber(Forcing.ForcingBase):
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState, TimeSteppingController
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

        zl = self._Grid.z_local
        self._ug = np.zeros_like(zl) + 2.0
        self._vg = np.zeros_like(zl)
        self._f = 1.0e-4
        self._TimeSteppingController = TimeSteppingController

        return

    def update(self):

        #self._Timers.start_timer("ForcingBomex_update")
        exner = self._Ref.exner
        
        
        vol = self._Grid._dx[0] * self._Grid._dx[1] * self._Grid._dx[2]


        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        w = self._VelocityState.get_field("w")
        s = self._ScalarState.get_field("s")
        qv = self._ScalarState.get_field("qv")


        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        wt = self._VelocityState.get_tend("w")
        st = self._ScalarState.get_tend("s")
        qvt = self._ScalarState.get_tend("qv")
        qct = self._ScalarState.get_tend("qc")
        sprayt = self._ScalarState.get_tend("spray")

        lx = self._Grid.l[0]
        ly = self._Grid.l[1]
        lz = self._Grid.l[2]
    

        xlocs = [1.0 - 4*0.078125, 1.0, 1.0  + 4*0.078125]
        ylocs = [7.5 - 4*0.078125, 7.5, 7.5  + 4*0.078125]
        zlocs  = [7.5 - 4*0.078125, 7.5,  7.5  + 4*0.078125] 

        noz_points = []
        for xl in xlocs:
            for yl in ylocs:
                for zl in zlocs:
                    noz_points.append((xl, yl, zl))

        for noz_point in noz_points:
            
            on_rank = self._Grid.point_on_rank(noz_point[0], noz_point[1], noz_point[2])
        
            if on_rank and  self._TimeSteppingController.time >= 240.0 and self._TimeSteppingController.time <= 250.0:
                
                xl = self._Grid.x_local
                yl = self._Grid.y_local
                zl = self._Grid.z_local
                
                
                xp = np.argmin(np.abs(xl - noz_point[0]))
                yp = np.argmin(np.abs(yl  - noz_point[1]))
                zp = np.argmin(np.abs(zl  - noz_point[2])) - 1
                
                
                #print(vol)
                wt[xp, yp, zp] += ((2.155/1000.0) * 16.0)/ self._Ref.rho0[zp]/vol
                qct[xp, yp, zp + 1] += (2.155/1000.0) / self._Ref.rho0[zp]/vol
                sprayt[xp, yp, zp + 1] += (2.155/1000.0) / self._Ref.rho0[zp]/vol
                st[xp, yp, zp + 1] -=  parameters.LV * (2.155/1000.0) / self._Ref.rho0[zp]/vol*parameters.ICPD

        Forcing_impl.large_scale_pgf(self._ug, self._vg, self._f, u, v, 0.0, 0.0, vt, ut)


        #self._Timers.end_timer("ForcingBomex_update")
        return
    
class SurfaceChamber(Surface.SurfaceBase):
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

        self._theta_flux = 0.0
        self._z0 = 0.1

        nl = self._Grid.ngrid_local

        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._taux_sfc = np.zeros_like(self._windspeed_sfc)
        self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
        self._bflx_sfc = np.zeros_like(self._windspeed_sfc)
        self._ustar_sfc = np.zeros_like(self._windspeed_sfc)
        self._tflx = np.zeros_like(self._windspeed_sfc)

        self._Timers.add_timer("SurfaceSullivanAndPatton_update")
        return

    def update(self):

        self._Timers.start_timer("SurfaceSullivanAndPatton_update")
        self.bflux_from_thflux()

        self._bflx_sfc[:, :] = self._buoyancy_flux
        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        alpha0 = self._Ref.alpha0
        alpha0_edge = self._Ref.alpha0_edge
        rho0_edge = self._Ref.rho0_edge

        exner_edge = self._Ref.exner_edge

        # Get Fields
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")

        u0 = self._Ref.u0
        v0 = self._Ref.v0

        # Get Tendencies
        ut = self._VelocityState.get_tend("u")
        vt = self._VelocityState.get_tend("v")
        st = self._ScalarState.get_tend("s")

        # Get surface slices
        usfc = u[:, :, nh[2]]
        vsfc = v[:, :, nh[2]]
        utsfc = ut[:, :, nh[2]]
        vtsfc = vt[:, :, nh[2]]
        stsfc = st[:, :, nh[2]]

        # Compute the windspeed, friction velocity, and surface stresses
        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, u0, v0, self.gustiness, self._windspeed_sfc
        )
        Surface_impl.compute_ustar_sfc(
            self._windspeed_sfc,
            self._bflx_sfc,
            self._z0,
            self._Grid.dx[2] / 2.0,
            self._ustar_sfc,
        )
        Surface_impl.tau_given_ustar(
            self._ustar_sfc,
            usfc,
            vsfc,
            u0,
            v0,
            self._windspeed_sfc,
            self._taux_sfc,
            self._tauy_sfc,
        )

        # Compute the surface temperature flux
        self._tflx[:, :] = self._theta_flux * exner_edge[nh[2] - 1]

        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._taux_sfc, ut
        )
        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tauy_sfc, vt
        )
        Surface_impl.iles_surface_flux_application(
            1e-6, z_edge, dxi2, nh, alpha0, alpha0_edge, 250.0, self._tflx, st
        )

        self._Timers.end_timer("SurfaceSullivanAndPatton_update")
        return
