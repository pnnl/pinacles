import pinacles.Surface as Surface
import pinacles.Surface_impl as Surface_impl
import pinacles.externals.wrf_noahmp_wrapper.noahmp_via_cffi as NoahMP
import pinacles.externals.wrf_noahmp_wrapper.test_notebooks.noahmp_offline_mod as nom
import numpy as np
import xarray as xr
from scipy import interpolate
class SurfaceNoahMP(Surface.SurfaceBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        Micro,
        VelocityState,
        ScalarState,
        DiagnosticState,
    ):

        self._Micro = Micro

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



        self._NOAH_MP = NoahMP.noahmp()

        n_halo = self._Grid.n_halo

        #domain range
        ids=1
        ide=self._Grid.nl[0]
        jds=1
        jde=self._Grid.nl[1]
        kds=1
        kde=1


        ## further domain setting and other testcase-dependent setting
        #domain size & indices
        #memory range (used to define variable arrays)
        ims=ids
        ime=ide#-1
        jms=jds
        jme=jde#-1
        kms=kds
        kme=kde 

        #tile range
        its=ids
        ite=ide#-1
        jts=jds
        jte=jde#-1
        kts=kds
        kte=kde

        iones_float_2d =  np.ones((ime-ims+1,jme-jms+1),dtype=np.float64)
        iones_int_2d = np.ones((ime-ims+1,jme-jms+1),dtype=np.intc)


        nsoil=4
        nsnow = 3 #number of snow layers. set to 3 in the subroutine noahmplsm 
        dzs = np.asfortranarray([0.01, 0.3, 0.6, 1]) #soil layer thickness
        dz = 20.0


        # Container classes
        self._NoahMPvars = nom.NoahMPvars(ims,ime,jms,jme,nsoil,nsnow,dzs)
        self._NoahMPtoATM = nom.NoahMPtoATM(ims, ime, jms, jme)
        self._ATMtoNoahMP = nom.ATMtoNoahMP(ims,ime,jms,jme,kms,kme,dz)

        NMPvars = self._NoahMPvars
        NMP2ATM = self._NoahMPtoATM

        dx = self._Grid.dx[0]
        dy = self._Grid.dx[1]

        is_restart = False
        is_restart = False  #is restart or not
        allowed_to_read = True #True allows model to read land-use and soil parameters from tables
                            #i.e., call  SOIL_VEG_GEN_PARM( MMINLU, MMINSL )

        areaxy  = np.asfortranarray(iones_float_2d)  * dx * dy #grid cell area [m2] - 
        #no need to give actual value when dx, dy, msftx,msfty are given to noahmplsm; calculated in the module for iopt_run = 5
        msftx = np.asfortranarray(iones_float_2d)      #map factor
        msfty = np.asfortranarray(iones_float_2d)      #map factor


        IDVEG = 1    #dynamic vegetation (1 -> off ; 2 -> on) with opt_crs = 1      
        IOPT_CRS = 1 #canopy stomatal resistance (1-> Ball-Berry; 2->Jarvis)
        IOPT_BTR = 1 #soil moisture factor for stomatal resistance (1-> Noah; 2-> CLM; 3-> SSiB)
        IOPT_RUN = 1 #runoff and groundwater (1->SIMGM; 2->SIMTOP; 3->Schaake96; 4->BATS)
        IOPT_SFC = 1 #surface layer drag coeff (CH & CM) (1->M-O; 2->Chen97)
        IOPT_FRZ = 1 #supercooled liquid water (1-> NY06; 2->Koren99)
        IOPT_INF = 1 #frozen soil permeability (1-> NY06; 2->Koren99)
        IOPT_RAD = 1 #radiation transfer (1->gap=F(3D,cosz); 2->gap=0; 3->gap=1-Fveg)
        IOPT_ALB = 1 #snow surface albedo (1->BATS; 2->CLASS)
        IOPT_SNF = 1 #rainfall & snowfall (1-Jordan91; 2->BATS; 3->Noah)
        IOPT_TBOT = 1 #lower boundary of soil temperature (1->zero-flux; 2->Noah)
        IOPT_STC = 1 #snow/soil temperature time scheme (for only layer 1) 
                    # 1 -> semi-implicit; 2 -> full implicit (original Noah))
        IZ0TLND = 0 #option of Chen adjustment of Czil 
                    #! it seems to be used; value of 1 uses Chen & Zhang (2009) to modify 
                    #a constant parameter (CZIL) in the expression for thermal roughness length

        ISICE =15
        ISURBAN = 16
        ISWATER =  np.intc(17)


        data_in = xr.open_dataset('sensitivity5_d01_static.nc')
        

        ISLTYP = np.asfortranarray(np.ones_like(iones_int_2d)) #soil type, all 1 = sand
        IVGTYP = np.asfortranarray(np.ones_like(iones_int_2d)*5) #5 = all grid points are grass 

        print(np.shape(ISLTYP))

        lat_in = data_in['XLAT'].values[0,:,:]
        lon_in = data_in['XLONG'].values[0,:,:]
        ISLTYP = data_in['ISLTYP'].values[0,:,:]
        IVGTYP = data_in['IVGTYP'].values[0,:,:]


        #print(np.shape(lat_in))

        #lon_grid, lat_grid = np.meshgrid(lon_in[0,:,:], lat_in[0,:,:])
        lon_lat = (lon_in.flatten(), lat_in.flatten())

 
        ISLTYP = interpolate.griddata(
                lon_lat,
                ISLTYP.flatten(),
                (self._Grid.lon_local, self._Grid.lat_local),
                method="nearest",
            )

        IVGTYP = interpolate.griddata(
                lon_lat,
                IVGTYP.flatten(),
                (self._Grid.lon_local, self._Grid.lat_local),
                method="nearest",
            )
        


        ISLTYP = np.asfortranarray(ISLTYP[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1]], dtype=np.intc)
        IVGTYP = np.asfortranarray(IVGTYP[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1]], dtype=np.intc)
        print(np.shape(ISLTYP))
        import pylab as plt
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.pcolor(ISLTYP.T)
        plt.subplot(122)
        plt.pcolor(IVGTYP.T)
        plt.show()

        data_in.close()
        

        self._NOAH_MP.init(ids,ide,jds,jde,kds,kde,
            ims,ime,jms,jme,kms,kme,
            its,ite,jts,jte,kts,kte,
            nsoil,dzs,NMPvars.tsk,NMPvars.isnowxy,NMPvars.snow, NMPvars.snowh, NMPvars.canwat, 
            ISICE, ISWATER, ISURBAN, ISLTYP, IVGTYP,
            NMPvars.xice, IOPT_RUN, is_restart, allowed_to_read,
            NMPvars.smois, NMPvars.sh2o, NMPvars.tslb, NMPvars.tmn, NMPvars.zsnsoxy, 
            NMPvars.tsnoxy, NMPvars.snicexy, NMPvars.snliqxy,
            NMPvars.sneqvoxy, NMPvars.alboldxy, NMPvars.qsnowxy, NMPvars.tvxy, 
            NMPvars.tgxy, NMPvars.canicexy, NMPvars.canliqxy,
            NMPvars.eahxy,NMPvars.tahxy, NMP2ATM.cmxy, NMP2ATM.chxy, NMPvars.fwetxy, 
            NMPvars.wslakexy, NMPvars.zwtxy, NMPvars.waxy, NMPvars.wtxy,
            NMPvars.lfmassxy, NMPvars.rtmassxy, NMPvars.stmassxy, NMPvars.woodxy,
            NMPvars.stblcpxy, NMPvars.fastcpxy, NMPvars.xsaixy,
            NMPvars.t2mvxy, NMPvars.t2mbxy, NMP2ATM.chstarxy, 
            NMPvars.smoiseq, NMPvars.smcwtdxy, NMPvars.deeprechxy, NMPvars.rechxy, 
            NMPvars.qrfsxy, NMPvars.qspringsxy, NMPvars.qslatxy, areaxy,
            dx, dy, msftx, msfty,
            NMPvars.fdepthxy, NMPvars.ht, NMPvars.riverbedxy, NMPvars.eqzwt, 
            NMPvars.rivercondxy, NMPvars.pexpxy)


        return

    def initialize(self):
        return super().initialize()

    def update(self):

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

        self._tflx = -self._ch * self._windspeed_sfc * (Ssfc - self._TSKIN)
        self._qvflx = -self._cq * self._windspeed_sfc * (qvsfc - self._qv0)

        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        # Apply the surface fluxes
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

        return super().update()

    def io_initialize(self, rt_grp):
        return super().io_initialize(rt_grp)

    def io_update(self, rt_grp):
        return super().io_update(rt_grp)
