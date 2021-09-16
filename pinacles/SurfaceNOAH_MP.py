import pinacles.Surface as Surface
import pinacles.Surface_impl as Surface_impl
import pinacles.externals.wrf_noahmp_wrapper.noahmp_via_cffi as NoahMP
from pinacles.Radiation import cos_sza
from pinacles.WRFUtil import to_wrf_order
from pinacles import UtilitiesParallel
from pinacles import parameters
import pinacles.externals.wrf_noahmp_wrapper.test_notebooks.noahmp_offline_mod as nom
import numpy as np
import xarray as xr
from scipy import interpolate

import datetime


class SurfaceNoahMP(Surface.SurfaceBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        Micro,
        Radiation,
        VelocityState,
        ScalarState,
        DiagnosticState,
        TimeSteppingController,
    ):

        self._Micro = Micro
        self._Radiation = Radiation
        self._TimeSteppingController = TimeSteppingController
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

        self.itimestep = 1
        nl = self._Grid.ngrid_local
        self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
        self._NOAH_MP = NoahMP.noahmp()

        self.yr = namelist["time"]["year"]
        date = datetime.datetime(
            namelist["time"]["year"],
            namelist["time"]["month"],
            namelist["time"]["day"],
            namelist["time"]["hour"],
        )

        self.julian = date.timetuple().tm_yday

        n_halo = self._Grid.n_halo

        # domain range
        ids = 1
        ide = self._Grid.nl[0] + 1
        jds = 1
        jde = self._Grid.nl[1] + 1
        kds = 1
        kde = self._Grid.nl[2] 

        ## further domain setting and other testcase-dependent setting
        # domain size & indices
        # memory range (used to define variable arrays)
        ims = ids
        ime = ide   -1
        jms = jds
        jme = jde   -1
        kms = kds
        kme = kde

        # tile range
        its = ids
        ite = ide   -1
        jts = jds
        jte = jde   -1
        kts = kds
        kte = kde

        iones_float_2d = np.ones((ime - ims + 1, jme - jms + 1), dtype=np.float64)
        iones_int_2d = np.ones((ime - ims + 1, jme - jms + 1), dtype=np.intc)

        self.vegfra = np.asfortranarray(
            np.ones_like(iones_float_2d) * 0.5
        )  # vegetation fraction, all = 0.5
        self.vegmax = np.asfortranarray(
            np.ones_like(iones_float_2d)
        )  # annual maximum veg fraction all = 1

        self.nsoil = 4
        self.nsnow = 3  # number of snow layers. set to 3 in the subroutine noahmplsm
        self.dzs = np.asfortranarray([0.01, 0.3, 0.6, 1])  # soil layer thickness
        dz = self._Grid.dx[2]

        # Container classes
        self._NoahMPvars = nom.NoahMPvars(
            ims, ime, jms, jme, self.nsoil, self.nsnow, self.dzs
        )
        self._NoahMPtoATM = nom.NoahMPtoATM(ims, ime, jms, jme)
        self._ATMtoNoahMP = nom.ATMtoNoahMP(ims, ime, jms, jme, kms, kme, dz)

        NMPvars = self._NoahMPvars
        NMP2ATM = self._NoahMPtoATM

        NMPvars.tslb.fill(298.5068)

        dx = self._Grid.dx[0]
        dy = self._Grid.dx[1]

        is_restart = False
        is_restart = False  # is restart or not
        allowed_to_read = (
            True  # True allows model to read land-use and soil parameters from tables
        )
        # i.e., call  SOIL_VEG_GEN_PARM( MMINLU, MMINSL )

        areaxy = np.asfortranarray(iones_float_2d) * dx * dy  # grid cell area [m2] -
        # no need to give actual value when dx, dy, msftx,msfty are given to noahmplsm; calculated in the module for iopt_run = 5
        msftx = np.asfortranarray(iones_float_2d)  # map factor
        msfty = np.asfortranarray(iones_float_2d)  # map factor

        self.IDVEG = 1  # dynamic vegetation (1 -> off ; 2 -> on) with opt_crs = 1
        self.IOPT_CRS = 1  # canopy stomatal resistance (1-> Ball-Berry; 2->Jarvis)
        self.IOPT_BTR = 1  # soil moisture factor for stomatal resistance (1-> Noah; 2-> CLM; 3-> SSiB)
        self.IOPT_RUN = (
            1  # runoff and groundwater (1->SIMGM; 2->SIMTOP; 3->Schaake96; 4->BATS)
        )
        self.IOPT_SFC = 1  # surface layer drag coeff (CH & CM) (1->M-O; 2->Chen97)
        self.IOPT_FRZ = 1  # supercooled liquid water (1-> NY06; 2->Koren99)
        self.IOPT_INF = 1  # frozen soil permeability (1-> NY06; 2->Koren99)
        self.IOPT_RAD = (
            1  # radiation transfer (1->gap=F(3D,cosz); 2->gap=0; 3->gap=1-Fveg)
        )
        self.IOPT_ALB = 1  # snow surface albedo (1->BATS; 2->CLASS)
        self.IOPT_SNF = 1  # rainfall & snowfall (1-Jordan91; 2->BATS; 3->Noah)
        self.IOPT_TBOT = 1  # lower boundary of soil temperature (1->zero-flux; 2->Noah)
        self.IOPT_STC = 1  # snow/soil temperature time scheme (for only layer 1)
        # 1 -> semi-implicit; 2 -> full implicit (original Noah))
        self.IZ0TLND = 0  # option of Chen adjustment of Czil
        #! it seems to be used; value of 1 uses Chen & Zhang (2009) to modify
        # a constant parameter (CZIL) in the expression for thermal roughness length

        self.ISICE = 15
        self.ISURBAN = 16
        self.ISWATER = np.intc(17)

        data_in = xr.open_dataset("sensitivity5_d01_static.nc")

        ISLTYP = np.asfortranarray(
            np.ones_like(iones_int_2d)
        )  # soil type, all 1 = sand
        IVGTYP = np.asfortranarray(
            np.ones_like(iones_int_2d) * 5
        )  # 5 = all grid points are grass

        self._RAINC_last = np.copy(self._Micro._RAINNC)

        print(np.shape(ISLTYP))

        lat_in = data_in["XLAT"].values[0, :, :]
        lon_in = data_in["XLONG"].values[0, :, :]
        ISLTYP = data_in["ISLTYP"].values[0, :, :]
        IVGTYP = data_in["IVGTYP"].values[0, :, :]
        landmask = data_in["LANDMASK"].values[0, :, :]
        xlandin = landmask.copy
        xlandin = np.where((landmask < 1), 2, landmask)
        xlandin = np.where((IVGTYP == self.ISWATER), 2, landmask)
        xlandin = np.where((ISLTYP == 14), 2, landmask)

        # print(np.shape(lat_in))

        # lon_grid, lat_grid = np.meshgrid(lon_in[0,:,:], lat_in[0,:,:])
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

        xland = interpolate.griddata(
            lon_lat,
            xlandin.flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="nearest",
        )

        self._ISLTYP = np.asfortranarray(
            ISLTYP[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]], dtype=np.intc
        )
        self._IVGTYP = np.asfortranarray(
            IVGTYP[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]], dtype=np.intc
        )
        self._xland = np.asfortranarray(
            xlandin[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]], dtype=np.intc
        )

        data_in.close()
        
        # Now provide initial conditions
        
        #########################################
        # First for temperature
        #
        #########################################
        UtilitiesParallel.print_root('Initializing Soil Temperature')
        soil_temp_in = xr.open_dataset('TSLB.nc')        

        tslb = soil_temp_in['TSLB'][0,:,:,:].values
        assert(np.shape(tslb)[0] == self.nsoil)
        lat_in = soil_temp_in["XLAT"].values[0, :, :]
        lon_in = soil_temp_in["XLONG"].values[0, :, :]
        lon_lat = (lon_in.flatten(), lat_in.flatten())
        for k in range(self.nsoil):
            interp_field = interpolate.griddata(
            lon_lat,
            tslb[k,:,:].flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="nearest",
            )
            
            NMPvars.tslb[:,k,:] = np.asfortranarray(interp_field[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])
        soil_temp_in.close()
        ##########################################
        # Now for soil moisture
        #
        ##########################################
        UtilitiesParallel.print_root('Initializing Soil Moisture')
        soil_mois_in = xr.open_dataset('SMOIS.nc')
        SMOIS = soil_mois_in['SMOIS'][0,:,:,:].values
        assert(np.shape(SMOIS)[0] == self.nsoil)
        lat_in = soil_mois_in["XLAT"].values[0, :, :]
        lon_in = soil_mois_in["XLONG"].values[0, :, :]
        lon_lat = (lon_in.flatten(), lat_in.flatten())

        for k in range(self.nsoil):
            interp_field = interpolate.griddata(
            lon_lat,
            SMOIS[k,:,:].flatten(),
            (self._Grid.lon_local, self._Grid.lat_local),
            method="nearest",
            )
            
            NMPvars.smois[:,k,:] = np.asfortranarray(interp_field[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]])

        soil_mois_in.close()


        for t in [NMPvars.tsk, NMPvars.tmn, NMPvars.tvxy, NMPvars.tgxy, NMPvars.t2mvxy,  NMPvars.t2mbxy, NMPvars.tahxy] :
            t.fill(294.969329833984)


        self._NOAH_MP.init(
            ids,
            ide,
            jds,
            jde,
            kds,
            kde,
            ims,
            ime,
            jms,
            jme,
            kms,
            kme,
            its,
            ite,
            jts,
            jte,
            kts,
            kte,
            self.nsoil,
            self.dzs,
            NMPvars.tsk,
            NMPvars.isnowxy,
            NMPvars.snow,
            NMPvars.snowh,
            NMPvars.canwat,
            self.ISICE,
            self.ISWATER,
            self.ISURBAN,
            self._ISLTYP,
            self._IVGTYP,
            NMPvars.xice,
            self.IOPT_RUN,
            is_restart,
            allowed_to_read,
            NMPvars.smois,
            NMPvars.sh2o,
            NMPvars.tslb,
            NMPvars.tmn,
            NMPvars.zsnsoxy,
            NMPvars.tsnoxy,
            NMPvars.snicexy,
            NMPvars.snliqxy,
            NMPvars.sneqvoxy,
            NMPvars.alboldxy,
            NMPvars.qsnowxy,
            NMPvars.tvxy,
            NMPvars.tgxy,
            NMPvars.canicexy,
            NMPvars.canliqxy,
            NMPvars.eahxy,
            NMPvars.tahxy,
            NMP2ATM.cmxy,
            NMP2ATM.chxy,
            NMPvars.fwetxy,
            NMPvars.wslakexy,
            NMPvars.zwtxy,
            NMPvars.waxy,
            NMPvars.wtxy,
            NMPvars.lfmassxy,
            NMPvars.rtmassxy,
            NMPvars.stmassxy,
            NMPvars.woodxy,
            NMPvars.stblcpxy,
            NMPvars.fastcpxy,
            NMPvars.xsaixy,
            NMPvars.t2mvxy,
            NMPvars.t2mbxy,
            NMP2ATM.chstarxy,
            NMPvars.smoiseq,
            NMPvars.smcwtdxy,
            NMPvars.deeprechxy,
            NMPvars.rechxy,
            NMPvars.qrfsxy,
            NMPvars.qspringsxy,
            NMPvars.qslatxy,
            areaxy,
            dx,
            dy,
            msftx,
            msfty,
            NMPvars.fdepthxy,
            NMPvars.ht,
            NMPvars.riverbedxy,
            NMPvars.eqzwt,
            NMPvars.rivercondxy,
            NMPvars.pexpxy,
        )

        self.T_skin = NMPvars.tsk
        self.albedo = self._NoahMPtoATM.albedo
        self.emiss =  self._NoahMPtoATM.emiss
        return

    def initialize(self):
        return super().initialize()

    def update(self, dt=None):

        nh = self._Grid.n_halo
        dxi2 = self._Grid.dxi[2]
        z_edge = self._Grid.z_edge_global

        # domain range
        ids = 1
        ide = self._Grid.nl[0] + 1
        jds = 1
        jde = self._Grid.nl[1] + 1
        kds = 1
        kde = self._Grid.nl[2] 

        ## further domain setting and other testcase-dependent setting
        # domain size & indices
        # memory range (used to define variable arrays)
        ims = ids
        ime = ide   -1
        jms = jds
        jme = jde   -1
        kms = kds
        kme = kde

        # tile range
        its = ids
        ite = ide   -1
        jts = jds
        jte = jde   -1
        kts = kds
        kte = kde

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
        dflux_lw = self._DiagnosticState.get_field("dflux_lw")
        dflux_sw = self._DiagnosticState.get_field("dflux_sw")
        #import pylab as plt
        #plt.contourf(dflux_lw[:,:,5])
        #plt.colorbar()
        #plt.show()

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

        Surface_impl.compute_windspeed_sfc(
            usfc, vsfc, self._Ref.u0, self._Ref.v0, self.gustiness, self._windspeed_sfc
        )

        ATM2NMP = self._ATMtoNoahMP
        NMP2ATM = self._NoahMPtoATM
        NMPvars = self._NoahMPvars
        
        #Compute precipitation
        np.subtract(self._Micro._RAINNC, self._RAINC_last, ATM2NMP.rainbl)
        self._RAINC_last = np.copy(self._RAINC_last)

        dt = self._TimeSteppingController.dt
        if self.itimestep == 1:
            dt = 0.0
        
        dx = self._Grid.dx[0]
        xice_thres = 0.5  # fraction of grid determining seaice

        n_halo = self._Grid.n_halo
        xlat = np.asfortranarray(
            self._Grid.lat_local[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]]
        )
        xlon = np.asfortranarray(
            self._Grid.lon_local[n_halo[0] : -n_halo[0], n_halo[1] : -n_halo[1]]
        )
        coszin = cos_sza(
            self.julian, self._TimeSteppingController.time // 86400.0, xlat, xlon
        )

        print(coszin)
        import sys

        for wrf_array, pinacles_array in zip(
            [ATM2NMP.t3d, ATM2NMP.qv3d, ATM2NMP.u_phy, ATM2NMP.v_phy], [T, qv, u, v]
        ):
            to_wrf_order(n_halo, pinacles_array, wrf_array)


        ATM2NMP.glw[:,:] = np.asfortranarray(dflux_lw[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1], n_halo[2]])
        ATM2NMP.swdown[:,:] = np.asfortranarray(dflux_sw[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1], n_halo[2]])

        ATM2NMP.p8w3d[:,:,:] = self._Ref.p0[:][np.newaxis,n_halo[2]:-n_halo[2],np.newaxis]
        
        if self.itimestep %2 == 0 or self.itimestep==1:
            self._NOAH_MP.noahmplsm(
                self.itimestep,
                self.yr,
                self.julian,
                coszin,
                xlat,
                ATM2NMP.dz8w,
                dt,
                self.dzs,
                self.nsoil,
                dx,
                self._IVGTYP,
                self._ISLTYP,
                self.vegfra,
                self.vegmax,
                NMPvars.tmn,
                self._xland,
                NMPvars.xice,
                xice_thres,
                self.ISICE,
                self.ISURBAN,
                self.IDVEG,
                self.IOPT_CRS,
                self.IOPT_BTR,
                self.IOPT_RUN,
                self.IOPT_SFC,
                self.IOPT_FRZ,
                self.IOPT_INF,
                self.IOPT_RAD,
                self.IOPT_ALB,
                self.IOPT_SNF,
                self.IOPT_TBOT,
                self.IOPT_STC,
                self.IZ0TLND,
                ATM2NMP.t3d,
                ATM2NMP.qv3d,
                ATM2NMP.u_phy + self._Ref.u0,
                ATM2NMP.v_phy + self._Ref.v0,
                ATM2NMP.swdown,
                ATM2NMP.glw,
                ATM2NMP.p8w3d,
                ATM2NMP.rainbl,
                NMPvars.tsk,
                NMP2ATM.hfx,
                NMP2ATM.qfx,
                NMP2ATM.lh,
                NMPvars.grdflx,
                NMPvars.smstav,
                NMPvars.smstot,
                NMPvars.sfcrunoff,
                NMPvars.udrunoff,
                NMP2ATM.albedo,
                NMPvars.snowc,
                NMPvars.smois,
                NMPvars.sh2o,
                NMPvars.tslb,
                NMPvars.snow,
                NMPvars.snowh,
                NMPvars.canwat,
                NMPvars.acsnom,
                NMPvars.acsnow,
                NMP2ATM.emiss,
                NMPvars.qsfc,
                NMPvars.isnowxy,
                NMPvars.tvxy,
                NMPvars.tgxy,
                NMPvars.canicexy,
                NMPvars.canliqxy,
                NMPvars.eahxy,
                NMPvars.tahxy,
                NMP2ATM.cmxy,
                NMP2ATM.chxy,
                NMPvars.fwetxy,
                NMPvars.sneqvoxy,
                NMPvars.alboldxy,
                NMPvars.qsnowxy,
                NMPvars.wslakexy,
                NMPvars.zwtxy,
                NMPvars.waxy,
                NMPvars.wtxy,
                NMPvars.tsnoxy,
                NMPvars.zsnsoxy,
                NMPvars.snicexy,
                NMPvars.snliqxy,
                NMPvars.lfmassxy,
                NMPvars.rtmassxy,
                NMPvars.stmassxy,
                NMPvars.woodxy,
                NMPvars.stblcpxy,
                NMPvars.fastcpxy,
                NMPvars.xlaixy,
                NMPvars.xsaixy,
                NMPvars.taussxy,
                NMPvars.smoiseq,
                NMPvars.smcwtdxy,
                NMPvars.deeprechxy,
                NMPvars.rechxy,
                NMPvars.t2mvxy,
                NMPvars.t2mbxy,
                NMPvars.q2mvxy,
                NMPvars.q2mbxy,
                NMP2ATM.tradxy,
                NMP2ATM.neexy,
                NMPvars.gppxy,
                NMPvars.nppxy,
                NMPvars.fvegxy,
                NMPvars.runsfxy,
                NMPvars.runsbxy,
                NMPvars.ecanxy,
                NMPvars.edirxy,
                NMPvars.etranxy,
                NMPvars.fsaxy,
                NMPvars.firaxy,
                NMPvars.aparxy,
                NMPvars.psnxy,
                NMPvars.savxy,
                NMPvars.sagxy,
                NMPvars.rssunxy,
                NMPvars.rsshaxy,
                NMPvars.bgapxy,
                NMPvars.wgapxy,
                NMPvars.tgvxy,
                NMPvars.tgbxy,
                NMPvars.chvxy,
                NMPvars.chbxy,
                NMPvars.shgxy,
                NMPvars.shcxy,
                NMPvars.shbxy,
                NMPvars.evgxy,
                NMPvars.evbxy,
                NMPvars.ghvxy,
                NMPvars.ghbxy,
                NMPvars.irgxy,
                NMPvars.ircxy,
                NMPvars.irbxy,
                NMPvars.trxy,
                NMPvars.evcxy,
                NMPvars.chleafxy,
                NMPvars.chucxy,
                NMPvars.chv2xy,
                NMPvars.chb2xy,
                ids,
                ide,
                jds,
                jde,
                kds,
                kde,
                ims,
                ime,
                jms,
                jme,
                kms,
                kme,
                its,
                ite,
                jts,
                jte,
                kts,
                kte,
            )

    
        self.T_skin = NMPvars.tsk
        self.albedo = self._NoahMPtoATM.albedo
        self.emiss =  self._NoahMPtoATM.emiss

        #import pylab as plt
        #if self.itimestep % 2 == 0:
        #    plt.figure(1,figsize=(21,10))
        #    plt.subplot(121)
        #    plt.title('Latent Heat Flux')
        #    plt.contourf(NMP2ATM.lh,64)
        #    plt.colorbar()
        #    plt.subplot(122)
        #    plt.title('Sensible Heat Flux')
        #    plt.contourf(NMP2ATM.hfx,64)
        #    plt.colorbar()
        #    plt.savefig('./plot_figs/' + str(self.itimestep + 10000000) + '.png')
        #    plt.close()

        #sys.exit()

        



        #self._tflx = -self._ch * self._windspeed_sfc * (Ssfc - self._TSKIN)
        #self._qvflx = -self._cq * self._windspeed_sfc * (qvsfc - self._qv0)

        self._cm = np.pad(NMP2ATM.cmxy, pad_width=(n_halo[0], n_halo[1]),mode='edge')

        self._taux_sfc = -self._cm * self._windspeed_sfc * (usfc + self._Ref.u0)
        self._tauy_sfc = -self._cm * self._windspeed_sfc * (vsfc + self._Ref.v0)

        self._qvflx = np.pad(NMP2ATM.qfx, pad_width=(n_halo[0], n_halo[1]), mode='edge')
        self._tflx = np.pad(NMP2ATM.hfx, pad_width=(n_halo[0], n_halo[1]), mode='edge')

        #Apply the surface fluxes
        if self.itimestep > 1:
            #Surface_impl.iles_surface_flux_application_u(
            #    10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._taux_sfc, ut
            #)

            #Surface_impl.iles_surface_flux_application_v(
            #    10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tauy_sfc, vt
            #)

            Surface_impl.iles_surface_flux_application(
                10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._tflx* alpha0_edge[nh[2] - 1] / parameters.CPD, st
            )

            Surface_impl.iles_surface_flux_application(
                10, z_edge, dxi2, nh, alpha0, alpha0_edge, 10, self._qvflx, qvt
            )
        self.itimestep += 1
        return super().update()

    def io_initialize(self, rt_grp):
        return super().io_initialize(rt_grp)

    def io_update(self, rt_grp):
        return super().io_update(rt_grp)

    def io_fields2d_update(self, nc_grp):
        lhf = nc_grp.createVariable("latent_heat_flux", np.double, dimensions=("X", "Y",))
        lhf[:,:] = self._NoahMPtoATM.lh

        shf = nc_grp.createVariable("sensible_heat_flux", np.double, dimensions=("X", "Y",))
        shf[:,:] = self._NoahMPtoATM.hfx

        tskin = nc_grp.createVariable("T_skin", np.double, dimensions=("X", "Y",))
        tskin[:,:] = self.T_skin

        nc_grp.sync()
        return