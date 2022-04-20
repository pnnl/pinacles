from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
    water_path_lasso,
)
from pinacles.externals.sam_m2005_ma_wrapper import m2005_ma_via_cffi
from pinacles import UtilitiesParallel
from pinacles import parameters
from mpi4py import MPI
import numba
import numpy as np

class Micro_M2005_MA(MicrophysicsBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):

        MicrophysicsBase.__init__(
            self,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )

        self._m2005_ma_cffi = m2005_ma_via_cffi.M2005_MA()
        
        nhalo = self._Grid.n_halo
        self._sam_dims = np.array([
            self._Grid.ngrid_local[0] - 2 * nhalo[0],
            self._Grid.ngrid_local[1] - 2 * nhalo[1],
            self._Grid.ngrid_local[2] - 2 * nhalo[2],
            0,
        ], order="F", dtype=np.intc)
        nz    = self._sam_dims[2]+1
        #print("SAM_Micro_M2005_MA",self._sam_dims)
        
        self._iqarray = np.zeros((22,), order="F", dtype=np.intc)
        self._masterproc = False
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            self._masterproc = True
            
        self._tlatqi = np.zeros(
            (nz), order="F", dtype=np.double
        )
                
        try:
            mp_flags = namelist["microphysics"]["flags"]
            docloud = mp_flags["docloud"]
            doprecip = mp_flags["doprecip"]
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with custom flags")
        except:
            docloud = True
            doprecip = True
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with default flags")
                
        self._m2005_ma_cffi.setparm(
        self._sam_dims,
        self._iqarray,
        docloud,
        doprecip,
        self._masterproc,
        self._tlatqi,
        )
        
        self._micro_vars = ["" for x in range(self._sam_dims[3]+1)]
        # print("SAM_Micro_M2005_MA",self._iqarray)
        # print(self._iqarray[0])
        
        if(self._iqarray[0]<100):
            iqv = self._iqarray[0]   # total water (vapor + cloud liq) mass mixing ratio [kg H2O / kg dry air]
            self._micro_vars[iqv] = "qv"
            self._ScalarState.add_variable(
                "qv",
                long_name="water vapor mixing ratio",
                units="g kg^{-1}",
                latex_name="q_v",
                limit=True,
            )
        
        if(self._iqarray[1]<100):            
            iqcl = self._iqarray[1] # cloud water mass mixing ratio [kg H2O / kg dry air]
            self._micro_vars[iqcl] = "qc"
            self._ScalarState.add_variable(
                "qc",
                long_name="cloud water mixing ratio",
                units="g kg^{-1}",
                latex_name="q_c",
                limit=True,
            )
            
        if(self._iqarray[2]<100):
            iqci = self._iqarray[2]
            self._micro_vars[iqci] = "qci"
            self._ScalarState.add_variable(
                "qci",
                long_name="cloud ice mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{i}",
                limit=True,
            )
            
        if(self._iqarray[3]<100):
            iqr = self._iqarray[3]
            self._micro_vars[iqr] = "qr"
            self._ScalarState.add_variable(
                "qr",
                long_name="rain water mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{r}",
                limit=True,
            )
            
        if(self._iqarray[4]<100):
            iqs = self._iqarray[4]
            self._micro_vars[iqs] = "qs"
            self._ScalarState.add_variable(
                "qs",
                long_name="snow mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{s}",
                limit=True,
            )
            
        if(self._iqarray[5]<100):
            iqg = self._iqarray[5]
            self._micro_vars[iqg] = "qg"
            self._ScalarState.add_variable(
                "qg",
                long_name="graupel mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{g}",
                limit=True,
            )
            
        if(self._iqarray[6]<100):
            incl = self._iqarray[6]
            self._micro_vars[incl] = "qnc"
            self._ScalarState.add_variable(
                "qnc",
                long_name="cloud number concentration",
                units="# cm^{-3}",
                latex_name="q_{nc}",
                limit=True,
            )
            
        if(self._iqarray[7]<100):
            inci = self._iqarray[7]
            self._micro_vars[inci] = "qnci"
            self._ScalarState.add_variable(
                "qnci",
                long_name="cloud ice number concentration",
                units="# cm^{-3}",
                latex_name="q_{nci}",
                limit=True,
            )
            
        if(self._iqarray[8]<100):
            inr = self._iqarray[8]
            self._micro_vars[inr] = "qnr"
            self._ScalarState.add_variable(
                "qnr",
                long_name="rain number concentration",
                units="# cm^{-3}",
                latex_name="q_{nr}",
                limit=True,
            )
            
        if(self._iqarray[9]<100):
            ins = self._iqarray[9]
            self._micro_vars[ins] = "qns"
            self._ScalarState.add_variable(
                "qns",
                long_name="snow number concentration",
                units="# cm^{-3}",
                latex_name="q_{ns}",
                limit=True,
            )
            
        if(self._iqarray[10]<100):
            ing = self._iqarray[10]
            self._micro_vars[ing] = "qng"
            self._ScalarState.add_variable(
                "qng",
                long_name="graupel number concentration",
                units="# cm^{-3}",
                latex_name="q_{ng}",
                limit=True,
            )
            
        if(self._iqarray[11]<100):
            iqad = self._iqarray[11]
            self._micro_vars[iqad] = "qad"
            self._ScalarState.add_variable(
                "qad",
                long_name="dry aerosol mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{ad}",
                limit=True,
            )
            
        if(self._iqarray[12]<100):
            iqaw = self._iqarray[12]
            self._micro_vars[iqaw] = "qaw"
            self._ScalarState.add_variable(
                "qaw",
                long_name="wet aerosol mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{aw}",
                limit=True,
            )
            
        if(self._iqarray[13]<100):
            iqar = self._iqarray[13]
            self._micro_vars[iqar] = "qar"
            self._ScalarState.add_variable(
                "qar",
                long_name="wet aerosol mass mixing ratio in rain",
                units="g kg^{-1}",
                latex_name="q_{ar}",
                limit=True,
            )
            
        if(self._iqarray[14]<100):
            inad = self._iqarray[14]
            self._micro_vars[inad] = "qnad"
            self._ScalarState.add_variable(
                "qnad",
                long_name="dry aerosol number concentration",
                units="# cm^{-3}",
                latex_name="q_{nad}",
                limit=True,
            )
            
        if(self._iqarray[15]<100):
            iqad2 = self._iqarray[15]
            self._micro_vars[iqad2] = "qad2"
            self._ScalarState.add_variable(
                "qad2",
                long_name="aitken mode mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{ad2}",
                limit=True,
            )
            
        if(self._iqarray[16]<100):
            inad2 = self._iqarray[16]
            self._micro_vars[inad2] = "qnad2"
            self._ScalarState.add_variable(
                "qnad2",
                long_name="aitken mode mass number concentration",
                units="# cm^{-3}",
                latex_name="q_{nad2}",
                limit=True,
            )
            
        if(self._iqarray[17]<100):
            igas1 = self._iqarray[17]
            
        if(self._iqarray[18]<100):
            iDMS = self._iqarray[18]
            self._micro_vars[iDMS] = "DMS"
            self._ScalarState.add_variable(
                "DMS",
                long_name="DMS GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="DMS",
                limit=True,
            )
            
        if(self._iqarray[19]<100):
            iSO2 = self._iqarray[19]
            self._micro_vars[iSO2] = "SO2"
            self._ScalarState.add_variable(
                "SO2",
                long_name="SO2 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="SO2",
                limit=True,
            )
            
        if(self._iqarray[20]<100):
            iH2SO4 = self._iqarray[20]
            self._micro_vars[iH2SO4] = "H2SO4"
            self._ScalarState.add_variable(
                "H2SO4",
                long_name="H2SO4 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="H2SO4",
                limit=True,
            )
        
                
        # Allocate microphysical/thermodynamic variables

        nx    = self._Grid.ngrid_local[0] - 2 * nhalo[0]
        ny    = self._Grid.ngrid_local[1] - 2 * nhalo[1]
        nzm   = self._Grid.ngrid_local[2] - 2 * nhalo[2]
        nz    = nzm + 1
        nmicrofields = self._sam_dims[3]
        nx_gl = self._Grid.n[0]
        ny_gl = self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()
        nsubdomains_x = self._Grid.subcomms[0].Get_size()
        nsubdomains_y = self._Grid.subcomms[1].Get_size()
                
        self._microfield = np.empty(
            (nx, ny, nzm, nmicrofields),
            dtype=np.double,
            order="F",
        )
        
        if (namelist["restart"]["restart_simulation"] == False):
            nrestart = 0
        else:
            nrestart = 1
                
        self._itimestep = 0
                        
        self._nrainy = 0.0
        self._nrmn = 0.0
        self._ncmn = 0.0
        self._total_water_prec = 0.0
                
        self._fluxbq = np.zeros(
            (nx, ny), order="F", dtype=np.double
        )
        self._fluxtq = np.zeros_like(self._fluxbq)  # fluxes not being used anywhere
        self._u10arr = np.zeros_like(self._fluxbq)
        self._precsfc = np.zeros_like(self._fluxbq)
        self._prec_xy = np.zeros_like(self._fluxbq)
                
        self._tlat = np.zeros(
            (nz), dtype=np.double
        )
        self._precflux = np.asfortranarray(np.zeros_like(self._tlat))
        self._qpfall = np.zeros_like(self._tlat)
                
        z = self._Grid.z_global
        zi = self._Grid.z_edge_global
        
        dx = self._Grid.dx[0]
        dz = self._Grid.dx[2]

        p0 = self._Ref.p0
        rho0 = self._Ref.rho0
        tabs0 = self._Ref.T0
        rhow = self._Ref.rho0_edge
        t0 = tabs0
        qv0 = np.zeros_like(p0)
        self._q0 = np.zeros_like(p0)
        
        dostatis = False
        do_chunked_energy_budgets = False
        donudging_aerosol = False
        
        self._T = np.empty(
            (nx, ny, nzm),
            dtype=np.double,
            order="F",
        )
        #Temp = self._DiagnosticState.get_field("T")

        # For diagnostic variables

        self._diag_3d_vars = [
            "qtot_sed",
            "qice_sed",
            "Nacc_sct",
            "Nait_sct",
            "Nait2a_ct",
            "Mait2a_ct",
            "relhum",
            "diag_effc_3d",
            "diag_effr_3d",
            "diag_effi_3d",
            "diag_effs_3d",
            "dBZCLRAD",
            "NARG1",
            "NARG2",
            "NACTRATE",
            "QACTRATE",
            "NACTDIFF",
            "NATRANS",
            "QATRANS",
            "ISACT",
            "DC1",
            "DC2",
            "DG1",
            "DG2",
            "SSPK",
            "NCPOSLM",
            "NCNEGLM",
        ]

        qc_units = "kg kg-1"
        qc_tend_units = "kg kg-1 s-1"
        n_tend_units = "kg-1"
        n_rate_units = "kg-1 s-1"
        diag_3d_long_names = {
            "qtot_sed" : ("total liquid water sedimentation", ""),
            "qice_sed" : ("ice sedimentation", ""),
            "Nacc_sct" : ("Nacc self coagulation tendency", ""),
            "Nait_sct" : ("Nait self coagulation tendency", ""),
            "Nait2a_ct" : ("Nait2acc coagulation tendency", ""),
            "Mait2a_ct" : ("Mait2acc coagulation tendency", ""),
            "relhum" : ("relative humidity", ""),
            "diag_effc_3d" : ("cloud droplet effective radius", "m"),
            "diag_effr_3d" : ("rain droplet effective radius", "m"),
            "diag_effi_3d" : ("cloud ice effective radius", "m"),
            "diag_effs_3d" : ("snow effective radius", "m"),
            "dBZCLRAD": ("Cloud Radar Reflectivity", "dBZ"),
            "NARG1": ("A-R&G. Activated Number Accumulation", n_tend_units),
            "NARG2": ("A-R&G Activated Number Aitken", n_tend_units),
            "NACTRATE": ("Activation Rate Accumulation Aerosol", n_rate_units),
            "QACTRATE": ("Activation Rate Accumulation Aerosol", qc_tend_units),
            "NACTDIFF": ("Difference from A-R&G activ number and and old cloud number", n_tend_units),
            "NATRANS": ("Aitken to Accum Transfer Number", n_tend_units),
            "QATRANS": ("Aitken to Accum Transfer Mass", qc_units),
            "ISACT": ("ARG Activation run on particular point","None"),
            "DC1": ("Critical activation diameter mode 1", "m"),
            "DC2": ("Critical activation diameter mode 2", "m"),
            "DG1": ("Modal diameter 1", "m"),
            "DG2": ("Modal diameter 2", "m"),
            "SSPK": ("Peak activation supersaturation ARG", "fraction"),
            "NCPOSLM": ("Change in NC due to positive rate limiter", n_rate_units),
            "NCNEGLM": ("Mass Activation Rate Aitken Aerosol", n_rate_units),
        }

        self.n_diag_3d = len(self._diag_3d_vars)

        self._diag_3d = np.empty(
            (nx, ny, nzm, self.n_diag_3d),
            dtype=np.double,
            order="F",
        )
        
        # Add a diagnostic variable for each of the process rates
        for i, vn in enumerate(self._diag_3d_vars):
            self._DiagnosticState.add_variable(
                "m2005_ma_" + vn,
                long_name=diag_3d_long_names[vn][0],
                units=diag_3d_long_names[vn][1],
            )
            
        #print("SAM Before calling init")
        
        time = self._TimeSteppingController.time        
              
        self._m2005_ma_cffi.init(
            nx,
            ny,
            nzm,
            nmicrofields,
            nx_gl, 
            ny_gl,
            my_rank, 
            nsubdomains_x, 
            nsubdomains_y, 
            nrestart,
            self._T,
            self._microfield,
            self.n_diag_3d,
            self._diag_3d,
            z, 
            p0,
            rho0, 
            tabs0, 
            zi, 
            rhow, 
            t0, 
            dx,
            dz,
            time,
            self._q0, 
            qv0, 
            dostatis, 
            do_chunked_energy_budgets, 
            donudging_aerosol,
            parameters.LV, 
            parameters.LS, 
            parameters.CPD, 
            parameters.RD, 
            parameters.RV, 
            parameters.G,
        )
        
        # print("Init succesfully called",iqv)
        
        qv = self._microfield[:,:,:,iqv]
        
        self._Timers.add_timer("MicroM2005_MA_update")
        return

    def update(self):
        self._Timers.start_timer("MicroM2005_MA_update")
        
        # print("SAM_Micro_M2005_MA",self._sam_dims)
        
        # Get variables from the model state
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        w = self._VelocityState.get_field("w")
        
        nhalo = self._Grid.n_halo
        th_3d = np.asfortranarray(T[nhalo[0]:self._Grid.ngrid_local[0] - nhalo[0],nhalo[1]:self._Grid.ngrid_local[1] - nhalo[1],nhalo[2]:self._Grid.ngrid_local[2] - nhalo[2]])
        s_3d = np.asfortranarray(s[nhalo[0]:self._Grid.ngrid_local[0] - nhalo[0],nhalo[1]:self._Grid.ngrid_local[1] - nhalo[1],nhalo[2]:self._Grid.ngrid_local[2] - nhalo[2]])
        w_3d = np.asfortranarray(w[nhalo[0]:self._Grid.ngrid_local[0] - nhalo[0],nhalo[1]:self._Grid.ngrid_local[1] - nhalo[1],nhalo[2] - 1:self._Grid.ngrid_local[2] - nhalo[2]])
                
        dt = self._TimeSteppingController.dt
        time = self._TimeSteppingController.time
        
        icycle = 1 
        ncycle = 1
        nsaveMSE = 1
        nstat = 1 
        nstatis = 1 
        nz = self._sam_dims[2] + 1
        # nstep = self._TimeSteppingController._n_timesteps
        
        # print("SAM Before calling main")
        
        self._m2005_ma_cffi.update(
            self._sam_dims[0],
            self._sam_dims[1],
            self._sam_dims[2],
            nz,
            self._sam_dims[3],
            self.n_diag_3d,
            icycle, 
            ncycle, 
            nsaveMSE, 
            nstat, 
            nstatis, 
            self._itimestep,
            th_3d,
            s_3d,
            w_3d,
            self._microfield,
            self._diag_3d,
            self._nrainy,
            self._nrmn,
            self._ncmn,
            self._total_water_prec,
            self._tlat,
            self._fluxbq,
            self._fluxtq,
            self._u10arr,
            self._precflux,
            self._qpfall,
            self._precsfc,
            self._prec_xy,
            dt,
            time,
        )
        
        print("SAM after main")
#        qv = self._ScalarState.getfield("qv")
#        qv = self._microfield(1:nx,1:ny,1:nz,iqv)
#        
#        if(not dototalwater):
#            self._microfield(:,:,:,iqcl) =   self._ScalarState.getfield("qc")
#
#        if(dopredictNc):
#            self._microfield(:,:,:,incl) =   self._ScalarState.getfield("qnc")
#
#        if(doprogaerosol):
#            self._microfield(:,:,:,iqad)  =  self._ScalarState.getfield("qad")
#            self._microfield(:,:,:,iqad2) =  self._ScalarState.getfield("qad2")
#            self._microfield(:,:,:,inad)  =  self._ScalarState.getfield("qnad")
#            self._microfield(:,:,:,inad2) =  self._ScalarState.getfield("qnad2")
#            self._microfield(:,:,:,iqaw)  =  self._ScalarState.getfield("qaw")
#            self._microfield(:,:,:,iqgas) =  self._ScalarState.getfield("qgas")
#
#        if(doprecip):
#            self._microfield(:,:,:,iqr)   =  self._ScalarState.getfield("qr")
#            self._microfield(:,:,:,iqnr)  =  self._ScalarState.getfield("qnr")
#
#            if(doprogaerosol):
#                self._microfield(:,:,:,iqar) =   self._ScalarState.getfield("qar")
#
#        if(doicemicro):
#            self._microfield(:,:,:,iqci)  =  self._ScalarState.getfield("qci")
#            self._microfield(:,:,:,iqnci) =  self._ScalarState.getfield("qnci")
#            self._microfield(:,:,:,iqs)   =  self._ScalarState.getfield("qs")
#            self._microfield(:,:,:,iqns)  =  self._ScalarState.getfield("qns")
#
#        if(dograupel):
#            self._microfield(:,:,:,iqg)   =  self._ScalarState.getfield("qg")
#            self._microfield(:,:,:,iqng)  =  self._ScalarState.getfield("qng")                

        # Compute and apply sedimentation sources of static energy
        np.multiply(qtot_sed, parameters.LV / parameters.CPD, out=s_tend_liq_sed)
        np.multiply(qice_sed, parameters.LS / parameters.CPD, out=s_tend_ice_sed)

        # Convert sedimentation sources to units of tendency
        np.multiply(qtot_sed, 1.0 / self._TimeSteppingController.dt, out=liq_sed)
        np.multiply(qice_sed, 1.0 / self._TimeSteppingController.dt, out=ice_sed)

        self._itimestep += 1

        self._Timers.end_timer("MicroM2005_MA_update")
        return

    def io_initialize(self, nc_grp):
        timeseries_grp = nc_grp["timeseries"]
        profiles_grp = nc_grp["profiles"]

        v = timeseries_grp.createVariable("CF", np.double, dimensions=("time",))
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        v = timeseries_grp.createVariable("RF", np.double, dimensions=("time",))
        v.long_name = "Rain Fraction"
        v.standard_name = "RF"
        v.units = ""

        v = timeseries_grp.createVariable("LWP", np.double, dimensions=("time",))
        v.long_name = "Liquid Water Path"
        v.standard_name = "LWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("LWP_LASSO", np.double, dimensions=("time",))
        v.long_name = "LASSO Liquid Water Path"
        v.standard_name = "LWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("RWP", np.double, dimensions=("time",))
        v.long_name = "Rain Water Path"
        v.standard_name = "RWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("VWP", np.double, dimensions=("time",))
        v.long_name = "Water Vapor Path"
        v.standard_name = "VWP"
        v.units = "kg/m^2"

        v = timeseries_grp.createVariable("precsfc", np.double, dimensions=("time",))
        v.long_name = "accumulated surface precip"
        v.units = "mm"
        v.latex_name = "precsfc"

        v = timeseries_grp.createVariable("prec_xy", np.double, dimensions=("time",))
        v.long_name = "one time step accumulated surface precip"
        v.units = "mm"
        v.latex_name = "prec_xy"

        # Now add cloud fraction and rain fraction profiles
        v = profiles_grp.createVariable(
            "CF",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        v = profiles_grp.createVariable(
            "RF",
            np.double,
            dimensions=(
                "time",
                "z",
            ),
        )
        v.long_name = "Rain Fraction"
        v.standard_name = "RF"
        v.units = ""

        return

    def io_update(self, nc_grp):

        my_rank = MPI.COMM_WORLD.Get_rank()

        n_halo = self._Grid.n_halo
        dz = self._Grid.dx[2]
        rho0 = self._Ref.rho0
        npts = self._Grid.n[0] * self._Grid.n[1]

        qc = self._ScalarState.get_field("qc")
        qv = self._ScalarState.get_field("qv")
        qr = self._ScalarState.get_field("qr")

        # First compute liqud water path
        lwp = water_path(n_halo, dz, npts, rho0, qc)
        lwp = UtilitiesParallel.ScalarAllReduce(lwp)

        # First compute liqud water path
        lwp_lasso, npts_lasso = water_path_lasso(n_halo, dz, rho0, qc + qr)
        lwp_lasso = UtilitiesParallel.ScalarAllReduce(lwp_lasso)
        npts_lasso = UtilitiesParallel.ScalarAllReduce(npts_lasso)
        if npts_lasso > 0:
            lwp_lasso /= npts_lasso

        rwp = water_path(n_halo, dz, npts, rho0, qr)
        rwp = UtilitiesParallel.ScalarAllReduce(rwp)

        vwp = water_path(n_halo, dz, npts, rho0, qv)
        vwp = UtilitiesParallel.ScalarAllReduce(vwp)

        # Compute cloud and rain fraction
        cf = water_fraction(n_halo, npts, qc, threshold=1e-5)
        cf = UtilitiesParallel.ScalarAllReduce(cf)

        cf_prof = water_fraction_profile(n_halo, npts, qc, threshold=1e-5)
        cf_prof = UtilitiesParallel.ScalarAllReduce(cf_prof)

        rf = water_fraction(n_halo, npts, qr)
        rf = UtilitiesParallel.ScalarAllReduce(rf)

        precsfc = np.sum(self._precsfc) / npts
        precsfc = UtilitiesParallel.ScalarAllReduce(precsfc)

        rf_prof = water_fraction_profile(n_halo, npts, qr, threshold=1e-5)
        rf_prof = UtilitiesParallel.ScalarAllReduce(rf_prof)

        prec_xy = np.sum(self._prec_xy) / npts
        prec_xy = UtilitiesParallel.ScalarAllReduce(prec_xy)

        if my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]
            profiles_grp = nc_grp["profiles"]

            timeseries_grp["CF"][-1] = cf
            timeseries_grp["RF"][-1] = rf
            timeseries_grp["LWP"][-1] = lwp
            timeseries_grp["LWP_LASSO"][-1] = lwp_lasso
            timeseries_grp["RWP"][-1] = rwp
            timeseries_grp["VWP"][-1] = vwp

            timeseries_grp["precsfc"][-1] = precsfc
            timeseries_grp["prec_xy"][-1] = prec_xy

            profiles_grp["CF"][-1, :] = cf_prof[n_halo[2] : -n_halo[2]]
            profiles_grp["RF"][-1, :] = rf_prof[n_halo[2] : -n_halo[2]]

        return

    @staticmethod
    @numba.njit
    def _update_static_energy(z, T_sam, s_sam, qc_sam, qr_sam, qi1_sam):
        shape = T_sam.shape
        # Fortran ordered so we need to loop differently
        for k in range(shape[2]):
            for j in range(shape[1]):
                for i in range(shape[0]):
                    _T = T_sam[i, j, k]
                    T_sam[i, j, k] = _T
                    s_sam[i, j, k] = (
                        _T
                        + (
                            parameters.G * z[i, j, k]
                            - parameters.LV * (qc_sam[i, j, k] + qr_sam[i, j, k])
                            - parameters.LS * (qi1_sam[i, j, k])
                        )
                        * parameters.ICPD
                    )

        return

    def io_fields2d_update(self, nc_grp):

        start = self._Grid.local_start
        end = self._Grid._local_end
        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        if nc_grp is not None:
            precsfc = nc_grp.create_dataset(
                        "precsfc",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                precsfc.dims[i].attach_scale(nc_grp[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = self._precsfc
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        if nc_grp is not None:
            precsfc[:, :] = recv_buffer

        if nc_grp is not None:
            prec_xy = nc_grp.create_dataset(
                        "prec_xy",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                prec_xy.dims[i].attach_scale(nc_grp[d])

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = self._prec_xy
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            prec_xy[:, :] = recv_buffer

        # Compute and output the LWP
        if nc_grp is not None:
            lwp = nc_grp.create_dataset(
                        "LWP",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                lwp.dims[i].attach_scale(nc_grp[d])
                
        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field("qc")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        lwp_compute = np.sum(
            qc * rho0[np.newaxis, np.newaxis, 0] * self._Grid.dx[2], axis=2
        )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = lwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            lwp[:, :] = recv_buffer

        # Compute and output the LWP
        if nc_grp is not None:
            iwp = nc_grp.create_dataset(
                        "IWP",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                iwp.dims[i].attach_scale(nc_grp[d])
                
        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field("qci")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        iwp_compute = np.sum(qc * rho0[np.newaxis, np.newaxis, 0], axis=2)

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = iwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            iwp[:, :] = recv_buffer

        return

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    def get_reffc(self):
        return self._DiagnosticState.get_field("diag_effc_3d")

    def get_reffi(self):
        return self._DiagnosticState.get_field("diag_effi_3d")
