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
        
        try:
            
            mp_flags = namelist["microphysics"]["flags"]

            self._m2005_ma_cffi.init(
                docloud = mp_flags["docloud"],
                dototalwater = mp_flags["dototalwater"],
                dopredictNc = mp_flags["dopredictNc"],
                doprogaerosol = mp_flags["doprogaerosol"],
                dospecifyaerosol = mp_flags["dospecifyaerosol"],
                doprecoff = mp_flags["doprecoff"],
                doprecip = mp_flags["doprecip"], 
                doicemicro = mp_flags["doicemicro"],
                dograupel = mp_flags["dograupel"],
                doactivdiagoutput = mp_flags["doactivdiagoutput"],
                doreflectivity_cloudradar = mp_flags["doreflectivity_cloudradar"],
                donudging_aerosol = mp_flags["donudging_aerosol"],
                n_gas_chem_fields = mp_flags["n_gas_chem_fields"],
            )

            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with custom flags")
        except:
            self._m2005_ma_cffi.init()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("\tM2005_MA: Initialized with default flags")
                
        iqv = 1   ! total water (vapor + cloud liq) mass mixing ratio [kg H2O / kg dry air]
        
        self._ScalarState.add_variable(
            "qv",
            long_name="water vapor mixing ratio",
            units="g kg^{-1}",
            latex_name="q_v",
            limit=True,
        )
        
        if(dototalwater):
            count = 1
            
        else:
            iqcl = 2  ! cloud water mass mixing ratio [kg H2O / kg dry air]
            count = 2
            self._ScalarState.add_variable(
                "qc",
                long_name="cloud water mixing ratio",
                units="g kg^{-1}",
                latex_name="q_c",
                limit=True,
            )

        if(dopredictNc):
            incl = count + 1  ! cloud water number mixing ratio [#/kg dry air]
            count = count + 1
            self._ScalarState.add_variable(
                "qnc",
                long_name="cloud number concentration",
                units="# cm^{-3}",
                latex_name="q_{nc}",
                limit=True,
            )

        if(doprogaerosol):
            iqad = count + 1 ! dry aerosol mass mixing ratio [kg aerosol/kg dry air]
            iqad2 = count + 2 ! aitken mode
            inad = count + 3 ! dry aerosol number mixing ratio [#/kg dry air]
            inad2 = count + 4 ! aitken mode
            iqaw = count + 5 ! wet aerosol mass mixing ratio [kg activated aerosol/kg dry air]
            count = count + 5
            self._ScalarState.add_variable(
                "qad",
                long_name="dry aerosol mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{ad}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qad2",
                long_name="aitken mode mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{ad2}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qnad",
                long_name="dry aerosol number concentration",
                units="# cm^{-3}",
                latex_name="q_{nad}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qnad2",
                long_name="aitken mode mass number concentration",
                units="# cm^{-3}",
                latex_name="q_{nad2}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qaw",
                long_name="wet aerosol mass mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{aw}",
                limit=True,
            )

            igas1 = count + 1 ! first gas chem field
            iDMS = igas1
            iSO2 = igas1 + 1
            iH2SO4 = igas1 + 2
            count = count + 3 ! save space for gas chem fields # salt in our case
            self._ScalarState.add_variable(
                "DMS",
                long_name="DMS GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="DMS",
                limit=True,
            )
            self._ScalarState.add_variable(
                "SO2",
                long_name="SO2 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="SO2",
                limit=True,
            )
            self._ScalarState.add_variable(
                "H2SO4",
                long_name="H2SO4 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="H2SO4",
                limit=True,
            )

        if(doprecip):
            iqr = count + 1 ! rain mass mixing ratio [kg H2O / kg dry air]
            inr = count + 2 ! rain number mixing ratio [#/kg dry air]
            count = count + 2
            self._ScalarState.add_variable(
                "qr",
                long_name="rain water mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{r}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qnr",
                long_name="rain number concentration",
                units="# cm^{-3}",
                latex_name="q_{nr}",
                limit=True,
            )

            if(doprogaerosol):
                iqar = count + 1 ! ! wet aerosol mass mixing ratio in rain [kg aerosol in rain/kg dry air]
                count = count + 1
                self._ScalarState.add_variable(
                    "qar",
                    long_name="wet aerosol mass mixing ratio in rain",
                    units="g kg^{-1}",
                    latex_name="q_{ar}",
                    limit=True,
                )

        if(doicemicro):
            iqci = count + 1  ! cloud ice mass mixing ratio [kg H2O / kg dry air]
            inci = count + 2  ! cloud ice number mixing ratio [#/kg dry air]
            iqs = count + 3   ! snow mass mixing ratio [kg H2O / kg dry air]
            ins = count + 4   ! snow number mixing ratio [#/kg dry air]
            count = count + 4
            self._ScalarState.add_variable(
                "qci",
                long_name="cloud ice mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{i}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qnci",
                long_name="cloud ice number concentration",
                units="# cm^{-3}",
                latex_name="q_{nci}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qs",
                long_name="snow mixing ratio",
                units="g kg^{-1}",
                latex_name="q_{s}",
                limit=True,
            )
            self._ScalarState.add_variable(
                "qns",
                long_name="snow number concentration",
                units="# cm^{-3}",
                latex_name="q_{ns}",
                limit=True,
            )

        if(dograupel):
            iqg = count + 1   ! graupel mass mixing ratio [kg H2O / kg dry air]
            ing = count + 2  ! graupel number mixing ratio [#/kg dry air]
            count = count + 2
            if(dohail):
                self._ScalarState.add_variable(
                    "qh",
                    long_name="hail mixing ratio",
                    units="g kg^{-1}",
                    latex_name="q_{h}",
                    limit=True,
                )
                self._ScalarState.add_variable(
                    "qnh",
                    long_name="hail number concentration",
                    units="# cm^{-3}",
                    latex_name="q_{nh}",
                    limit=True,
                )
            else:
                self._ScalarState.add_variable(
                    "qg",
                    long_name="graupel mixing ratio",
                    units="g kg^{-1}",
                    latex_name="q_{g}",
                    limit=True,
                )
                self._ScalarState.add_variable(
                    "qng",
                    long_name="graupel number concentration",
                    units="# cm^{-3}",
                    latex_name="q_{ng}",
                    limit=True,
                )
        
        self._nmicrofields = count
                
        self._microfield = np.empty(
            (self._sam_dims[0], self._sam_dims[1], self._sam_dims[2], self._nmicrofields),
            dtype=np.double,
            order="F",
        )
                
        # Allocate microphysical/thermodynamic variables

        nhalo = self._Grid.n_halo
        self._sam_dims = (
            self._Grid.ngrid_local[0] - 2 * nhalo[0],
            self._Grid.ngrid_local[1] - 2 * nhalo[1],
            self._Grid.ngrid_local[2] - 2 * nhalo[2],
        )
        nx_gl = self._Grid.n[0]
        ny_gl = self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()
        nsubdomains_x = self._Grid.subcomms[0].Get_size()
        nsubdomains_y = self._Grid.subcomms[1].Get_size()
                
        self._itimestep = 0
                        
        self._nrainy = 0.0
        self._nrmn = 0.0
        self._ncmn = 0.0
        self._total_water_prec = 0.0
                
        self._fluxbq = np.zeros(
            (self._sam_dims[0], self._sam_dims[1]), order="F", dtype=np.double
        )
        self._fluxtq = np.zeros_like(self._fluxbq)
        self._u10arr = np.zeros_like(self._fluxbq)
        self._precsfc = np.zeros_like(self._fluxbq)
        self._prec_xy = np.zeros_like(self._fluxbq)
                
        self._tlat = np.zeros(
            (self._sam_dims[2]+1), order="F", dtype=np.double
        )
        self._tlatqi = np.zeros_like(self._tlat)
        self._precflux = np.zeros_like(self._tlat)
        self._qpfall = np.zeros_like(self._tlat)
                
        z = self.Grid.z_global
        zi = self.Grid.z_edge_global
        
        dx = self._Grid.dx[0]
        dz = self._Grid.dx[2]

        p0 = self._Ref.p0
        rho0 = self._Ref.rho0
        tabs0 = self._Ref.T0
        rhow = self._Ref.rho0_edge

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
            "qtot_sed" : ("total liquid water sedimentation", "")
            "qice_sed" : ("ice sedimentation", "")
            "Nacc_sct" : ("Nacc self coagulation tendency", "")
            "Nait_sct" : ("Nait self coagulation tendency", "")
            "Nait2a_ct" : ("Nait2acc coagulation tendency", "")
            "Mait2a_ct" : ("Mait2acc coagulation tendency", "")
            "relhum" : ("relative humidity", "")
            "diag_effc_3d" : ("cloud droplet effective radius", "m")
            "diag_effr_3d" : ("rain droplet effective radius", "m")
            "diag_effi_3d" : ("cloud ice effective radius", "m")
            "diag_effs_3d" : ("snow effective radius", "m")
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

        self._n_diag_3d = len(self._diag_3d_vars)

        self._diag_3d = np.empty(
            (self._sam_dims[0], self._sam_dims[1], self._sam_dims[2], self._n_diag_3d),
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
                        
        self._Timers.add_timer("MicroM2005_MA_update")
        return

    def update(self):
        self._Timers.start_timer("MicroM2005_MA_update")

        # Get variables from the model state
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        w = self._VelocityState.get_field("w")
                
        dt = self._TimeSteppingController.dt
        time = self._TimeSteppingController.time
                
        LCOND = self.Parameters.LV
        LSUB = self.Parameters.LS
        CP = self.Parameters.CPD
        RGAS = self.Parameters.RD
        RV = self.Parameters.RV
        G = self.Parameters.G
        
        self._microfield(:,:,:,iqv) =   self._ScalarState.getfield("qv")
        
        if(Not dototalwater):
            self._microfield(:,:,:,iqcl) =   self._ScalarState.getfield("qc")

        if(dopredictNc):
            self._microfield(:,:,:,incl) =   self._ScalarState.getfield("qnc")

        if(doprogaerosol):
            self._microfield(:,:,:,iqad) =   self._ScalarState.getfield("qad")
            self._microfield(:,:,:,iqad2) =   self._ScalarState.getfield("qad2")
            self._microfield(:,:,:,inad) =   self._ScalarState.getfield("qnad")
            self._microfield(:,:,:,inad2) =   self._ScalarState.getfield("qnad2")
            self._microfield(:,:,:,iqaw) =   self._ScalarState.getfield("qaw")
            self._microfield(:,:,:,iqgas) =   self._ScalarState.getfield("qgas")

        if(doprecip):
            self._microfield(:,:,:,iqr) =   self._ScalarState.getfield("qr")
            self._microfield(:,:,:,iqnr) =   self._ScalarState.getfield("qnr")

            if(doprogaerosol):
                self._microfield(:,:,:,iqar) =   self._ScalarState.getfield("qar")

        if(doicemicro):
            self._microfield(:,:,:,iqci) =   self._ScalarState.getfield("qci")
            self._microfield(:,:,:,iqnci) =   self._ScalarState.getfield("qnci")
            self._microfield(:,:,:,iqs) =   self._ScalarState.getfield("qs")
            self._microfield(:,:,:,iqns) =   self._ScalarState.getfield("qns")

        if(dograupel):
            self._microfield(:,:,:,iqg) =   self._ScalarState.getfield("qg")
            self._microfield(:,:,:,iqng) =   self._ScalarState.getfield("qng")                

        self._m2005_ma_cffi.update(
            self.sam_dims[0],
            self.sam_dims[1],
            self.sam_dims[2],
            self.nmicrofields,
            nx_gl,
            ny_gl,
            my_rank,
            nsubdomains_x,
            nsubdomains_y,
            T,
            s,
            w,
            self._microfield,
            self._n_diag_3d,
            self._diag_3d,
            z,
            p0,
            rho0,
            tabs0,
            zi,
            rhow,
            dx,
            dz,
            self._nrainy,
            self._nrmn,
            self._ncmn,
            self._total_water_prec,
            self._tlat,
            self._tlatqi,
            self._precflux,
            self._qpfall,
            self.fluxbq,
            self.fluxtq,
            self.u10arr,
            self._precsfc,
            self._prec_xy,
            dt,
            time,
            self._itimestep,
            LCOND,
            LSUB,
            CPD,
            RV,
            G,
        )

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

        v = timeseries_grp.createVariable("RAINNC", np.double, dimensions=("time",))
        v.long_name = "accumulated surface precip"
        v.units = "mm"
        v.latex_name = "rainnc"

        v = timeseries_grp.createVariable("RAINNCV", np.double, dimensions=("time",))
        v.long_name = "one time step accumulated surface precip"
        v.units = "mm"
        v.latex_name = "rainncv"

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

        rainnc = np.sum(self._RAINNC) / npts
        rainnc = UtilitiesParallel.ScalarAllReduce(rainnc)

        rf_prof = water_fraction_profile(n_halo, npts, qr, threshold=1e-5)
        rf_prof = UtilitiesParallel.ScalarAllReduce(rf_prof)

        rainncv = np.sum(self._RAINNCV) / npts
        rainncv = UtilitiesParallel.ScalarAllReduce(rainncv)

        if my_rank == 0:
            timeseries_grp = nc_grp["timeseries"]
            profiles_grp = nc_grp["profiles"]

            timeseries_grp["CF"][-1] = cf
            timeseries_grp["RF"][-1] = rf
            timeseries_grp["LWP"][-1] = lwp
            timeseries_grp["LWP_LASSO"][-1] = lwp_lasso
            timeseries_grp["RWP"][-1] = rwp
            timeseries_grp["VWP"][-1] = vwp

            timeseries_grp["RAINNC"][-1] = rainnc
            timeseries_grp["RAINNCV"][-1] = rainncv

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
            rainnc = nc_grp.createVariable(
                "RAINNC",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        send_buffer[start[0] : end[0], start[1] : end[1]] = self._RAINNC
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        if nc_grp is not None:
            rainnc[:, :] = recv_buffer

        if nc_grp is not None:
            rainncv = nc_grp.createVariable(
                "RAINNCV",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = self._RAINNCV
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            rainncv[:, :] = recv_buffer

        # Compute and output the LWP
        if nc_grp is not None:
            lwp = nc_grp.createVariable(
                "LWP",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )
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
            iwp = nc_grp.createVariable(
                "IWP",
                np.double,
                dimensions=(
                    "X",
                    "Y",
                ),
            )

        nh = self._Grid.n_halo
        rho0 = self._Ref.rho0
        qc = self._ScalarState.get_field("qi1")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        iwp_compute = np.sum(qc * rho0[np.newaxis, np.newaxis, 0], axis=2)

        send_buffer.fill(0.0)
        send_buffer[start[0] : end[0], start[1] : end[1]] = iwp_compute
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if nc_grp is not None:
            iwp[:, :] = recv_buffer

        if nc_grp is not None:
            nc_grp.sync()

        return

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    def get_qi(self):
        return self._ScalarState.get_field("qi1")

    def get_reffc(self):
        return self._DiagnosticState.get_field("diag_effc_3d")

    def get_reffi(self):
        return self._DiagnosticState.get_field("diag_effi_3d")
