from pinacles.Microphysics import (
    MicrophysicsBase,
    water_path,
    water_fraction,
    water_fraction_profile,
    water_path_lasso,
)
from pinacles.externals.sam_m2005_ma_wrapper import m2005_ma_via_cffi
from pinacles import UtilitiesParallel
from pinacles.WRFUtil import sam_to_our_order
from pinacles import parameters
import pinacles.ThermodynamicsMoist_impl as MoistThermo
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

        self._nhalo = self._Grid.n_halo
        self._sam_dims = np.array(
            [
                self._Grid.ngrid_local[0] - 2 * self._nhalo[0],
                self._Grid.ngrid_local[1] - 2 * self._nhalo[1],
                self._Grid.ngrid_local[2] - 2 * self._nhalo[2],
                0,
            ],
            order="F",
            dtype=np.intc,
        )
        nz = self._sam_dims[2] + 1

        self._iqarray = np.zeros((22,), order="F", dtype=np.intc)
        self._masterproc = False

        if MPI.COMM_WORLD.Get_rank() == 0:
            self._masterproc = True

        self._tlatqi = np.zeros((nz), order="F", dtype=np.double)

        try:
            mp_flags = namelist["m2005_ma"]["flags"]
            docloud = mp_flags["docloud"]
            doprecip = mp_flags["doprecip"]
            doprogaero = mp_flags["doprogaerosol"]
            
            UtilitiesParallel.print_root("\tM2005_MA: Initialized with custom flags")
        except:
            docloud = True
            doprecip = True
            doprogaero = True
            
            UtilitiesParallel.print_root("\tM2005_MA: Initialized with default flags")
            
        try:
            aero_in = namelist["m2005_ma"]["aero"]

            rm_acc = aero_in["rm_acc"]
            N_acc = aero_in["N_acc"]
            sigma_acc = aero_in["sigma_acc"]

            rm_ait = aero_in["rm_ait"]
            N_ait = aero_in["N_ait"]
            sigma_ait = aero_in["sigma_ait"]
            
            UtilitiesParallel.print_root("\tM2005_MA: Initialized with custom aerosol parameters")
            
        except:
            rm_acc    = 0.06e-6
            N_acc     = 65.0e6
            sigma_acc = 1.7
            
            rm_ait    = 0.011e-6
            N_ait     = 125.0e6
            sigma_ait = 1.2
            
            UtilitiesParallel.print_root("\tM2005_MA: Initialized with default DYCOMS aerosol parameters")

        self._m2005_ma_cffi.setparm(
            self._sam_dims,
            self._iqarray,
            docloud,
            doprecip,
            doprogaero,
            rm_acc,
            N_acc,
            sigma_acc,
            rm_ait,
            N_ait,
            sigma_ait,
            self._masterproc,
            self._tlatqi,
        )

        self._micro_vars = ["" for x in range(self._sam_dims[3])]

        if self._iqarray[0] > 0:
            iqv = self._iqarray[0] - 1  # vapor mass mixing ratio [kg H2O / kg dry air]
            self._micro_vars[
                iqv
            ] = "qv"  # -1 in the index to go from fortran (starting from 1) to python (starting from 0)
            self._ScalarState.add_variable(
                "qv",
                long_name="water vapor mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{v}",
                limit=True,
            )

        if self._iqarray[1] > 0:
            iqcl = (
                self._iqarray[1] - 1
            )  # cloud water mass mixing ratio [kg H2O / kg dry air]
            self._micro_vars[iqcl] = "qc"
            self._ScalarState.add_variable(
                "qc",
                long_name="cloud water mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{c}",
                limit=True,
                is_prognosed_liquid=True,
            )

        if self._iqarray[2] > 0:
            iqi = self._iqarray[2] - 1
            self._micro_vars[iqi] = "qi"
            self._ScalarState.add_variable(
                "qi",
                long_name="cloud ice mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{i}",
                limit=True,
                is_prognosed_ice=True,
            )

        if self._iqarray[3] > 0:
            iqr = self._iqarray[3] - 1
            self._micro_vars[iqr] = "qr"
            self._ScalarState.add_variable(
                "qr",
                long_name="rain water mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{r}",
                limit=True,
                is_prognosed_liquid=True,
            )

        if self._iqarray[4] > 0:
            iqs = self._iqarray[4] - 1
            self._micro_vars[iqs] = "qs"
            self._ScalarState.add_variable(
                "qs",
                long_name="snow mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{s}",
                limit=True,
                is_prognosed_ice=True,
            )

        if self._iqarray[5] > 0:
            iqg = self._iqarray[5] - 1
            self._micro_vars[iqg] = "qg"
            self._ScalarState.add_variable(
                "qg",
                long_name="graupel mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{g}",
                limit=True,
                is_prognosed_ice=True,
            )

        if self._iqarray[6] > 0:
            incl = self._iqarray[6] - 1
            self._micro_vars[incl] = "qnc"
            self._ScalarState.add_variable(
                "qnc",
                long_name="cloud number concentration",
                units="# kg^{-1}",
                latex_name="q_{nc}",
                limit=True,
            )

        if self._iqarray[7] > 0:
            inci = self._iqarray[7] - 1
            self._micro_vars[inci] = "qnci"
            self._ScalarState.add_variable(
                "qnci",
                long_name="cloud ice number concentration",
                units="# kg^{-1}",
                latex_name="q_{nci}",
                limit=True,
            )

        if self._iqarray[8] > 0:
            inr = self._iqarray[8] - 1
            self._micro_vars[inr] = "qnr"
            self._ScalarState.add_variable(
                "qnr",
                long_name="rain number concentration",
                units="# kg^{-1}",
                latex_name="q_{nr}",
                limit=True,
            )

        if self._iqarray[9] > 0:
            ins = self._iqarray[9] - 1
            self._micro_vars[ins] = "qns"
            self._ScalarState.add_variable(
                "qns",
                long_name="snow number concentration",
                units="# kg^{-1}",
                latex_name="q_{ns}",
                limit=True,
            )

        if self._iqarray[10] > 0:
            ing = self._iqarray[10] - 1
            self._micro_vars[ing] = "qng"
            self._ScalarState.add_variable(
                "qng",
                long_name="graupel number concentration",
                units="# kg^{-1}",
                latex_name="q_{ng}",
                limit=True,
            )

        if self._iqarray[11] > 0:
            iqad = self._iqarray[11] - 1
            self._micro_vars[iqad] = "qad"
            self._ScalarState.add_variable(
                "qad",
                long_name="dry aerosol mass mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{ad}",
                limit=True,
            )

        if self._iqarray[12] > 0:
            iqaw = self._iqarray[12] - 1
            self._micro_vars[iqaw] = "qaw"
            self._ScalarState.add_variable(
                "qaw",
                long_name="wet aerosol mass mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{aw}",
                limit=True,
            )

        if self._iqarray[13] > 0:
            iqar = self._iqarray[13] - 1
            self._micro_vars[iqar] = "qar"
            self._ScalarState.add_variable(
                "qar",
                long_name="wet aerosol mass mixing ratio in rain",
                units="kg kg^{-1}",
                latex_name="q_{ar}",
                limit=True,
            )

        if self._iqarray[14] > 0:
            inad = self._iqarray[14] - 1
            self._micro_vars[inad] = "qnad"
            self._ScalarState.add_variable(
                "qnad",
                long_name="dry aerosol number concentration",
                units="# kg^{-1}",
                latex_name="q_{nad}",
                limit=True,
            )

        if self._iqarray[15] > 0:
            iqad2 = self._iqarray[15] - 1
            self._micro_vars[iqad2] = "qad2"
            self._ScalarState.add_variable(
                "qad2",
                long_name="aitken mode mass mixing ratio",
                units="kg kg^{-1}",
                latex_name="q_{ad2}",
                limit=True,
            )

        if self._iqarray[16] > 0:
            inad2 = self._iqarray[16] - 1
            self._micro_vars[inad2] = "qnad2"
            self._ScalarState.add_variable(
                "qnad2",
                long_name="aitken mode mass number concentration",
                units="# kg^{-1}",
                latex_name="q_{nad2}",
                limit=True,
            )

        if self._iqarray[17] > 0:
            igas1 = self._iqarray[17] - 1

        if self._iqarray[18] > 0:
            iDMS = self._iqarray[18] - 1
            self._micro_vars[iDMS] = "DMS"
            self._ScalarState.add_variable(
                "DMS",
                long_name="DMS GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="DMS",
                limit=True,
            )

        if self._iqarray[19] > 0:
            iSO2 = self._iqarray[19] - 1
            self._micro_vars[iSO2] = "SO2"
            self._ScalarState.add_variable(
                "SO2",
                long_name="SO2 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="SO2",
                limit=True,
            )

        if self._iqarray[20] > 0:
            iH2SO4 = self._iqarray[20] - 1
            self._micro_vars[iH2SO4] = "H2SO4"
            self._ScalarState.add_variable(
                "H2SO4",
                long_name="H2SO4 GAS CONCENTRATION",
                units="kg kg^{-1}",
                latex_name="H2SO4",
                limit=True,
            )

        # Allocate microphysical/thermodynamic variables

        # For diagnostic variables

        # self._diag_3d_vars = [
        #     "qtot_sed",
        #     "qice_sed",
        #     "Nacc_sct",
        #     "Nait_sct",
        #     "Nait2a_ct",
        #     "Mait2a_ct",
        #     "relhum",
        #     "diag_effc_3d",
        #     "diag_effr_3d",
        #     "diag_effi_3d",
        #     "diag_effs_3d",
        #     "dBZCLRAD",
        #     "NARG1",
        #     "NARG2",
        #     "NACTRATE",
        #     "QACTRATE",
        #     "NACTDIFF",
        #     "NATRANS",
        #     "QATRANS",
        #     "ISACT",
        #     "DC1",
        #     "DC2",
        #     "DG1",
        #     "DG2",
        #     "SSPK",
        #     "NCPOSLM",
        #     "NCNEGLM",
        # ]

        # qc_units = "kg kg-1"
        # qc_tend_units = "kg kg-1 s-1"
        # n_tend_units = "kg-1"
        # n_rate_units = "kg-1 s-1"
        # diag_3d_long_names = {
        #     "qtot_sed" : ("total liquid water sedimentation", ""),
        #     "qice_sed" : ("ice sedimentation", ""),
        #     "Nacc_sct" : ("Nacc self coagulation tendency", ""),
        #     "Nait_sct" : ("Nait self coagulation tendency", ""),
        #     "Nait2a_ct" : ("Nait2acc coagulation tendency", ""),
        #     "Mait2a_ct" : ("Mait2acc coagulation tendency", ""),
        #     "relhum" : ("relative humidity", ""),
        #     "diag_effc_3d" : ("cloud droplet effective radius", "m"),
        #     "diag_effr_3d" : ("rain droplet effective radius", "m"),
        #     "diag_effi_3d" : ("cloud ice effective radius", "m"),
        #     "diag_effs_3d" : ("snow effective radius", "m"),
        #     "dBZCLRAD": ("Cloud Radar Reflectivity", "dBZ"),
        #     "NARG1": ("A-R&G. Activated Number Accumulation", n_tend_units),
        #     "NARG2": ("A-R&G Activated Number Aitken", n_tend_units),
        #     "NACTRATE": ("Activation Rate Accumulation Aerosol", n_rate_units),
        #     "QACTRATE": ("Activation Rate Accumulation Aerosol", qc_tend_units),
        #     "NACTDIFF": ("Difference from A-R&G activ number and and old cloud number", n_tend_units),
        #     "NATRANS": ("Aitken to Accum Transfer Number", n_tend_units),
        #     "QATRANS": ("Aitken to Accum Transfer Mass", qc_units),
        #     "ISACT": ("ARG Activation run on particular point","None"),
        #     "DC1": ("Critical activation diameter mode 1", "m"),
        #     "DC2": ("Critical activation diameter mode 2", "m"),
        #     "DG1": ("Modal diameter 1", "m"),
        #     "DG2": ("Modal diameter 2", "m"),
        #     "SSPK": ("Peak activation supersaturation ARG", "fraction"),
        #     "NCPOSLM": ("Change in NC due to positive rate limiter", n_rate_units),
        #     "NCNEGLM": ("Mass Activation Rate Aitken Aerosol", n_rate_units),
        # }

        # self.n_diag_3d = len(self._diag_3d_vars)

        # self._diag_sam = np.empty(
        #     (nx, ny, nzm, self.n_diag_3d),
        #     dtype=np.double,
        #     order="F",
        # )

        #         # Add a diagnostic variable for each of the process rates
        #         for i, vn in enumerate(self._diag_3d_vars):
        #             self._DiagnosticState.add_variable(
        #                 "m2005_ma_" + vn,
        #                 long_name=diag_3d_long_names[vn][0],
        #                 units=diag_3d_long_names[vn][1],
        #             )

        self._DiagnosticState.add_variable(
            "qtot_sed", long_name="total liquid water sedimentation", units=""
        )
        self._DiagnosticState.add_variable(
            "qice_sed", long_name="ice sedimentation", units=""
        )

        self._DiagnosticState.add_variable(
            "s_tend_liq_sed", long_name="s tend liquid water sedimentation", units=""
        )
        self._DiagnosticState.add_variable(
            "s_tend_ice_sed", long_name="s tend ice water sedimentation", units=""
        )
        
        self._DiagnosticState.add_variable(
            "diag_effc_3d",
            long_name="cloud droplet effective radius",
            units="m",
            latex_name="r_e",
        )
        self._DiagnosticState.add_variable(
            "diag_effi_3d",
            long_name="cloud ice effective radius",
            units="m",
            latex_name="r_{e,i}",
        )

        nx = self._Grid.ngrid_local[0] - 2 * self._nhalo[0]
        ny = self._Grid.ngrid_local[1] - 2 * self._nhalo[1]
        nzm = self._Grid.ngrid_local[2] - 2 * self._nhalo[2]
        nz = nzm + 1
        nmicrofields = self._sam_dims[3]
        self._nx_gl = self._Grid.n[0]
        self._ny_gl = self._Grid.n[1]
        self._my_rank = MPI.COMM_WORLD.Get_rank()
        self._nsubdomains_x = self._Grid.subcomms[0].Get_size()
        self._nsubdomains_y = self._Grid.subcomms[1].Get_size()

        self._microfield = np.empty(
            (nx, ny, nzm, nmicrofields), dtype=np.double, order="F",
        )

        if namelist["restart"]["restart_simulation"] == False:
            self._nrestart = 0
        else:
            self._nrestart = 1

        self._itimestep = 0

        self._nrainy = np.zeros((1), order="F", dtype=np.double)
        self._nrmn = np.zeros_like(self._nrainy)
        self._ncmn = np.zeros_like(self._nrainy)
        self._total_water_prec = np.zeros_like(self._nrainy)

        self._fluxbq = np.zeros((nx, ny), order="F", dtype=np.double)
        self._fluxtq = np.zeros_like(self._fluxbq)  # fluxes not being used anywhere
        self._u10arr = np.zeros_like(self._fluxbq)
        self._precsfc = np.zeros_like(self._fluxbq)
        self._prec_xy = np.zeros_like(self._fluxbq)

        self._tlat = np.zeros((nz), order="F", dtype=np.double)
        self._precflux = np.zeros_like(self._tlat)
        # np.asfortranarray(np.zeros_like(self._tlat))
        self._qpfall = np.zeros_like(self._tlat)

        self._z = self._Grid.z_global
        self._zi = self._Grid.z_edge_global

        self._dx = self._Grid.dx[0]
        self._dz = self._Grid.dx[2]

        self._dostatis = False
        self._do_chunked_energy_budgets = False
        self._donudging_aerosol = False

        self._Temp = np.empty((nx, ny, nzm), dtype=np.double, order="F",)

        self._time = self._TimeSteppingController.time

        return

    def initialize(self):
        self._p0 = self._Ref.p0
        self._rho0 = self._Ref.rho0
        self._tabs0 = self._Ref.T0
        self._rhow = self._Ref.rho0_edge

        qv = self._ScalarState.get_field("qv")
        qc = self._ScalarState.get_field("qc")
        self._qv0 = qv.mean(axis=1).mean(axis=0)[
            self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2]
        ]
        self._qc0 = qc.mean(axis=1).mean(axis=0)[
            self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2]
        ]

        s = self._ScalarState.get_field("s")
        self._t0 = s.mean(axis=1).mean(axis=0)[
            self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2]
        ]

        # self._tabs0 = np.zeros_like(self._p0)
        for k in range(self._sam_dims[2]):
            self._tabs0[k] = MoistThermo.T(self._z[k], self._t0[k], self._qc0[k], 0.0)
        
        diag_effc_3d = self._DiagnosticState.get_field("diag_effc_3d")
        diag_effi_3d = self._DiagnosticState.get_field("diag_effi_3d")
        
        diag_effc_3d_sam = np.asfortranarray(
            diag_effc_3d[
                self._nhalo[0] : self._Grid.ngrid_local[0] - self._nhalo[0],
                self._nhalo[1] : self._Grid.ngrid_local[1] - self._nhalo[1],
                self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2],
            ]
        )
        diag_effi_3d_sam = np.asfortranarray(
            diag_effi_3d[
                self._nhalo[0] : self._Grid.ngrid_local[0] - self._nhalo[0],
                self._nhalo[1] : self._Grid.ngrid_local[1] - self._nhalo[1],
                self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2],
            ]
        )    

        for i, vn in enumerate(self._micro_vars):
            dv = self._ScalarState.get_field(vn)
            self._microfield[:, :, :, i] = np.asfortranarray(
                dv[
                    self._nhalo[0] : self._Grid.ngrid_local[0] - self._nhalo[0],
                    self._nhalo[1] : self._Grid.ngrid_local[1] - self._nhalo[1],
                    self._nhalo[2] : self._Grid.ngrid_local[2] - self._nhalo[2],
                ]
            )

        self._m2005_ma_cffi.init(
            self._sam_dims[0],
            self._sam_dims[1],
            self._sam_dims[2],
            self._sam_dims[3],
            self._nx_gl,
            self._ny_gl,
            self._my_rank,
            self._nsubdomains_x,
            self._nsubdomains_y,
            self._nrestart,
            self._Temp,
            diag_effc_3d_sam,
            diag_effi_3d_sam,
            self._microfield,
            self._z,
            self._p0,
            self._rho0,
            self._tabs0,
            self._zi,
            self._t0,
            self._dx,
            self._dz,
            self._time,
            self._qc0,
            self._qv0,
            self._dostatis,
            self._do_chunked_energy_budgets,
            self._donudging_aerosol,
            parameters.LV,
            parameters.LS,
            parameters.CPD,
            parameters.RD,
            parameters.RV,
            parameters.G,
        )

        for i, vn in enumerate(self._micro_vars):
            dv = self._ScalarState.get_field(vn)
            sam_to_our_order(self._nhalo, self._microfield[:, :, :, i], dv)
            
        sam_to_our_order(self._nhalo, diag_effc_3d_sam, diag_effc_3d)
        sam_to_our_order(self._nhalo, diag_effi_3d_sam, diag_effi_3d)

        self._Timers.add_timer("MicroM2005_MA_update")
        return

    def update(self):
        self._Timers.start_timer("MicroM2005_MA_update")

        p0 = self._Ref.p0
        rho0 = self._Ref.rho0
        # tabs0 = self._Ref.T0
        rhow = self._Ref.rho0_edge

        # Get variables from the model state
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        w = self._VelocityState.get_field("w")

        qtot_sed = self._DiagnosticState.get_field("qtot_sed")
        qice_sed = self._DiagnosticState.get_field("qice_sed")

        s_tend_liq_sed = self._DiagnosticState.get_field("s_tend_liq_sed")
        s_tend_ice_sed = self._DiagnosticState.get_field("s_tend_ice_sed")
        
        diag_effc_3d = self._DiagnosticState.get_field("diag_effc_3d")
        diag_effi_3d = self._DiagnosticState.get_field("diag_effi_3d")

        nhalo = self._Grid.n_halo

        th_sam = np.asfortranarray(
            T[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )

        s_sam = np.asfortranarray(
            s[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )

        w_sam = np.asfortranarray(
            w[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] - 1 : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )

        qtot_sed_sam = np.asfortranarray(
            qtot_sed[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )
        qice_sed_sam = np.asfortranarray(
            qice_sed[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )

        diag_effc_3d_sam = np.asfortranarray(
            diag_effc_3d[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )
        diag_effi_3d_sam = np.asfortranarray(
            diag_effi_3d[
                nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
            ]
        )

        for i, vn in enumerate(self._micro_vars):
            dv = self._ScalarState.get_field(vn)
            self._microfield[:, :, :, i] = np.asfortranarray(
                dv[
                    nhalo[0] : self._Grid.ngrid_local[0] - nhalo[0],
                    nhalo[1] : self._Grid.ngrid_local[1] - nhalo[1],
                    nhalo[2] : self._Grid.ngrid_local[2] - nhalo[2],
                ]
            )

        dt = self._TimeSteppingController.dt
        time = self._TimeSteppingController.time

        icycle = 1
        ncycle = 1
        nsaveMSE = 1
        nstat = 1
        nstatis = 1

        self._m2005_ma_cffi.update(
            self._sam_dims[0],
            self._sam_dims[1],
            self._sam_dims[2],
            self._sam_dims[3],
            icycle,
            ncycle,
            nsaveMSE,
            nstat,
            nstatis,
            self._itimestep,
            dt,
            time,
            p0,
            rho0,
            self._tabs0,
            rhow,
            self._nrainy,
            self._nrmn,
            self._ncmn,
            self._total_water_prec,
            self._tlat,
            self._precflux,
            self._qpfall,
            self._precsfc,
            self._prec_xy,
            self._fluxbq,
            self._fluxtq,
            self._u10arr,
            th_sam,
            s_sam,
            w_sam,
            self._microfield,
            qtot_sed_sam,
            qice_sed_sam,
            diag_effc_3d_sam,
            diag_effi_3d_sam,
        )

        for i, vn in enumerate(self._micro_vars):
            dv = self._ScalarState.get_field(vn)
            sam_to_our_order(nhalo, self._microfield[:, :, :, i], dv)

        # sam_to_our_order(nhalo, th_sam, T)
        sam_to_our_order(nhalo, s_sam, s)

        sam_to_our_order(nhalo, qtot_sed_sam, qtot_sed)
        sam_to_our_order(nhalo, qice_sed_sam, qice_sed)

        sam_to_our_order(nhalo, diag_effc_3d_sam, diag_effc_3d)
        sam_to_our_order(nhalo, diag_effi_3d_sam, diag_effi_3d)

        # Compute and apply sedimentation sources of static energy
        np.multiply(qtot_sed, parameters.LV / parameters.CPD, out=s_tend_liq_sed)
        np.multiply(qice_sed, parameters.LS / parameters.CPD, out=s_tend_ice_sed)

        # Convert sedimentation sources to units of tendency
        np.multiply(qtot_sed, 1.0 / self._TimeSteppingController.dt, out=qtot_sed)
        np.multiply(qice_sed, 1.0 / self._TimeSteppingController.dt, out=qice_sed)

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
        v = profiles_grp.createVariable("CF", np.double, dimensions=("time", "z",),)
        v.long_name = "Cloud Fraction"
        v.standard_name = "CF"
        v.units = ""

        v = profiles_grp.createVariable("RF", np.double, dimensions=("time", "z",),)
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
                "precsfc", (1, self._Grid.n[0], self._Grid.n[1]), dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                precsfc.dims[i].attach_scale(nc_grp[d])

        send_buffer[start[0] : end[0], start[1] : end[1]] = self._precsfc
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)

        if nc_grp is not None:
            precsfc[:, :] = recv_buffer

        if nc_grp is not None:
            prec_xy = nc_grp.create_dataset(
                "prec_xy", (1, self._Grid.n[0], self._Grid.n[1]), dtype=np.double,
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
                "LWP", (1, self._Grid.n[0], self._Grid.n[1]), dtype=np.double,
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
                "IWP", (1, self._Grid.n[0], self._Grid.n[1]), dtype=np.double,
            )

            for i, d in enumerate(["time", "X", "Y"]):
                iwp.dims[i].attach_scale(nc_grp[d])

        # nh = self._Grid.n_halo
        # rho0 = self._Ref.rho0
        # qc = self._ScalarState.get_field("qi")[nh[0] : -nh[0], nh[1] : -nh[1], :]
        # iwp_compute = np.sum(qc * rho0[np.newaxis, np.newaxis, 0], axis=2)

        #        send_buffer.fill(0.0)
        #        send_buffer[start[0] : end[0], start[1] : end[1]] = iwp_compute
        #        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        #        if nc_grp is not None:
        #            iwp[:, :] = recv_buffer

        return

    def get_qc(self):
        return self._ScalarState.get_field("qc") + self._ScalarState.get_field("qr")

    def get_qcloud(self):
        return self._ScalarState.get_field("qc")

    # def get_qi(self):
    #     return self._ScalarState.get_field("qi")

    def get_reffc(self):
        return self._DiagnosticState.get_field("diag_effc_3d")

    def get_reffi(self):
        return self._DiagnosticState.get_field("diag_effi_3d")
