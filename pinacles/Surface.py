from pinacles import parameters
from pinacles import UtilitiesParallel
from pinacles import Surface_impl
import numpy as np

class SurfaceBase:
    def __init__(
        self, namelist, Timers, Grid, Ref, VelocityState, ScalarState, DiagnosticState
    ):

        self._name = "Surface"

        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        self._theta_flux = None
        self._buoyancy_flux = None

        self._lhf_sfc = None
        self._shf_sfc = None
        self._taux_sfc = None
        self._tauy_sfc = None
        self._windspeed_sfc = None
        self.gustiness = 0.1
        self.T_surface = 300.0

        self._z0 = None
        self.aero_flux = False
        
        self._scheme = namelist["microphysics"]["scheme"]
        
        try:
            mp_flags = namelist["m2005_ma"]["flags"]
            doprogaero = mp_flags["doprogaerosol"]
            
            UtilitiesParallel.print_root("\tCustom prog. aerosol flag")
        except:
            doprogaero = True
            
            UtilitiesParallel.print_root("\tDefault prog. aerosol flag (true)")
                
        if (self._scheme == "m2005_ma" and doprogaero == True):
            
            UtilitiesParallel.print_root("\tAerosol fluxes init")
            self.aero_flux = True
            # self._ustar = 0.25  # m/s
            self._WHITECAP_COEF = 3.84e-6  # for surface salt aerosol flux,  from eq (5) for whitecap coverage in Clarke etal (2006)
            self._RHO_AEROSOL = 2160.0  # kg/m^3 aerosol density set for NaCl
            
            self._SFLUX_NACC_COEF = 4.37e7  # coefficient of surface accumulation number flux
            self._SFLUX_NAIT_COEF = 4.37e7  # coefficient of surface aitken number flux
            
            try:
                aero_in = namelist["m2005_ma"]["aero"]
                
                self._SFLUX_RACC = aero_in["rm_acc"]
                self._SFLUX_RAIT = aero_in["rm_ait"]
                self._SIGMA_ACCUM = aero_in["sigma_acc"]
                self._SIGMA_AITKEN = aero_in["sigma_ait"]
                
                UtilitiesParallel.print_root("\tCustom aerosol parameters")
            except:
                self._SFLUX_RACC = 0.06  # median radius of surface accum. flux  (micron)
                self._SFLUX_RAIT = 0.011  # median radius of aitken flux
                self._SIGMA_ACCUM = 1.7  # sig=geom standard deviation of aer size distn.
                self._SIGMA_AITKEN = 1.2  # sig=geom standard deviation of aer size distn.
                
                UtilitiesParallel.print_root("\tDefault DYCOMS aerosol parameters")
                        
            nl = self._Grid.ngrid_local

            zl = self._Grid.z_local
            for k in range(nl[2]):
                if zl[k] > 10.0:
                    break
            self.ind10 = k - 1
            self.fac1 = (zl[k] - 10.0) / (zl[k] - zl[k - 1])
            self.fac2 = (10.0 - zl[k - 1]) / (zl[k] - zl[k - 1])

            # self._windspeed_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
            # self._taux_sfc = np.zeros_like(self._windspeed_sfc)
            # self._tauy_sfc = np.zeros_like(self._windspeed_sfc)
            # self._ustar_sfc = np.zeros_like(self._windspeed_sfc) + self._ustar
            self._naflux_sfc = np.zeros((nl[0], nl[1]), dtype=np.double)
            self._qaflux_sfc = np.zeros_like(self._naflux_sfc)
            self._na2flux_sfc = np.zeros_like(self._naflux_sfc)
            self._qa2flux_sfc = np.zeros_like(self._naflux_sfc)
            self._u10_arr = np.zeros_like(self._naflux_sfc)
        
        return

    def update(self):
                
        if self.aero_flux == True:
            # UtilitiesParallel.print_root("\tAerosol fluxes") # Get Fields
            
            nh = self._Grid.n_halo
            dxi2 = self._Grid.dxi[2]
            z_edge = self._Grid.z_edge_global

            alpha0 = self._Ref.alpha0
            alpha0_edge = self._Ref.alpha0_edge
            
            u = self._VelocityState.get_field("u")
            v = self._VelocityState.get_field("v")

            u10 = u[:, :, self.ind10 : self.ind10 + 1]
            v10 = v[:, :, self.ind10 : self.ind10 + 1]

            nadt = self._ScalarState.get_tend("qnad")
            qadt = self._ScalarState.get_tend("qad")
            nad2t = self._ScalarState.get_tend("qnad2")
            qad2t = self._ScalarState.get_tend("qad2")

            Surface_impl.compute_u10_arr(
                u10,
                v10,
                self._Ref.u0,
                self._Ref.v0,
                self.fac1,
                self.fac2,
                self.gustiness,
                self._u10_arr,
            )

            Surface_impl.compute_aerosol_flux(
                self._u10_arr,
                self._SFLUX_NACC_COEF,
                self._SFLUX_RACC,
                self._SIGMA_ACCUM,
                self._WHITECAP_COEF,
                self._RHO_AEROSOL,
                self._naflux_sfc,
                self._qaflux_sfc,
            )

            Surface_impl.iles_surface_flux_application(
                1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._naflux_sfc, nadt
            )

            Surface_impl.iles_surface_flux_application(
                1e-5, z_edge, dxi2, nh, alpha0, alpha0_edge, 100, self._qaflux_sfc, qadt
            )

            Surface_impl.compute_aerosol_flux(
                self._u10_arr,
                self._SFLUX_NAIT_COEF,
                self._SFLUX_RAIT,
                self._SIGMA_AITKEN,
                self._WHITECAP_COEF,
                self._RHO_AEROSOL,
                self._na2flux_sfc,
                self._qa2flux_sfc,
            )

            Surface_impl.iles_surface_flux_application(
                1e-5,
                z_edge,
                dxi2,
                nh,
                alpha0,
                alpha0_edge,
                100,
                self._na2flux_sfc,
                nad2t,
            )

            Surface_impl.iles_surface_flux_application(
                1e-5,
                z_edge,
                dxi2,
                nh,
                alpha0,
                alpha0_edge,
                100,
                self._qa2flux_sfc,
                qad2t,
            )

        return

    def bflux_from_thflux(self):
        assert self._theta_flux is not None

        nh = self._Grid.n_halo
        nh2 = nh[2]
        self._buoyancy_flux = (
            self._theta_flux * parameters.G / self._Ref.T0_edge[nh2 - 1]
        )

        return

    @property
    def name(self):
        return self._name

    def io_initialize(self, rt_grp):
        return

    def io_update(self, rt_grp):
        return

    def restart(self, data_dict, **kwargs):
        return

    def dump_restart(self, data_dict):
        return
