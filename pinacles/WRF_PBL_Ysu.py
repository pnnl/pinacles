from pinacles.externals.ysu import ysu_via_cffi
from pinacles import parameters
import numpy as np

from pinacles.WRFUtil import (
    to_wrf_order,
    to_wrf_order_4d,
    to_our_order_4d,
    to_our_order,
)

from pinacles.WRFUtil import (
    to_wrf_order_halo,
    to_wrf_order_4d_halo,
    to_our_order_4d_halo,
    to_our_order_halo,
)


class PBL_Ysu:
    def __init__(
        self,
        namelist,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        Surface,
        Radiation,
        TimeSteppingController,
    ):

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._Surface = Surface
        self._Radiation = Radiation
        self._TimesteppingController = TimeSteppingController

        nhalo = self._Grid.n_halo
        self._our_dims = self._Grid.ngrid_local
        self._wrf_dims = (
            self._our_dims[0] - 2 * nhalo[0],
            self._our_dims[2] - 2 * nhalo[2],
            self._our_dims[1] - 2 * nhalo[1],
        )

        # Allocate WRF arrays
        self.wrf_vars_3d = [
            "u3d",
            "v3d",
            "qv3d",
            "qc3d",
            "qi3d",
            "p3d",
            "p3di",
            "pi3d",
            "th3d",
            "t3d",
            "rublten",
            "rvblten",
            "rthblten",
            "rqvblten",
            "rqcblten",
            "rqiblten",
            "dz8w",
            "exch_h",
            "exch_m",
            "rthraten",
            "a_u_bep",
            "a_v_bep",
            "a_t_bep",
            "a_q_bep",
            "a_e_bep",
            "b_u_bep",
            "b_v_bep",
            "b_t_bep",
            "b_q_bep",
            "b_e_bep",
            "dlg_bep",
            'sf_bep',
            "dl_u_bep",
            "vl_bep",
            "exch_h",
            "exch_m",
        ]

        self.flag_qi = True

        for i in range(len(self.wrf_vars_3d)):
            self.wrf_vars_3d[i] += "_wrf"

            setattr(
                self,
                "_" + self.wrf_vars_3d[i],
                np.empty(tuple(self._wrf_dims), order="F", dtype=np.double),
            )

        self.wrf_vars_2d = [
            "psfc",
            "znt",
            "ust",
            "hpbl",
            "psim",
            "psih",
            "xland",
            "hfx",
            "qfx",
            "wspd",
            "br",
            "wstar",
            "delta",
            "u10",
            "v10",
            "uoce",
            "voce",
            "ctopo",
            "ctopo2",
            "frc_urb2d",
        ]


        for i in range(len(self.wrf_vars_2d)):
            self.wrf_vars_2d[i] += "_wrf"

            setattr(
                self,
                "_" + self.wrf_vars_2d[i],
                np.zeros((self._wrf_dims[0], self._wrf_dims[2]), order="F", dtype=np.double),
            )
        self._ysu_cffi = ysu_via_cffi.YSU()

        import sys
        
        self._kbl2d_wrf = np.zeros((self._wrf_dims[0], self._wrf_dims[2]), order="F", dtype=np.intc)

        #sys.exit()

        return

    def update(self):

        nhalo = self._Grid.n_halo

        # Reorder essential arrays
        T = self._DiagnosticState.get_field("T")
        s = self._ScalarState.get_field("s")
        u = self._VelocityState.get_field("u")
        v = self._VelocityState.get_field("v")
        qv = self._ScalarState.get_field("qv")
        qc = self._ScalarState.get_field("qc")
        qi = self._ScalarState.get_field("qi1")

        to_wrf_order(nhalo, u, self._u3d_wrf)
        to_wrf_order(nhalo, v, self._v3d_wrf)
        to_wrf_order(nhalo, qv, self._qv3d_wrf)
        to_wrf_order(nhalo, qc, self._qc3d_wrf)
        to_wrf_order(nhalo, qi, self._qi3d_wrf)
        to_wrf_order(nhalo, T, self._t3d_wrf)

        p0 = self._Ref.p0
        p0i = self._Ref.p0_edge
        self._p3d_wrf[:, :, :] = p0[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]
        self._p3di_wrf[:, :, :] = p0i[
            np.newaxis, nhalo[2] - 1 : -nhalo[2] -1, np.newaxis
        ]
        
        
        print('p0!!!!!!', np.amax(self._p3d_wrf[:, :, :]), p0)
        
        
        exner = self._Ref.exner
        self._pi3d_wrf[:, :, :] = exner[np.newaxis, nhalo[2] : -nhalo[2], np.newaxis]

        self._dz8w_wrf[:, :, :] = self._Grid.dx[2]

        self._th3d_wrf.fill(0.0)

        self._xland_wrf.fill(2)
        self._uoce_wrf.fill(0.0)
        self._voce_wrf.fill(0.0)
        self._ctopo_wrf.fill(1.0)
        self._ctopo2_wrf.fill(1.0)
        self._frc_urb2d_wrf.fill(0.0)
        self._br_wrf[:,:] = self._Surface._Ri[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._wspd_wrf[:,:] = self._Surface._windspeed_sfc[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._znt_wrf[:,:] = self._Surface._z0[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._hfx_wrf[:,:] = self._Surface._shf[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._qfx_wrf[:,:] = self._Surface._lhf[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._psim_wrf[:,:] = self._Surface._psi_m[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._psih_wrf[:,:] = self._Surface._psi_h[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._psfc_wrf[:,:] = self._Ref._P0_edge[nhalo[2] - 1] 
        self._ust_wrf[:,:] = self._Surface._ustar[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._u10_wrf[:,:] = self._Surface._u10[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]
        self._v10_wrf[:,:] = self._Surface._v10[nhalo[0] : -nhalo[0],nhalo[1] : -nhalo[1]]

        # Get grid dimensions
        ids = 1
        jds = 1
        kds = 1
        ide = self._wrf_dims[0]
        jde = self._wrf_dims[2]
        kde = self._wrf_dims[1]
        ims = 1
        jms = 1
        kms = 1
        ime = self._wrf_dims[0]
        jme = self._wrf_dims[2]
        kme = self._wrf_dims[1]
        its = 1
        jts = 1
        kts = 1
        ite = ime
        jte = jme
        kte = kme

        self._ysu_cffi.ysu(
            True,
            self._u3d_wrf,
            self._v3d_wrf,
            self._qv3d_wrf,
            self._qc3d_wrf,
            self._qi3d_wrf,
            self._p3d_wrf,
            self._p3di_wrf,
            self._pi3d_wrf,
            self._th3d_wrf,
            self._t3d_wrf,
            self._rublten_wrf,
            self._rvblten_wrf,
            self._rthblten_wrf,
            self._rqvblten_wrf,
            self._rqcblten_wrf,
            self._rqiblten_wrf,
            True,
            parameters.CPD,
            parameters.G,
            parameters.RD / parameters.CPD,
            parameters.RD,
            parameters.RD / parameters.G,
            (parameters.RV / parameters.RD - 1.0),
            parameters.RD / parameters.RV,
            0.41,
            parameters.LV,
            parameters.RV,
            self._dz8w_wrf,
            self._psfc_wrf,
            self._znt_wrf,
            self._ust_wrf,
            self._hpbl_wrf,
            self._psim_wrf,
            self._psih_wrf,
            self._xland_wrf,
            self._hfx_wrf,
            self._qfx_wrf,
            self._wspd_wrf,
            self._br_wrf,
            self._TimesteppingController.dt,
            self._kbl2d_wrf,
            self._exch_h_wrf,
            self._exch_m_wrf,
            self._wstar_wrf,
            self._delta_wrf,
            self._u10_wrf,
            self._v10_wrf,
            self._uoce_wrf,
            self._voce_wrf,
            self._rthraten_wrf,
            1,
            self._ctopo_wrf,
            self._ctopo2_wrf,
            0,
            False,
            self._frc_urb2d_wrf,
            self._a_u_bep_wrf,
            self._a_v_bep_wrf,
            self._a_t_bep_wrf,
            self._a_q_bep_wrf,
            self._a_e_bep_wrf,
            self._b_u_bep_wrf,
            self._b_v_bep_wrf,
            self._b_t_bep_wrf,
            self._b_q_bep_wrf,
            self._b_e_bep_wrf,
            self._dlg_bep_wrf,
            self._dl_u_bep_wrf,
            self._sf_bep_wrf,
            self._vl_bep_wrf,
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

        print(np.amax(self._rublten_wrf))

        return
