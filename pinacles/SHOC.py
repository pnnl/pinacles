from pinacles.externals.shoc import shoc_wrapper
from pinacles import parameters
import numpy as np
import numba
import time



@numba.njit
def call_shoc_main(
    nh,
    ngrid_local,
    nlev,
    nlevi,
    dtime,
    nadv,
    host_dx,
    host_dy,
    thv,
    zt_grid,
    zi_grid,
    pres,
    presi,
    pdel,
    wthl_sfc,
    wqw_sfc,
    uw_sfc,
    vw_sfc,
    wtracer_sfc,
    num_qtracers,
    w_field,
    inv_exner,
    phis,
    tke,
    thetal,
    qw,
    u_wind,
    v_wind,
    scalar_arrays,
    wthv_sec,
    tkh,
    tk,
    qc,
    shoc_ql,
    shoc_cldfrac,
    pblh,
    shoc_mix,
    isotropy,
    w_sec,
    thl_sec,
    qw_sec,
    qwthl_sec,
    wthl_sec,
    wqw_sec,
    wtke_sec,
    uw_sec,
    vw_sec,
    w3,
    wqls_sec,
    brunt,
    shoc_ql2,
    T,
    uc,
    vc,
    ut,
    vt,
    tke_t,
    s_t,
    qv_t, 
    qc_t,
    qr_idx,  
    scalar_tendencies
):
    shcol = 1

    nz = ut.shape[2]
    
    host_dse = np.zeros((nz), dtype=np.double)
    host_dse_c = np.zeros((nz), dtype=np.double)
    thlm =  np.zeros((nz), dtype=np.double)
    
    qc_c = np.zeros((nz), dtype=np.double)
    tke_c =np.zeros((nz), dtype=np.double)
    
    qw_c = np.zeros((nz), dtype=np.double)
    
    uc_c = np.zeros((nz), dtype=np.double)
    vc_c = np.zeros((nz), dtype=np.double)


    num_qtracers = len(scalar_tendencies)

    # This is logic that allows us to deal with the case when there are NO 
    # tracers... we need to have at least some memory allocated in the array
    # so bump the zero dimension by one
    bump = 0
    if num_qtracers == 0:
        bump = 1
    qtracers = np.zeros((nz-2*nh[2]) *  num_qtracers + bump, dtype=np.double)
    wtracer_sfc = np.zeros(num_qtracers + bump, dtype=np.double)
        
    
    # Reduce the ranges later
    pblh_a = np.empty((1,), dtype=np.double)
    for i in range(nh[0]-1, ngrid_local[0]-nh[0]+1):
        for j in range(nh[1]-1, ngrid_local[1]-nh[1]+1):
            for k in range(nz):
                # Center the velocity components
                uc[k] = 0.5 * (u_wind[i, j, k] + u_wind[i - 1, j, k])
                vc[k] = 0.5 * (v_wind[i, j, k] + v_wind[i, j - 1, k])
                
                vc_c[k] = vc[k]
                uc_c[k] = uc[k]
                
                
                host_dse[k] = parameters.CPD * T[i,j,k] +  parameters.G * zt_grid[k] + phis[i,j]
                host_dse_c[k] = host_dse[k]
                
                tke_c[k] = tke[i,j,k]
                qc_c[k] = qc[i,j,k]
                qw_c[k] = qw[i,j,k] + qc[i,j,k]
                
                pdel[k] = presi[k] - presi[k-1]
                
                thlm[k] = (T[i,j,k] - (parameters.LV/parameters.CPD) * qc_c[k])*inv_exner[k]
                
            # Pack the array with fortran order
            count = 0
            for sp in range(num_qtracers):
                for k in range(nz-2*nh[2]):
                    qtracers[count] = scalar_arrays[sp][i,j,k+nh[2]]
                    count += 1

            pblh_a[0] = pblh[i,j]
            
            
            shoc_wrapper.shoc_main(
                shcol,
                nlev,
                nlevi,
                dtime,
                nadv,
                host_dx,
                host_dy,
                thv[i,j,nh[2]:-nh[2]],
                zt_grid[nh[2]:-nh[2]],
                zi_grid[nh[2]-1:-nh[2]],
                pres[nh[2]:-nh[2]],
                presi[nh[2]-1:-nh[2]],
                pdel[nh[2]:-nh[2]],
                np.array(wthl_sfc[i,j]),
                np.array(wqw_sfc[i,j]),
                np.array(uw_sfc[i,j]),
                np.array(vw_sfc[i,j]),
                wtracer_sfc,
                num_qtracers,
                w_field[i,j,nh[2]:-nh[2]],
                inv_exner[nh[2]:-nh[2]],
                np.array(phis[i,j]),
                host_dse_c[nh[2]:-nh[2]],
                tke_c[nh[2]:-nh[2]],
                thlm[nh[2]:-nh[2]],
                qw_c[nh[2]:-nh[2]],
                uc_c[nh[2]:-nh[2]],
                vc_c[nh[2]:-nh[2]],
                qtracers,
                wthv_sec[i,j,nh[2]:-nh[2]],
                tkh[i,j,nh[2]:-nh[2]],
                tk[i,j,nh[2]:-nh[2]],
                qc_c[nh[2]:-nh[2]],
                shoc_cldfrac[i,j,nh[2]:-nh[2]],
                pblh_a,
                shoc_mix[i,j,nh[2]:-nh[2]],
                isotropy[i,j,nh[2]:-nh[2]],
                w_sec[i,j,nh[2]:-nh[2]],
                thl_sec[i,j,nh[2]:-nh[2]],
                qw_sec[i,j,nh[2]:-nh[2]],
                qwthl_sec[i,j,nh[2]:-nh[2]],
                wthl_sec[i,j,nh[2]:-nh[2]],
                wqw_sec[i,j,nh[2]:-nh[2]],
                wtke_sec[i,j,nh[2]:-nh[2]],
                uw_sec[i,j,nh[2]:-nh[2]],
                vw_sec[i,j,nh[2]:-nh[2]],
                w3[i,j,nh[2]:-nh[2]],
                wqls_sec[i,j,nh[2]:-nh[2]],
                brunt[i,j,nh[2]:-nh[2]],
                shoc_ql2[i,j,nh[2]:-nh[2]],    
            )
            pblh[i,j] = pblh_a[0]
            
            # Compute tendencies for the prognostics
            for k in range(nz):
                
                
                tke_t[i,j,k] += (tke_c[k] - tke[i,j,k])/dtime
                
                # qw is the water vapor mixing ratio
                qv = qw[i,j,k]
                
                # qw is the water vapor plus the cloud water mixing ratio
                # so we need to subtract off the condensed water 
                qv_c = qw_c[k] - qc_c[k]

                qv_t[i,j,k] += (qv_c - qv)/dtime
                
                qc_t_=  (qc_c[k] - qc[i,j,k])/dtime
                qc_t[i,j,k] += qc_t_
                
                
                # copy chock cloud water into 
                # shoc_ql variable for output purposes only
                shoc_ql[i,j,k] = qc_c[k]
                
                # Compute the change in temperature
                T_t = (host_dse_c[k] - host_dse[k])/dtime / parameters.CPD
                
                #Compute the change in static energy
                s_t[i,j,k] += T_t - parameters.LV / parameters.CPD * qc_t_
                
                u_tend = (uc_c[k] - uc[k])/dtime
                v_tend = (vc_c[k] - vc[k])/dtime
                
                # Average tendencies of velocities back to staggered position 
                ut[i,j,k] += u_tend * 0.5
                ut[i-1,j,k] += u_tend * 0.5
                
                vt[i,j,k] += v_tend * 0.5
                vt[i,j-1,k] += v_tend * 0.5
             
             
            # Unpack the scalars array    
            count = 0
            for sp in range(num_qtracers):
                for k in range(nz-2*nh[2]):

                    scalar_tend = (qtracers[count] -  scalar_arrays[sp][i,j,k+nh[2]])/dtime
                    scalar_tendencies[sp][i,j,k+nh[2]] += scalar_tend
                    count += 1
                    
                    if sp == qr_idx:
                        # Here we make sure that transport of qr conserves static energy
                        s_t[i,j,k] -= parameters.LV / parameters.CPD * scalar_tend

                
    return


class SHOC:
    def __init__(
        self,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
        Surface,
    ):
        self.Grid = Grid
        self.Ref = Ref
        self.ScalarState = ScalarState
        self.VelocityState = VelocityState
        self.DiagnosticState = DiagnosticState
        self.TimeSteppingController = TimeSteppingController
        self.Surface = Surface

        self.nlev = None
        self.pblh = None

        ScalarState.add_variable("shoc_tke")

        DiagnosticState.add_variable("shoc_tk")
        DiagnosticState.add_variable("shoc_tkh")
        DiagnosticState.add_variable("shoc_cldfrac")
        DiagnosticState.add_variable("shoc_ql")
        DiagnosticState.add_variable("shoc_ql2")
        DiagnosticState.add_variable("shoc_mix")
        DiagnosticState.add_variable("shoc_w_sec")
        DiagnosticState.add_variable("shoc_thl_sec")
        DiagnosticState.add_variable("shoc_qw_sec")
        DiagnosticState.add_variable("shoc_qwthl_sec")
        DiagnosticState.add_variable("shoc_wthl_sec")
        DiagnosticState.add_variable("shoc_wthv_sec")
        DiagnosticState.add_variable("shoc_wqw_sec")
        DiagnosticState.add_variable("shoc_wtke_sec")
        DiagnosticState.add_variable("shoc_uw_sec")
        DiagnosticState.add_variable("shoc_vw_sec")
        DiagnosticState.add_variable("shoc_uw_sec")
        DiagnosticState.add_variable("shoc_w3")
        DiagnosticState.add_variable("shoc_wqls_sec")
        DiagnosticState.add_variable("shoc_brunt")
        DiagnosticState.add_variable("shoc_isotropy")
        #

    def initialize(self):
        n_halo = self.Grid.n_halo[2]
        zvir = parameters.RV / parameters.RD - 1
        nlev = self.Grid.n[2]
        nbot_shoc= nlev
        ntop_shoc = 1
        pref_mid = self.Ref.p0[n_halo:-n_halo]


        shoc_wrapper.shoc_init(
            nlev,
            parameters.G,
            parameters.RD,
            parameters.RV,
            parameters.CPD,
            zvir,
            parameters.LV,
            parameters.LF,
            0.40,
            pref_mid,
            nbot_shoc,
            ntop_shoc,
        )

        self.nlev = nlev
        nl = self.Grid.ngrid_local

        self.pblh = np.zeros((nl[0], nl[1]), dtype=np.double)
        self.uc = np.zeros((nl[2],), dtype=np.double)
        self.vc = np.zeros((nl[2],), dtype=np.double)

    def update(self):
        t1 = time.perf_counter()
        nh = self.Grid.n_halo
        z_global = self.Grid.z_global
        z_edge_global = self.Grid.z_edge_global
        ngrid_local = self.Grid.ngrid_local
        host_dx = np.array(self.Grid.dx[0], dtype=np.double)
        host_dy = np.array(self.Grid.dx[1], dtype=np.double)

        nlev = self.nlev
        nlevi = nlev + 1

        # We can precompute these to save time
        zt_grid = z_global
        zi_grid = z_edge_global

        dtime = self.TimeSteppingController.dt

        # Get pressure level data
        pres = self.Ref.p0[:]
        presi = self.Ref.p0_edge[:]

        # Be careful here
        
        
        pdel = np.zeros((z_global.shape[0],), dtype=np.double)
        thetav = self.DiagnosticState.get_field("thetav")

        wthl_sfc = self.Surface._tflx * (1.0/ self.Ref.exner_edge[nh[2]-1])
        wqw_sfc  = self.Surface._qvflx
        uw_sfc = self.Surface._taux_sfc#self.Surface._taux_sfc
        vw_sfc = self.Surface._tauy_sfc#self.Surface._tauy_sfc

        #uw_sfc = self.Surface._taux_sfc
        #vw_sfc = self.Surface._tauy_sfc
        #wqw_sfc = self.Surface._qv_flux_sfc
        #wthl_sfc = self.Surface._theta_flux_sfc

        num_qtracers = 0
        wtracer_sfc = np.zeros((0,), dtype=np.double)
        w_field = self.VelocityState.get_field("w")
        inv_exner = 1.0 / self.Ref.exner
        phis = np.zeros_like(wthl_sfc)


        u_field = self.VelocityState.get_field("u")
        v_field = self.VelocityState.get_field("v")
        w_field = self.VelocityState.get_field("w")

        ut = self.VelocityState.get_tend("u")
        vt = self.VelocityState.get_tend("v")

        tke = self.ScalarState.get_field("shoc_tke")
        tke_t = self.ScalarState.get_tend("shoc_tke")
        
        qv = self.ScalarState.get_field('qv')
        qv_t = self.ScalarState.get_tend('qv')
        
        qc = self.ScalarState.get_field('qc')
        qc_t = self.ScalarState.get_tend("qc")
        
        s = self.ScalarState.get_field('s')
        s_t = self.ScalarState.get_tend('s')
        
        thetal = self.DiagnosticState.get_field("thetal")
        qt = self.DiagnosticState.get_field("qt")
    

        #qtracers = np.zeros((ngrid_local[0], 0), dtype=np.double)
    

        # Get Shoc Specific Fields
        shoc_tk = self.DiagnosticState.get_field("shoc_tk")
        shoc_tkh = self.DiagnosticState.get_field("shoc_tkh")
        shoc_cldfrac = self.DiagnosticState.get_field("shoc_cldfrac")
        shoc_ql = self.DiagnosticState.get_field("shoc_ql")
        shoc_ql2 = self.DiagnosticState.get_field("shoc_ql2")
        shoc_mix = self.DiagnosticState.get_field("shoc_mix")
        shoc_w_sec = self.DiagnosticState.get_field("shoc_w_sec")
        shoc_thl_sec = self.DiagnosticState.get_field("shoc_thl_sec")
        shoc_qw_sec = self.DiagnosticState.get_field("shoc_qw_sec")
        shoc_qwthl_sec = self.DiagnosticState.get_field("shoc_qwthl_sec")
        shoc_wthv_sec = self.DiagnosticState.get_field("shoc_wthv_sec")
        shoc_wthl_sec = self.DiagnosticState.get_field("shoc_wthl_sec")
        shoc_wqw_sec = self.DiagnosticState.get_field("shoc_wqw_sec")
        shoc_wtke_sec = self.DiagnosticState.get_field("shoc_wtke_sec")
        shoc_uw_sec = self.DiagnosticState.get_field("shoc_uw_sec")
        shoc_vw_sec = self.DiagnosticState.get_field("shoc_vw_sec")
        shoc_uw_sec = self.DiagnosticState.get_field("shoc_uw_sec")
        shoc_w3 = self.DiagnosticState.get_field("shoc_w3")
        shoc_wqls_sec = self.DiagnosticState.get_field("shoc_wqls_sec")
        shoc_brunt = self.DiagnosticState.get_field("shoc_brunt")
        shoc_isotropy = self.DiagnosticState.get_field("shoc_isotropy")
        T = self.DiagnosticState.get_field("T")


        # Pack Scalars that need to be transported
        scalar_arrays =  numba.typed.List.empty_list(item_type=numba.float64[:,:,:], allocated=0)
        scalar_tendencies =  numba.typed.List.empty_list(item_type=numba.float64[:,:,:], allocated=0)
        qr_idx = None
        count = 0 
        for name in self.ScalarState._dofs:
            if name != "qv" and name != 'qc' and name != 's' and name != 'shoc_tke':
                scalar_arrays.append(self.ScalarState.get_field(name))
                scalar_tendencies.append(self.ScalarState.get_tend(name))
                if name == "qr":
                    qr_idx = count
            count += 1 
        nadv = 1
        # Actually call the shoc driver
        call_shoc_main(
            nh,
            ngrid_local,
            nlev,
            nlevi,
            dtime/nadv,
            nadv,
            host_dx,
            host_dy,
            thetav,
            zt_grid,
            zi_grid,
            pres,
            presi,
            pdel,
            wthl_sfc,
            wqw_sfc,
            uw_sfc,
            vw_sfc,
            wtracer_sfc,
            num_qtracers,
            w_field,
            inv_exner,
            phis,
            tke,
            thetal,
            qv,
            u_field,
            v_field,
            scalar_arrays,
            shoc_wthv_sec,
            shoc_tkh,
            shoc_tk,
            qc,
            shoc_ql,
            shoc_cldfrac,
            self.pblh,
            shoc_mix,
            shoc_isotropy,
            shoc_w_sec,
            shoc_thl_sec,
            shoc_qw_sec,
            shoc_qwthl_sec,
            shoc_wthl_sec,
            shoc_wqw_sec,
            shoc_wtke_sec,
            shoc_uw_sec,
            shoc_vw_sec,
            shoc_w3,
            shoc_wqls_sec,
            shoc_brunt,
            shoc_ql2,
            T,
            self.uc,
            self.vc,
            ut,
            vt,
            tke_t,
            s_t, 
            qv_t, 
            qc_t,
            qr_idx,
            scalar_tendencies
        )
        

        t2 = time.perf_counter()
        print("SHOC: ", t2 - t1)
