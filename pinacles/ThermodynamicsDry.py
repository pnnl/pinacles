import numpy as np
import numba
from mpi4py import MPI
from pinacles import Thermodynamics, ThermodynamicsDry_impl
from pinacles import parameters
from functools import partial
import jax 

class ThermodynamicsDry(Thermodynamics.ThermodynamicsBase):
    def __init__(self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState):
        Thermodynamics.ThermodynamicsBase.__init__(
            self, Timers, Grid, Ref, ScalarState, VelocityState, DiagnosticState, None
        )

        ScalarState.add_variable(
            "qv",
            long_name="Water vapor mixing ratio",
            latex_name="q_v",
            units="kg kg^{-1}",
        )

        DiagnosticState.add_variable(
            "bvf",
            long_name="Brunt–Väisälä frequency squared",
            latex_name="N^2",
            units="s^-2",
        )

        DiagnosticState.add_variable(
            "thetav",
            long_name="Virtual Potential Temperature",
            latex_name="\theta_v",
            units="K",
        )

        DiagnosticState.add_variable(
            "s_dry",
            long_name = "Dry Static Energy", 
            latex_name="s_d",
            units="K"
        )

        self._Timers.add_timer("ThermoDynamicsDry_update")

        return

    def update(self, apply_buoyancy=True):

        self._Timers.start_timer("ThermoDynamicsDry_update")

        n_halo = self._Grid.n_halo
        z = self._Grid.z_global
        dz = self._Grid.dx[2]
        dxi = self._Grid.dxi
        p0 = self._Ref.p0
        alpha0 = self._Ref.alpha0
        T0 = self._Ref.T0
        exner = self._Ref.exner
        tref = self._Ref.T0

        theta_ref = T0 / exner

        s = self._ScalarState.get_field("s")
        s_dry = self._DiagnosticState.get_field("s_dry")
        qv = self._ScalarState.get_field("qv")
        T = self._DiagnosticState.get_field("T")
        alpha = self._DiagnosticState.get_field("alpha")
        buoyancy = self._DiagnosticState.get_field("buoyancy")
        thetav = self._DiagnosticState.get_field("thetav")
        bvf = self._DiagnosticState.get_field("bvf")
        w_t = self._VelocityState.get_tend("w")
        buoyancy_gradient_mag = self._DiagnosticState.get_field("buoyancy_gradient_mag")

        ThermodynamicsDry_impl.eos(z, p0, alpha0, s, qv, T, tref, alpha, buoyancy)
        
        
        s_dry[:,:,:] = s[:,:,:]
        
        ThermodynamicsDry_impl.compute_bvf(
            n_halo, theta_ref, exner, T, qv, dz, thetav, bvf
        )

        if apply_buoyancy:
            ThermodynamicsDry_impl.apply_buoyancy(buoyancy, w_t)

        self._DiagnosticState.remove_mean("buoyancy")

        self.compute_buoyancy_gradient(dxi, buoyancy, buoyancy_gradient_mag)

        self._Timers.end_timer("ThermoDynamicsDry_update")

        return

    def io_fields2d_update(self, fx):

        start = self._Grid.local_start
        end = self._Grid._local_end
        nh = self._Grid.n_halo

        send_buffer = np.zeros((self._Grid.n[0], self._Grid.n[1]), dtype=np.double)
        recv_buffer = np.empty_like(send_buffer)

        # Output Temperature
        if fx is not None:
            t = fx.create_dataset(
                        "T",
                        (1, self._Grid.n[0], self._Grid.n[1]),
                        dtype=np.double,
                    )

            for i, d in enumerate(["time", "X", "Y"]):
                t.dims[i].attach_scale(fx[d])
                

        T = self._DiagnosticState.get_field("T")
        send_buffer[start[0] : end[0], start[1] : end[1]] = T[
            nh[0] : -nh[0], nh[1] : -nh[1], nh[2]
        ]
        MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
        if fx is not None:
            t[:, :] = recv_buffer

        return
    


    @staticmethod
    @numba.njit()
    def compute_thetali(exner, T, thetali):
        shape = T.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    thetali[i, j, k] = T[i, j, k] / exner[k]
        return

    def get_thetali(self):
        exner = self._Ref.exner
        T = self._DiagnosticState.get_field("T")
        thetali = np.empty_like(T)
        self.compute_thetali(exner, T, thetali)
        return thetali

    def get_qt(self):
        # Todo this gets a copy. So modifying it does nothing!
        qv = self._ScalarState.get_field("qv")
        return np.copy(qv)
    
@jax.jit
def update_jax(Grid, Ref, Scalars, Velocities, Diagnostics):
    
    nh = Grid.n_halo
    
    p0 = Ref.p0
    exner = Ref.exner
    T0 = Ref.T0
    alpha0 = Ref.alpha0
    z = Grid.z_global
    idz = Grid.dxi[2]
    
    theta_ref = T0 / exner
    
    s = Scalars.get_field('s')    
    qv = Scalars.get_field('qv')
    wt = Velocities.get_tend('w')
    
    thetav = Diagnostics.get_field('thetav')
    bvf = Diagnostics.get_field('bvf')
    
    #Compute thermodynamic quantities
    T = s - parameters.G * z[np.newaxis, np.newaxis, :] * parameters.ICPD
    alpha = 1.0/(p0[np.newaxis, np.newaxis,:])*parameters.RD * T
    buoyancy = parameters.G * (alpha - alpha0[np.newaxis, np.newaxis, :]) / alpha
    
    #Compute Velocity Tendency
    wt = wt.at[:,:,:-1].add(0.5 * (buoyancy[:,:,:-1] + buoyancy[:,:,1:]))
    
    #Compute virtual potential temperature
    thetav = T / exner[np.newaxis, np.newaxis, :] * (1.0 + 0.61 * qv)
 
    #Compute buoyancy frequency
    # @ surface
    k = nh[2]
    bvf = bvf.at[:,:,k].set(parameters.G / theta_ref[np.newaxis, np.newaxis, k] * (thetav[:,:,k+1] - thetav[:,:,k])*idz)

    # @ interior 
    kstart = nh[2] + 1 
    kend = -nh[2]
    bvf = bvf.at[:,:,kstart:kend].set(parameters.G / theta_ref[np.newaxis, np.newaxis, kstart:kend] * (thetav[:,:,kstart:kend] - thetav[:,:,kstart-1:kend-1])*idz)
    
    # @ domain top
    k = -nh[2] - 1
    bvf = bvf.at[:,:,k].set(parameters.G / theta_ref[np.newaxis, np.newaxis, k] * (thetav[:,:,k] - thetav[:,:,k-1])*idz)
    
    
    #Update containers
    Diagnostics = Diagnostics.set_field('thetav', thetav)
    Diagnostics = Diagnostics.set_field('T', T)
    Diagnostics = Diagnostics.set_field('buoyancy', buoyancy)
    Diagnostics = Diagnostics.set_field('bvf', bvf)
    Velocities = Velocities.set_tend('w', wt)
    
    return  Velocities, Diagnostics