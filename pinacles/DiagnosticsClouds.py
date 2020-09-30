import numpy as np
from mpi4py import MPI
import numba
from Columbia import UtilitiesParallel

class DiagnosticsClouds:

    def __init__(self, Grid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState):

        self._name = 'DiagnosticsClouds'
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._Micro = Micro
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        return

    def io_initialize(self, this_grp):

        if 'qc' not in self._ScalarState.names:
            return

        #Get aliases to the timeseries and profiles groups
        timeseries_grp = this_grp['timeseries']
        profiles_grp = this_grp['profiles']

        profiles_grp.createVariable('cloud_frac', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('core_frac', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('u_core', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('v_core', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('w_core', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('u_cloud', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('v_cloud', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('w_cloud', np.double, dimensions=('time', 'z',))


        return


    @staticmethod
    @numba.njit()
    def _compute_cloud_conditional_velocities(n_halo, qc, u, v, w,
                                              cloud_count, core_count, u_cloud, v_cloud, w_cloud,
                                              u_core, v_core, w_core):


        shape = qc.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):

                    # Cloud stats
                    if qc[i,j,k] > 1e-5:
                        # Get a cell centered velocity
                        uc = 0.5 * (u[i,j,k] + u[i-1,j,k])
                        vc = 0.5 * (v[i,j,k] + v[i,j-1,k])
                        wc = 0.5 * (w[i,j,k] + w[i,j,k-1])

                        cloud_count[k] += 1.0
                        u_cloud[k] += uc
                        v_cloud[k] += vc
                        w_cloud[k] += wc

                        # Cloud core stats
                        if wc > 0.0:
                            core_count[k] += 1.0
                            u_core[k] += uc
                            v_core[k] += vc
                            w_core[k] += wc

        return


    def _update_cloud_conditional_velocities(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')
        qc = self._ScalarState.get_field('qc')

        cloud_count = np.zeros((qc.shape[2],), dtype=np.double, order='C')
        core_count =  np.zeros_like(cloud_count)

        u_cloud = np.zeros_like(cloud_count)
        v_cloud = np.zeros_like(cloud_count)
        w_cloud = np.zeros_like(cloud_count)

        u_core = np.zeros_like(cloud_count)
        v_core = np.zeros_like(cloud_count)
        w_core = np.zeros_like(cloud_count)

        self._compute_cloud_conditional_velocities(n_halo, qc, u, v, w,
                                              cloud_count, core_count, u_cloud, v_cloud, w_cloud,
                                              u_core, v_core, w_core)


        cloud_count = UtilitiesParallel.ScalarAllReduce(cloud_count)
        cloud_frac = cloud_count/npts
        cloud_points = cloud_frac > 0
        for var in [u_cloud, v_cloud, w_cloud]:
            var[:] = UtilitiesParallel.ScalarAllReduce(var)
            var[cloud_points] = var[cloud_points]/cloud_count[cloud_points]

        core_count = UtilitiesParallel.ScalarAllReduce(core_count)
        core_frac = core_count/npts
        core_points = core_frac > 0
        for var in [u_core, v_core, w_core]:
            var[:] = UtilitiesParallel.ScalarAllReduce(var)
            var[core_points] = var[core_points]/core_count[core_points]


        MPI.COMM_WORLD.barrier()
        if my_rank == 0:
            profiles_grp = this_grp['profiles']
            profiles_grp['cloud_frac'][-1,:] = cloud_frac[n_halo[2]:-n_halo[2]]
            profiles_grp['core_frac'][-1,:] = core_frac[n_halo[2]:-n_halo[2]]

            profiles_grp['u_cloud'][-1,:] = u_cloud[n_halo[2]:-n_halo[2]]
            profiles_grp['v_cloud'][-1,:] = v_cloud[n_halo[2]:-n_halo[2]]
            profiles_grp['w_cloud'][-1,:] = w_cloud[n_halo[2]:-n_halo[2]]

            profiles_grp['u_core'][-1,:] = u_core[n_halo[2]:-n_halo[2]]
            profiles_grp['v_core'][-1,:] = v_core[n_halo[2]:-n_halo[2]]
            profiles_grp['w_core'][-1,:] = w_core[n_halo[2]:-n_halo[2]]



        return

    def _update_cloud_conditional_scalars(self, this_grp):

        return


    def io_update(self, this_grp):

        if 'qc' not in self._ScalarState.names:
            return

        self._update_cloud_conditional_velocities(this_grp)
        self._update_cloud_conditional_scalars(this_grp)

        return

    @property
    def  name(self):
        return self._name


