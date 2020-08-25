import numpy as np
from mpi4py import MPI
import numba
from Columbia import UtilitiesParallel

class DiagnosticsTurbulence:

    def __init__(self, Grid, Ref, Thermo, Micro, VelocityState, ScalarState, DiagnosticState):

        self._name = 'DiagnosticsTurbulence'
        self._Grid = Grid
        self._Ref = Ref
        self._Thermo = Thermo
        self._Micro = Micro
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState

        return

    def io_initialize(self, this_grp):

        #Get aliases to the timeseries and profiles groups
        timeseries_grp = this_grp['timeseries']
        profiles_grp = this_grp['profiles']

        #Add velocity moments
        # 2nd moments
        profiles_grp.createVariable('u2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('v2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('w2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('tke', np.double, dimensions=('time', 'z',))

        # 3rd moments
        profiles_grp.createVariable('u3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('v3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('w3', np.double, dimensions=('time', 'z',))

        # 4th moments
        profiles_grp.createVariable('u4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('v4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('w4', np.double, dimensions=('time', 'z',))


        #Add thermodynamic field moments
        profiles_grp.createVariable('thetali', np.double, dimensions=('time', 'z',))

        # 2nd moments
        profiles_grp.createVariable('s2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali2', np.double, dimensions=('time', 'z',))

        # 3rd moments
        profiles_grp.createVariable('s3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali3', np.double, dimensions=('time', 'z',))

        # 4th moments
        profiles_grp.createVariable('s4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali4', np.double, dimensions=('time', 'z',))

        return


    @staticmethod
    @numba.njit()
    def velocity_moments(n_halo, u, v, w, umean, vmean, wmean,
                        uu, vv, ww,
                        uuu, vvv, www,
                        uuuu, vvvv, wwww):
        shape = u.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):

                    # Compute cell centered fluctations
                    up = 0.5 * (u[i-1,j,k] + u[i,j,k]) - umean[k]
                    vp = 0.5 * (v[i,j-1,k] + v[i,j,k]) - vmean[k]
                    wp = 0.5 * (w[i,j,k-1] + w[i,j,k]) - 0.5 * (wmean[k-1] + wmean[k])

                    # Second central moment
                    uu[k] += up * up
                    vv[k] += vp * vp
                    ww[k] += wp * wp

                    # Third central moment
                    uuu[k] += up * up * up
                    vvv[k] += vp * vp * vp
                    www[k] += wp * wp * wp

                    # Fourth cental moments
                    uuuu[k] += up * up * up * up
                    vvvv[k] += vp * vp * vp * vp
                    wwww[k] += wp * wp * wp * wp

        return


    @staticmethod
    @numba.njit()
    def scalar_moments(n_halo, phi, phimean, phi2, phi3, phi4):

        shape = phi.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):

                    phip = phi[i,j,k] - phimean[k]
                    phip2 = phip * phip
                    phip3 =  phip2 * phip
                    phi2[k] += phip2
                    phi3[k] += phip3
                    phi4[k] += phip3 * phip

        return

    def io_update(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]

        #First compute velocity moments
        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        umean = self._VelocityState.mean('u')
        vmean = self._VelocityState.mean('v')
        wmean = self._VelocityState.mean('w')

        uu = np.zeros_like(umean)
        vv = np.zeros_like(vmean)
        ww = np.zeros_like(wmean)

        uuu = np.zeros_like(umean)
        vvv = np.zeros_like(vmean)
        www = np.zeros_like(wmean)

        uuuu = np.zeros_like(umean)
        vvvv = np.zeros_like(vmean)
        wwww = np.zeros_like(wmean)

        self.velocity_moments(n_halo, u, v, w, umean, vmean, wmean,
                        uu, vv, ww,
                        uuu, vvv, www,
                        uuuu, vvvv, wwww)

        uu = UtilitiesParallel.ScalarAllReduce(uu/npts)
        vv = UtilitiesParallel.ScalarAllReduce(vv/npts)
        ww = UtilitiesParallel.ScalarAllReduce(ww/npts)

        uuu = UtilitiesParallel.ScalarAllReduce(uuu/npts)
        vvv = UtilitiesParallel.ScalarAllReduce(vvv/npts)
        www = UtilitiesParallel.ScalarAllReduce(www/npts)

        uuuu = UtilitiesParallel.ScalarAllReduce(uuuu/npts)
        vvvv = UtilitiesParallel.ScalarAllReduce(vvvv/npts)
        wwww = UtilitiesParallel.ScalarAllReduce(wwww/npts)


        #Only do IO on rank 0
        my_rank = MPI.COMM_WORLD.Get_rank()
        if my_rank == 0:
            profiles_grp = this_grp['profiles']
            profiles_grp['u2'][-1,:] = uu[n_halo[2]:-n_halo[2]]
            profiles_grp['v2'][-1,:] = vv[n_halo[2]:-n_halo[2]]
            profiles_grp['w2'][-1,:] = ww[n_halo[2]:-n_halo[2]]
            profiles_grp['tke'][-1,:] = 0.5 * (uu[n_halo[2]:-n_halo[2]] + vv[n_halo[2]:-n_halo[2]] + ww[n_halo[2]:-n_halo[2]])


            profiles_grp['u3'][-1,:] = uuu[n_halo[2]:-n_halo[2]]
            profiles_grp['v3'][-1,:] = vvv[n_halo[2]:-n_halo[2]]
            profiles_grp['w3'][-1,:] = www[n_halo[2]:-n_halo[2]]

            profiles_grp['u4'][-1,:] = uuuu[n_halo[2]:-n_halo[2]]
            profiles_grp['v4'][-1,:] = vvvv[n_halo[2]:-n_halo[2]]
            profiles_grp['w4'][-1,:] = wwww[n_halo[2]:-n_halo[2]]


        for v in ['s', 'qv']:
            #Now compute scalar moments
            phi = self._ScalarState.get_field(v)
            phimean = self._ScalarState.mean('s')


            phi2 = np.zeros_like(phimean)
            phi3 = np.zeros_like(phimean)
            phi4 = np.zeros_like(phimean)

            self.scalar_moments(n_halo, phi, phimean, phi2, phi3, phi4)
            phi2 = UtilitiesParallel.ScalarAllReduce(phi2/npts)
            phi3 = UtilitiesParallel.ScalarAllReduce(phi3/npts)
            phi4 = UtilitiesParallel.ScalarAllReduce(phi4/npts)

            if my_rank == 0:
                profiles_grp = this_grp['profiles']
                profiles_grp[v+'2'][-1,:]  = phi2[n_halo[2]:-n_halo[2]]
                profiles_grp[v+'3'][-1,:]  = phi3[n_halo[2]:-n_halo[2]]
                profiles_grp[v+'4'][-1,:]  = phi4[n_halo[2]:-n_halo[2]]



        # Compute moments of liquid ice potential temperature
        thetali = self._Thermo.get_thetali()
        thetali_mean = UtilitiesParallel.ScalarAllReduce(np.sum(np.sum(thetali[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1],:],axis=0),axis=0)/npts)
        thetali2 = np.zeros_like(thetali_mean)
        thetali3 = np.zeros_like(thetali_mean)
        thetali4 = np.zeros_like(thetali_mean)
        self.scalar_moments(n_halo, thetali, thetali_mean, thetali2, thetali3, thetali4)
        thetali2= UtilitiesParallel.ScalarAllReduce(thetali2/npts)
        thetali3= UtilitiesParallel.ScalarAllReduce(thetali3/npts)
        thetali4= UtilitiesParallel.ScalarAllReduce(thetali4/npts)

        if my_rank == 0:
            profiles_grp['thetali'][-1,:]  = thetali_mean[n_halo[2]:-n_halo[2]]
            profiles_grp['thetali2'][-1,:]  = thetali2[n_halo[2]:-n_halo[2]]
            profiles_grp['thetali3'][-1,:]  = thetali3[n_halo[2]:-n_halo[2]]
            profiles_grp['thetali4'][-1,:]  = thetali4[n_halo[2]:-n_halo[2]]




        return

    @property
    def name(self):
        return self._name