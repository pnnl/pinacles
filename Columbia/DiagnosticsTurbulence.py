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
        profiles_grp.createVariable('qt', np.double, dimensions=('time', 'z',))


        # 2nd moments
        profiles_grp.createVariable('s2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali2', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qt2', np.double, dimensions=('time', 'z',))

        # 3rd moments
        profiles_grp.createVariable('s3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali3', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qt3', np.double, dimensions=('time', 'z',))

        # 4th moments
        profiles_grp.createVariable('s4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qv4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('thetali4', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('qt4', np.double, dimensions=('time', 'z',))

        #  Add resolved turbulent fluxes
        profiles_grp.createVariable('wqt', np.double, dimensions=('time', 'z',))
        profiles_grp.createVariable('wthetali', np.double, dimensions=('time', 'z',))
        for var in self._ScalarState.names:
            profiles_grp.createVariable('w' + var, np.double, dimensions=('time', 'z',))

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


    def _update_velocity_moments(self, this_grp):

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
        MPI.COMM_WORLD.barrier()
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


        return


    def _update_scalar_moments(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()
        for v in ['s', 'qv']:
            #Now compute scalar moments
            phi = self._ScalarState.get_field(v)
            phimean = self._ScalarState.mean(v)

            phi2 = np.zeros_like(phimean)
            phi3 = np.zeros_like(phimean)
            phi4 = np.zeros_like(phimean)

            self.scalar_moments(n_halo, phi, phimean, phi2, phi3, phi4)
            phi2 = UtilitiesParallel.ScalarAllReduce(phi2/npts)
            phi3 = UtilitiesParallel.ScalarAllReduce(phi3/npts)
            phi4 = UtilitiesParallel.ScalarAllReduce(phi4/npts)

            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp['profiles']
                profiles_grp[v+'2'][-1,:]  = phi2[n_halo[2]:-n_halo[2]]
                profiles_grp[v+'3'][-1,:]  = phi3[n_halo[2]:-n_halo[2]]
                profiles_grp[v+'4'][-1,:]  = phi4[n_halo[2]:-n_halo[2]]



        # Compute moments of liquid ice potential temperature
        derived_vars = {}
        derived_vars['thetali'] = self._Thermo.get_thetali()
        derived_vars['qt'] = self._Thermo.get_qt()
        for key, item in derived_vars.items():
            item_mean = UtilitiesParallel.ScalarAllReduce(np.sum(np.sum(item[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1],:],axis=0),axis=0)/npts)
            item2 = np.zeros_like(item_mean)
            item3 = np.zeros_like(item_mean)
            item4 = np.zeros_like(item_mean)
            self.scalar_moments(n_halo, item, item_mean, item2, item3, item4)
            item2= UtilitiesParallel.ScalarAllReduce(item2/npts)
            item3= UtilitiesParallel.ScalarAllReduce(item3/npts)
            item4= UtilitiesParallel.ScalarAllReduce(item4/npts)

           #Only do IO on rank 0
            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp['profiles']
                profiles_grp[key][-1,:]  = item_mean[n_halo[2]:-n_halo[2]]
                profiles_grp[key+'2'][-1,:]  = item2[n_halo[2]:-n_halo[2]]
                profiles_grp[key+'3'][-1,:]  = item3[n_halo[2]:-n_halo[2]]
                profiles_grp[key+'4'][-1,:]  = item4[n_halo[2]:-n_halo[2]]

        return


    @staticmethod
    @numba.njit()
    def compute_vertical_fluxes(n_halo, w_mean, phi_mean, w, phi, fluxz):
        # Compute a resolved vertival flux of phi
        shape = phi.shape
        for i in range(n_halo[0], shape[0] - n_halo[0]):
            for j in range(n_halo[1], shape[1] - n_halo[1]):
                for k in range(n_halo[2], shape[2] - n_halo[2]):
                    flux_above =  w[i,j,k] *  0.5 * (phi[i,j,k+1]-phi_mean[k+1] + phi[i,j,k]-phi_mean[k])
                    flux_below =  w[i,j,k-1] * 0.5 * (phi[i,j,k]-phi_mean[k] + phi[i,j,k-1]-phi_mean[k-1])#fluxz[k] +=  #0.5 * (w[i,j,k-1] + w[i,j,k] - (w_mean[k-1] + w_mean[k])) * (phi[i,j,k]-phi_mean[k])
                    fluxz[k] += 0.5 * (flux_above + flux_below)
        return

    def _update_scalar_fluxes(self, this_grp):

        n_halo = self._Grid.n_halo
        npts = self._Grid.n[0] * self._Grid.n[1]
        my_rank = MPI.COMM_WORLD.Get_rank()

        w = self._VelocityState.get_field('w')
        w_mean = UtilitiesParallel.ScalarAllReduce(np.sum(np.sum(w[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1],:],axis=0),axis=0)/npts)

        derived_vars = {}
        derived_vars['thetali'] = self._Thermo.get_thetali()
        derived_vars['qt'] = self._Thermo.get_qt()

        for key, item in derived_vars.items():
            item_mean = UtilitiesParallel.ScalarAllReduce(np.sum(np.sum(item[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1],:],axis=0),axis=0)/npts)
            item_fluxz = np.zeros_like(w_mean)
            self.compute_vertical_fluxes(n_halo, w_mean, item_mean, w, item, item_fluxz)
            item_fluxz = UtilitiesParallel.ScalarAllReduce(item_fluxz/npts)


            #Only do IO on rank 0
            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp['profiles']
                profiles_grp['w'+key][-1,:]  = item_fluxz[n_halo[2]:-n_halo[2]]

        for key in self._ScalarState.names:
            item = self._ScalarState.get_field(key)
            item_mean = UtilitiesParallel.ScalarAllReduce(np.sum(np.sum(item[n_halo[0]:-n_halo[0],n_halo[1]:-n_halo[1],:],axis=0),axis=0)/npts)
            item_fluxz = np.zeros_like(w_mean)
            self.compute_vertical_fluxes(n_halo, w_mean, item_mean, w, item, item_fluxz)
            item_fluxz = UtilitiesParallel.ScalarAllReduce(item_fluxz/npts)


            #Only do IO on rank 0
            MPI.COMM_WORLD.barrier()
            if my_rank == 0:
                profiles_grp = this_grp['profiles']
                profiles_grp['w'+key][-1,:]  = item_fluxz[n_halo[2]:-n_halo[2]]

        return


    def io_update(self, this_grp):

        self._update_velocity_moments(this_grp)
        self._update_scalar_moments(this_grp)
        self._update_scalar_fluxes(this_grp)

        return

    @property
    def name(self):
        return self._name