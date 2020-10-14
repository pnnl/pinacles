import numpy as np
import numba
import pylab as plt
from numba.core import types
from numba.typed import Dict, List
import time

class ParticlesBase:

    def __init__(self, Grid, TimeSteppingController, VelocityState, ScalarState, DiagnosticState):

        # Initialize data
        self._Grid = Grid
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._particle_dofs = 0    # The first three dofs are always position
        self._interp_particle_dofs = 0
        self._nointerp_particle_dofs = 0
        self._particle_data = None
        self._initialzied = False
        self._n_buffer =  4 #The size of the buffer

        # These are numba dictionaries
        self._particle_varnames = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        self._interp_particle_varnames = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        self._nointerp_particle_varnames = Dict.empty(key_type=types.unicode_type, value_type=types.int64)

        self.add_particle_variable('valid')
        self.add_particle_variable('x')
        self.add_particle_variable('y')
        self.add_particle_variable('z')


        self.add_particle_variable('u')
        self.add_particle_variable('v')
        self.add_particle_variable('w')


        n = 0
        xp = np.random.uniform(2600, 2650, n) 
        yp = np.random.uniform(2600, 2650, n)
        zp = np.random.uniform(50,100.0, n)


        self._allocate_memory()
        self.initialize_particles(xp, yp, zp)
        self.call_count = 1000000000

        print(np.sum(self._n))
        return

    def _allocate_memory(self):
        local_shape = self._Grid.local_shape

        #This is list of floatng point numpy arrays (that are not necessarily of the same size)
        self._particle_data = List.empty_list(item_type=types.float64[:,:])

        #Store the valid particle here
        self._n = np.zeros((local_shape[0], local_shape[1], local_shape[2]), dtype=np.int64)

        return


    def initialize_particles(self, xp, yp, zp):
        assert(xp.shape == yp.shape)
        assert(xp.shape == zp.shape)
        low_corner = (self._Grid.x_range[0], self._Grid.y_range[0], self._Grid.z_range[0])
        local_shape = self._Grid.local_shape
        dx = self._Grid.dx
        n_per_cell = np.zeros(local_shape, dtype=np.int64)

        self.map_particles_allocate(low_corner, dx, xp, yp, zp, self._particle_varnames,  self._n_buffer, self._particle_dofs, self._n, self._particle_data)

        return

    @staticmethod
    @numba.njit
    def map_particles_allocate(low_corner, dx, xp, yp, zp, particle_varnames, n_buffer, particle_dofs, n, particle_data):

        #First determine how many points are valid
        valid_dof = particle_varnames['valid']
        x_dof = particle_varnames['x']
        y_dof = particle_varnames['y']
        z_dof = particle_varnames['z']

        npart = xp.shape[0]
        for pi in range(npart):
            i = int((xp[pi] - low_corner[0])//dx[0])
            j = int((yp[pi] - low_corner[1])//dx[1])
            k = int((zp[pi] - low_corner[2])//dx[2])
            n[i,j,k] += 1

        #Loop over the grid and allocate particle arrays
        shape = n.shape
        ishift = shape[1]* shape[2]
        jshift =  shape[2]
        for i in range(shape[0]):
            ii = i * ishift
            for j in range(shape[1]):
                jj = j * jshift
                for k in range(shape[2]):
                    particle_data.append(np.zeros((particle_dofs, max(0, n[i,j,k])), dtype=np.double))
                    arr = particle_data[ii + jj + k]
                    arr[valid_dof,:] = 0.0

        for pi in range(npart):
            i = int((xp[pi] - low_corner[0])//dx[0])
            j = int((yp[pi] - low_corner[1])//dx[1])
            k = int((zp[pi] - low_corner[2])//dx[2])
            arr = particle_data[i * ishift + j * jshift + k]

            fits = False
            for n in range(arr.shape[1]):

                if arr[valid_dof, n] == 0.0:
                   arr[valid_dof, n] = 1.0
                   arr[x_dof, n] = xp[pi]
                   arr[y_dof, n] = yp[pi]
                   arr[z_dof, n] = zp[pi]
                   fits = True
                   break

            if not fits:
                n = arr.shape[1]
                new_buf = np.zeros((arr.shape[0], n_buffer), dtype=np.double)
                particle_data[i * ishift + j * jshift + k] = np.concatenate((arr, new_buf),axis=1)
                arr = particle_data[i * ishift + j * jshift + k]
                arr[valid_dof, n] = 1.0
                arr[x_dof, n] = xp[pi]
                arr[y_dof, n] = yp[pi]
                arr[z_dof, n] = zp[pi]

        return


    @staticmethod
    #@numba.njit()
    def point_inject(low_corner, high_corner, local_shape, dx, n, particle_varnames, particle_data, x, y, z, n_total):

        #If the point is not on this rank then reutrn
        for lp, li, hi in zip([x,y,z], low_corner, high_corner):
            if lp  <= li or lp > hi:
                return


        #Now that we are on the correct rank let's find compute the i,j,k index
        i = int((x - low_corner[0])//dx[0])
        j = int((y - low_corner[1])//dx[1])
        k = int((z - low_corner[2])//dx[2])


        # Now let's check and see if we need to add particles to this rank
        if n[i,j,k] >= n_total:
            return

        #Get the particle array at this point
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        ii = i * ishift
        jj = j * jshift
        arr = particle_data[ii + jj + k]
        n_arr = arr.shape[1]

        #Get the particle  dofs for the position
        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        zdof = particle_varnames['z']
        valid = particle_varnames['valid']

        #Now let check an see if we need to allocate more memory
        if n_arr < n_total:
            new_buf = np.zeros((arr.shape[0], n_total - n_arr), dtype=np.double)
            particle_data[ii + jj + k] = np.concatenate((arr, new_buf),axis=1)
            arr = particle_data[ii + jj + k]

       # print("adding particles", arr.shape)
        for pi in range(arr.shape[1]):
            if arr[valid, pi] == 0.0:
                arr[xdof, pi] = np.random.uniform(low_corner[0] + dx[0]*i, low_corner[0] + dx[0]*(i+1))
                arr[ydof, pi] = np.random.uniform(low_corner[1] + dx[1]*j, low_corner[1] + dx[1]*(j+1))
                arr[zdof, pi] = np.random.uniform(low_corner[2] + dx[2]*k, low_corner[2] + dx[2]*(k+1))
                arr[valid,pi] = 1.0
                n[i,j,k] += 1

        #xdof = particle_varnames['x']
        #ydof = particle_varnames['y']
        #zdof = particle_varnames['z']
        #valid = particle_varnames['val']

        #ishift = local_shape[1] * local_shape[2]
        #jshift = local_shape[2]
        #ii = i * ishift
        #jj = j * jshift
        #arr = particle_data[ii + jj + k]
        #n_arr = arr.shape[1]
        #if n[i,j,k] < ntotal:
        #    if n_arr != ntotal:
        #        #Create a new buffer
        #        new_buf = np.zeros((arr.shape[0], ntotal - n_arr), dtype=np.double)
        #    particle_data[ii + jj + k] = np.concatenate((arr, new_buf),axis=1)




        return

    def get_particle_var(self, name):
        if name in self._interp_particle_varnames:
            indx = self._interp_particle_varnames[name]
            return indx
        elif name in self._nointerp_particle_varnames:
            indx = self._nointerp_particle_varnames[name] + self._interp_particle_dofs
            return  indx

    def update(self):


        return


    def update_position(self):


        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        local_shape = self._Grid.local_shape
        low_corner = (self._Grid.x_range[0], self._Grid.y_range[0], self._Grid.z_range[0])
        high_corner = (self._Grid.x_range[1], self._Grid.y_range[1], self._Grid.z_range[1])
        l = self._Grid.l
       # #print(xdof, ydof, zdof)

        t0 = time.perf_counter()
        self.compute_new_position(local_shape, self._n, self._particle_varnames, self._particle_data, self._TimeSteppingController.dt) 

        self._surface_bounce(local_shape, self._n, self._particle_varnames, self._particle_data)

        self._boundary_exit_serial(local_shape, l, self._n, self._particle_varnames, self._particle_data)

        self.move_particles_on_grid(low_corner, local_shape, self._Grid.dx, self._n_buffer, self._n, self._particle_varnames, self._particle_data)

        self.point_inject(low_corner, high_corner, local_shape, self._Grid.dx, self._n, self._particle_varnames, self._particle_data, 10240./10.0, 10240.0/2, 80.0, 1024)

        self.interpolate_pt(local_shape, self._Grid.n_halo, low_corner, self._Grid.dx, self._n, self._particle_varnames, self._particle_data, u, 'u', 0)
        self.interpolate_pt(local_shape, self._Grid.n_halo, low_corner, self._Grid.dx, self._n, self._particle_varnames, self._particle_data, v, 'v', 1)
        self.interpolate_pt(local_shape, self._Grid.n_halo, low_corner, self._Grid.dx, self._n, self._particle_varnames, self._particle_data, w, 'w', 2)
        t1 = time.perf_counter()
    
        print(low_corner)
        print('Timing: ', t1 - t0)
        print(np.sum(self._n))
        #import sys; sys.exit()

        xp = np.empty((np.sum(self._n),), dtype=np.double)
        yp = np.empty((np.sum(self._n),), dtype=np.double)
        zp = np.empty((np.sum(self._n),), dtype=np.double)
        self.distill_dof(self._particle_varnames, self._n,  'x', self._particle_data, xp)
        self.distill_dof(self._particle_varnames, self._n,  'y', self._particle_data, yp)
        self.distill_dof(self._particle_varnames, self._n,  'z', self._particle_data, zp)



        #print(np.amin(xp), np.amax(xp), np.amin(yp), np.amax(yp), np.amin(zp), np.amax(zp), np.shape(xp))
        import pylab as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xp, yp, zp, s=0.1)
        ax.axes.set_xlim3d(left=0, right=10240.0)
        ax.axes.set_ylim3d(bottom=0, top=10240.0)
        ax.axes.set_zlim3d(bottom=0, top=2048.0)
        plt.savefig('./part_figs/' + str(self.call_count) + '.png' ,dpi=300)
        plt.close()
        #import sys; sys.exit()
        self.call_count += 1
        print('plot finished')
        return

    @staticmethod
    @numba.njit()
    def compute_new_position(local_shape, n, particle_varnames, particle_data, dt):

        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        zdof = particle_varnames['z']
        valid = particle_varnames['valid']

        udof = particle_varnames['u']
        vdof = particle_varnames['v']
        wdof = particle_varnames['w']

        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid,p] != 0.0:
                                arr[xdof,p] += (arr[udof,p] + np.random.normal(loc=0.0, scale=0.4)) * dt
                                arr[ydof,p] += (arr[vdof,p] + np.random.normal(loc=0.0, scale=0.4))* dt
                                arr[zdof,p] += (arr[wdof,p]+ np.random.normal(loc=0.0, scale=0.4)) * dt

        return


    @staticmethod
    @numba.njit()
    def _surface_bounce(local_shape, n, particle_varnames, particle_data):
        zdof = particle_varnames['z']
        valid = particle_varnames['valid']
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid,p] != 0.0:
                                arr[zdof,p] = np.abs(arr[zdof,p])
        return

    @staticmethod
    @numba.njit()
    def _boundary_periodic_serial(local_shape, high_corner, n, particle_varnames, particle_data):
        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        valid = particle_varnames['valid']
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid,p] != 0.0:
                                arr[xdof,p] = arr[xdof, p]%high_corner[0]
                                arr[ydof,p] = arr[ydof, p]%high_corner[1]
        return


    @staticmethod
    @numba.njit()
    def _boundary_exit_serial(local_shape, high_corner, n, particle_varnames, particle_data):
        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        valid = particle_varnames['valid']
        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid,p] != 0.0:
                                if arr[xdof,p] >= high_corner[0] or arr[ydof,p] <= 0  or arr[ydof,p] >= high_corner[1] or arr[ydof,p] <= 0:
                                    arr[valid,p] = 0.0
                                    n[i,j,k] -= 1
                                #arr[xdof,p] = arr[xdof, p]%high_corner[0]
                                #arr[ydof,p] = arr[ydof, p]%high_corner[1]
        return



    @staticmethod
    @numba.njit()
    def interpolate_pt(local_shape, n_halo,low_corner, dx, n, particle_varnames, particle_data, u, var, loc):


        #Compute points
        if loc ==0:
            xpos_shift = 0.0
            ypos_shift = 0.5 * dx[1]
            zpos_shift = 0.5 * dx[2]
        elif loc==1:
            xpos_shift = 0.5 * dx[0]
            ypos_shift = 0.0
            zpos_shift = 0.5 * dx[2]
        elif loc==2:
            xpos_shift = 0.5 * dx[0]
            ypos_shift = 0.5 * dx[1]
            zpos_shift = 0.0
        else:
            xpos_shift = 0.0
            ypos_shift = 0.0
            zpos_shift = 0.0


        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        zdof = particle_varnames['z']
        valid = particle_varnames['valid']
        var_dof = particle_varnames[var]


        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]
        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for p in range(arr.shape[1]):
                            if arr[valid,p] != 0.0:
                                xl = (arr[xdof, p] - low_corner[0] - xpos_shift)
                                yl = (arr[ydof, p] - low_corner[1] - ypos_shift)
                                zl = (arr[zdof, p] - low_corner[2] - zpos_shift)

                                ix = int(xl//dx[0]) - 1 + n_halo[0]
                                iy = int(yl//dx[1]) - 1 + n_halo[1]
                                iz = int(zl//dx[2]) - 1 + n_halo[2]

                                xd = (xl%dx[0])/dx[0]
                                yd = (yl%dx[1])/dx[1]
                                zd = (zl%dx[2])/dx[2]

                                c00 = (1.-xd)*u[ix,iy,iz] + xd*u[ix+1,iy,iz]
                                c01 = (1.-xd)*u[ix,iy,iz+1] + xd*u[ix+1,iy,iz+1]
                                c10 = (1.-xd)*u[ix,iy+1,iz] + xd*u[ix+1,iy+1,iz]
                                c11 = (1.-xd)*u[ix,iy+1,iz+1] + xd*u[ix+1,iy+1,iz+1]

                                c0 = c00*(1. - yd) + c10 * yd
                                c1 = c01*(1. - yd) + c11 * yd

                                arr[var_dof, p] = c0 * (1.0 - zd) + c1 * zd

        return



    @staticmethod
    @numba.njit()
    def move_particles_on_grid(low_corner, local_shape, dx, n_buffer, n, particle_varnames, particle_data):

        xdof = particle_varnames['x']
        ydof = particle_varnames['y']
        zdof = particle_varnames['z']
        valid = particle_varnames['valid']


        ishift = local_shape[1] * local_shape[2]
        jshift = local_shape[2]

        for i in range(local_shape[0]):
            ii = i * ishift
            for j in range(local_shape[1]):
                jj = j * jshift
                for k in range(local_shape[2]):
                    if n[i,j,k] > 0:
                    #if n[i,j,k] > 0:
                    #    print(i,j,k, n[i,j,k],np.max(n))
                #if n[i,j,k] > 0:
                        arr = particle_data[ii + jj + k]
                        for pi in range(arr.shape[1]):
                            if arr[valid,pi]!= 0.0:

                                #Compute the location of  the particle on the grid
                                inew = int((arr[xdof,pi] - low_corner[0])//dx[0])
                                jnew = int((arr[ydof,pi] - low_corner[1])//dx[1])
                                knew = int((arr[zdof,pi] - low_corner[2])//dx[2])

                                #If the new grid cell is different than the old, move it!
                                if i != inew or j != jnew or k != knew:

                                    #Get the particle array at the new grid point
                                    arr_new = particle_data[inew * ishift + jnew * jshift + knew]

                                    # Bool to hold if the moving aprticle can fit on the array at the new point
                                    fits = False

                                    #Loop over all points in the new array
                                    for pi_new in range(arr_new.shape[1]):

                                        #Find if there is a point that is currently empty
                                        if arr_new[valid,pi_new] == 0.0:
                                            #If we are here we found an empty location

                                            # Loop over all degress of freedom
                                            for d in range(arr.shape[0]):
                                                #print('Moving Particle')
                                                # Copy all degrees of freedom into the new array
                                                arr_new[d,pi_new] = arr[d,pi]

                                            # If we made it here, it must fit.
                                            fits = True
                                            break
                                    if not fits:
                                        #Get the index in the resized array
                                        n_arr = arr_new.shape[1]

                                        #Create a new buffer
                                        new_buf = np.zeros((arr_new.shape[0], n_buffer), dtype=np.double)

                                        #Write the particle data
                                        particle_data[inew * ishift + jnew * jshift + knew] = np.concatenate((arr_new, new_buf),axis=1)
                                        arr_new = particle_data[inew * ishift + jnew * jshift + knew]
                                        for d in range(arr.shape[0]):
                                            arr_new[d,n_arr] =  arr[d,pi]

                                    n[i,j,k] -= 1
                                    n[inew, jnew, knew] += 1


                                    arr[valid,pi] = 0.0
        #print(imin, imax, jmin, jmax, kmin, kmax, local_shape)
        return



    @staticmethod
    @numba.jit()
    def distill_dof(particle_varnames, n, var, particle_data, data):
        dof = particle_varnames[var]
        valid = particle_varnames['valid']
        count = 0
        for arr in particle_data:
            for p in range(arr.shape[1]):
                if arr[valid,p] != 0.0:
                    data[count] = arr[dof,p]
                    count += 1
        return



    def add_particle_variable(self, name, interp=False):
        # Assert if variable names are duplicated
        assert(name not in self._interp_particle_varnames)
        assert(name not in self._nointerp_particle_varnames)

        if interp:
            self._interp_particle_varnames[name] = self._interp_particle_dofs
            self._interp_particle_dofs += 1
        else:
            self._nointerp_particle_varnames[name] = self._nointerp_particle_dofs
            self._nointerp_particle_dofs += 1

        self._particle_varnames[name] = self._particle_dofs

        self._particle_dofs += 1
        return

    
class ParticlesSimple(ParticlesBase):

    def __init__(self, Grid, TimeSteppingController, VelocityState, ScalarState, DiagnsoticState):

        ParticlesBase.__init__(self,Grid, TimeSteppingController, VelocityState, ScalarState, DiagnsoticState)

        return

    def update(self):

        self.update_position()

        return