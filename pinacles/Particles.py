import numpy as np
import numba

class ParticlesBase:

    def __init__(self, Grid, TimeSteppingController, VelocityState, ScalarState, DiagnosticState):

        # Initialize data
        self._Grid = Grid
        self._VelocityState = VelocityState
        self._ScalarState = ScalarState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController


        self._DiagnosticState.add_variable('part_count')

        # Number of arrays
        self._n_buffer = 16 * 16 * 200 * 10
        self._n_particles = 16 * 16 * 200 * 10

        self._particle_dofs = 3    # The first three dofs are always position
        self._interp_particle_dofs = 3
        self._nointerp_particle_dofs = 0
        self._particle_data = None
        self._initialzied = False


        self._interp_particle_varnames = {}
        self._nointerp_particle_varnames = {}

        self.add_particle_variable('u', interp=True)
        self.add_particle_variable('v', interp=True)
        self.add_particle_variable('w', interp=True)

        self._allocate_memory()

        xp = self._particle_data[0,:]
        yp = self._particle_data[1,:]
        zp = self._particle_data[2,:]

        #xp[:] = np.random.uniform(2048.0, 3072.0, xp.shape[0])
        #yp[:] = np.random.uniform(2048.0, 3072.0, yp.shape[0])

        self.sortsx = None
        self.sortsy = None
        self.sortsz = None


        xp[:] =  np.random.uniform(0, 5120.0, xp.shape[0])
        yp[:] = np.random.uniform(0.0, 5120.0, yp.shape[0])
        #zp[:] =  np.random.uniform(0.0, 2048.0, yp.shape[0])
        self.call_count = 1000000000

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

        self._particle_dofs += 1
        return

    def get_particle_var(self, name):
        if name in self._interp_particle_varnames:
            indx = self._interp_particle_varnames[name]
            return self._particle_data[indx,:]
        elif name in self._nointerp_particle_varnames:
            indx = self._nointerp_particle_varnames[name] + self._interp_particle_dofs 
            return  self._particle_data[indx,:]


    def _allocate_memory(self):
        self._particle_data = np.zeros((self._particle_dofs, self._n_buffer), dtype=np.double)
        return

    def update_position(self):

        low_corner = (self._Grid.x_range[0], self._Grid.y_range[0], self._Grid.z_range[0])
        high_corner = (self._Grid.x_range[1], self._Grid.y_range[1], self._Grid.z_range[1])
        if self._n_particles == 0:
            # Do nothing if there are no particles here
            return

        xp = self._particle_data[0,:]
        yp = self._particle_data[1,:]
        zp = self._particle_data[2,:]


        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        part_count = self._DiagnosticState.get_field('part_count')

        up = self.get_particle_var('u')
        vp = self.get_particle_var('v')
        wp = self.get_particle_var('w')



        import time
        t0 = time.perf_counter()


        self.compute_new_positions(low_corner, self._Grid.dx, self._n_particles,
            self._TimeSteppingController.dt, up, vp, wp, xp, yp, zp)
        self.serial_periodic_bc(high_corner, self._n_particles, xp, yp)
        self.serial_reflect_bc(self._n_particles, zp)
        self.interpolate_pt(self._Grid.n_halo, low_corner, self._Grid.dx, self._n_particles, xp, yp, zp, u, up,0)
        self.interpolate_pt(self._Grid.n_halo,low_corner, self._Grid.dx, self._n_particles, xp, yp, zp, v, vp,1)
        self.interpolate_pt(self._Grid.n_halo,low_corner, self._Grid.dx, self._n_particles, xp, yp, zp, w, wp,2)

        self.sortsx = np.argsort(xp)
        self.sortsy = np.argsort(yp)
        self.sortsz = np.argsort(zp)
        t0 = time.perf_counter()
        #ind = np.lexsort((zp, yp, xp))
        
        #for i in ind:
        #    print(zp[i], yp[i], xp[i])



        part_count[:,:,:] = 0
        t0 = time.perf_counter()  
        self.particle_count(self._Grid.n_halo, self._Grid.dx, xp, yp, zp, part_count)
        t1 = time.perf_counter()

        print('Timing: ', t1 - t0)
        #print('X pos:', np.amin(xp), np.amax(xp))
        #print('Y pos:', np.amin(yp), np.amax(yp))
        #print('Z pos:', np.amin(zp), np.amax(zp))
        #print('U:', np.amin(up), np.amax(up))
        #print('V:', np.amin(vp), np.amax(vp))
        #print('W:', np.amin(wp), np.amax(wp))
        #print('Post update time: ', t1 - t0)

        #if self._TimeSteppingController.time % 60.0 == 0:
        #import pylab as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(xp, yp, zp, s=0.1)
        #plt.xlim(low_corner[0], high_corner[0])
        #plt.ylim(low_corner[0], high_corner[1])
        #ax.axes.set_xlim3d(left=0, right=5120.0)
        #ax.axes.set_ylim3d(bottom=0, top=5120.0) 
        #ax.axes.set_zlim3d(bottom=0, top=2048.0) 
        #plt.savefig('./part_figs/' + str(self.call_count) + '.png' ,dpi=300)
        #plt.close()
        self.call_count += 1
        return

    @property
    def n_particles(self):
        return self._n_particles

    @property
    def n_memory(self):
        return self._n_buffer

    @property
    def initialized(self):
        return self._initialzied


    @staticmethod
    @numba.njit()
    def particle_count(n_halo, dx, xp, yp, zp, part_count):
        #Now loop over sorted indicies
        for i in range(xp.shape[0]):
            ig = int(xp[i]//dx[0])
            jg = int(yp[i]//dx[1])
            kg = int(zp[i]//dx[2])
            part_count[ig + n_halo[0], jg + n_halo[1], kg + n_halo[2]] += 1.0

            #print(ig, jg, kg, xp[i], yp[i], zp[i])

        return

    @staticmethod
    @numba.njit()
    def compute_new_positions(low_corner, dx, n_particles, dt, up, vp, wp, xp, yp, zp):
        for pi in range(n_particles):
            xp[pi] += up[pi] * dt
            yp[pi] += vp[pi] * dt
            zp[pi] += wp[pi] * dt
        return


    @staticmethod
    @numba.njit()
    def serial_periodic_bc(high_corner, n_particles, xp, yp):
        for pi in range(n_particles):
            xp[pi] = xp[pi]%high_corner[0]
            yp[pi] = yp[pi]%high_corner[1]

        return

    @staticmethod
    @numba.njit()
    def serial_reflect_bc(n_particles, zp):
        for pi in range(n_particles):
            zp[pi] = np.abs(zp[pi])

        return


    @staticmethod
    @numba.njit()
    def interpolate_pt(n_halo,low_corner, dx, n_particles, x, y, z, u, up, loc):


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


        for pi in range(n_particles):
            xl = (x[pi] - low_corner[0] - xpos_shift)
            yl = (y[pi] - low_corner[1] - ypos_shift)
            zl = (z[pi] - low_corner[2] - zpos_shift)



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

            up[pi] = c0 * (1.0 - zd) + c1 * zd

        return


class ParticlesSimple(ParticlesBase):

    def __init__(self, Grid, TimeSteppingController, VelocityState, ScalarState, DiagnsoticState):

        ParticlesBase.__init__(self,Grid, TimeSteppingController, VelocityState, ScalarState, DiagnsoticState)

        return

    def update(self):

        self.update_position()


        return