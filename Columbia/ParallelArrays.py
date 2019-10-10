from Columbia import Grid 
import numpy as np
from mpi4py import MPI 
import pylab as plt

class GhostArrayBase: 
    def __init__(self, _Grid, dof=None): 
        self._Grid = _Grid
        return 


class GhostArray(GhostArrayBase): 
    def __init__(self, _Grid, dtype=np.double,ndof=1): 

        GhostArrayBase.__init__(self, _Grid)
        self._shape = tuple(np.append(ndof, self._Grid.ngrid_local))
        self._n_halo = self._Grid.n_halo 
        self.array = np.empty(self._shape, dtype=np.double)  

        return 

    @property
    def shape(self): 
        return self._shape 

    def zero(self, dof=None): 
        if dof is None: 
            self.array[:,:,:,:] = 0.0
        else: 
            self.array[dof, :,:,:] = 0.0 

        return

    def set(self, value, dof=None):
        if dof is None: 
            self.array[:,:,:,:] = value
        else: 
            self.array[dof,:,:,:] = value 
        return 

    def max(self, dof=0, profile=False, halo=False):
        if not halo: 
            local_max = np.array(np.amax(self.array[dof,self._n_halo[0]:-self._n_halo[0],
                                                self._n_halo[1]:-self._n_halo[1],
                                                self._n_halo[2]:-self._n_halo[2]]),dtype=np.double) 

        if halo: 
            local_max = np.array(np.amax(self.array[dof,:,:,:]),dtype=np.double) 

        global_max = np.empty_like(local_max)
        MPI.COMM_WORLD.Allreduce(local_max, global_max, op=MPI.MAX)

        return global_max 

    def min(self, dof=0, profile=False, halo=False):
        if not halo:
            local_min = np.array(np.amin(self.array[self._n_halo[0]:-self._n_halo[0],
                                                self._n_halo[1]:-self._n_halo[1],
                                                self._n_halo[2]:-self._n_halo[2],dof]),dtype=np.double)
        if halo:
            local_min = np.array(np.amin(self.array[dof,:,:,:]),dtype=np.double)

        global_min = np.empty_like(local_min)
        MPI.COMM_WORLD.Allreduce(local_min, global_min, op=MPI.MIN)

        return global_min

    def boundary_exchange(self):
        #TODO At present this boundary exchange sends all dofs. We could do this DOF by DOF.

        for dim in range(2):

            comm = self._Grid.subcomms[dim]
            comm_size = comm.Get_size()

            #First do the right exchange
            source, dest = comm.Shift(0,1)

            if source == MPI.PROC_NULL:
                source = comm_size - 1

            if dest == MPI.PROC_NULL:
                dest = 0

            #Construct the buffers
            nh = self._n_halo[dim]

            if dim == 0:
                send_buf = np.copy(self.array[:,nh:2*nh,:,:])
                recv_buf = np.empty_like(send_buf)
                comm.Sendrecv(send_buf, dest, 113, recv_buf)
                self.array[:,-nh:,:,:] = recv_buf

            if dim == 1:
                send_buf = np.copy(self.array[:,:,nh:2*nh,:])
                recv_buf = np.empty_like(send_buf)
                comm.Sendrecv(send_buf, dest, 113, recv_buf) 
                self.array[:,:,-nh:,:] = recv_buf

            #Now do the left exchange 
            source, dest = comm.Shift(0,-1)
            
            if source == MPI.PROC_NULL:
                source = 0
            
            if dest == MPI.PROC_NULL: 
                dest = comm_size - 1

            if dim == 0: 
                send_buf = np.copy(self.array[:,-2*nh:-nh,:,:]) 
                recv_buf = np.empty_like(send_buf)
                comm.Sendrecv(send_buf, dest, 113, recv_buf)
                self.array[:,:nh,:,:] = recv_buf

            if dim == 1:
                send_buf = np.copy(self.array[:,:,-2*nh:-nh,:])
                recv_buf = np.empty_like(send_buf)
                comm.Sendrecv(send_buf, dest, 113, recv_buf) 
                self.array[:,:,:nh,:] = recv_buf

        return 

    def mean(self, dof=0): 
            
        local_sum = np.sum(self.array[dof,
            self._n_halo[0]:-self._n_halo[0],
            self._n_halo[1]:-self._n_halo[1],
            :],axis=(0,1))

        n = self._Grid.n
        local_sum /= n[0] * n[1] 
        mean = np.empty_like(local_sum)
        MPI.COMM_WORLD.Allreduce(local_sum, mean, op=MPI.SUM)
        
        return mean

    def remove_mean(self, dof): 
        #TODO perhpas use numba here? 
        mean = self.mean(dof)
        self.array[dof,:,:,:] -= mean[np.newaxis, np.newaxis, :]
    
        return 
