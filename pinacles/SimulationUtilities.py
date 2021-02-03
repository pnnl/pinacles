import numpy as np
from mpi4py import MPI
import pickle
import os
class HorizontalSlice:
    def __init__(self, name, height, frequency, var, state, Sim):

        self._var = var
        self._state = state
        self._Sim = Sim
        self._name = name
        self._height = height
        self.frequency = frequency
        
        self._level = None 
        self._compute_height_index()

        self._out_path = os.path.join(self._Sim._namelist['meta']['output_directory'], self._Sim._namelist['meta']['simname'])

        self._out_path = os.path.join(self._out_path, 'HorizontalSlice_' + str(self._height) + '_' + var)
        if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.exists(self._out_path):
            os.makedirs(self._out_path)
        MPI.COMM_WORLD.Barrier()

        return

    def _compute_height_index(self):

        zl = self._Sim.ModelGrid.z_local
        self._level = np.where(np.abs(zl - self._height) == np.amin(np.abs(zl - self._height) ))[0][0]

        return
    
    def update(self):

        #Total number of grid points
        n =self._Sim.ModelGrid.n
        n_halo = self._Sim.ModelGrid.n_halo
        local_start = self._Sim.ModelGrid.local_start 
        local_end = self._Sim.ModelGrid.local_end


        local_array = np.zeros((n[0], n[1]), dtype=np.double, order='C')
        global_array = np.empty_like(local_array)

        state = self._Sim.__dict__[self._state].get_field(self._var)[:,:,self._level]

        local_array[local_start[0]:local_end[0], local_start[1]:local_end[1]] = state[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1]]


        MPI.COMM_WORLD.Allreduce(local_array, global_array)       

        out = {}
        out[self._var] = global_array

        time = self._Sim.TimeSteppingController.time
        fname = os.path.join(self._out_path, str(100000000 + np.round(time, 1)) + '.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(out, f)

        return

class Albedo:
    def __init__(self, frequency, Sim):


        self._Sim = Sim
        self.frequency = frequency
        

        self._out_path = os.path.join(self._Sim._namelist['meta']['output_directory'], self._Sim._namelist['meta']['simname'])

        self._out_path = os.path.join(self._out_path, 'Albedo')
        if MPI.COMM_WORLD.Get_rank() == 0 and not os.path.exists(self._out_path):
            os.makedirs(self._out_path)
        MPI.COMM_WORLD.Barrier()

        return


    
    def update(self):

        #Total number of grid points
        n =self._Sim.ModelGrid.n
        n_halo = self._Sim.ModelGrid.n_halo
        dx = self._Sim.ModelGrid.dx
        local_start = self._Sim.ModelGrid.local_start 
        local_end = self._Sim.ModelGrid.local_end

        rho = self._Sim.Ref.rho0


        local_array = np.zeros((n[0], n[1]), dtype=np.double, order='C')
        global_array = np.empty_like(local_array)

        state = self._Sim.ScalarState.get_field('qc')[:,:,:]

        Nc = 70.0

        tau = np.sum(0.19 * Nc**(1.0/3.0) * ((state * 1000.0 * rho[np.newaxis, np.newaxis,:] * dx[0])**(5.0/6.0)), axis=2)
        tau = tau/(6.8 + tau)


        local_array[local_start[0]:local_end[0], local_start[1]:local_end[1]] = tau[n_halo[0]:-n_halo[0], n_halo[1]:-n_halo[1]]


    
        MPI.COMM_WORLD.Allreduce(local_array, global_array)       
  

        out = {}
        out['Albdeo'] = global_array

        time = self._Sim.TimeSteppingController.time
        fname = os.path.join(self._out_path, str(100000000 + np.round(time, 1)) + '.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(out, f)

        return
