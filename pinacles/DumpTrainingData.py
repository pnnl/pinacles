import os
import h5py
import time
import numba
from mpi4py import MPI
import numpy as np
from pinacles import UtilitiesParallel
import zarr
import random

def DumpCloudOnlyDataFactory(namelist, Timers, Grid, TimeSteppingController):

    if "cloud_cond_fields" not in namelist:
        UtilitiesParallel.print_root("Namelist item cloud cond fields not found")
        return DumpCloudCondFieldsBase(namelist, Timers, Grid, TimeSteppingController)

    return DumpCloudCondFields(namelist, Timers, Grid, TimeSteppingController)

    return


class DumpCloudCondFieldsBase:
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        ScalarState,
        VelocityState,
        DiagnosticState,
        Micro,
        TimeSteppingController,
    ):

        self._Timers = Timers
        self._Grid = Grid
        self._ScalerState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._Micro = Micro
        self._TimeSteppingController = TimeSteppingController

        self.collective = True
        try:
            if "collective" in namelist["cloud_cond_fields"]["hdf5"]:
                self.collective = namelist["cloud_cond_fields"]["hdf5"]["collective"]
        except:
            pass

        self._this_rank = MPI.COMM_WORLD.Get_rank()
        self._output_root = str(namelist["meta"]["output_directory"])
        self._casename = str(namelist["meta"]["simname"])
        self._output_path = self._output_path = os.path.join(
            self._output_root, self._casename
        )
        self._output_path = os.path.join(self._output_path, "cloud_cond_fields")

        try:
            self._frequency = namelist["cloud_cond_fields"]["frequency"]
        except:
            self._frequency = 1e9

        if self._this_rank == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        self._classes = {}
        self._namelist = namelist

        self._Timers.add_timer("DumpFields")

        return

    @property
    def frequency(self):
        return self._frequency

    def add_class(self, aclass):
        assert aclass not in self._classes
        self._classes[aclass.name] = aclass
        return

    def update(self):
        self._Timers.start_timer("DumpFields")

        self._Timers.add_timer("DumpFields")
        return


class DumpCloudCondFields(DumpCloudCondFieldsBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        ScalerState,
        DiagnosticState,
        VelocityState,
        Micro,
        TimeSteppingController,
    ):
        return DumpCloudCondFieldsBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            ScalerState,
            DiagnosticState,
            VelocityState,
            Micro,
            TimeSteppingController,
        )

    @staticmethod
    @numba.njit
    def to_global_flat_index(local_start, n, qc_3d, flat_indx):

        shape = qc_3d.shape
        count = 0
        for i in range(shape[0]):
            ishift = (i + local_start[0]) * n[1] * n[2]
            for j in range(shape[1]):
                jshift = (local_start[1] + j) * n[2]
                for k in range(shape[2]):
                    if qc_3d[i, j, k] > 0:
                        flat_indx[count] = ishift + jshift + k
                        count += 1

        return

    @staticmethod
    @numba.njit
    def condition_to_flat(local_start, n, qc_3d, var, var_flat):

        shape = qc_3d.shape
        count = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if qc_3d[i, j, k] > 0:
                        var_flat[count] = var[i, j, k]
                        count += 1

        return

    def update(self):

        t0 = time.perf_counter()
        self._Timers.start_timer("DumpFields")

        qc = self._Micro.get_qc()

        nh = self._Grid.n_halo
        n_qc = np.count_nonzero(qc[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] : -nh[1]])

        n_on_rank = MPI.COMM_WORLD.allgather(n_qc)

        my_rank = MPI.COMM_WORLD.Get_rank()
        my_start = 0
        if my_rank != 0:
            my_start = np.sum(n_on_rank[:my_rank])

        my_end = my_start + n_on_rank[my_rank]

        n_total = np.sum(n_on_rank)
        if np.sum(n_on_rank) == 0:
            return

        output_here = self._output_path

        output_here = os.path.join(output_here)

        MPI.COMM_WORLD.barrier()
        zarr_fname = os.path.join(
                output_here, str(np.round(self._TimeSteppingController.time)) + ".zarr"
            )
        
        
        zarr_sync = zarr.ProcessSynchronizer(os.path.join(
                output_here, str(np.round(self._TimeSteppingController.time)) + ".sync"
            ))
    
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)
                
            root = zarr.open(zarr_fname, mode='w', synchronizer=zarr_sync)
            
 
 
            root.create_dataset('flat_index', shape=(n_total,), dtype='i8')
            for state in [self._ScalerState, self._DiagnosticState, self._VelocityState]:
                for v in state._dofs:
                    root.create_dataset(v, shape=(n_total,))

        MPI.COMM_WORLD.barrier()

        if not self._this_rank == 0:
            root = zarr.open(zarr_fname, mode='r+', synchronizer=zarr_sync)


        MPI.COMM_WORLD.barrier()


        # These are needed to compute the 3D indexing
        root.attrs["nx"] = int(self._Grid.n[0])
        root.attrs["ny"] = int(self._Grid.n[1])
        root.attrs["nz"] = int(self._Grid.n[2])

        # Compute the index
        flat_indx = np.empty((n_on_rank[MPI.COMM_WORLD.Get_rank()],), dtype=np.int)
        self.to_global_flat_index(
            self._Grid._local_start,
            self._Grid._n,
            qc[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] : -nh[2]],
            flat_indx,
        )
        
        
        #if self._this_rank == 0:
        #    flat_indx = root.create_dataset('flat_indx', shape=n_total, type='i8')
        #MPI.COMM_WORLD.barrier()

            
        

        
        
        #dset = fx.create_dataset(
        #    "flat_indx",
        #    (n_total,),
        #    dtype="i",
        #)

        #if my_start != my_end:
        #    root['flat_index'][my_start:my_end] = flat_indx[:]

        var_flat = np.empty((n_on_rank[MPI.COMM_WORLD.Get_rank()],), dtype=np.double)
        states = [self._ScalerState, self._DiagnosticState, self._VelocityState]
        
        
        list_of_vars = []
        for i, state in enumerate(states):
            for v in state._dofs:
                list_of_vars.append((i, v))
        
        list_of_vars.append((999, 'flat_index'))
        
        
        states_permute = random.sample(list_of_vars, len(list_of_vars))
        
        for sp in states_permute:
            
            if sp[0] == 999:
                if my_start != my_end:
                    root[sp[1]][my_start:my_end] = flat_indx[:]

            else:
                var = states[sp[0]].get_field(sp[1])
                self.condition_to_flat(
                    self._Grid._local_start,
                    self._Grid._n,
                    qc[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] : -nh[2]],
                    var[nh[0] : -nh[0], nh[1] : -nh[1], nh[2] : -nh[2]],
                    var_flat,
                )

                if my_start != my_end:
                    root[sp[1]][my_start:my_end] = var_flat[:]
                
                var_flat.fill(0.0)
                MPI.COMM_WORLD.barrier()


        t1 = time.perf_counter()
        UtilitiesParallel.print_root(
            "\t Parallel of IO of conditioned fields to zarr: " + str(t1 - t0) + " seconds"
        )
        self._Timers.end_timer("DumpFields")

        return
