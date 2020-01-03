import numpy as np
import netCDF4 as nc
import os
from mpi4py import MPI

class DumpFields:
    def __init__(self, namelist, Grid, TimeSteppingController):


        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController

        self._this_rank = MPI.COMM_WORLD.Get_rank()
        self._output_root = str(namelist['meta']['output_directory'])
        self._casename = str(namelist['meta']['casename'])
        self._output_path = self._output_path = os.path.join(self._output_root, self._casename)
        self._output_path = os.path.join(self._output_path, 'fields')

        if self._this_rank == 0:
            if not os.path.exists(self._output_path):
                os.makedirs(self._output_path)

        self._classes = {}

        return


    def add_class(self, aclass):
        assert(aclass not in self._classes)
        self._classes[aclass.name] = aclass
        return

    def update(self):

        output_here = os.path.join(self._output_path, str(self._TimeSteppingController.time))
        MPI.COMM_WORLD.barrier()
        if self._this_rank == 0:
            if not os.path.exists(output_here):
                os.makedirs(output_here)

        MPI.COMM_WORLD.barrier()
        rt_grp = nc.Dataset(os.path.join(output_here, str(self._this_rank) + '.nc'), 'w', format="NETCDF4_CLASSIC")

        self.setup_nc_dims(rt_grp)

        nhalo = self._Grid.n_halo
        for ac in self._classes:
            #Loop over all variables
            ac = self._classes[ac]
            for v in ac._dofs:
                v_nc = rt_grp.createVariable(v, np.double, dimensions=('T','X','Y','Z'))

                v_nc[0,:,:,:] =  ac.get_field(v)[nhalo[0]:-nhalo[0],
                    nhalo[1]:-nhalo[1],
                    nhalo[2]:-nhalo[2]]



            rt_grp.sync()

        rt_grp.close()
        return


    def setup_nc_dims(self, rt_grp):


        rt_grp.createDimension('T', size=1)
        rt_grp.createDimension('X', size=self._Grid.local_shape[0])
        rt_grp.createDimension('Y', size=self._Grid.local_shape[1])
        rt_grp.createDimension('Z', size=self._Grid.n[2])

        nhalo = self._Grid.n_halo

        T = rt_grp.createVariable('T', np.double, dimensions=('T'))
        T[0] =  self._TimeSteppingController.time
        X = rt_grp.createVariable('X', np.double, dimensions=('X'))
        X[:] = self._Grid.x_local[nhalo[0]:-nhalo[0]]
        Y = rt_grp.createVariable('Y', np.double, dimensions=('Y'))
        Y[:] = self._Grid.y_local[nhalo[1]:-nhalo[1]]
        Z = rt_grp.createVariable('Z', np.double, dimensions=('Z'))
        Z[:] = self._Grid.z_local[nhalo[2]:-nhalo[2]]
        rt_grp.sync()

        return