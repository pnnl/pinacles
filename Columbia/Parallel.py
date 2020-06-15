from mpi4py import MPI

class Parallel: 
    def __init__(self, local_world): 

        self._local_world = local_world


        self._rank = self._local_world.Get_rank()
        self._size = self._local_world.Get_size()

        return
        
    @property
    def world(self):
        return self._local_world

    @property
    def rank(self): 
        return self._rank

    @property
    def size(self):
        return self._size

    def root_print(self, what_to_print):
        if self._rank == 0:
            print(what_to_print)
        return

    def barrier():
        self._local_world.barrier()