from mpi4py import MPI
import numpy as np

def ScalarAllReduce(value, op=MPI.SUM): 
    sbuf = np.array(value)
    rbuf = np.empty_like(sbuf)

    MPI.COMM_WORLD.Allreduce(sbuf, rbuf,op=op)

    return rbuf