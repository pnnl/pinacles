from mpi4py import MPI
import numpy as np

def ScalarAllReduce(comm, value, op=MPI.SUM): 
    sbuf = np.array(value)
    rbuf = np.empty_like(sbuf)

    comm.Allreduce(sbuf, rbuf,op=op)

    return rbuf