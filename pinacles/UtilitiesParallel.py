from mpi4py import MPI
import numpy as np


def ScalarAllReduce(value, op=MPI.SUM):
    sbuf = np.array(value)
    rbuf = np.empty_like(sbuf)

    MPI.COMM_WORLD.Allreduce(sbuf, rbuf, op=op)

    return rbuf


def print_root(msg):

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(msg)

    return
