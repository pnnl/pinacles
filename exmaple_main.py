from mpi4py import MPI 
import mpi4py_fft as mpfft
from mpi4py_fft.pencil import Subcomm
import numpy as np
import time 
scomm = Subcomm(MPI.COMM_WORLD,dims=[0,0,1])
u = mpfft.DistArray((64,64,100), scomm, dtype=np.complex)

ud = u.redistribute(0)


fft = mpfft.PFFT(scomm, darray=ud, axes=(1,0))#(scomm,(64,64,32), axes=(0,1), dtype=np.complex)
##u = mpfft.newDistArray(fft, False)
t0 = time.time() 
ud = u.redistribute(0)
u_hat = fft.forward(ud, normalize=True)
uj = np.zeros_like(ud)
uj = fft.backward(u_hat, uj).redistribute(2)
t1 = time.time() 
print(MPI.COMM_WORLD.Get_rank(), u.shape, ud.shape, u_hat.shape, uj.shape, t1-t0)
#print(u.shape)
#print(u.redistribute(1).shape)
#print(u.commsizes)
#fft = mpfft.PFFT(MPI.COMM_WORLD, (64, 32, 32), axes=(1,2))
#u = mpfft.newDistArray(scomm)
        
#print(u.commsizes) 
