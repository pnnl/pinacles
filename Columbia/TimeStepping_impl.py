import numba 

@numba.njit
def rk2ssp_s0(state_0, tend_0, dt):
    shape = state_0.shape 
    for n in range(shape[0]): 
        for i in range(shape[1]): 
            for j in range(shape[2]): 
                for k in range(shape[3]): 
                    state_0[n,i,j,k] += tend_0[n,i,j,k]*dt 
                    tend_0[n,i,j,k] = 0.0 

    return 

@numba.njit
def rk2ssp_s1(state_0, state_1, tend_1, dt):
    shape = state_0.shape
    for n in range(shape[0]): 
        for i in range(shape[1]): 
            for j in range(shape[2]): 
                for k in range(shape[3]):
                    state_1[n,i,j,k]  = 0.5*(state_0[n,i,j,k] + (state_1[n,i,j,k] + tend_1[n,i,j,k]* dt) )
                    tend_1[n,i,j,k] = 0.0 
    return 
