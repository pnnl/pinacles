import numba 

@numba.njit
def rk2ssp_s0(state_0, tend_0, dt):
    shape = state_0.shape 
    for k in range(shape[2]): 
        for j in range(shape[1]): 
            for i in range(shape[0]): 
                state_0[i,j,k] += tend_0[i,j,k]*dt 
                tend_0[i,j,k] = 0.0 

    return 

@numba.njit
def rk2ssp_s1(state_0, state_1, tend_1, dt):
    shape = state_0.shape
    for k in range(shape[2]): 
        for j in range(shape[1]): 
            for i in range(shape[0]):
                state_1[i,j,k]  = 0.5*(state_0[i,j,k] + (state_1[i,j,k] + tend_1[i,j,k]* dt) )
                tend_1[i,j,k] = 0.0 
    return 
