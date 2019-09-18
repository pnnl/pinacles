import numba
from Columbia import interpolation_impl 

@numba.njit
def u_advection_weno5(u, v, w, fluxx, fluxy, fluxz, ut): 
    shape = ut.shape 
    for k in range(2,shape[2]-3): 
        for j in range(2,shape[1]-3): 
            for i in range(2,shape[0]-3): 
               #Compute advection of u by u-wind 
               up = interpolation_impl.centered_second(u[i,j,k], u[i+1,j,k])
               
               if up >= 0.0: 
                   fluxx[i,j,k] = up  * interpolation_impl.interp_weno5(
                                                     u[i-2,j,k],
                                                     u[i-1,j,k],
                                                     u[i,j,k],
                                                     u[i+1,j,k],
                                                     u[i+2,j,k] )
               else: 
                   fluxx[i,j,k] = up * interpolation_impl.interp_weno5(u[i+3,j,k],
                                                     u[i+2, j, k],
                                                     u[i+1, j, k],
                                                     u[i,j,k],
                                                     u[i-1,j,k])

               #Copute advection of u by v-wind 
               vp = interpolation_impl.centered_second(v[i,j,k], v[i,j+1,k]) 
               if vp >= 0.0: 
                   fluxy[i,j,k] = vp  * interpolation_impl.interp_weno5(
                                                     u[i,j-2,k],
                                                     u[i,j-1,k],
                                                     u[i,j,k],
                                                     u[i,j+1,k],
                                                     u[i,j+2,k] )
               else: 
                   fluxy[i,j,k] = vp * interpolation_impl.interp_weno5(u[i,j+3,k],
                                                     u[i, j+2, k],
                                                     u[i, j+1, k],
                                                     u[i,j,k],
                                                     u[i,j-1,k])


               #Compute advection of u by w-wind 
               wp = interpolation_impl.centered_second(w[i,j,k], w[i,j,k+1])
               if wp >= 0.0:
                   fluxy[i,j,k] = wp  * interpolation_impl.interp_weno5(
                                                     u[i,j,k-2],
                                                     u[i,j,k-1],
                                                     u[i,j,k],
                                                     u[i,j,k+1],
                                                     u[i,j,k+2] )
               else: 
                   fluxy[i,j,k] = wp * interpolation_impl.interp_weno5(u[i,j,k+3],
                                                     u[i, j, k+2],
                                                     u[i, j, k+1],
                                                     u[i,j,k],
                                                     u[i,j,k-1])


    return 

@numba.njit 
def v_advection_weno5(dy,u, v, w, fluxx, fluxy, fluxz, vt): 
    shape = vt.shape

    return 

@numba.njit
def w_advection_weno5(u, v, w, fluxx, fluxy, fluxz, wt): 
    shape = wt.shape 

    return 