import numba
from Columbia import interpolation_impl 

@numba.njit
def u_advection_weno5(rho0, rho_edge0, u, v, w, fluxx, fluxy, fluxz, ut): 
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
                                                     u[i+2,j,k] ) * rho0[k]
               else: 
                   fluxx[i,j,k] = up * interpolation_impl.interp_weno5(u[i+3,j,k],
                                                     u[i+2, j, k],
                                                     u[i+1, j, k],
                                                     u[i,j,k],
                                                     u[i-1,j,k]) * rho0[k]

               #Copute advection of u by v-wind 
               vp = interpolation_impl.centered_second(v[i,j,k], v[i,j+1,k]) 
               if vp >= 0.0: 
                   fluxy[i,j,k] = vp  * interpolation_impl.interp_weno5(
                                                     u[i,j-2,k],
                                                     u[i,j-1,k],
                                                     u[i,j,k],
                                                     u[i,j+1,k],
                                                     u[i,j+2,k] ) * rho0[k]
               else: 
                   fluxy[i,j,k] = vp * interpolation_impl.interp_weno5(u[i,j+3,k],
                                                     u[i, j+2, k],
                                                     u[i, j+1, k],
                                                     u[i,j,k],
                                                     u[i,j-1,k]) * rho0[k]


               #Compute advection of u by w-wind 
               wp = interpolation_impl.centered_second(w[i,j,k], w[i,j,k+1])
               if wp >= 0.0:
                   fluxz[i,j,k] = wp  * interpolation_impl.interp_weno5(
                                                     u[i,j,k-2],
                                                     u[i,j,k-1],
                                                     u[i,j,k],
                                                     u[i,j,k+1],
                                                     u[i,j,k+2] ) * rho_edge0[k]
               else: 
                   fluxz[i,j,k] = wp * interpolation_impl.interp_weno5(u[i,j,k+3],
                                                     u[i, j, k+2],
                                                     u[i, j, k+1],
                                                     u[i,j,k],
                                                     u[i,j,k-1]) * rho_edge0[k]


    return 

@numba.njit 
def v_advection_weno5(rho0, rho0_edge, u, v, w, fluxx, fluxy, fluxz, vt): 
    shape = vt.shape
    for k in range(2,shape[2]-3): 
        for j in range(2,shape[1]-3): 
            for i in range(2,shape[0]-3): 
                #Compute v advection by the u wind
                up = interpolation_impl.centered_second(u[i,j,k], u[i+1,j,k])
                if up >= 0.0: 
                    fluxx[i,j,k] = up  * interpolation_impl.interp_weno5(
                                                     v[i-2,j,k],
                                                     v[i-1,j,k],
                                                     v[i,j,k],
                                                     v[i+1,j,k],
                                                     v[i+2,j,k] ) * rho0[k]
                else: 
                    fluxx[i,j,k] = up * interpolation_impl.interp_weno5(
                                                     v[i+3,j,k],
                                                     v[i+2, j, k],
                                                     v[i+1, j, k],
                                                     v[i,j,k],
                                                     v[i-1,j,k]) * rho0[k]


                #Compute v advection by the v wind 
                vp = interpolation_impl.centered_second(v[i,j,k], v[i,j+1,k])                
                if vp >= 0.0: 
                    fluxy[i,j,k] = vp * interpolation_impl.interp_weno5(
                                                     v[i,j-2,k],
                                                     v[i,j-1,k],
                                                     v[i,j,k],
                                                     v[i,j+1,k],
                                                     v[i,j+2,k] ) * rho0[k]
                else: 
                    fluxy[i,j,k] = vp * interpolation_impl.interp_weno5(
                                                     v[i,j+3,k],
                                                     v[i, j+2, k],
                                                     v[i, j+1, k],
                                                     v[i,j,k],
                                                     v[i,j-1,k]) * rho0[k]

                #Compute v advection by the w wind 
                wp = interpolation_impl.centered_second(w[i,j,k], w[i,j,k+1])
                if wp >= 0.0: 
                    fluxz[i,j,k] = wp * interpolation_impl.interp_weno5(
                                                     v[i,j,k-2],
                                                     v[i,j,k-1],
                                                     v[i,j,k],
                                                     v[i,j,k+1],
                                                     v[i,j,k+2] ) * rho0_edge[k]
                else: 
                    fluxz[i,j,k] = wp * interpolation_impl.interp_weno5(
                                                     v[i,j,k+3],
                                                     v[i, j, k+2],
                                                     v[i, j, k+1],
                                                     v[i,j,k],
                                                     v[i,j,k-1]) * rho0_edge[k]
    return 

@numba.njit
def w_advection_weno5(rho0, rho0_edge, u, v, w, fluxx, fluxy, fluxz, wt): 
    shape = wt.shape 
    for k in range(2,shape[2]-3): 
        for j in range(2,shape[1]-3): 
            for i in range(2,shape[0]-3): 
                #Compute w advection by the u wind 
                up = interpolation_impl.centered_second(u[i,j,k], u[i+1,j,k])
                if up >= 0.0: 
                    pass 
                else: 
                    pass 

                #Compute w advection by the v wind 
                vp = interpolation_impl.centered_second(v[i,j,k], v[i,j+1,k])                
                if vp >= 0.0: 
                    pass 
                else: 
                    pass 


                #Compute w advection by the w wind 
                wp = interpolation_impl.centered_second(w[i,j,k], w[i,j,k+1])
                if wp >= 0.0: 
                    pass 
                else: 
                    pass 


    return 