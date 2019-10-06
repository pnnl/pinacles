import numpy as np 
import Columbia.ThermodynamicsDry_impl as ThermoDry 
import Columbia.parameters as parameters 

def test_sT(): 

    z_test = 100.0 
    T_test = 293.15
    s = ThermoDry.s(z_test, T_test)
    T = ThermoDry.T(z_test, s)

    assert(T_test==T)

    return 

def test_rhoalpha(): 

    T_test = 293.15
    P_test = 1e5
    rho_test = P_test/(parameters.RD * T_test)
    rho = ThermoDry.rho(P_test, T_test)
    assert(rho_test == rho)
    
    alpha_test = 1.0/rho_test 
    alpha = ThermoDry.alpha(P_test, T_test)
    assert(alpha_test == alpha)

    return 

def test_buoyancy(): 

    def truth(alpha0, alpha): 
        return parameters.G * (alpha - alpha0)/alpha0

    #First test zero buoyancy case
    alpha0 = 1.0 
    alpha = 1.0 
    buoyancy = ThermoDry.buoyancy(alpha0, alpha)
    assert(buoyancy == 0.0)

    #Test a positively buoyant case 
    alpha = 1.1
    buoyancy = ThermoDry.buoyancy(alpha0, alpha)
    assert(buoyancy == truth(alpha0, alpha))
    assert(buoyancy>0.0)

    #Test a negative buoyany case 
    alpha = 0.9 
    buoyancy = ThermoDry.buoyancy(alpha0, alpha)
    assert(buoyancy == truth(alpha0, alpha))
    assert(buoyancy<0.0)

    return 

def test_eos(): 

    dims = (10, 10, 10)
    z_in = np.linspace(0.0, 1000.0, dims[2], dtype=np.double)
    P_in = np.linspace(1e5, 8.5e4, dims[2], dtype=np.double)
    alpha0 = np.ones_like(z_in, dtype=np.double)

    s_test = np.empty(dims, dtype=np.double, order='F')
    T_test = np.empty_like(s_test)
    alpha_test = np.empty_like(s_test)
    buoyancy_test = np.empty_like(s_test)
    
    for i in range(dims[0]): 
        for j in range(dims[1]): 
            for k in range(dims[2]): 
                T_test[i,j,k] = 300.0 + np.random.randn()
                s_test[i,j,k] = ThermoDry.s(z_in[k], T_test[i,j,k])
                alpha_test[i,j,k] = ThermoDry.alpha(P_in[k], T_test[i,j,k])
                buoyancy_test[i,j,k] = ThermoDry.buoyancy(alpha0[k], alpha_test[i,j,k])
    
    T_out = np.empty_like(s_test)
    alpha_out = np.empty_like(s_test)
    buoyancy_out = np.empty_like(s_test)

    ThermoDry.eos(z_in, P_in, alpha0, s_test, T_out, alpha_out, buoyancy_out)
    
    assert(np.allclose(T_out,T_test))
    assert(np.allclose(alpha_out, alpha_test))
    assert(np.allclose(buoyancy_out, buoyancy_test))


    return 