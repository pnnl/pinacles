import numpy as np 
import Columbia.ThermodynamicsDry_impl as ThermoDry_impl
import Columbia.Thermodynamics as Thermodynamics
import Columbia.parameters as parameters 
import Columbia.Grid as Grid 
import Columbia.Containers as Containers
import Columbia.ReferenceState as ReferenceState
def test_sT(): 

    z_test = 100.0 
    T_test = 293.15
    s = ThermoDry_impl.s(z_test, T_test)
    T = ThermoDry_impl.T(z_test, s)

    assert(T_test==T)

    return 

def test_rhoalpha(): 

    T_test = 293.15
    P_test = 1e5
    rho_test = P_test/(parameters.RD * T_test)
    rho = ThermoDry_impl.rho(P_test, T_test)
    assert(rho_test == rho)
    
    alpha_test = 1.0/rho_test 
    alpha = ThermoDry_impl.alpha(P_test, T_test)
    assert(alpha_test == alpha)

    return 

def test_buoyancy(): 

    def truth(alpha0, alpha): 
        return parameters.G * (alpha - alpha0)/alpha0

    #First test zero buoyancy case
    alpha0 = 1.0 
    alpha = 1.0 
    buoyancy = ThermoDry_impl.buoyancy(alpha0, alpha)
    assert(buoyancy == 0.0)

    #Test a positively buoyant case 
    alpha = 1.1
    buoyancy = ThermoDry_impl.buoyancy(alpha0, alpha)
    assert(buoyancy == truth(alpha0, alpha))
    assert(buoyancy>0.0)

    #Test a negative buoyany case 
    alpha = 0.9 
    buoyancy = ThermoDry_impl.buoyancy(alpha0, alpha)
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
                s_test[i,j,k] = ThermoDry_impl.s(z_in[k], T_test[i,j,k])
                alpha_test[i,j,k] = ThermoDry_impl.alpha(P_in[k], T_test[i,j,k])
                buoyancy_test[i,j,k] = ThermoDry_impl.buoyancy(alpha0[k], alpha_test[i,j,k])
    
    T_out = np.empty_like(s_test)
    alpha_out = np.empty_like(s_test)
    buoyancy_out = np.empty_like(s_test)

    ThermoDry_impl.eos(z_in, P_in, alpha0, s_test, T_out, alpha_out, buoyancy_out)
    
    assert(np.allclose(T_out,T_test))
    assert(np.allclose(alpha_out, alpha_test))
    assert(np.allclose(buoyancy_out, buoyancy_test))

    return 


def test_update(): 

    # Build input dictionary to test update
    namelist = {} 
    namelist['grid'] = {} 
    namelist['grid']['n'] = [16, 16, 40]
    namelist['grid']['l'] = [400, 400, 400]
    namelist['grid']['n_halo'] = [3, 3, 3]

    ModelGrid = Grid.RegularCartesian(namelist)
    ScalarState = Containers.ModelState(ModelGrid, prognostic=True)
    VelocityState = Containers.ModelState(ModelGrid, prognostic=True)
    DiagnosticState = Containers.ModelState(ModelGrid)

    # Need to add vertical velocity component so we can update it
    VelocityState.add_variable('w')

    # Set up the reference state class
    Ref =  ReferenceState.factory(namelist, ModelGrid)
    Ref.set_surface()
    Ref.integrate()

    # Set up the thermodynamics class
    Thermo = Thermodynamics.factory(namelist, ModelGrid, Ref, 
        ScalarState, VelocityState, DiagnosticState)

    # Allocate all of the big parallel arrays needed for the container classes
    ScalarState.allocate()
    VelocityState.allocate()
    DiagnosticState.allocate()

    #Now let's create an isothermal state
    h = ScalarState.get_field('h')
    z = ModelGrid.z_local

    for i in range(h.shape[0]): 
        for j in range(h.shape[1]): 
            for k in range(h.shape[2]): 
                h[i,j,k] = ThermoDry_impl.s(z[k], 293.15)

    #Now run update 
    Thermo.update()

    #Now test the value of temperature 
    T = DiagnosticState.get_field('T')
    assert(np.all(T == 293.15))

    #Check that the mean is zero
    DiagnosticState.remove_mean('buoyancy')
    buoyancy_mean = DiagnosticState.mean('buoyancy')
    assert(np.allclose(buoyancy_mean[ModelGrid.n_halo[2]:-ModelGrid.n_halo[2]],0))