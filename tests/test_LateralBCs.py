import numpy as np
from pinacles import LateralBCs
from pinacles import Grid
from pinacles import Containers

namelist = {}
key = "grid"
namelist[key] = {}
namelist[key]["n"] = (5, 5, 5)
namelist[key]["n_halo"] = (1, 1, 1)
namelist[key]["l"] = (1000.0, 1000.0, 1000.0)


def test_LateralBCs():

    ModelGrid = Grid.RegularCartesian(namelist)
    VelocityState = Containers.ModelState(ModelGrid, "VelocityState")
    ScalarState = Containers.ModelState(ModelGrid, "ScalarState")

    VelocityState.add_variable("u")
    VelocityState.add_variable("v")
    VelocityState.add_variable("w")

    ScalarState.add_variable("s")
    ScalarState.add_variable("qv")

    VelocityState.allocate()
    ScalarState.allocate()

    LBC = LateralBCs.LateralBCs(ModelGrid, ScalarState, VelocityState)
    LBCVel = LateralBCs.LateralBCs(ModelGrid, VelocityState, VelocityState)

    LBC.init_vars_on_boundary()
    LBCVel.init_vars_on_boundary()

    # Check and make sure that vars_on_boundary are propert initialized
    for var in ScalarState._dofs:
        assert var in LBC._var_on_boundary
        assert "x_low" in LBC._var_on_boundary[var]
        assert "x_high" in LBC._var_on_boundary[var]
        assert "y_low" in LBC._var_on_boundary[var]
        assert "y_high" in LBC._var_on_boundary[var]

    u = VelocityState.get_field("u")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")
    u.fill(-1.0)
    s.fill(300.0)
    qv.fill(0.1)

    s_x_low, s_x_high, s_y_low, s_y_high = LBC.get_vars_on_boundary('s')
    
    
    # Check that the x lateral boundary conditions are getting set correctly
    s_x_low.fill(299.0) 
    s_x_high.fill(301.0)
    LBC.update()

    assert np.all(s[-1,1:-1,:] == 301.0)
    s.fill(300.0)
    u.fill(1.0)
    LBC.update()
    assert np.all(s[0,1:-1,:] == 299.0)

    # Check that the y lateral boundary conditions are getting set correctly
    s_y_low.fill(302.0)
    s_y_high.fill(303.0)
    s.fill(300.0)
    u.fill(-1.0)
    LBC.update()
    assert np.all(s[1:-1,-1,:] == 303.0)

    s.fill(300.0)
    u.fill(1.0)
    LBC.update()
    assert(np.all(s[1:-1,0,:]==302.0))

    # Tests now for the normal velocity components
    s_x_low, s_x_high, s_y_low, s_y_high = LBCVel.get_vars_on_boundary('u')


    return
