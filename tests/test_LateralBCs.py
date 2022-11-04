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

namelist['lbc'] ={}
namelist['lbc']['type'] = 'open'
def test_LateralBCs():

    ModelGrid = Grid.RegularCartesian(namelist)
    VelocityState = Containers.ModelState(namelist,ModelGrid, "VelocityState")
    ScalarState = Containers.ModelState(namelist,ModelGrid, "ScalarState")

    VelocityState.add_variable("u")
    VelocityState.add_variable("v")
    VelocityState.add_variable("w")

    ScalarState.add_variable("s")
    ScalarState.add_variable("qv")

    VelocityState.allocate()
    ScalarState.allocate()

    LBC = LateralBCs.LateralBCsBase(ModelGrid, ScalarState, VelocityState)
    LBCVel = LateralBCs.LateralBCsBase(ModelGrid, VelocityState, VelocityState)

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
    v = VelocityState.get_field("v")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")

    u.fill(-1.0)
    s.fill(300.0)
    qv.fill(0.1)

    s_x_low, s_x_high, s_y_low, s_y_high = LBC.get_vars_on_boundary("s")

    # Check that the x lateral boundary conditions are getting set correctly
    s_x_low.fill(299.0)
    s_x_high.fill(301.0)
    LBC.update()

    assert np.all(s[-1, 1:-1, :] == 301.0)
    s.fill(300.0)
    u.fill(1.0)
    LBC.update()
    assert np.all(s[0, 1:-1, :] == 299.0)

    # Check that the y lateral boundary conditions are getting set correctly
    s_y_low.fill(302.0)
    s_y_high.fill(303.0)
    s.fill(300.0)
    u.fill(-1.0)
    LBC.update()
    assert np.all(s[1:-1, -1, :] == 303.0)

    s.fill(300.0)
    u.fill(1.0)
    LBC.update()
    assert np.all(s[1:-1, 0, :] == 302.0)

    # Tests for the normal velocity components on the x lbc
    u_x_low, u_x_high, u_y_low, u_y_high = LBCVel.get_vars_on_boundary("u")
    u_x_low.fill(2.0)
    u_x_high.fill(3.0)
    LBCVel.update()
    assert np.all(u[0, 1:-1, :] == 2.0)
    assert np.all(u[-2:, 1:-1, :] == 3.0)

    # Tests for the normal velocity components on the y lbc
    v = VelocityState.get_field("v")
    v_x_low, v_x_high, v_y_low, v_y_high = LBCVel.get_vars_on_boundary("v")
    v_y_low.fill(2.0)
    v_y_high.fill(3.0)
    LBCVel.update()
    assert np.all(v[1:-1, 0, :] == 2.0)
    assert np.all(v[1:-1, -2:, :] == 3.0)

    return
