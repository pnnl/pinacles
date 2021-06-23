from pinacles import LateralBCs
from pinacles import Grid
from pinacles import Containers

namelist = {}
key = "grid"
namelist[key] = {}
namelist[key]["n"] = (5, 5, 5)
namelist[key]["n_halo"] = (3, 3, 3)
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
    
    
    LBC.init_vars_on_boundary()

    #Check and make sure that vars_on_boundary are propert initialized
    for var in ScalarState._dofs:
        assert var in LBC._var_on_boundary
        assert 'x_low' in LBC._var_on_boundary[var]
        assert 'x_high' in LBC._var_on_boundary[var]
        assert 'y_low' in LBC._var_on_boundary[var]
        assert 'y_high' in LBC._var_on_boundary[var]




    u = VelocityState.get_field("u")
    s = ScalarState.get_field("s")
    qv = ScalarState.get_field("qv")
    u.fill(-1.0)
    s.fill(300.0)
    qv.fill(0.1)



    LBC.update()
    print(qv[:,2,4])

    return
