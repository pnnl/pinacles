from pinacles import LateralBCs
from pinacles import Grid
from pinacles import Containers

namelist = {}
key = "grid"
namelist[key] = {}
namelist[key]['n'] = (5, 5, 5)
namelist[key]['n_halo'] = (3,3,3)
namelist[key]['l'] = (1000.0, 1000.0, 1000.0)


def test_LateralBCs():

    ModelGrid = Grid.RegularCartesian(namelist)
    VelocityState = Containers.ModelState(ModelGrid, 'VelocityState')
    ScalarState = Containers.ModelState(ModelGrid, 'ScalarState')

    VelocityState.add_variable('u')
    VelocityState.add_variable('v')
    VelocityState.add_variable('w')

    ScalarState.add_variable('s')
    ScalarState.add_variable('qv')

    VelocityState.allocate()
    ScalarState.allocate()

    print('Hello')
    print('VelocityState')
    
    LBC = LateralBCs.LateralBCs(Grid, VelocityState, ScalarState)


    u = VelocityState.get_field('u')
    u.fill(1.0)


    LBC.update()


    return
