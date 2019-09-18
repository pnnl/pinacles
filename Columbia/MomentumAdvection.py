class MomentumAdvectionBase: 
    def __init__(self, Grid, Ref, ScalarState, VelocityState): 
        self._Grid = Grid
        self._Ref = Ref 
        self._ScalarState = ScalarState 
        self._VelocityState = VelocityState 

        return 

    def update(self): 

        return 

class MomentumWENO5(MomentumAdvectionBase): 
    def __init__(self, Grid, Ref, ScalarState, VelocityState): 
        MomentumAdvectionBase.__init__(self, Grid, Ref, ScalarState, VelocityState)

    def update(self): 

        u = self._VelocityState.get_field('u')
        v = self._VelocityState.get_field('v')
        w = self._VelocityState.get_field('w')

        u_t = self._VelocityState.get_tend('u')
        v_t = self._VelocityState.get_tend('v')
        w_t = self._VelocityState.get_tend('w')


        return 

def factory(namelist, Grid, Ref, ScalarState, VelocityState): 
    return MomentumWENO5(Grid, Ref, ScalarState, VelocityState)
    

    return 