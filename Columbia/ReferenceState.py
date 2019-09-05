class ReferenceBase: 
    def __init__(self, Grid, Thermo):
        self._Grid = Grid
        self._Thermo = Thermo 

        return 


class ReferenceDry(ReferenceBase): 
    def __init__(self, namelist, Grid, Thermo):
        ReferenceBase.__init__(self, Grid, Thermo)


        return 


def factory(namelist, Grid, Thermo): 
    return ReferenceDry(namelist, Grid, Thermo)