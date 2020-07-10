from Columbia.ScalarAdvection import ScalarWENO5
from Columbia.SLScalarAdvection import CTU

def factory(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping):
   return CTU(Grid, Ref, ScalarState, VelocityState, TimeStepping)
   #return ScalarWENO5(Grid, Ref, ScalarState, VelocityState, TimeStepping)