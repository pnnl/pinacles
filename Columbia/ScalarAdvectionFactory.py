from Columbia.ScalarAdvection import ScalarWENO5
from Columbia.SLScalarAdvection import CTU

def factory(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping):
   
   try:
      adv_type = namelist['scalar_advection']['type']
   except:
      adv_type = 'sl2'


   if adv_type == 'sl2':
      return CTU(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping)
   else:
      return ScalarWENO5(Grid, Ref, ScalarState, VelocityState, TimeStepping)