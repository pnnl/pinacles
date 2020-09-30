from pinacles.ScalarAdvection import ScalarWENO5
from pinacles.SLScalarAdvection import CTU

def factory(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping):
   
   try:
      adv_type = namelist['scalar_advection']['type']
   except:
      adv_type = 'weno'


   if adv_type == 'sl2':
      return CTU(namelist, Grid, Ref, ScalarState, VelocityState, TimeStepping)
   else:
      return ScalarWENO5(Grid, Ref, ScalarState, VelocityState, TimeStepping)