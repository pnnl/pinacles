import jax
import jax.numpy as jnp
from simple_pytree import Pytree, static_field

class TimeStepping(Pytree):

    def __init__(self, TimeStepping, PrognosticState):
        
        self.Tn = jax.device_put(jnp.zeros_like(PrognosticState.state_))
        self.n_rk_step = 2
        
        return 
    
    
    def  update(self, PrognosticState):
        
        present_state = PrognosticState.array
        present_tend = PrognosticState.tend_array
        
        #Need to branch here
        
        return PrognosticState