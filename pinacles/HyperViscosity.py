import numba

@numba.njit
def hyper_op(c, h, pm2, pm1, p, p1, p2):
    return  c * (pm2-4.0*pm1+6.0*p-4.0*p1+p2)/(h * h * h * h)


@numba.njit
def compute_hyper(dx0, dx1,c, p, pt):
    shape = p.shape
    for i in range(2,shape[0]-2):
        for j in range(2,shape[1]-2):
            for k in range(2,shape[2]-2):
                pt[i,j,k] += hyper_op(c, dx0, p[i-2,j,k], p[i-1, j, k], p[i,j,k], p[i+1, j, k], p[i+1, j, k])
                pt[i,j,k] += hyper_op(c, dx1, p[i,j-2, k], p[i, j-1, k], p[i, j, k], p[i,j+1,k], p[i,j+2,k])


class HyperViscosity:
    def __init__(self, namelist,  Grid, ScalarState, VelocityState):
  
  
        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        
        
        return
    
    def initialize(self,):
        

        
        return 
    
    def update(self):
        
        dx = self._Grid.dx
        
        c = (dx[0] * dx[1])**(3.0/2.0) * 0.125
        for v in ['u', 'v', 'w']:
            p = self._VelocityState.get_field(v)
            p_t = self._VelocityState.get_tend(v)
            
            compute_hyper(dx[0], dx[1], c, p, p_t)    
        
        
        
        return 