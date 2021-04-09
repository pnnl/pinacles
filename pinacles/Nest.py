import numba 
import numpy as np

class Nest:

    def __init__(self, TimeSteppingController, Grid, ScalarState, VelocityState):

        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._TimeSteppingController = TimeSteppingController
        
        self.x_left_bdys = {}
        self.x_right_bdys = {}
        self.y_left_bdys = {}
        self.y_right_bdys = {}


        self.factor = 3
        self.partent_pts = 64

        self.root_point = (32, 32, 32)

        return


    @staticmethod
    @numba.njit()
    def relax_x_parallel(n_halo, factor,  indx_range, tau_i, parent_var, var, var_tend):
        # The index range applies to y because we are relaxing an x boundary

        # Rexlaxtaion along the x boundary
        var_shape = var.shape

        x_indx_range = (n_halo[0], var_shape[0] - n_halo[0])   #Loop over the full range of x w/o halos
        y_indx_range = (indx_range[0], indx_range[0] + (indx_range[1]-indx_range[0])*factor)
        z_indx_range = (n_halo[2], var_shape[2] - n_halo[2])   #Loop over the full range of z w/o halos
        

        # Loop over only the subset of points that needs to be updated.
        for i in range(x_indx_range[0],  x_indx_range[1]):
            i_parent = (i -  x_indx_range[0]) //factor
            for j in range(y_indx_range[0], y_indx_range[1]):
                j_parent = (j- y_indx_range[0])//factor
                for k in range(z_indx_range[0],  z_indx_range[1]):
                    k_parent = k - z_indx_range[0]
                    #if k == 5:
                    #    print(parent_var[i_parent,j_parent,k_parent], var[i,j,k], i_parent, j_parent, i, j)
                    var_tend[i,j,k] += tau_i * (1.0/(np.abs(y_indx_range[0]-j)+ 1)) * (parent_var[i_parent,j_parent,k_parent] - var[i,j,k])

        return

    @staticmethod
    @numba.njit()
    def relax_y_parallel(n_halo, factor,  indx_range, tau_i, parent_var, var, var_tend):

        # The index range applies to y because we are relaxing an x boundary

        # Rexlaxtaion along the x boundary
        var_shape = var.shape

        x_indx_range = (indx_range[0], indx_range[0] + (indx_range[1]-indx_range[0])*factor) #(n_halo[0], var_shape[0] - n_halo[0])   #Loop over the full range of x w/o halos
        y_indx_range =  (n_halo[1], var_shape[1]- n_halo[1])#(2*n_halo[1], var_shape[1] - 2*n_halo[1]) #(n_halo[1] + factor, var_shape[1] - n_halo[1]-factor) #(indx_range[0], indx_range[0] + (indx_range[1]-indx_range[0])*factor)
        z_indx_range = (n_halo[2], var_shape[2] - n_halo[2])   #Loop over the full range of z w/o halos
        

        # Loop over only the subset of points that needs to be updated.
        for i in range(x_indx_range[0],  x_indx_range[1]):
            i_parent = (i- x_indx_range[0])//factor 
            for j in range(y_indx_range[0], y_indx_range[1]):
                if j >= 2*n_halo[1] and j  < var_shape[1]- 2*n_halo[1]:
                    j_parent = (j-y_indx_range[0]) //factor
                    for k in range(z_indx_range[0],  z_indx_range[1]):
                        k_parent = k - z_indx_range[0]
                        var_tend[i,j,k] += tau_i * (1.0/(np.abs(x_indx_range[0]-i) + 1)) * (parent_var[i_parent,j_parent,k_parent] - var[i,j,k])

        return


    def update_boundaries(self, ParentNest):
        
        parent_nhalo = ParentNest.ModelGrid.n_halo
        x  = ParentNest.ModelGrid.x_local
        y  = ParentNest.ModelGrid.y_local

        local_start = self._Grid._local_start
        local_end = self._Grid._local_end

        for v in self._ScalarState._dofs:
            
            #This is the location of the lower corenr of the nest in the parent's index
            center_point_x = parent_nhalo[0] + self.root_point[0] 
            center_point_y = parent_nhalo[1] + self.root_point[1] 

            #First we get the low-x boundary
            slab_range = (center_point_y, center_point_y+1)

            #Now get the indicies of the subset on this rank
            local_part_of_parent =  ((local_start[0])//self.factor + self.root_point[0],
                (local_end[0] )//self.factor + self.root_point[0])
           
            self.x_left_bdys[v] = ParentNest.ScalarState.get_slab_y(v,
                                                                slab_range 
                                                                )[local_part_of_parent[0]:local_part_of_parent[1],:,:]
          
            slab_range = (center_point_y + self.partent_pts-1, center_point_y+self.partent_pts)
   
            self.x_right_bdys[v] = ParentNest.ScalarState.get_slab_y(v,
                                                                 slab_range
                                                                 )[local_part_of_parent[0]:local_part_of_parent[1],:,:]

            #Now get the indicies of the subset on this rank
            local_part_of_parent =  ((local_start[1])//self.factor + self.root_point[1], 
                (local_end[1] )//self.factor + self.root_point[1])

            slab_range = (center_point_x, center_point_x+1)
            
            self.y_left_bdys[v] = ParentNest.ScalarState.get_slab_x(v,
                                                                slab_range
                                                                )[:,local_part_of_parent[0]:local_part_of_parent[1],:]

            slab_range = (center_point_x + self.partent_pts-1 , center_point_x+self.partent_pts )
            self.y_right_bdys[v] = ParentNest.ScalarState.get_slab_x(v, 
                                                                    slab_range
                                                                    )[:,local_part_of_parent[0]:local_part_of_parent[1],:]
        
        v = 'w'
        #This is the location of the lower corenr of the nest in the parent's index
        center_point_x = parent_nhalo[0] + self.root_point[0] 
        center_point_y = parent_nhalo[1] + self.root_point[1]

    
        #First we get the low-x boundary
        slab_range = (center_point_y, center_point_y+1)

        #Now get the indicies of the subset on this rank
        local_part_of_parent =  ((local_start[0])//self.factor + self.root_point[0],(local_end[0] )//self.factor + self.root_point[0])
        self.x_left_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                            )[local_part_of_parent[0]:local_part_of_parent[1],:,:]
        slab_range = (center_point_y + self.partent_pts-1, center_point_y+self.partent_pts)

        self.x_right_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                                )[local_part_of_parent[0]:local_part_of_parent[1],:,:]

        #Now get the indicies of the subset on this rank

        local_part_of_parent =  ((local_start[1])//self.factor + self.root_point[1],(local_end[1] )//self.factor + self.root_point[1])

        slab_range = (center_point_x, center_point_x+1)
        self.y_left_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                            slab_range
                                                            )[:,local_part_of_parent[0]:local_part_of_parent[1],:]
        
        slab_range = (center_point_x + self.partent_pts-1, center_point_x+self.partent_pts)       
        self.y_right_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                                   slab_range
                                                                   )[:,local_part_of_parent[0]:local_part_of_parent[1],:]

        v = 'u'
        #This is the location of the lower corenr of the nest in the parent's index
        center_point_x = parent_nhalo[0] + self.root_point[0] - 1    # Shift to account for stagger
        center_point_y = parent_nhalo[1] + self.root_point[1]

    
        #First we get the low-x boundary
        slab_range = (center_point_y, center_point_y+1)

        #Now get the indicies of the subset on this rank
        local_part_of_parent =  ((local_start[0])//self.factor + self.root_point[0],(local_end[0] )//self.factor + self.root_point[0])
        self.x_left_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                            )[local_part_of_parent[0]:local_part_of_parent[1],:,:]
        slab_range = (center_point_y + self.partent_pts-1, center_point_y+self.partent_pts)

        self.x_right_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                                )[local_part_of_parent[0]:local_part_of_parent[1],:,:]

        #Now get the indicies of the subset on this rank

        local_part_of_parent =  ((local_start[1])//self.factor + self.root_point[1],(local_end[1] )//self.factor + self.root_point[1])

        slab_range = (center_point_x, center_point_x+1)
        self.y_left_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                            slab_range
                                                            )[:,local_part_of_parent[0]:local_part_of_parent[1],:]
        
        slab_range = (center_point_x + self.partent_pts-1, center_point_x+self.partent_pts)       
        self.y_right_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                                   slab_range
                                                                   )[:,local_part_of_parent[0]:local_part_of_parent[1],:]

        v = 'v'
        #This is the location of the lower corenr of the nest in the parent's index
        center_point_x = parent_nhalo[0] + self.root_point[0]    # Shift to account for stagger
        center_point_y = parent_nhalo[1] + self.root_point[1] - 1

    
        #First we get the low-x boundary
        slab_range = (center_point_y, center_point_y+1)

        #Now get the indicies of the subset on this rank
        local_part_of_parent =  ((local_start[0])//self.factor + self.root_point[0],(local_end[0] )//self.factor + self.root_point[0])
        self.x_left_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                            )[local_part_of_parent[0]:local_part_of_parent[1],:,:]
        slab_range = (center_point_y + self.partent_pts-1, center_point_y+self.partent_pts)

        self.x_right_bdys[v] = ParentNest.VelocityState.get_slab_y(v,
                                                            slab_range
                                                                )[local_part_of_parent[0]:local_part_of_parent[1],:,:]

        #Now get the indicies of the subset on this rank

        local_part_of_parent =  ((local_start[1])//self.factor + self.root_point[1],(local_end[1] )//self.factor + self.root_point[1])

        slab_range = (center_point_x, center_point_x+1)
        self.y_left_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                            slab_range
                                                            )[:,local_part_of_parent[0]:local_part_of_parent[1],:]
        
        slab_range = (center_point_x + self.partent_pts-1, center_point_x+self.partent_pts)       
        self.y_right_bdys[v] = ParentNest.VelocityState.get_slab_x(v, 
                                                                   slab_range
                                                                   )[:,local_part_of_parent[0]:local_part_of_parent[1],:]

        return

    def update(self, ParentNest):

        factor = self.factor
        partent_pts = self.partent_pts

        root_point = self.root_point


        parent_nhalo = ParentNest.ModelGrid.n_halo
        n_halo = self._Grid.n_halo
        
    
        x_local = self._Grid.x_local
        y_local = self._Grid.y_local

        x_local_parent = ParentNest.ModelGrid.x_local
        y_local_parent = ParentNest.ModelGrid.y_local

        tau_i = 1.0/self._TimeSteppingController.dt

        for v in ['s']:

            var = self._ScalarState.get_field(v)
            var_tend = self._ScalarState.get_tend(v)
            
            indx_range = (n_halo[1], n_halo[1]+1)
            self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_left_bdys[v], var, var_tend)

            indx_range = (var.shape[1] - 2 * n_halo[1],var.shape[1] - 2 * n_halo[1]+1)
            self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_right_bdys[v], var, var_tend)

            indx_range = (n_halo[0], n_halo[0]+1)
            self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_left_bdys[v], var, var_tend)  

            indx_range = (var.shape[0] - 2 * n_halo[0],var.shape[0] - 2 * n_halo[0]+1)
            self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_right_bdys[v], var, var_tend) 

        return
        v = 'u'
        var = self._VelocityState.get_field(v)
        var_tend = self._VelocityState.get_tend(v)

        indx_range = (n_halo[1], n_halo[1]+1)
        self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_left_bdys[v], var, var_tend)
        
        indx_range = (var.shape[1] - 2 * n_halo[1],var.shape[1] - 2 * n_halo[1]+1)
        self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_right_bdys[v], var, var_tend)

        indx_range = (n_halo[0], n_halo[0]+1)
        self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_left_bdys[v], var, var_tend)  
        
        indx_range = (var.shape[0] - 2 * n_halo[0],var.shape[0] - 2 * n_halo[0]+1)
        self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_right_bdys[v], var, var_tend) 

        v = 'v'
        var = self._VelocityState.get_field(v)
        var_tend = self._VelocityState.get_tend(v)
         
        indx_range = (n_halo[1], n_halo[1]+1)
        self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_left_bdys[v], var, var_tend)
        
        indx_range = (var.shape[1] - 2 * n_halo[1],var.shape[1] - 2 * n_halo[1]+1)
        self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_right_bdys[v], var, var_tend)

        indx_range = (n_halo[0], n_halo[0]+1)
        self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_left_bdys[v], var, var_tend)  
        
        indx_range = (var.shape[0] - 2 * n_halo[0],var.shape[0] - 2 * n_halo[0]+1)
        self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_right_bdys[v], var, var_tend) 


        #v = 'w'
        #var = self._VelocityState.get_field(v)
        #var_tend = self._VelocityState.get_tend(v)
        # 
        #indx_range = (n_halo[1], n_halo[1]+1)
        #self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_left_bdys[v], var, var_tend)
        
        #indx_range = (var.shape[1] - 2 * n_halo[1],var.shape[1] - 2 * n_halo[1]+1)
        #self.relax_x_parallel(n_halo, self.factor,  indx_range, tau_i, self.x_right_bdys[v], var, var_tend)

        #indx_range = (n_halo[0], n_halo[0]+1)
        #self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_left_bdys[v], var, var_tend)  
        
        #indx_range = (var.shape[0] - 2 * n_halo[0],var.shape[0] - 2 * n_halo[0]+1)
        #self.relax_y_parallel(n_halo, self.factor,  indx_range, tau_i, self.y_right_bdys[v], var, var_tend) 

        return