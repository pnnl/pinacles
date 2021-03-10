import numba 

class Nest:

    def __init__(self, TimeSteppingController, Grid, ScalarState, VelocityState):

        self._Grid = Grid
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._TimeSteppingController = TimeSteppingController
        return


    @staticmethod
    @numba.njit()
    def relax_parent(x_local, x_local_parent, 
                      root_point, 
                      parent_points, 
                      parent_halo,
                      factor, 
                      parent_scalar, 
                      n_halo,
                      scalar, 
                      scalars_tend):



        parent_shape = parent_scalar.shape
        our_shape = scalar.shape

        # First set the boundarines on y
        for i in range(2 + n_halo[0] + root_point[0], n_halo[0] + root_point[0] + parent_points - 2):
            i_nest = (i - root_point[0] - n_halo[0]) * factor + n_halo[0]  
            for j in range(2 + n_halo[1] + root_point[1], n_halo[1] + root_point[1] + parent_points -2):
                j_nest = (j - root_point[1] - n_halo[1]) * factor + n_halo[1]  
                #print(i, j, i_nest, j_nest, x_local_parent[i], x_local[i_nest])
                #return
                for k in range(parent_shape[2]):
                    parent_scalar[i,j,k] = scalar[i_nest ,j_nest,k] 


        return 



    @staticmethod
    @numba.njit()
    def relax_scalars(root_point, 
                      parent_points, 
                      parent_halo,
                      factor, tau_i, 
                      parent_scalar, 
                      n_halo,
                      scalar, 
                      scalars_tend):


        parent_shape = parent_scalar.shape
        our_shape = scalar.shape

        # First set the boundarines on y
        for i in range(n_halo[0], our_shape[0]):
            iparent = root_point[0] + (parent_halo[0]-1) +  (i-1) // factor 
            for j in range(n_halo[1], n_halo[1]+2*factor):
                jparent = (parent_halo[1]-1) +  root_point[1] +  (j-1)  // factor
                #print(x_local[i], x_local_parent[iparent], y_local[j], y_local_parent[jparent])
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += tau_i * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])
            le = our_shape[1] - n_halo[1]
            for j in range(le - 2*factor, le):
                jparent = root_point[1] + (parent_halo[1]-1) +  (j-1)  // factor

                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += tau_i * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])

        # Now set the boundaries on x
        for i in range(n_halo[0], n_halo[0] + 2*factor):
            iparent = root_point[0] +  (parent_halo[0]-1) +  (i-1) // factor
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] + (parent_halo[1]-1) +  (j-1)  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += tau_i * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])

        le = our_shape[0] - n_halo[0]
        for i in range(le - 2*factor, le):
            iparent = root_point[0] +  (parent_halo[0]-1) +  (i-1) // factor
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] + (parent_halo[1]-1) +  (j-1)  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += tau_i * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])

        return



    @staticmethod
    @numba.njit()
    def relax_u(root_point, 
                      parent_point, 
                      parent_halo,
                      factor, 
                      parent_scalar, 
                      n_halo,
                      scalar, 
                      scalars_tend):

        parent_shape = parent_scalar.shape
        our_shape = scalar.shape

        # First set the boundarines on y
        for i in range(n_halo[0]+factor, our_shape[0]-n_halo[0]-factor):
            #i = i -1 
            iparent = root_point[0] +  parent_halo[0] +  i // factor
            for j in range(n_halo[1], n_halo[1]+2*factor):
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])

            le = our_shape[1] - n_halo[1]
            for j in range(le - 2*factor, le):
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])


        # Now set the boundaries on x
        for i in range(n_halo[0], n_halo[0] + 2*factor):
            #i = i -1 
            iparent = root_point[0] +  parent_halo[0] +  i // factor 
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])
        
        
        le = our_shape[0] - n_halo[0]
        for i in range(le - 2*factor, le):
            iparent = root_point[0] +  parent_halo[0] +  i // factor
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])
        
        return


    @staticmethod
    @numba.njit()
    def relax_v(root_point, 
                      parent_points, 
                      parent_halo,
                      factor, 
                      parent_scalar, 
                      n_halo,
                      scalar, 
                      scalars_tend):

        parent_shape = parent_scalar.shape
        our_shape = scalar.shape

        # First set the boundarines on y
        for i in range(n_halo[0]+factor, our_shape[0]-n_halo[0]-factor):
            iparent = root_point[0] +  parent_halo[0] +  i // factor
            for j in range(n_halo[1], n_halo[1]+2*factor):
                #j = j -1 
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                
                #print()
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])

            le = our_shape[1] - n_halo[1]
            for j in range(le - 2*factor, le):
                jparent = root_point[1] + parent_halo[1] +  j  // factor 
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])


        # Now set the boundaries on x
        for i in range(n_halo[0], n_halo[0] + 2*factor):
            iparent = root_point[0] +  parent_halo[0] +  i // factor
            #j = j - 1 
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] +  parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])
        
        
        le = our_shape[0] - n_halo[0]
        for i in range(le - 2*factor, le):
            iparent = root_point[0] +  parent_halo[0] +  i // factor
            for j in range(n_halo[1], our_shape[1]-n_halo[1]):
                jparent = root_point[1] + parent_halo[1] +  j  // factor
                for k in range(parent_shape[2]):
                    scalars_tend[i,j,k] += 1.0/10.0 * (parent_scalar[iparent, jparent, k] - scalar[i,j,k])
        
        return




    def update(self, ParentNest):

        factor = 3
        partent_pts = 32

        root_point = (32, 32, 32)


        parent_nhalo = ParentNest.ModelGrid.n_halo
        n_halo = self._Grid.n_halo
        
    
        x_local = self._Grid.x_local
        y_local = self._Grid.y_local

        x_local_parent = ParentNest.ModelGrid.x_local
        y_local_parent = ParentNest.ModelGrid.y_local

        #print(x_local)
        #print(x_local_parent)


        tau_i = 0.5 * 1.0/self._TimeSteppingController.dt

        for v in ['s', 'qv']:
            parent_scalar = ParentNest.ScalarState.get_field(v)
            scalar = self._ScalarState.get_field(v)
            scalar_tend = self._ScalarState.get_tend(v)
            
            self.relax_scalars(root_point, partent_pts, parent_nhalo, factor, tau_i, parent_scalar, n_halo, scalar, scalar_tend)
        


        for v in ['u', 'v', 'w']:
            parent_scalar = ParentNest.VelocityState.get_field(v)
            scalar = self._VelocityState.get_field(v)
            scalar_tend = self._VelocityState.get_tend(v)


            if v == 'w':
                self.relax_scalars(root_point, partent_pts, parent_nhalo, factor, tau_i, parent_scalar, n_halo, scalar, scalar_tend)
            elif v == 'u':
                self.relax_scalars(root_point, partent_pts, parent_nhalo, factor, tau_i, parent_scalar, n_halo, scalar, scalar_tend)
                #self.relax_u(root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
            elif v == 'v':
                self.relax_scalars(root_point, partent_pts, parent_nhalo, factor, tau_i, parent_scalar, n_halo, scalar, scalar_tend)
                #self.relax_v(root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)   

        return


    def update_parent(self, ParentNest):

        factor = 3
        partent_pts = 64

        root_point = (32, 32, 32)


        parent_nhalo = ParentNest.ModelGrid.n_halo
        n_halo = self._Grid.n_halo
        
    
        x_local = self._Grid.x_local
        y_local = self._Grid.y_local

        x_local_parent = ParentNest.ModelGrid.x_local
        y_local_parent = ParentNest.ModelGrid.y_local

        #print(x_local)
        #print(x_local_parent)


        for v in ['s', 'qv']:
            parent_scalar = ParentNest.ScalarState.get_field(v)
            scalar = self._ScalarState.get_field(v)
            scalar_tend = self._ScalarState.get_tend(v)
            
            self.relax_parent(x_local, x_local_parent, root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
        


        for v in ['u', 'v', 'w']:
            parent_scalar = ParentNest.VelocityState.get_field(v)
            scalar = self._VelocityState.get_field(v)
            scalar_tend = self._VelocityState.get_tend(v)


            if v == 'w':
                self.relax_parent(x_local, x_local_parent, root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
            elif v == 'u':
                self.relax_parent(x_local, x_local_parent, root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
                #self.relax_u(root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
            elif v == 'v':
                self.relax_parent(x_local, x_local_parent, root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)
                #self.relax_v(root_point, partent_pts, parent_nhalo, factor, parent_scalar, n_halo, scalar, scalar_tend)   


        ParentNest.PSolver.update()
        return

