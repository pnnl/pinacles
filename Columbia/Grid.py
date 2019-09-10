from mpi4py import MPI 
import mpi4py_fft  
import numpy as np 
from mpi4py_fft.pencil import Subcomm

class GridBase: 
    def __init__(self, namelist): 

        #The total number of points in the domain NOT including halo/ghost points 
        self._n = np.array(namelist['grid']['n'], dtype=np.int) 
        assert len(self._n) == 3 

        #The number of halo points
        self._n_halo = np.array(namelist['grid']['n_halo'], dtype=np.int) 
        assert len(self._n_halo) == 3 

        #The total number of points in the CRM domain including halo/ghost points 
        self._ngrid = self._n + 2 * self._n_halo

        #Lenght of each CRM domain side 
        self._l = np.array(namelist['grid']['l'], dtype=np.double) 
        assert len(self._l) == 3 

        #The global x,y,z coordiantes 
        self._global_axes  = None 
        self._local_axes = None

        #Store sub-cummunicators created by mpi4py_fft 
        self.subcomms = None 
        self._create_subcomms() 

        #Get local grid information 
        self._get_local_grid_indicies() 

        return 
        
    @property
    def n(self): 
        return self._n 

    @property
    def n_halo(self): 
        return self._n_halo

    @property
    def l(self): 
        return self._l 

    @property
    def ngrid(self): 
        return self._ngrid

    @property
    def x_global(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return np.copy(self._global_axes[0]) 

    @property 
    def y_global(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return np.copy(self._global_axes[1])

    @property 
    def z_global(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return np.copy(self._global_axes[2]) 

    @property
    def z_global_edge(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return np.copy(self._global_axes_edge[2]) 


    @property
    def global_axes(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return self._global_axes.copy() 

    @property
    def x_local(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        start = self._local_start[0] 
        end = self._local_end[0] + self._n_halo
        return np.copy(self._global_axes[0][start:end])
    
    @property
    def y_local(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        start = self._local_start[1]
        end = self._local_end[1] + self._n_halo
        return np.copy(self._global_axes[1][start:end])
    
    @property
    def z_local(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        start = self._local_start[2]
        end = self._local_end[2] + self._n_halo 
        return np.copy(self._global_axes[2][start:end])

    @property
    def local_axes(self): 
        ''' Copy here is forced to keep _global_axes externally immutable,  
        if performace becomes an issue we can provide a property that return a 
        view so that copy occurs. 
        '''
        return np.copy(self._local_axes)

    @property 
    def local_shape(self): 
        return self._local_shape



    def _create_subcomms(self): 
        self.subcomms = Subcomm(MPI.COMM_WORLD, dims=[0,0,1])
        return 

    def _get_local_grid_indicies(self): 
       
        #Create a dummy array using the given subcomms and the 
        # global domain size 
        dum_array = mpi4py_fft.DistArray(self._n, self.subcomms)
        
        #Get shape of dum_array 
        self._local_shape = np.array(dum_array.shape)
        
        #Get the starting index and ending of this ranks part of the dist array 
        #these are global indicies 
        self._local_start = np.array(dum_array.substart)
        self._local_end = self._local_start + self._local_shape




        return 

class RegularCartesian(GridBase): 
    def __init__(self, namelist): 

        GridBase.__init__(self, namelist)
        self._compute_globalcoordiantes()

        return 

    def _compute_globalcoordiantes(self): 
    
        self._global_axes = [] 
        self._global_axes_edge = [] 
        for i in range(3): 
            dx = self._l[i]/self._n[i]
            
            #Location of lowest most halo point 
            lx = (-self._n_halo[i] +0.5)  * dx 

            #Location of upper most halo point 
            ux = ((self._n[i]+self._n_halo[i]) - 0.5) * dx

            #Generate an axis based on upper and lower points 
            self._global_axes.append(np.linspace(lx, ux, self.ngrid[i]))
            self._global_axes_edge.append(self._global_axes[i] + 0.5 * dx)


        return 

