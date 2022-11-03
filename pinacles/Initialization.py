import numpy as np
<<<<<<< HEAD
from scipy import interpolate
from pinacles import CaseStableBubble
=======
import pinacles.ThermodynamicsDry_impl as DryThermo
import pinacles.ThermodynamicsMoist_impl as MoistThermo
from pinacles.CaseReal import InitializeReanalysis
import netCDF4 as nc
from scipy import interpolate
from pinacles import UtilitiesParallel
>>>>>>> f5fc59432c51a8e42e904ff73084e6c98f8f8f5f
from mpi4py import MPI
from pinacles import UtilitiesParallel
import pinacles.CaseSullivanAndPatton as CaseSullivanAndPatton
import pinacles.CaseStableBubble as CaseStableBubble
import pinacles.CaseBOMEX as CaseBomex
import pinacles.CaseDYCOMS as CaseDycoms
import pinacles.CaseRICO as CaseRico
import pinacles.CaseATEX as CaseATEX
import pinacles.CaseTestbed as CaseTestbed
import netCDF4 as nc
CASENAMES = [
    "sullivan_and_patton",
    "stable_bubble",
    "bomex",
    "dycoms",
    "dycoms_rotated",
    "rico",
    "atex",
    "testbed",
    "real",
]


def real(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest):

    init_class = InitializeReanalysis(
        namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest
    )
    init_class.initialize()

    return


def factory(namelist):
    assert namelist["meta"]["casename"] in CASENAMES

    if namelist["meta"]["casename"] == "sullivan_and_patton":
        return CaseSullivanAndPatton.initialize
    elif namelist["meta"]["casename"] == "stable_bubble":
        return CaseStableBubble.initialize
    elif namelist["meta"]["casename"] == "bomex":
        return CaseBomex.initialize
    elif (namelist["meta"]["casename"] == "dycoms" or namelist["meta"]["casename"] == "dycoms_rotated"):
        return CaseDycoms.initialize
    elif namelist["meta"]["casename"] == "rico":
        return CaseRico.initialize
    elif namelist["meta"]["casename"] == "atex":
        return CaseATEX.initialize
    elif namelist["meta"]["casename"] == "testbed":
        return CaseTestbed.initialize
    elif namelist["meta"]["casename"] == "real":
        return real
    else:
        UtilitiesParallel.print_root(
            "Caanot find initialization for: ", namelist["meta"]["casename"]
        )


<<<<<<< HEAD
def initialize(namelist, ModelGrid, TimeStepManager, Ref, ScalarState, VelocityState):
    init_function = factory(namelist)
    
    init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    
    try:
        interp_restart = namelist["restart"]["interp_restart"]
        infile = namelist['restart']["infile"]
    except:
        interp_restart = False
        infile = None
        
    
    # If this is not an interpoalted restart simply return
    if not interp_restart:
        return
        
    UtilitiesParallel.print_root("\t \t Interpolating Restart From: " + infile)
    
    # Interpolate restarted data
    if MPI.COMM_WORLD.Get_rank() == 0:
        df = nc.Dataset(infile, 'r')
        
        x_in = df.x
        y_in =  df.y 
        z_in = df.z 
        
        x_e_in = df.x_edge
        y_e_in = df.y_edge
        z_e_in = df.z_edge
        
        time_in = df.time
        
    else:
        x_in, y_in, z_in = None, None, None
        x_e_in, y_e_in, z_e_in = None, None, None
        time_in = None
        
    x_in = MPI.COMM_WORLD.bcast(x_in)
    y_in = MPI.COMM_WORLD.bcast(y_in)
    z_in = MPI.COMM_WORLD.bcast(z_in)
    
    x_e_in = MPI.COMM_WORLD.bcast(x_e_in)
    y_e_in = MPI.COMM_WORLD.bcast(y_e_in)
    z_e_in = MPI.COMM_WORLD.bcast(z_e_in)
    
    time_in = MPI.COMM_WORLD.bcast(time_in)
    
    TimeStepManager._time = time_in
    
    x= ModelGrid._local_axes[0]
    y = ModelGrid._local_axes[1]
    z  = ModelGrid._local_axes[2]
    
    x_e = ModelGrid._local_axes_edge[0]
    y_e = ModelGrid._local_axes_edge[1]
    z_e  = ModelGrid._local_axes_edge[2]
    #Loop over scalar fields and interpolate
    for c in [ScalarState, VelocityState]:
        for v in c._dofs:

            if v == 'w':
                inpts = (x_in, y_in, z_e_in)
                xg, yg, zg = np.meshgrid(x, y, z_e, indexing = 'ij')
            elif v == 'v':
                inpts = (x_in, y_e_in, z_in)
                xg, yg, zg = np.meshgrid(x, y_e, z, indexing = 'ij')
            elif v == 'u':
                inpts = (x_e_in, y_in, z_in)
                xg, yg, zg = np.meshgrid(x_e, y, z, indexing = 'ij')
            else:
                inpts = (x_in, y_in, z_in)
                xg, yg, zg = np.meshgrid(x, y, z, indexing = 'ij')
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                in_data = df[v][:,:,:]
            else:
                in_data = None
                
            in_data = MPI.COMM_WORLD.bcast(in_data)
                
            d = c.get_field(v)
            g =  np.stack((xg.flatten(), yg.flatten(), zg.flatten()),axis=1)
            interp = interpolate.interpn( inpts, in_data , g, method="linear", bounds_error=False, fill_value=None)


            d[:,:,:] = interp.reshape(x.shape[0], y.shape[0], z.shape[0])
     
    if MPI.COMM_WORLD.Get_rank() == 0:
        df.close() 
=======
def initialize(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest=None):
    init_function = factory(namelist)

    try:
        UtilitiesParallel.print_root('\t Initializing without ingest option')
        init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState)
    except:
        UtilitiesParallel.print_root('\t Initializing with ingest option')
        init_function(namelist, ModelGrid, Ref, ScalarState, VelocityState, Ingest)
>>>>>>> f5fc59432c51a8e42e904ff73084e6c98f8f8f5f
    
    return
