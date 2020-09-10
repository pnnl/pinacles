from Columbia.Radiation import RadiationBase, RadiationRRTMG


from mpi4py import MPI

def factory(namelist, Grid, Ref, ScalarState, DiagnosticState):
    try:
        radiation_model = namelist['radiation']['model']
    except:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('Looks like there is no radiation model specified in the namelist!')
        radiation_model = None
    

    if radiation_model is None:
        return RadiationBase(namelist, Grid, Ref, ScalarState, DiagnosticState)
    elif radiation_model == 'rrtmg':
        return RadiationRRTMG(namelist, Grid, Ref, ScalarState, DiagnosticState)
