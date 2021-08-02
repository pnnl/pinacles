import argparse
import numpy as np
import uuid
from mpi4py import MPI
import json
import datetime
from pinacles import SimulationStandard
from pinacles import SimulationUtilities
import copy

def main(namelist):

    # Instantiate and Initialize the Simulation class for the outermost domain
    Sim = SimulationStandard.SimulationStandard(namelist, nest_num=0)

    # Put this the outermost domain in the zero-location of a Python LIst
    ListOfSims = [Sim]

    #Get the toal number of lists
    n_nests = namelist['nests']['n']
    
    # Now we create a class for each of the nests
    for i in range(n_nests):

        # Create a namelist file for each of the nests 
        nest_namelist = copy.deepcopy(namelist)      # This will be used as a namelist for the i + 1 th nest
        parent_namelist = ListOfSims[i]._namelist    # This is the namelist of the i-th nest
        parent_sim = ListOfSims[i]                   # This is the namelist of the i-th simulation class
        
        # Loop over the dimensions 
        for dim in range(2):
            
            #assert(namelist['nests']['factor'][i]%2 != 0) # Check that the nest refinment factor is odd

            # Compute the number of points and domain physical size for the i+1th nest 
            nest_namelist['grid']['n'][dim] = namelist['nests']['parent_points'][i][dim]* namelist['nests']['factor'][i][dim]
            nest_namelist['grid']['l'][dim] =  namelist['nests']['parent_points'][i][dim]* parent_namelist['grid']['l'][dim]/parent_namelist['grid']['n'][dim]

        nest_namelist['grid']['n'][2] = parent_namelist['grid']['n'][2] * namelist['nests']['factor'][i][2]

        # Here we use the i + 1 th nest namelist to communicat informaton to the nest class
        # TODO Perhaps we could clean this up as may get confusing
        nest_namelist['nest'] = {}
        nest_namelist['nest']['factor'] = namelist['nests']['factor'][i]
        nest_namelist['nest']['parent_pts'] = namelist['nests']['parent_points'][i]
        nest_namelist['nest']['root_point'] = namelist['nests']['ll_corner'][i]

        nest_namelist['pressure'] = "open"
        nest_namelist['lbc']['type'] = "open"
        nest_namelist['lbc']['open_boundary_treatment'] = "nest"
        
        # Create a new simname for i + 1 th nest
        nest_namelist['meta']['simname'] = namelist['meta']['simname'] + '_nest_' + str(i)
        
        # Compute the lower left corner point for the i + 1 th nest
        llx = parent_sim.ModelGrid.x_edge_global[parent_sim.ModelGrid.n_halo[0]-1 + namelist['nests']['ll_corner'][i][0]]
        lly = parent_sim.ModelGrid.y_edge_global[parent_sim.ModelGrid.n_halo[1]-1 + namelist['nests']['ll_corner'][i][1]]
        llz = 0.0  # This will always be zero

        # Instantiate and initialize the 
        this_sim = SimulationStandard.SimulationStandard(nest_namelist, llx, lly, llz, ParentNest=parent_sim,  nest_num=i+1)
        
        # Append this simulation to the end of the list of simulations
        ListOfSims.append(this_sim)

    # Put all of the output classes into a list (these are just references)
    io_classes = []
    for Nest_i in ListOfSims:
        io_classes.append([Nest_i.StatsIO,
                 Nest_i.FieldsIO,
                 Nest_i.IOTower, 
                 Nest_i.Restart])


    # Determine all of the output frequencies
    io_frequencies = []
    for nest_i in range(len(ListOfSims)):
        for ic in io_classes[nest_i]:
            io_frequencies.append(ic.frequency)
    io_frequencies = np.array(io_frequencies)

    print(io_frequencies)

    # Iterate through io classes and do first IO
    for nest_i in range(len(ListOfSims)):
        Nest = ListOfSims[nest_i]
        for item in io_classes[nest_i]:
            if hasattr(item, 'update'):
                item.update()
            elif hasattr(item, 'dump_restart'):
                #print(nest_i)
                item.dump_restart(Nest.TimeSteppingController.time)

    # Compute how long the first integration step should be
    last_io_time = np.zeros_like(io_frequencies) + Sim.TimeSteppingController.time
    integrate_by_dt = np.amin(io_frequencies)

    # This is the outerloop over time
    while Sim.TimeSteppingController.time < Sim.TimeSteppingController.time_max:

        #for n, Nest_i in enumerate(ListOfSims):
        #    print('Nest: ', n)
        #    # Integrate model forward by integrate_by_dt seconds
        #    if n >= 1:
        #        parent = ListOfSims[n-1]
        #    else:
        #        parent = None
        #
        #    Nest_i.update(ParentNest=parent, integrate_by_dt=integrate_by_dt)

            #Adjust the integration to to make sure output is at the correct time
        
        ListOfSims[0].update(integrate_by_dt=integrate_by_dt, ListOfSims=ListOfSims)

        time = ListOfSims[0].TimeSteppingController.time
        for n, Nest_i in enumerate(ListOfSims):
            for idx, item in enumerate(io_classes[n]):
                if time - io_frequencies[n * len(io_classes[n]) + idx] == last_io_time[n * len(io_classes[n]) + idx]:
                    if hasattr(item, 'update'):
                        item.update()
                    elif hasattr(item, 'dump_restart'):
                        item.dump_restart(Sim.TimeSteppingController.time)
                    # We did output here so lets update last io-time
                    last_io_time[n * len(io_classes[n]) + idx] = time

        # Coupute how long t
        integrate_by_dt = np.amin(last_io_time + io_frequencies - time)
        
        if integrate_by_dt == 0.0:
            print(last_io_time , io_frequencies, last_io_time + io_frequencies,  time)
            import sys; sys.exit()

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input for pinacles an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    #Broadcast a uuid and wall time
    unique_id= None
    wall_time = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        unique_id = uuid.uuid4()
        wall_time = datetime.datetime.now()


    unique_id = MPI.COMM_WORLD.bcast(str(unique_id))
    wall_time = MPI.COMM_WORLD.bcast(str(wall_time))

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        namelist['meta']['unique_id'] = unique_id
        namelist['meta']['wall_time'] = wall_time
        main(namelist)

