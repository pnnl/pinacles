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

    # Instantiate and Initialize the Simulation
    
    factor = 3
    partent_pts = 64


    root_point = (32, 32, 32)

    
    Sim = SimulationStandard.SimulationStandard(namelist)

    nest_namelist = copy.deepcopy(namelist)
    for i in range(2):
        nest_namelist['grid']['n'][i] = partent_pts * factor
        nest_namelist['grid']['l'][i] =  partent_pts * namelist['grid']['l'][i]/namelist['grid']['n'][i]    
    
    nest_namelist['meta']['simname'] = namelist['meta']['simname'] + '_nest'
    print(nest_namelist)


    llx = Sim.ModelGrid.x_edge_global[Sim.ModelGrid.n_halo[0]-1 + root_point[0]]
    lly = Sim.ModelGrid.y_edge_global[Sim.ModelGrid.n_halo[1]-1 + root_point[1]]
    llz = 0.0

    Nest1 = SimulationStandard.SimulationStandard(nest_namelist, llx, lly, llz)

    ListOfSims = [Sim, Nest1]


    #S_slice = SimulationUtilities.HorizontalSlice('qv_20m', height=20, frequency=10, var='s', state='ScalarState', Sim=Sim)
    #Albedo = SimulationUtilities.Albedo(20.0, Sim)

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


    # Iterate through io classes and do first IO
    for nest_i in range(len(ListOfSims)):
        Nest = ListOfSims[nest_i]
        for item in io_classes[nest_i]:
            if hasattr(item, 'update'):
                item.update()
            elif hasattr(item, 'dump_restart'):
                print(nest_i)
                item.dump_restart(Nest.TimeSteppingController.time)

    # Compute how long the first integration step should be
    last_io_time = np.zeros_like(io_frequencies) + Sim.TimeSteppingController.time
    integrate_by_dt = np.amin(io_frequencies)


    # This is the outerloop over time
    while Sim.TimeSteppingController.time < Sim.TimeSteppingController.time_max:

        for n, Nest_i in enumerate(ListOfSims):
            print('Nest: ', n)
            # Integrate model forward by integrate_by_dt seconds
            if n >= 1:
                parent = ListOfSims[0]
            else:
                parent = None

            Nest_i.update(ParentNest=parent, integrate_by_dt=integrate_by_dt)



            #Adjust the integration to to make sure output is at the correct time
            time = Nest_i.TimeSteppingController.time
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

