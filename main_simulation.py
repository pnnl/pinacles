import argparse
import numpy as np
import uuid
from mpi4py import MPI
import json
import datetime
from pinacles import SimulationStandard
from pinacles import SimulationUtilities

def main(namelist):

    # Instantiate and Initialize the Simulation
    Sim = SimulationStandard.SimulationStandard(namelist)

    #S_slice = SimulationUtilities.HorizontalSlice('qv_20m', height=20, frequency=10, var='s', state='ScalarState', Sim=Sim)
    #Albedo = SimulationUtilities.Albedo(20.0, Sim)

    # Put all of the output classes into a list (these are just references)
    io_classes = [Sim.StatsIO,
                 Sim.FieldsIO,
                 Sim.IOTower, 
                 Sim.Restart]

    # Determine all of the output frequencies
    io_frequencies = []
    for ic in io_classes:
        io_frequencies.append(ic.frequency)
    io_frequencies = np.array(io_frequencies)

    # Iterate through io classes and do first IO
    for item in io_classes:

        if hasattr(item, 'update'):
            item.update()
        elif hasattr(item, 'dump_restart'):
            item.dump_restart(Sim.TimeSteppingController.time)

    # Compute how long the first integration step should be
    last_io_time = np.zeros_like(io_frequencies) + Sim.TimeSteppingController.time
    integrate_by_dt = np.amin(io_frequencies)
   
    # This is the outerloop over time
    while Sim.TimeSteppingController.time < Sim.TimeSteppingController.time_max:

        # Integrate model forward by integrate_by_dt seconds
        Sim.update(integrate_by_dt=integrate_by_dt)

        #Adjust the integration to to make sure output is at the correct time
        time = Sim.TimeSteppingController.time
        for idx, item in enumerate(io_classes):
            if time - io_frequencies[idx] == last_io_time[idx]:
                if hasattr(item, 'update'):
                    item.update()
                elif hasattr(item, 'dump_restart'):
                    item.dump_restart(Sim.TimeSteppingController.time)
                # We did output here so lets update last io-time
                last_io_time[idx] = time
        
        # Coupute how long t
        integrate_by_dt = np.amin(last_io_time + io_frequencies - time)


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

