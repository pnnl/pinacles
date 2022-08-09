import argparse
import numpy as np
import uuid
from mpi4py import MPI
import json
import datetime
from pinacles import SimulationStandard


def main(namelist):

    # Instantiate and Initialize the Simulation
    Sim = SimulationStandard.SimulationStandard(namelist)

    # Put all of the output classes into a list (these are just references)


    io_classes = [
        Sim.StatsIO,
        Sim.FieldsIO,
        Sim.Fields2d,
        Sim.CoarseGrain,
        Sim.IOTower,
        Sim.Restart,
        Sim.Timers,
        Sim.Parts,
        Sim.PlatSim
    ]


    if Sim.Rad.time_synced :
        io_classes.append(Sim.Rad)

    # Determine all of the output frequencies
    io_frequencies = []
    for ic in io_classes:
        io_frequencies.append(ic.frequency)
    io_frequencies = np.array(io_frequencies)

    # Iterate through io classes and do first IO
    for item in io_classes:

        if hasattr(item, "output"):
            item.output()
        elif hasattr(item, "update"):
            item.update()
        elif hasattr(item, "dump_restart"):
            Sim.Timers.start_timer("Restart")
            item.dump_restart(Sim.TimeSteppingController.time)
            Sim.Timers.start_timer("Restart")

    # Compute how long the first integration step should be
    last_io_time = np.zeros_like(io_frequencies) + Sim.TimeSteppingController.time
    integrate_by_dt = np.amin(io_frequencies)

    # This is the outer loop over time
    while Sim.TimeSteppingController.time < Sim.TimeSteppingController.time_max:

        # Integrate model forward by integrate_by_dt seconds
        Sim.update(integrate_by_dt=integrate_by_dt)

        # Adjust the integration to make sure output is at the correct time
        time = Sim.TimeSteppingController.time
        for idx, item in enumerate(io_classes):
            if np.round(time - io_frequencies[idx],10) == np.round(last_io_time[idx],10):
                if hasattr(item, "output"):
                    item.output()
                elif hasattr(item, "update"):
                    item.update()
                elif hasattr(item, "dump_restart"):
                    item.dump_restart(Sim.TimeSteppingController.time)
                # We did output here so lets update last io-time
                last_io_time[idx] = np.round(time, 10)

        integrate_by_dt = np.amin(last_io_time + io_frequencies - time)
        
        integrate_by_dt = np.round(integrate_by_dt, 10)
        
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Input for pinacles an LES!")
    parser.add_argument("inputfile")
    args = parser.parse_args()

    # Broadcast a uuid and wall time
    unique_id = None
    wall_time = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        unique_id = uuid.uuid4()
        wall_time = datetime.datetime.now()

    unique_id = MPI.COMM_WORLD.bcast(str(unique_id))
    wall_time = MPI.COMM_WORLD.bcast(str(wall_time))

    with open(args.inputfile, "r") as namelist_h:
        namelist = json.load(namelist_h)
        namelist["meta"]["unique_id"] = unique_id
        namelist["meta"]["wall_time"] = wall_time
        main(namelist)
