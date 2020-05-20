from mpi4py import MPI
import argparse
import json
import numpy as np

import SimulationClass

def main(namelist):

    n = 1 
    gcm_res = 50*1e3
    couple_dt = 600.0
    forced_fields = ['qv', 's']
    domains = []
    uls = 10.0
    for i in range(n):
        namelist["meta"]["output_directory"] = './couple_' + str(i)
        domains.append(SimulationClass.Simulation(namelist, i))
        domains[i].initialize()

    ls_state = []
    #Get GCM initial condition
    for i in range(n):
        ls_state.append({})
        for v in forced_fields:
            ls_state[i][v] = domains[i].ScalarState.mean(v)
            np.shape(ls_state[i][v])

    #Set up storage for LES
    ls_forcing = []
    ss_forcing = []
    for i in range(n):
        ss_forcing.append({})
        ls_forcing.append({})

    for couple_time in np.arange(couple_dt,21600.0+couple_dt, couple_dt):
        print(couple_time)

    for couple_time in np.arange(couple_dt,10*86400+couple_dt, couple_dt):
        for i in range(n):
            for v in forced_fields:
                ls_forcing[i][v] = (ls_state[i][v] - domains[i].ScalarState.mean(v) )/couple_dt
            domains[i].update(couple_time, ls_forcing[i])
            for v in forced_fields:
                ss_forcing[i][v] = (domains[i].ScalarState.mean(v) - ls_state[i][v])/couple_dt

        for i in range(n):
            for v in forced_fields:
                adv = uls * (ls_state[(i-1)%n][v] - ls_state[i%n][v] )/gcm_res
                ls_state[i][v] += adv * couple_dt + ss_forcing[i][v] * couple_dt

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input for Columbia an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        main(namelist)


