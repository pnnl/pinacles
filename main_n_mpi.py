from mpi4py import MPI
import argparse
import json
import numpy as np
import pylab as plt
import netCDF4 as nc 

import SimulationClass


COMM_WORLD = MPI.COMM_WORLD

def main(namelist):

    n = 2
    gcm_res = 50*1e3
    couple_dt =600.0
    forced_fields = ['qv', 's']
    # domain
    uls = 10.0



    #Check to make sure that we have 1 mpi rank for each CRM column
    size = COMM_WORLD.Get_size() 
    rank = COMM_WORLD.Get_rank()
    if not size == n: 
        import sys; sys.exit()


    LOCAL_COMM = COMM_WORLD.Split(rank)

    #Initialize on domain on each rank
    for i in range(n):
        if i == rank:
            namelist["meta"]["output_directory"] = './couple_' + str(i)
            domain = SimulationClass.Simulation(namelist, i, LOCAL_COMM)
            domain.initialize()


    if MPI.COMM_WORLD.Get_rank() == 0:
        rt_grp = nc.Dataset('couple_out.nc', 'w')
        
        for i in range(n):
            cpl_grp = rt_grp.createGroup('couple_' + str(i))
            cpl_grp.createDimension('z', size=domain.ModelGrid.n[2])
            z = cpl_grp.createVariable('z', np.double,  dimensions=('z',))
            
            cpl_grp.createDimension('t')
            t = cpl_grp.createVariable('t', np.double, dimensions=('t',))
            
            nh = domain.ModelGrid.n_halo
            z[:] = domain.ModelGrid.z_global[nh[2]:-nh[2]]


            #Now add variables for each of the large scale focing
            for v in forced_fields:
                cpl_grp.createVariable(v + '_ss_forcing', np.double, dimensions=('t','z'))
                cpl_grp.createVariable(v + '_ls_forcing', np.double, dimensions=('t','z'))
                cpl_grp.createVariable(v + '_ls_state', np.double, dimensions=('t','z'))
        rt_grp.close()

    #On each rank
    gcm_state_local = {}
    for v in forced_fields:
        gcm_state_local[v] = domain.ScalarState.mean(v)
    
    #Now put the 
    gcm_state_global = MPI.COMM_WORLD.allgather(gcm_state_local)

    #import sys; sys.exit()
    #ls_state = []
    #Get GCM initial condition
    #for i in range(n):
    #    ls_state.append({})
    #    for v in forced_fields:
    #        ls_state[i][v] = domains[i].ScalarState.mean(v)
    #        np.shape(ls_state[i][v])

    #Set up storage for LES
    ls_forcing = []
    ss_forcing = []
    for i in range(n):
        ss_forcing.append({})
        ls_forcing.append({})



    for couple_time in np.arange(couple_dt,20*86400+couple_dt, couple_dt):
        crm_state_local = {}
        for v in forced_fields + ['qc' , 'qr']:
            crm_state_local[v] = domain.ScalarState.mean(v)

        #Now put the 
        crm_state_global = MPI.COMM_WORLD.allgather(crm_state_local)

        for i in range(n):
            for v in forced_fields:
                if not v == 'qv':
                    ls_forcing[i][v] = (gcm_state_global[i][v] - crm_state_global[i][v] )/couple_dt
                else:
                    ls_forcing[i][v] = (gcm_state_global[i][v] - crm_state_global[i][v] 
                        - crm_state_global[i]['qc'] - crm_state_global[i]['qr'] )/couple_dt
            
        domain.update(couple_time, ls_forcing[rank])
        
        for i in range(n):
            for v in forced_fields + ['qc' , 'qr']:
                crm_state_local[v] = domain.ScalarState.mean(v)

        crm_state_global = MPI.COMM_WORLD.allgather(crm_state_local)
        
        for i in range(n):
            for v in forced_fields:
                if not v == 'qv':
                    ss_forcing[i][v] = (crm_state_global[i][v] - gcm_state_global[i][v])/couple_dt
                else:
                    ss_forcing[i][v] = (crm_state_global[i][v] + crm_state_global[i]['qc'] + crm_state_global[i]['qr']
                     - gcm_state_global[i][v])/couple_dt

        for i in range(n):
            for v in forced_fields:
                adv = uls * (gcm_state_global[(i-1)%n][v] - gcm_state_global[i%n][v] )/gcm_res
                gcm_state_global[i][v] += adv * couple_dt + ss_forcing[i][v] * couple_dt


        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset('couple_out.nc', 'r+')
            for i in range(n):
                nh = domain.ModelGrid.n_halo

                this_grp = rt_grp['couple_' + str(i)]
                t = this_grp['t']
                t[t.shape[-1]] = couple_time
                rt_grp.sync()
                for v in forced_fields:
                    this_grp[v + '_ss_forcing'][-1,:] = ss_forcing[i][v][nh[2]:-nh[2]]
                    this_grp[v + '_ls_forcing'][-1,:] = ls_forcing[i][v][nh[2]:-nh[2]]
                    this_grp[v + '_ls_state'][-1,:] = gcm_state_global[i][v][nh[2]:-nh[2]]
            rt_grp.close()

        if rank == 0: 
            print('Time: ', str(couple_time))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input for Columbia an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        main(namelist)


