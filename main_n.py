from mpi4py import MPI
import argparse
import json
import numpy as np
import pylab as plt
import netCDF4 as nc 
import os 
import SimulationClass


def main(namelist):

    n = 4
    gcm_res = 50*1e3
    couple_dt = 600.0
    forced_fields = ['qv', 's']
    domains = []
    uls = 5.0
    
    
    #Get wind profiles
    infile = nc.Dataset('./stats.nc', 'r')
    
    u_ls_profile = infile['VelocityState']['profiles']['u'][-1,:] - 9.9
    v_ls_profile = infile['VelocityState']['profiles']['v'][-1,:] - 3.8
        
    
    
    w_spd = np.sqrt(u_ls_profile**2.0 + v_ls_profile**2.0)
    
    print(u_ls_profile)
    print(v_ls_profile)
    print(w_spd)
    
    infile.close()
    
    
    out_root = namelist["meta"]["output_directory"]
    sim_name = namelist["meta"]['simname']
    couple_out_path = os.path.join(namelist["meta"]["output_directory"], 'couple_out_' + sim_name + '.nc')
    
    for i in range(n):
        namelist["meta"]["output_directory"] = os.path.join(out_root, 'couple_' + str(i))
        domains.append(SimulationClass.Simulation(namelist, i))
        domains[i].initialize()

    if MPI.COMM_WORLD.Get_rank() == 0:
        rt_grp = nc.Dataset(couple_out_path, 'w')

        for i in range(n):
            cpl_grp = rt_grp.createGroup('couple_' + str(i))
            cpl_grp.createDimension('z', size=domains[i].ModelGrid.n[2])
            z = cpl_grp.createVariable('z', np.double,  dimensions=('z',))

            cpl_grp.createDimension('t')
            t = cpl_grp.createVariable('t', np.double, dimensions=('t',))

            nh = domains[i].ModelGrid.n_halo
            z[:] = domains[i].ModelGrid.z_global[nh[2]:-nh[2]]


            #Now add variables for each of the large scale focing
            for v in forced_fields:
                cpl_grp.createVariable(v + '_ss_forcing', np.double, dimensions=('t','z'))
                cpl_grp.createVariable(v + '_ls_forcing', np.double, dimensions=('t','z'))
                cpl_grp.createVariable(v + '_ls_state', np.double, dimensions=('t','z'))
        rt_grp.close()


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

        


    for couple_time in np.arange(couple_dt,10*86400+couple_dt, couple_dt):
        for i in range(n):
            for v in forced_fields:
                if not v == 'qv':
                    ls_forcing[i][v] = (ls_state[i][v] - domains[i].ScalarState.mean(v) )/couple_dt
                else:
                    ls_forcing[i][v] = (ls_state[i][v] - domains[i].ScalarState.mean(v) 
                        - domains[i].ScalarState.mean('qc') - domains[i].ScalarState.mean('qr') )/couple_dt
                    
            nh = domains[i].ModelGrid.n_halo
            nprof = len(u_ls_profile)
            u_adv = np.zeros(nprof + 2*nh[2], dtype=np.double)
            v_adv = np.zeros(nprof + 2*nh[2], dtype=np.double)
            #print(np.shape(u_adv), nprof, 2*nh[2])
            u_adv[nh[2]:-nh[2]] = u_ls_profile
            v_adv[nh[2]:-nh[2]] = v_ls_profile        
            
            #print('qv', (domains[i].ScalarState.mean('qv') - ls_state[i]['qv'])/couple_dt)
            
            domains[i].update(couple_time, ls_forcing[i], u_adv, v_adv)
            
            #print('qv', (domains[i].ScalarState.mean('qv') - ls_state[i]['qv'])/couple_dt)
            
            for v in forced_fields:
                if not v == 'qv':
                    ss_forcing[i][v] = (domains[i].ScalarState.mean(v) - ls_state[i][v])/couple_dt
                else:
                    ss_forcing[i][v] = (domains[i].ScalarState.mean(v) + domains[i].ScalarState.mean('qc') + domains[i].ScalarState.mean('qr') 
                     - ls_state[i][v])/couple_dt


                    
        for i in range(n):
            
            nh = domains[i].ModelGrid.n_halo
            nprof = len(u_ls_profile)
            u_adv = np.zeros(nprof + 2*nh[2], dtype=np.double)
            v_adv = np.zeros(nprof + 2*nh[2], dtype=np.double)
            wind_adv = np.zeros(nprof + 2*nh[2], dtype=np.double)
            #print(np.shape(u_adv), nprof, 2*nh[2])
            u_adv[nh[2]:-nh[2]] = u_ls_profile
            v_adv[nh[2]:-nh[2]] = v_ls_profile        
            wind_adv[nh[2]:-nh[2]] = w_spd  
            
            
            for v in forced_fields:
                adv = np.zeros_like(u_adv)
                for k in range(u_adv.shape[0]):
                    if wind_adv[k] >= 0.0:
                        adv[k] = -wind_adv[k] * (ls_state[(i)%n][v][k] - ls_state[(i-1)%n][v][k] )/gcm_res
                    else:
                        adv[k] = -wind_adv[k] * (ls_state[(i+1)%n][v][k] - ls_state[i%n][v][k] )/gcm_res
                ls_state[i][v] += adv * couple_dt + ss_forcing[i][v] * couple_dt


        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset(couple_out_path, 'r+')
            for i in range(n):
                nh = domains[i].ModelGrid.n_halo

                this_grp = rt_grp['couple_' + str(i)]
                t = this_grp['t']
                t[t.shape[-1]] = couple_time
                rt_grp.sync()
                for v in forced_fields:
                    this_grp[v + '_ss_forcing'][-1,:] = ss_forcing[i][v][nh[2]:-nh[2]]
                    this_grp[v + '_ls_forcing'][-1,:] = ls_forcing[i][v][nh[2]:-nh[2]]
                    this_grp[v + '_ls_state'][-1,:] = ls_state[i][v][nh[2]:-nh[2]]
            rt_grp.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input for Columbia an LES!')
    parser.add_argument('inputfile')
    args = parser.parse_args()

    with open(args.inputfile, 'r') as namelist_h:
        namelist = json.load(namelist_h)
        main(namelist)


