from mpi4py import MPI
import argparse
import json
import numpy as np
import pylab as plt
import netCDF4 as nc 
import uuid, datetime
import SimulationClass


def main(namelist):

    n = 4 
    gcm_res = 25*1e3
    couple_dt = 600.0
    forced_fields = ['qv', 's']
    domains = []
    uls = 10.0
    for i in range(n):
        namelist["meta"]["output_directory"] = './couple_' + str(i)
        domains.append(SimulationClass.Simulation(namelist, i))
        domains[i].initialize()

    if MPI.COMM_WORLD.Get_rank() == 0:
        rt_grp = nc.Dataset('couple_out.nc', 'w')

        for i in range(n):
            cpl_grp = rt_grp.createGroup('couple_' + str(i))
            print(domains[i].ModelGrid.n[2])
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


    edge_nudging = True
    center_min = 3 + 32
    center_max = 3 + 64

    for couple_time in np.arange(couple_dt,10*86400+couple_dt, couple_dt):
        for i in range(n):

            if not edge_nudging:
            #With out edge nudging
                for v in forced_fields:
                    if not v == 'qv':
                        ls_forcing[i][v] = (ls_state[i][v] - domains[i].ScalarState.mean(v) )/couple_dt
                    else:
                        ls_forcing[i][v] = (ls_state[i][v] - domains[i].ScalarState.mean(v) 
                            - domains[i].ScalarState.mean('qc') - domains[i].ScalarState.mean('qr') )/couple_dt
                domains[i].update(couple_time, ls_forcing[i])
                for v in forced_fields:
                    if not v == 'qv':
                        ss_forcing[i][v] = (domains[i].ScalarState.mean(v) - ls_state[i][v])/couple_dt
                    else:
                        ss_forcing[i][v] = (domains[i].ScalarState.mean(v) + domains[i].ScalarState.mean('qc') + domains[i].ScalarState.mean('qr') 
                        - ls_state[i][v])/couple_dt
            else:
            #with edge nudgeing
                for v in forced_fields:
                    if not v == 'qv':

                        phi = domains[i].ScalarState.get_field(v)
                        mean = np.mean(phi[center_min:center_max,3,:], axis=0)

                        ls_forcing[i][v] = (ls_state[i][v] - mean)/couple_dt
                    else:
                        qv = domains[i].ScalarState.get_field('qv')
                        qc = domains[i].ScalarState.get_field('qc')
                        qr = domains[i].ScalarState.get_field('qr')

                        mean = np.mean(qv[center_min:center_max,3,:], axis=0) + np.mean(qc[center_min:center_max,3,:], axis=0) + np.mean(qr[center_min:center_max,3,:], axis=0)

                        ls_forcing[i][v] = (ls_state[i][v] - mean )/couple_dt
                domains[i].update(couple_time, ls_forcing[i])
                for v in forced_fields:
                    if not v == 'qv':
                        phi = domains[i].ScalarState.get_field(v)
                        mean = np.mean(phi[center_min:center_max,3,:], axis=0)

                        ss_forcing[i][v] = (mean - ls_state[i][v])/couple_dt
                    else:

                        qv = domains[i].ScalarState.get_field('qv')
                        qc = domains[i].ScalarState.get_field('qc')
                        qr = domains[i].ScalarState.get_field('qr')

                        mean = np.mean(qv[center_min:center_max,3,:], axis=0) + np.mean(qc[center_min:center_max,3,:], axis=0) + np.mean(qr[center_min:center_max,3,:], axis=0)

                        ss_forcing[i][v] = (mean - ls_state[i][v])/couple_dt

        for i in range(n):
            for v in forced_fields:
                adv = uls * (ls_state[(i-1)%n][v] - ls_state[i%n][v] )/gcm_res
                ls_state[i][v] += adv * couple_dt + ss_forcing[i][v] * couple_dt


        if MPI.COMM_WORLD.Get_rank() == 0:
            rt_grp = nc.Dataset('couple_out.nc', 'r+')
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


