from pinacles.externals.parcel_model_wrapper import parcel_model_via_cffi
from pinacles import parameters
import pinacles.ThermodynamicsMoist_impl as MoistThermo
from mpi4py import MPI
import numba
import numpy as np

class AerosolBinBase():
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):
        self._Timers = Timers
        self._Grid = Grid
        self._Ref = Ref
        self._ScalarState = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self.name = "AerosolBinBase"

        return
    
    def update(self):
        return
    def set_aerosol_scalars(self, loc_indices_x, loc_indices_y, loc_indices_z, injection_rate,grid_cell_mass):
        # Do nothing in the base class
        return
    def boundary_outflow(self):
        return

class AerosolBinModel(AerosolBinBase):
    def __init__(
        self,
        namelist,
        Timers,
        Grid,
        Ref,
        ScalarState,
        VelocityState,
        DiagnosticState,
        TimeSteppingController,
    ):
        AerosolBinBase.__init__(
            self,
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController)

        self.name = "AerosolBinModel"
        self._nbins_aero = 20
        
        # Set up the chemistry
        try:
            self._aerosol_species = namelist['aerosolbin']['species']
        except:
            self._aerosol_species = 'ammonium_sulfate'
        if self._aerosol_species == 'ammonium_sulfate':
            ichem = 1
            self._rhos=1.769
            self._rhosap=2.093
            self._soluba=0.1149
            self._solubb=-4.489e-4
            self._solubc=1.385e-6
            self._Ms=132.14
            self._kap=0.61
        elif self._aerosol_species == 'sea_salt':
            ichem = 2
            self._rhos=2.165
            self._rhosap=2.726
            self._soluba=0.1805
            self._solubb=-5.310e-4
            self._solubc=9.965e-7
            self._Ms=58.44
            self._kap=1.28
        elif self._aerosol_species == 'biomass_burning':
            ichem = 3
            self._rhos=1.662
            self._rhosap=2.000
            self._soluba=0.1149
            self._solubb=-4.489e-4
            self._solubc=1.385e-6
            self._Ms=111.00
            self._kap=0.1
        else:
            print('UNRECOGNIZED AEROSOL, DEFAULT TO AMMONIUM SULFATE')
            self._aerosol_species = 'ammonium_sulfate'
            ichem =1
            self._rhos=1.769
            self._rhosap=2.093
            self._soluba=0.1149
            self._solubb=-4.489e-4
            self._solubc=1.385e-6
            self._Ms=132.14
            self._kap=0.61
        
        # Insoluble component--not used currently but included for consistency
        self._solfracm = 1.0
      
        self._rhois = 2.6  # DUST
        # self._rhois = 2.0 # SOOT
        self._solfracv=(self._solfracm*self._rhois/(self._rhos + self._solfracm*(self._rhois-self._rhos)))

   
        self._aero_cffi = parcel_model_via_cffi.AerosolModel()
        self._aero_cffi.init(ichem, self._nbins_aero)


        self._ScalarState.add_variable(
                'ql_aero',
                long_name='mixing ratio of aerosol liquid',
                units="kg kg^{-1}",
                latex_name = "q_{l,aero}",
                limit=True,
            )
        self._DiagnosticState.add_variable(
                'aerosol_conc',
                long_name='aerosol concentration',
                units="# m^{-3}",
                latex_name = "n_{aero}",
            )
        # Create the prognosed and diagnostic scalars
        for i in range(self._nbins_aero):
            
            self._ScalarState.add_variable(
                'pmass_aero_'+str(i),
                long_name='mixing ratio of dry aerosol bin '+str(i),
                units="kg kg^{-1}",
                latex_name = "q_{dry}",
                limit=True,
            )
            self._ScalarState.add_variable(
                'pmass_total_'+str(i),
                long_name='mixing ratio of dry aerosol bin '+str(i),
                units="kg kg^{-1}",
                latex_name = "q_{dry}",
                limit=True,
            )
            self._DiagnosticState.add_variable(
                'pmass_total_cbase_'+str(i),
                long_name = 'cloud base total mass of particles in bin ' + str(i),
                units = "m",
                latex_name = 'pmass_total_cbase',
            )
            self._DiagnosticState.add_variable(
                'pmass_aero_cbase_'+str(i),
                long_name = ' cloud base total mass of aerosols in bin ' + str(i),
                units = "m",
                latex_name = 'pmass_aero_cbase',
            )
            self._DiagnosticState.add_variable(
                'prad_'+str(i),
                long_name = 'radius of particle in bin ' + str(i),
                units = "m",
                latex_name = 'particleradius',
            )
            self._DiagnosticState.add_variable(
                'pconc_'+str(i),
                long_name = 'number of particles bin ' + str(i),
                units = "# m^{-3}",
                latex_name = 'particle concentration',
            )
            self._DiagnosticState.add_variable(
                'pmass1_'+str(i),
                long_name = 'mass of particle in bin ' + str(i),
                units = "kg",
                latex_name = 'particle mass',
            )
            self._DiagnosticState.add_variable(
                'T_tendency_aerosol',
                long_name = 'temperature tendency from aerosol processes ',
                units = "K s^{-1}",
                latex_name = 'temperature tendency aerosol',
            )
            self._DiagnosticState.add_variable(
                'qv_tendency_aerosol',
                long_name = 'qv tendency from aerosol processes ',
                units = "kg kg s^{-1}",
                latex_name = 'qv tendency aerosol',
            )



        # Parameters of aerosol bins--below are cgs
        self._dev = 1.5
        self._geo = 0.06e-4
        radmin=0.01e-4 # CM
        radmax=1.0e-4  #CM
        self._bprad0 = np.zeros((self._nbins_aero + 1),dtype=np.double)
        self._prad0 = np.zeros((self._nbins_aero),dtype=np.double)
        self._pmass0 = np.zeros_like(self._prad0)
        self._concfrac_init = np.zeros_like(self._prad0)
        
        raddiff = (np.log(radmax)-np.log(radmin))/self._nbins_aero
        self._bprad0[0] = radmin
        # Droplet related setup
        drop_radmin = 20.0  * 1.0e-4 # CM
        drop_radmax = 30.0 * 1.0e-4 # CM
        self._drop_br0 = np.zeros_like(self._bprad0)

        drop_raddiff = (np.log(drop_radmax)-np.log(drop_radmin))/self._nbins_aero
        self._drop_br0[0]  = drop_radmin
        for ii in range(self._nbins_aero):
            self._bprad0[ii+1] = np.exp(np.log(self._bprad0[ii])+raddiff)
            self._prad0[ii]  = 0.5 * (self._bprad0[ii]+self._bprad0[ii+1])
            self._pmass0[ii] = self._rhos*self._solfracv*4.0/3.0*np.pi*self._prad0[ii]**3.0
            self._pmass0[ii] +=self._rhois*(1.0-self._solfracv)*4.0/3.0*np.pi*self._prad0[ii]**3.0 # GRAMS

            coe1 = 1.0/ np.sqrt(2.0 * np.pi)/np.log(self._dev)
            coe2 = 2.0 * np.log(self._dev)**2.0
            self._concfrac_init[ii] = (coe1/self._prad0[ii])*np.exp(-(np.log(self._prad0[ii])-np.log(self._geo))**2.0/coe2)  # units  are (fraction of #)/CM3/CM
            self._concfrac_init[ii] *= (self._bprad0[ii+1]-self._bprad0[ii]) # (fraction of  #)/CM3   

            # Droplet sizes
            self._drop_br0[ii+1] = np.exp(np.log(self._drop_br0[ii])+drop_raddiff)

      
        self._Timers.add_timer("AerosolBinModel_update")

        self.dt_cpm = 1.0e-3
        return
    
    def update(self):
        # print('aerosol bin update')
        self._Timers.start_timer("AerosolBinModel_update")

        zl = self._Grid.z_local

        # Get variables from the model state

        T = self._DiagnosticState.get_field("T")
        qv = self._ScalarState.get_field("qv")
        ql_aero = self._ScalarState.get_field("ql_aero")
        qc = self._ScalarState.get_field("qc")
        s = self._ScalarState.get_field("s")
        p0 = self._Ref.p0
        rho0 = self._Ref.rho0
        nlocal = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        npts = ( 
            (nlocal[0] - 2 * nhalo[0]) 
            * (nlocal[1] - 2 * nhalo[1]) 
            * (nlocal[2] - 2 * nhalo[2])
            )
        
        pmass_total_m = np.zeros((npts,self._nbins_aero),dtype=np.double, order='F')
        pmass_aero_m = np.zeros((npts, self._nbins_aero),dtype=np.double, order='F')
        T_1d = np.zeros((npts),dtype=np.double, order='F')
        qv_1d = np.zeros((npts), dtype=np.double, order='F')
        pressure_1d = np.zeros((npts), dtype=np.double, order='F')
        density_1d = np.zeros((npts),dtype=np.double, order='F')
        ql_1d = np.zeros((npts), dtype=np.double, order='F')

        ipt = 0
        for i in range(nhalo[0],nlocal[0]-nhalo[0]):
            for j in range(nhalo[1],nlocal[1]-nhalo[1]):
                for k in range(nhalo[2],nlocal[2]-nhalo[2]):
                    # if i == 8 and j==8 and k ==4:
                    #     ipt_injection = ipt
                    T_1d[ipt] = T[i,j,k]
                    qv_1d[ipt] = qv[i,j,k]
                    pressure_1d[ipt] = p0[k]
                    density_1d[ipt] = rho0[k]
                    ipt +=1
        for ibin in range(self._nbins_aero):
            pmass_aero_3d = self._ScalarState.get_field('pmass_aero_'+str(ibin))
            pmass_total_3d = self._ScalarState.get_field('pmass_total_'+str(ibin))
            pmass_aero_cbase = self._DiagnosticState.get_field('pmass_aero_cbase_'+str(ibin))
            pmass_total_cbase = self._DiagnosticState.get_field('pmass_aero_cbase_'+str(ibin))
            ipt = 0
            for i in range(nhalo[0],nlocal[0]-nhalo[0]):
                for j in range(nhalo[1],nlocal[1]-nhalo[1]):
                    for k in range(nhalo[2],nlocal[2]-nhalo[2]):
                        if qc[i,j,k] > 1e-10:
                            pmass_aero_cbase[i,j,k]+=pmass_aero_3d[i,j,k]
                            pmass_total_cbase[i,j,k]+=pmass_total_3d[i,j,k]
                            pmass_aero_3d[i,j,k] = 0.0
                            pmass_total_3d[i,j,k] = 0.0
    
                        pmass_total_m[ipt,ibin] = pmass_total_3d[i,j,k]
                        pmass_aero_m[ipt,ibin] = pmass_aero_3d[i,j,k]                    
                        ipt +=1

        # print('ipt_injection', ipt_injection)
        # print('Injection point properties prior to update')
        # print(T_1d[ipt_injection], qv_1d[ipt_injection])
        # print(pmass_total_m[ipt_injection,:])
        # print(pmass_aero_m[ipt_injection,:])
        
        dt_les = self._TimeSteppingController.dt
        # print('going to cffi')
        self._aero_cffi.update(
            npts,
            self._nbins_aero, 
            dt_les,
            self.dt_cpm, 
            T_1d, 
            pressure_1d, 
            qv_1d,
            ql_1d,
            density_1d,
            pmass_aero_m,
            pmass_total_m)

        # print('returned from cffi')
        dTaero = self._DiagnosticState.get_field('T_tendency_aerosol')
        dqaero = self._DiagnosticState.get_field('qv_tendency_aerosol')
        ipt = 0
        # print('Injection point properties after  update')
        # print(T_1d[ipt_injection], qv_1d[ipt_injection])
        # print(pmass_total_m[ipt_injection,:])
        # print(pmass_aero_m[ipt_injection,:])

        for i in range(nhalo[0],nlocal[0]-nhalo[0]):
            for j in range(nhalo[1],nlocal[1]-nhalo[1]):
                for k in range(nhalo[2],nlocal[2]-nhalo[2]):
                    dTaero[i,j,k] = (T_1d[ipt] - T[i,j,k])/dt_les
                    dqaero[i,j,k] = (qv_1d[ipt] - qv[i,j,k])/dt_les
                    T[i,j,k] = T_1d[ipt]
                    qv[i,j,k] = qv_1d[ipt]
                    ql_aero[i,j,k] = ql_1d[ipt]
                    s[i,j,k] = MoistThermo.s(zl[k],T[i,j,k],qc[i,j,k]+ql_aero[i,j,k],0.0)
                    ipt +=1
        for ibin in range(self._nbins_aero):
            pmass_aero_3d = self._ScalarState.get_field('pmass_aero_'+str(ibin))
            pmass_total_3d = self._ScalarState.get_field('pmass_total_'+str(ibin))
            ipt = 0
            for i in range(nhalo[0],nlocal[0]-nhalo[0]):
                for j in range(nhalo[1],nlocal[1]-nhalo[1]):
                    for k in range(nhalo[2],nlocal[2]-nhalo[2]):
                        pmass_total_3d[i,j,k] = pmass_total_m[ipt,ibin]
                        pmass_aero_3d[i,j,k]  = pmass_aero_m[ipt,ibin]                
                        ipt +=1

        self.update_aerosol_diagnostics()

        self.dt_cpm = 1.0e-3

        self._Timers.end_timer("AerosolBinModel_update")
        return

    def update_aerosol_diagnostics(self):
        # print('update aerosol diagnostics')
        nlocal = self._Grid.ngrid_local
        nhalo = self._Grid.n_halo
        rhow_cgs = parameters.RHOW * 1.0e-3
        n_a = self._DiagnosticState.get_field('aerosol_conc')
        
        for ii in range(self._nbins_aero):
            prad_i = self._DiagnosticState.get_field('prad_'+str(ii))
            pconc_i = self._DiagnosticState.get_field('pconc_'+str(ii))
            pmass1_i = self._DiagnosticState.get_field('pmass1_'+str(ii))
            pmass_aero_3d = self._ScalarState.get_field('pmass_aero_'+str(ii))
            pmass_total_3d = self._ScalarState.get_field('pmass_total_'+str(ii))
            for i in range(nhalo[0],nlocal[0]-nhalo[0]):
                for j in range(nhalo[1],nlocal[1]-nhalo[1]):
                    for k in range(nhalo[2],nlocal[2]-nhalo[2]):
                        if pmass_aero_3d[i,j,k] < 1e-3/self._Ref.rho0[k] * self._pmass0[ii]*1e-3:
                            pmass1_i[i,j,k] = self._pmass0[ii] * 1.0e-3 # convert to kg
                            prad_i[i,j,k] = self._prad0[ii] * 1.0e-2 # convert to m
                            pconc_i[i,j,k] = pmass_aero_3d[i,j,k]/pmass1_i[i,j,k] * self._Ref.rho0[k]

                        elif pmass_total_3d[i,j,k] <= pmass_aero_3d[i,j,k]:
                            pmass1_i[i,j,k] = self._pmass0[ii] * 1.0e-3 # convert to kg
                            prad_i[i,j,k] = self._prad0[ii] * 1.0e-2 # convert to m
                            pconc_i[i,j,k] = pmass_aero_3d[i,j,k]/pmass1_i[i,j,k] * self._Ref.rho0[k] # Number/m^3
                        else:
                            conc_aero_m= pmass_aero_3d[i,j,k]/(self._pmass0[ii] * 1e-3) # aerosol number = drop number
                            pconc_i[i,j,k] = conc_aero_m * self._Ref.rho0[k] # Number/m^3
                            pmass1_i[i,j,k] = pmass_total_3d[i,j,k]/conc_aero_m # mass of drop in kg
                            pmass1_g = pmass1_i[i,j,k] * 1000.0
                
                            mola = self._pmass0[ii]*self._solfracv/self._Ms/(pmass1_g-self._pmass0[ii])
                            rhows = rhow_cgs+(self._rhosap - rhow_cgs)*mola/(self._rhosap/rhow_cgs/self._Ms+mola)
                            prad_cm=0.75*(pmass1_g-(1.0-self._solfracm)*self._pmass0[ii])/rhows/np.pi
                            prad_cm=(prad_cm+(1.0-self._solfracv)*self._prad0[ii]**3.0)**(1.0/3.0)
                            prad_i[i,j,k] = prad_cm * 1e-2
                        if ii == 0:
                            n_a[i,j,k] = pconc_i[i,j,k]
                        else:
                            n_a[i,j,k] += pconc_i[i,j,k]

                        

        return





    # replicating the aerospec and dropspec code in python
    # This might not be the best approach but we can clean up later
    # when the overall algorithm is better defined
    def set_aerosol_scalars(self, loc_indices_x, loc_indices_y, loc_indices_z, injection_rate,grid_cell_mass):
        # injection_rate ==> number of particles per second
        # print("indices", loc_indices_x,loc_indices_y,loc_indices_z)

        rhow_cgs = parameters.RHOW * 1.0e-3
        
      
        
        # print('bin masses at injection')
        for ii in range(self._nbins_aero):
            
            drop_r0_est = (self._drop_br0[ii] + self._drop_br0[ii+1])/2.0
            drop_m0 = rhow_cgs*4.0/3.0*np.pi*drop_r0_est**3 # assuming pure water
            qaero_tend = self._ScalarState.get_tend('pmass_aero_'+str(ii))
            qtot_tend = self._ScalarState.get_tend('pmass_total_'+str(ii))
            
            i = loc_indices_x
            j = loc_indices_y
            k = loc_indices_z
            # print('Before setting',ii,qaero[i,j,k],qtot[i,j,k])
            
            qaero_tend[i,j,k] = injection_rate /grid_cell_mass * self._concfrac_init[ii] * (self._pmass0[ii] * 1.0e-3) # (# particle /kg air) * single particle mass in kg
            qtot_tend[i,j,k] = injection_rate /grid_cell_mass * self._concfrac_init[ii] *   (drop_m0 * 1.0e-3)

            # str_aero +=str(qaero[i,j,k]) + ' , '
            # str_tot +=str(qtot[i,j,k]) + ' , '
        # print('thermo conditions at injection', T[i,j,k], qv[i,j,k],self._Ref.rho0[loc_indices_z],self._Ref.p0[loc_indices_z])
        # print('aero masses', str_aero)
        # print('total masses', str_tot)
        
        
            # print('After setting',ii,qaero[i,j,k],qtot[i,j,k])

        return
    def boundary_outflow(self):
        
        n_halo = self._Grid.n_halo

        x_local = self._Grid.x_local
        x_global = self._Grid.x_global

        y_local = self._Grid.y_local
        y_global = self._Grid.y_global

        for ii in range(self._nbins_aero):
            qaero = self._ScalarState.get_field('pmass_aero_'+str(ii))
            qtot = self._ScalarState.get_field('pmass_total_'+str(ii))


            if np.amin(x_local) == np.amin(x_global):
                qaero[: n_halo[0], :, :] = 0
                qtot[: n_halo[0], :, :] = 0

            if np.max(x_local) == np.amax(x_global):
                qaero[-n_halo[0] :, :, :] = 0
                qtot[-n_halo[0] :, :, :] = 0

            if np.amin(y_local) == np.amin(y_global):
                qaero[:, : n_halo[1], :] = 0
                qtot[:, : n_halo[1], :] = 0

            if np.max(y_local) == np.amax(y_global):
                qaero[:, -n_halo[1] :, :] = 0
                qtot[:, -n_halo[1] :, :] = 0
        return





def factory(
    namelist,
    Timers,
    Grid,
    Ref,
    ScalarState,
    VelocityState,
    DiagnosticState,
    TimeSteppingController,
):

    try:
        scheme = namelist["aerosolbin"]["scheme"]
    except:
        scheme = "binmodel"

    if scheme == "base":
        return AerosolBinBase(
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
    else:
        return AerosolBinModel(
            namelist,
            Timers,
            Grid,
            Ref,
            ScalarState,
            VelocityState,
            DiagnosticState,
            TimeSteppingController,
        )
   
