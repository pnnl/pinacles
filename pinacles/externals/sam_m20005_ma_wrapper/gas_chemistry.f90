module gas_chemistry

use micro_params, only: avgd, MW_H2SO4, MW_NO3, MW_SO2, MW_DMS, &
  MW_H2O2, MW_air, cOH, cNO3, dofixedH2O2, dofixedDMS

implicit none

contains

subroutine gas_chem_tendencies(SO2, DMS, H2SO4, rhobar, Tbar, &
     nx, ny, nzm, dtn, SO2_tend, DMS_tend, H2SO4_tend) 

   implicit none

   real, dimension(:,:,:), intent(in) :: SO2, DMS, H2SO4 ! kg/kg
   real, dimension(1:nzm), intent(in) :: rhobar
   integer, intent(in) :: nx, ny, nzm
   real, intent(in) ::  dtn   ! timestep (s)
   real, dimension(1:nx, 1:ny, 1:nzm), intent(out) :: SO2_tend, DMS_tend, &
        H2SO4_tend ! kg/kg

   real :: cNO3_k  ! molecules/cm^3
   real, dimension(1:nx,1:ny) :: SO2_k, DMS_k, H2SO4_k !molecules/cm^3
   real, dimension(1:nx,1:ny) :: s1, s2 ! limiter factors
   real, dimension(1:nx,1:ny) :: delDMS, delSO2 !molecules/cm^3
   real, dimension(1:nx,1:ny) :: SO2src, SO2sink  ! molecules/cm^3/s   
   real, dimension(1:nzm) :: Tbar !K
   real :: m ! air molecule concentration /cm^3
   integer :: i,j,k
   real :: j1, r1, r2, r3, r4, r5, r6  ! rate constants (cm^3/molecule)

   do k = 1,nzm
      m = rhobar(k) * avgd / (MW_air * 1000)
      cNO3_k = CNO3 * rhobar(k) * avgd /(MW_NO3 * 1000)
      ! compute the rate constants at this level
      ! rate constants in (cm^3/molecule)
      r2 = simple_rate(2.9e-12, -160., Tbar(k))
      r4 = simple_rate(9.6e-12, -234., Tbar(k))
      r6 = simple_rate(1.9e-13, -520., Tbar(k))
      r1 = 0.
      j1 = 0.
      r3 = rate_SO2_OH(Tbar(k), m)
      r5 = rate_DMS_OH(Tbar(k), m)
  
      ! compute the scaled concentrations at ths level (molecules/cm^3)
      SO2_k = SO2(1:nx,1:ny,k) * rhobar(k) * avgd / (MW_SO2 * 1000)
      DMS_k = DMS(1:nx,1:ny,k) * rhobar(k) * avgd / (MW_DMS * 1000)
      H2SO4_k = H2SO4(1:nx,1:ny,k) * rhobar(k) * avgd / (MW_H2SO4 * 1000)
      !H2O2_k = H2O2(1:nx,1:ny,k) * rhobar(k) * avgd / (MW_H2O2 * 1000)         
      
      ! limit the sinks of DMS and SO2 so the concentrations can never be negative
      delDMS = dtn*((-r4 - r5 )*cOH*DMS_k - r6*cNO3_k*DMS_k)
      where (DMS_k + delDMS < 0.)
          s1 = -DMS_k/delDMS
      elsewhere
          s1 = 1.0
      end where
      DMS_tend(:,:,k) = s1 * delDMS      


      SO2src = s1 * ( (r4 + r5/2.)* cOH*DMS_k + r6*cNO3_k*DMS_k)
      SO2sink = r3*cOH*SO2_k
      delSO2 = dtn*(SO2src - SO2sink)
      where (SO2_k + delSO2 < 0.)
          s2 = (SO2_k + SO2src)/SO2sink
      elsewhere
          s2 = 1.0
      end where
      SO2_tend(:,:,k) = dtn* (SO2src - s2*SO2sink)
    

      ! compute the tendencies at this level
!      SO2_tend(:,:,k) = (r4 + r5/2.)* cOH*DMS_k + r6*cNO3_k*DMS_k - r3*cOH*SO2_k
!      DMS_tend(:,:,k) = (-r4 - r5 )*cOH*DMS_k - r6*cNO3_k*DMS_k
      H2SO4_tend(:,:,k) = dtn * r3*cOH*SO2_k
      !H2O2_tend(:,:,k) = dtn * (r1 - j1*H2O2_k - r2*cOH*H2O2_k)
      ! need a limiter for H2O2 
          
      ! un-scale the tendencies and multiply by dt
      SO2_tend(:,:,k) = SO2_tend(:,:,k) * MW_SO2 * 1000 / (avgd * rhobar(k))
      DMS_tend(:,:,k) = DMS_tend(:,:,k) * MW_DMS * 1000 / (avgd * rhobar(k))
      H2SO4_tend(:,:,k) = H2SO4_tend(:,:,k) * MW_H2SO4 * 1000 / (avgd * rhobar(k))
      !H2O2_tend(:,:,k) = H2O2_tend(:,:,k) * MW_H2O2 * 1000 / (avgd * rhobar(k))
      
      !if (dofixedH2O2) then
      !   H2O2_tend = 0.
      !end if

      if (dofixedDMS) then
         DMS_tend = 0.
      end if
      ! avoid negative values TODO
       
      

   enddo

end subroutine gas_chem_tendencies

subroutine gas_chemistry_init(nrestart)


   implicit none    
   
   integer, intent(in) :: nrestart
 
   
   if (nrestart.eq.0) then
      ! run initialization code
      !   e.g assign profiles or constant values

      ! may need some namelist values (NO3, OH concentrations)
      ! will need temperature in some forefor some rates

   end if
!    Set up chemistry output variables
    
end subroutine gas_chemistry_init

subroutine gas_chemistry_flux()
! surface fluxes of at least DMS, possibly others
end subroutine gas_chemistry_flux

subroutine gas_chemistry_proc()
! convert all micro fields to appropriate units
! loop over all columns
!   column by column compute gas tendencies
! insure positivity
! update micro fields
! convert all new fields back to micro fields
! update statistics
end subroutine gas_chemistry_proc

subroutine gas_chemistry_hbuf_init()
! do nothing
end subroutine gas_chemistry_hbuf_init

subroutine gas_chemistry_statistics()
! do nothing 
end subroutine gas_chemistry_statistics

real function simple_rate(c1, c2, T)

    real :: c1, c2, T
    simple_rate = c1 * exp(c2/T)

end function simple_rate

real function rate_SO2_OH(T, m)
    ! Tin Kelvins
    ! m in /cm^3
  
    real :: T,m
    real :: ko,fc
    
    ! from mo_usrrxt.F90 in CAM MOZART code
    fc = 3.e-31 * (300./T)**3.3
    ko = fc*m/(1. + fc*m/1.5e-12) 
    rate_SO2_OH = ko*0.6**(1.0 + (log10(fc*m/1.5e-12))**2.0)**(-1.)

end function rate_SO2_OH

real function rate_DMS_OH(T,m)

    real :: T,m
    real :: efac, efac2, ko 
    efac = exp(7460./T)
    ko = 1.0 + 5.5e-31 * efac * m * 0.21
    efac2 = exp(7810./T)
    rate_DMS_OH = 1.7e-42 * efac2 * m * 0.21 / ko

end function rate_DMS_OH

end module gas_chemistry
