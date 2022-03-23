module gas_uptake

use micro_params, only: rho_aerosol, rho_water, hygro ! (kg/m^3, kg/m^3 liquid water, dimensionless)
use cam_cldaero, only: cldaero_uptakerate
use cam_modal_aero_gasaerexch, only: gas_aer_uptkrates
use mirage_wateruptake, only: modal_aero_kohler

contains

subroutine gas_uptake_tendencies(H2SO4, Nc, Nac, Nak, Qac, Qak, &
     sigmag_acc, sigmag_ait, qcl, T, nx, ny, nzm, & 
     rhobar, Tbar, pbar, relhum, dtn, &
     H2SO4_cld_uptake_tend, H2SO4_acc_uptake_tend, H2SO4_ait_uptake_tend)

   implicit none

   real, dimension(:,:,:), intent(in) :: H2SO4, Nc, Nac, Nak, Qac, Qak, qcl
   real, dimension(1:nx, 1:ny, 1:nzm), intent(in) :: T, relhum
   real, dimension(1:nzm), intent(in) :: rhobar, Tbar, pbar
   real, intent(in) :: sigmag_acc, sigmag_ait
   
   integer, intent(in) :: nx, ny, nzm
   real, intent(in) ::  dtn   
   real, dimension(1:nx, 1:ny) :: cld_uptake_rate, acc_uptake_rate, ait_uptake_rate
   real, dimension(1:nx, 1:ny) :: frac_change, s  !  limiter variables to keep H2SO4 positive
   real, dimension(1:nx, 1:ny, 1:nzm), intent(out) :: &
        H2SO4_cld_uptake_tend, H2SO4_acc_uptake_tend, H2SO4_ait_uptake_tend 
   

  
   real :: cld_uptk_rt, acc_uptk_rt, ait_uptk_rt, tot_aero_uptk   
   integer:: i,j,k
   real :: rdry(1), rwet(1), hygro_arr(1), relhum_arr(1)
   real :: pi = 3.141592654
   real :: qcl_vmr  ! qc volume mixing ratio

   real :: qcl_cutoff = 1.e-6 ! kg/kg
   real :: qa_cutoff = 1.e-20 !kg/kg
   real :: del = 1.e-5  ! #/m^3 roundoff noise to avoid divide by zero
   integer :: npoints = 1
   
   hygro_arr(1) = hygro

   do k = 1,nzm
      cld_uptake_rate(:,:) = 0.
      acc_uptake_rate(:,:) = 0.
      ait_uptake_rate(:,:) = 0.
      ! At each point compute cloud uptake, or aerosol uptake
      do j = 1,ny
         do i = 1,nx
 
            if(qcl(i,j,k).gt. qcl_cutoff) then
               ! do bulk cloud uptake
               qcl_vmr = qcl(i,j,k) * rhobar(k)/rho_water
               call cldaero_uptakerate(&
                    qcl_vmr,  Nc(i,j,k)/1.e6 ,T(i,j,k), pbar(k),cld_uptk_rt)
                       ! convert number to #/cm^3 for this routine
               cld_uptake_rate(i,j) = cld_uptk_rt
            else
               ! aerosol uptake
               acc_uptk_rt = 0.
               ait_uptk_rt = 0.
               relhum_arr(1) = relhum(i,j,k)
               if (Qac(i,j,k).gt.qa_cutoff) then
                  rdry = (0.75 * Qac(i,j,k) * rhobar(k) * &
                       exp(-4.5 * log(sigmag_acc))**2./ &
                       (pi * rho_aerosol * (Nac(i,j,k)+del)))**(1./3.)
                  ! rwet is the only output of the following call
                  
                  call modal_aero_kohler(rdry, hygro_arr, relhum_arr, rwet(:), npoints)
                  
                  call gas_aer_uptkrates(Nac(i,j,k), Tbar(k), pbar(k), &
                                          rwet(1), sigmag_acc, acc_uptk_rt) 
               end if
               if (Qak(i,j,k).gt.qa_cutoff) then
                  rdry = (0.75 * Qak(i,j,k) * rhobar(k) * &
                       exp(-4.5 * log(sigmag_ait))**2./ &
                       (pi*rho_aerosol * (Nak(i,j,k)+del)))**(1./3.)
                  call modal_aero_kohler(rdry, hygro_arr, relhum_arr, rwet(:), npoints)
                  
                  call gas_aer_uptkrates(Nak(i,j,k), Tbar(k), pbar(k), &
                                          rwet(1), sigmag_ait, ait_uptk_rt) 
               end if
               
               tot_aero_uptk = acc_uptk_rt + ait_uptk_rt 
               if (tot_aero_uptk.gt.1.0) then
                  acc_uptk_rt = acc_uptk_rt/tot_aero_uptk
                  ait_uptk_rt = ait_uptk_rt/tot_aero_uptk
               end if

               acc_uptake_rate(i,j) = acc_uptk_rt
               ait_uptake_rate(i,j) = ait_uptk_rt
            end if   

          end do
       end do

       ! Keep H2SO4 from becoming negative 
       frac_change = (cld_uptake_rate(:,:) + acc_uptake_rate(:,:) + ait_uptake_rate(:,:)) * dtn
       where(frac_change > 0.5)
          s = 1./frac_change
       elsewhere
          s = 1.
       end where
       
       ! save results into tendency arrays
       H2SO4_cld_uptake_tend(:,:,k) = -H2SO4(1:nx,1:ny,k) * cld_uptake_rate(:,:) * dtn * s
       H2SO4_acc_uptake_tend(:,:,k) = -H2SO4(1:nx,1:ny,k) * acc_uptake_rate(:,:) * dtn * s
       H2SO4_ait_uptake_tend(:,:,k) = -H2SO4(1:nx,1:ny,k) * ait_uptake_rate(:,:) * dtn * s

     end do   

   end subroutine gas_uptake_tendencies
end module gas_uptake
