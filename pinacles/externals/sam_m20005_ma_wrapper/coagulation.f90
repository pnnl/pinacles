module coagulation

use cam_modal_aero_coag, only: getcoags_wrapper_f

! Compute coagulation of non-activated aerosols
! Call CAM CMAQ code

contains

subroutine coagulation_tendencies(Nacc, Nait, Qacc, Qait, sigma_acc, sigma_ait, &
  rho_acc, rho_ait, rhobar, T, pbar, nx, ny, nzm, kmax, dtn, &
  Nait_self_tend, Nacc_self_tend, Nait2acc_transfer_tend, Mait2acc_transfer_tend)

  implicit none
  
  real, dimension(:,:,:), intent(in) :: Nacc, Nait, Qacc, Qait, T
  integer, intent(in) :: nx, ny, nzm, kmax
  real, dimension(1:nzm), intent(in) :: rhobar, pbar
  real, intent(in) :: sigma_acc, sigma_ait
  real, intent(in) :: rho_acc, rho_ait
  real, intent(in) ::  dtn    
  real, dimension(1:nx, 1:ny, 1:nzm), intent(out) :: &
       Nait_self_tend, Nacc_self_tend, &
       Nait2acc_transfer_tend, Mait2acc_transfer_tend

  real, dimension(1:nx, 1:ny, 1:nzm) :: Dacc, Dait, Qacc_temp, Qait_temp
  real :: betaij0, betaij2i, betaij2j, betaij3
  real :: betaii0, betaii2, betajj0, betajj2
  real :: log_sig_acc, log_sig_ait
  real :: pi = 3.141592654
  real :: eps = 1.
  integer :: i,j,k

  log_sig_acc = log(sigma_acc)
  log_sig_ait = log(sigma_ait)

  Qacc_temp = Qacc
  Qait_temp = Qait

  
  where(Qacc_temp < 0)
     Qacc_temp = 0
  end where
  where(Qait_temp < 0)
     Qait_temp = 0
  end where

  Nait_self_tend = 0.
  Nacc_self_tend = 0.
  Nait2acc_transfer_tend = 0.
  Mait2acc_transfer_tend = 0.
 
! need to put in eps for Nac, and then zero out D for these points
  ! Aerosol diameter in m
  Dacc = ((6./pi) * (1./rho_acc) * exp(-4.5 * log_sig_acc**2) * (Qacc_temp/(Nacc+eps)))**(0.333333333)  
  Dait = ((6./pi) * (1./rho_ait) * exp(-4.5 * log_sig_ait**2) * (Qait_temp/(Nait+eps)))**(0.333333333)
  where(Dacc < 1.e-10)
      Dacc = 1.e-10
  elsewhere(Dacc > 1.e-5)
      Dacc = 1.e-5
  end where
  where(Dait < 1.e-10)
      Dait = 1.e-10
  elsewhere(Dait > 1.e-5)
      Dait = 1.e-5
  end where  

! loop over the indices from surface to kmax - get the coag rates
  do k = 1,kmax
     do j = 1,ny
        do i = 1,nx
        
           call getcoags_wrapper_f(T(i,j,k), pbar(k), Dait(i,j,k), Dacc(i,j,k), &
                 sigma_ait, sigma_acc, log_sig_ait, log_sig_acc, rho_ait, rho_acc, &
            betaij0, betaij2i, betaij2j, betaij3, betaii0, betaii2, betajj0, betajj2)   
           
           ! number tendencies #/kg.s
           ! beta coeffs in m^3/s
           Nait_self_tend(i,j,k) = - betaii0 * Nait(i,j,k)**2 * rhobar(k)
           Nacc_self_tend(i,j,k) = - betajj0 * Nacc(i,j,k)**2 * rhobar(k)

           Nait2acc_transfer_tend(i,j,k) = betaij0 * Nait(i,j,k)*Nacc(i,j,k) * rhobar(k)
           ! mass tendency kg/kg.s
           Mait2acc_transfer_tend(i,j,k) = betaij3 * Nacc(i,j,k) *Qait(i,j,k) * rhobar(k) 
        end do
      end do
   end do 
!   multiply by timestep

    Nait_self_tend = Nait_self_tend * dtn
    Nacc_self_tend = Nacc_self_tend * dtn
    Nait2acc_transfer_tend = Nait2acc_transfer_tend * dtn
    Mait2acc_transfer_tend = Mait2acc_transfer_tend * dtn

end subroutine coagulation_tendencies

end module coagulation
