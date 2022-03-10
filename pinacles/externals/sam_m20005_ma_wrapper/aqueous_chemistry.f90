module aqueous_chemistry

use micro_params, only: rho_water, MW_H2O2, MW_SO2, MW_air, Rdry, cH2O2


contains

subroutine aq_chemistry_tendencies(SO2, qcl, T, Tbar, pbar, rhobar, pH, &
  nx, ny, nzm, dtn, SO2_aq_ox_tend)


   ! Compute aqueous oxidation of S(IV) to produce S(VI) using peroxide oxidation only
   ! Follows GOES-Chem module  
   ! Compute mass loss of SO2 in timestep dtn

   implicit none

   real, dimension(:,:,:), intent(in) :: SO2, qcl
   real, dimension(1:nx, 1:ny, 1:nzm), intent(in) :: T
   real, dimension(1:nzm), intent(in) :: Tbar,pbar,rhobar
   real, intent(in) :: pH
   
   integer, intent(in) :: nx, ny, nzm
   real, intent(in) ::  dtn   
   real, dimension(1:nx, 1:ny) :: aq_ox_rate
   real, dimension(1:nx, 1:ny, 1:nzm), intent(out) :: SO2_aq_ox_tend
   

   real, dimension(1:nx, 1:ny, 1:nzm) :: Kaq_ox  ! oxidation reaction rate moles/L sec
   integer:: i,j,k
   real :: LWC   ! m^3/m^3 = qc * rhobar/rho_water
   real :: patm = 101325  ! bar
   real :: Hplus
   real :: Kh1, Ks1, Ks2, kh2o2 ! reaction rates
   real :: HCH2O2, FHCH2O2, HCSO2, FHCSO2 ! Henry's (HC) and effective Henry (FHC) coefficients
   real :: XSO2g, XH2O2g ! :: atmospheric fractions of gases

   Hplus = 10**(-pH)  ! mol/L
!   do k = 1,nzm
!      print*, k, 'in aqueous:', qcl(1,1,k)
!   end do

   ! compute rate constant for reaction
   do k = 1,nzm
      do j = 1,ny
        do i = 1,nx
           Kh1 = 2.20e-12 * exp(-12.52 * ( 298.15 / T(i,j,k) - 1. ) ) ! mol/L                                                        

           Ks1    = 1.30e-2 * exp( 6.75 * ( 298.15 / T(i,j,k) - 1. ) )
           Ks2    = 6.31e-8 * exp( 5.05 * ( 298.15 / T(i,j,k) - 1. ) )

           kh2o2 = 6.31e+14 * exp( -4.76e+3 / T(i,j,k) )  

           ! Henry's constant (HC) [mol/l-atm] and Effective Henry's constant (FHC) for H2O2
           ! [Jacobson,1999]
           HCH2O2  = 7.45e4 * exp(22.21 * (298.15 / T(i,j,k) - 1.) )
           FHCH2O2 = HCH2O2 * (1. + (Kh1 / Hplus))

           ! Same for SO2
           HCSO2  = 1.22 * exp(10.55 * ( 298.15/ T(i,j,k) - 1.) )    
           FHCSO2 = HCSO2 * (1. + (Ks1/Hplus) +  (Ks1*Ks2 / (Hplus*Hplus)))

           LWC = qcl(i,j,k) * rhobar(k)/rho_water
           XSO2g  = 1. / ( 1. + ( FHCSO2 * Rdry * Tbar(k) * LWC ) )
           XH2O2g  = 1./ ( 1. + ( FHCH2O2 * Rdry * Tbar(k) * LWC ) )

           Kaq_ox(i,j,k) = kh2o2 * Ks1 * FHCH2O2 * HCSO2 * XH2O2g * XSO2g * (pbar(k)/patm)**2    

        end do
      end do
   end do

   ! apply rate constant to compute molar reaction rate (mol/L),  SO2 and H2O2 converted from input kg/kg to v/v (i.e. mol fraction)
   SO2_aq_ox_tend(:,:,:) = - Kaq_ox(:,:,:) * MW_air**2 * SO2(1:nx,1:ny,1:nzm) * cH2O2 * dtn/ (MW_SO2 * MW_H2O2)  ! positive rates become SO2 loss   
   SO2_aq_ox_tend(:,:,:) = SO2_aq_ox_tend(:,:,:) * 1000 * qcl(1:nx,1:ny,1:nzm)/rho_water * (MW_SO2/1000.) !   (molSO2/Lwat) *  (Lwat/kg_air)  * kg SO2/ molSO2 = kgSO2/kgair 
  

   ! apply a limiter t keep SO2 from becoming negative

   where (SO2 + SO2_aq_ox_tend < 0.)
      SO2_aq_ox_tend = -SO2
   end where
      
   end subroutine aq_chemistry_tendencies
end module aqueous_chemistry
