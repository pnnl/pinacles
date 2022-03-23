MODULE hoppel_transfer



contains  

REAL FUNCTION modal_diameter(N, M, Sg, rho_aerosol)
  implicit none
  ! returns the modal diameter
  real :: N, M, Sg, rho_aerosol
  real ::  pi = 3.1415926535
  modal_diameter = ((1/rho_aerosol) * (M/N) * (6/pi) * exp(-4.5 * Sg**2))**(1./3.)

END FUNCTION modal_diameter  

REAL FUNCTION mass_mixing_ratio(N,D,sg, rho_aero)
  implicit none
  real :: N,D,sg,rho_aero
  real :: pi =  3.1415926535
  mass_mixing_ratio = (pi/6)*rho_aero * (D**3) * exp(4.5*sg**2) * N

END FUNCTION mass_mixing_ratio  

REAL FUNCTION mass_fraction(nfrac, sigma)
  !    Compute mass fraction of lognormal distribution smaller than some size
  !   given a number fraction smaller than the same size
  implicit none
  double precision, external :: derfi
  double precision :: dnfrac 
  real :: xx 
  real :: nfrac, sigma
  real :: sqrt2 = 2.**0.5

  dnfrac = nfrac ! derfi needs double precision input
  if (nfrac <= 0.) then
     mass_fraction = 0.
  else if (nfrac < 0.999999) then
     xx = sqrt2 * derfi(2. * (dnfrac - 0.5))
     mass_fraction = 0.5 + 0.5 * erf((-3.*LOG(sigma)+ xx)/sqrt2)
  else
     mass_fraction = 1.
  end if   

END FUNCTION mass_fraction

SUBROUTINE hoppel_aitken_accum_transfer(N0, N1, M0, M1, sg0, sg1, Dc0, Dc1, &
     rho_aero0, rho_aero1, Ntrans, Mtrans)
  ! Compute number and mass of transfer from aitken to accumulation
  ! if aitken number at critical radius is larger than accumulation number at
  !   critical radius
  ! All particles transfered from Aitken have diameter Dc0

  ! Does not check that inputs are nonzero, N0 and N1. (diameter is
  ! computed and n in denomintor).

  implicit none

  
  real, intent(in) :: N0, N1  !number in /kg   0 is 'Aitken' 1 is 'accumulation'
  real, intent(in) :: M0, M1  ! mass in kg/kg
  real, intent(in) :: sg0, sg1  !natural log of sigma shape parameter
  real, intent(in) :: Dc0, Dc1  !critical diameter of activation (from ARG)
  real, intent(in) :: rho_aero0 ! rho_aero1  aerosol density
  real, intent(in) :: rho_aero1
  real, intent(out) :: Ntrans
  real, intent(out) :: Mtrans  ! in /kg and kg/kg

  real :: Dg0, Dg1
  real :: N0n, N1n, M0n, M1n, NT, MT, kT 
  real :: T, Tnew  ! Amoutn transferred scaled by N0
  real :: eta0, eta1 ! number distribution functions of modes
  real :: deta0_DT, deta1_DT
  
  real ::  pi = 3.1415926535
  integer, parameter :: maxiter = 15
  real, parameter :: deltaN_conv = 50. ! /m^3

  integer ::  count = 0
  
  ! N0, MO, N1, M1 are unchanged -> N0 = N00 in writeup

  
  
  N0n = N0
  N1n = N1
  M0n = M0
  M1n = M1
  
  kT = (pi/6.) * rho_aero0*Dc0**3 
  NT = 0.
  MT = 0.
  
  T = 0.
  count = 0
  DO
     count = count+1
     Dg0 = modal_diameter(N0n, M0n, sg0, rho_aero0)
     Dg1 = modal_diameter(N1n, M1n, sg1, rho_aero1)
!     print*, 'Dg0, Dg1', Dg0*1.e6, Dg1*1.e6
     eta0 = N0n*exp(-0.5 * (log(Dg0/Dc0)/sg0)**2)/(SQRT(2.*pi)*sg0)
     eta1 = N1n*exp(-0.5 * (log(Dg1/Dc1)/sg1)**2)/(SQRT(2.*pi)*sg1)
!     print*, 'eta0, eta1', eta0/1.e6, eta1/1.e6
     
     IF ((eta0-eta1).LE.0.AND.count.EQ.1) THEN
        EXIT
     END IF   
     IF (abs(eta0-eta1).LT.deltaN_conv) THEN
        EXIT
     END IF

     ! N0n nad N1n should not be identically zero
     deta0_DT = -eta0 * (N0/N0n) * (1. - 1./3. * sg0**(-2) * log(Dg0/Dc0)*(exp(-4.5 * sg0**2)*(Dc0/Dg0)**3 - 1.))
     deta1_DT = eta1 * (N0/N1n) * (1. - 1./3. * sg1**(-2) * log(Dg1/Dc1)*(exp(-4.5 * sg1**2)*(Dc1/Dg1)**3 - 1.))   

     Tnew = T - (eta0-eta1)/(deta0_dT - deta1_dT)

     IF (Tnew < 0) THEN
        print*, 'Aitken transfer negative. Count= ', count
        print*, 'N0,N1,M0,M1,Dc0,Dc1=', N0, N1, M0, M1, Dc0, Dc1
        NT = 0.
        MT = 0.
        EXIT
     END IF
     
     NT = Tnew * N0

     !put limiters here
     IF (NT.GE.N0) THEN
        DO WHILE (NT.GE.N0) 
           NT = NT/2.
        END DO
        Tnew = NT/N0
     END IF
     MT = NT*kT

     IF (MT.GE.M0) THEN 
        DO WHILE (MT.GE.M0)
           MT = MT/2
        END DO
        NT = MT/kT
    END IF

     IF (maxiter.EQ.count) THEN
        NT = 0.
        MT = 0.
        print*, 'Aitken Transfer Convergence Failure'
        print*, 'N0,N1,M0,M1,Dc0,Dc1=', N0, N1, M0, M1, Dc0, Dc1
        EXIT
     END IF


     
     N0n = N0 - NT
     N1n = N1 + NT
     M0n = M0 - MT 
     M1n = M1 + MT

     T = Tnew
    
!     print*,  '      N0          M0              N1           M1           F         T        NT'
!     print*, N0n/1.e6, M0n*1.e9, N1n/1.e6, M1n*1.e9, (eta0-eta1)/1.e6, Tnew, NT/1.e6
     
  END DO   

  Ntrans = NT
  Mtrans = MT

  
END SUBROUTINE hoppel_aitken_accum_transfer
  
END MODULE hoppel_transfer
