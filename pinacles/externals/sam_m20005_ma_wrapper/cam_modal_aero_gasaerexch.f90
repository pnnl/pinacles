module cam_modal_aero_gasaerexch

contains
!  MW: Small subset of modal_aero_gasaerexch.F90
!----------------------------------------------------------------------
!----------------------------------------------------------------------
subroutine gas_aer_uptkrates( num_a,          t,          p,       &
                              dgn_wet,  sigmag,      uptkrate    )

!
!                         /
!   computes   uptkrate = | dx  dN/dx  gas_conden_rate(Dp(x))
!                         /
!   using Gauss-Hermite quadrature of order nghq=2
!
!       Dp = particle diameter (cm)
!       x = ln(Dp)
!       dN/dx = log-normal particle number density distribution
!       gas_conden_rate(Dp) = 2 * pi * gasdiffus * Dp * F(Kn,ac) integer, parameter :: ntot_amode = 2    
! End-MW
!           F(Kn,ac) = Fuchs-Sutugin correction factor
!           Kn = Knudsen number
!           ac = accomodation coefficient
!


implicit none

! MW define cesm's r8 type from shr_kind_mod.F90
   integer, parameter :: r8 = selected_real_kind(12)
! End-MW
   real(r8), intent(in) :: num_a   ! Aerosol number concentration (#/m^3)
   real(r8), intent(in) :: t             ! Temperature in Kelvin
   real(r8), intent(in) :: p          ! Air pressure in Pa
   real(r8), intent(in) :: dgn_wet ! aerosol wet diameter in meters
   real(r8), intent(in) :: sigmag
   real(r8), intent(out) :: uptkrate  
                            ! gas-to-aerosol mass transfer rates (1/s)


! local 
   integer, parameter :: nghq = 2
   integer :: iq

   ! Can use sqrt here once Lahey is gone.
   real(r8), parameter :: tworootpi = 3.5449077_r8
   real(r8), parameter :: root2 = 1.4142135_r8
   real(r8), parameter :: beta = 2.0_r8

   real(r8) :: const
   real(r8) :: dp
   real(r8) :: gasdiffus, gasspeed
   real(r8) :: freepathx2, fuchs_sutugin
   real(r8) :: knudsen
   real(r8) :: lndp, lndpgn, lnsg
   real(r8) :: sumghq
   real(r8), save :: xghq(nghq), wghq(nghq) ! quadrature abscissae and weights

   data xghq / 0.70710678_r8, -0.70710678_r8 /
   data wghq / 0.88622693_r8,  0.88622693_r8 /
!   gasdiffus = h2so4 gas diffusivity from mosaic code (m^2/s)
!               (pmid must be Pa)
      gasdiffus = 0.557e-4_r8 * (t**1.75_r8) / p
!   gasspeed = h2so4 gas mean molecular speed from mosaic code (m/s)
      gasspeed  = 1.470e1_r8 * sqrt(t)
!   freepathx2 = 2 * (h2so4 mean free path)  (m)
      freepathx2 = 6.0_r8*gasdiffus/gasspeed

      lnsg   = log( sigmag )
      lndpgn = log( dgn_wet )   ! (m)
      const  = tworootpi * num_a * exp(beta*lndpgn + 0.5_r8*(beta*lnsg)**2)
         
!   sum over gauss-hermite quadrature points
      sumghq = 0.0_r8
      do iq = 1, nghq
         lndp = lndpgn + beta*lnsg**2 + root2*lnsg*xghq(iq)
         dp = exp(lndp)

!   knudsen number
         knudsen = freepathx2/dp
!  Changed by Manish Shrivastava on 7/17/2013 to use accom=1; because we do not know better
!   following assumes accomodation coefficient = ac = 1. instead 0.65 ! answer change needs to be tested
!   (Adams & Seinfeld, 2002, JGR, and references therein)
!           fuchs_sutugin = (0.75*ac*(1. + knudsen)) /
!                           (knudsen*(1.0 + knudsen + 0.283*ac) + 0.75*ac)
         fuchs_sutugin = (0.4875_r8*(1._r8 + knudsen)) /   &
                         (knudsen*(1.184_r8 + knudsen) + 0.4875_r8)
         sumghq = sumghq + wghq(iq)*dp*fuchs_sutugin/(dp**beta)
      end do
      uptkrate = const * gasdiffus * sumghq    



   return
   end subroutine gas_aer_uptkrates

end module cam_modal_aero_gasaerexch
