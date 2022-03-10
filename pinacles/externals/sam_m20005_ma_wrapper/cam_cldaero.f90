module cam_cldaero

contains

!----------------------------------------------------------------------------------
! low level utility module for cloud aerosols
!
! Created by Francis Vitt
!----------------------------------------------------------------------------------

! MW lightly edited to extract subroutine 

!----------------------------------------------------------------------------------

subroutine cldaero_uptakerate( xl, num_cd, tfld,  press , uptkrate )
   


    integer, parameter :: r8 = selected_real_kind(12)
!   xl is volume fraction of liquid water = ql * (rho/rhol)
    real(r8), intent(in) :: xl, num_cd, tfld,  press

    real(r8) :: uptkrate
    real(r8) :: pi = 3.141592654
    real(r8) :: &
         rad_cd, radxnum_cd, &
         gasdiffus, gasspeed, knudsen, &
         fuchs_sutugin, volx34pi_cd

!-----------------------------------------------------------------------
! compute uptake of h2so4 and msa to cloud water
!
!
! first-order uptake rate is
! 4*pi*(drop radius)*(drop number conc)
! *(gas diffusivity)*(fuchs sutugin correction)

! num_cd = (drop number conc in 1/cm^3)
!        num_cd = 1.0e-3_r8*cldnum*cfact/cldfrc
!        num_cd = max( num_cd, 0.0_r8 )

! rad_cd = (drop radius in cm), computed from liquid water and drop number,
! then bounded by 0.5 and 50.0 micrometers
! radxnum_cd = (drop radius)*(drop number conc)
! volx34pi_cd = (3/4*pi) * (liquid water volume in cm^3/cm^3)

        volx34pi_cd = xl*0.75_r8/pi

! following holds because volx34pi_cd = num_cd*(rad_cd**3)
        radxnum_cd = (volx34pi_cd*num_cd*num_cd)**0.3333333_r8

    
! apply bounds to rad_cd to avoid the occasional unphysical value
        if (radxnum_cd .le. volx34pi_cd*4.0e4_r8) then
            radxnum_cd = volx34pi_cd*4.0e4_r8
            rad_cd = 50.0e-4_r8
        else if (radxnum_cd .ge. volx34pi_cd*4.0e8_r8) then
            radxnum_cd = volx34pi_cd*4.0e8_r8
            rad_cd = 0.5e-4_r8
        else
            rad_cd = radxnum_cd/num_cd
        end if

!        print*, rad_cd
! gasdiffus = h2so4 gas diffusivity from mosaic code (cm^2/s)
! (pmid must be Pa)
        gasdiffus = 0.557_r8 * (tfld**1.75_r8) / press

! gasspeed = h2so4 gas mean molecular speed from mosaic code (cm/s)
        gasspeed = 1.455e4_r8 * sqrt(tfld/98.0_r8)

! knudsen number
        knudsen = 3.0_r8*gasdiffus/(gasspeed*rad_cd)

! following assumes accomodation coefficient = 0.65
! (Adams & Seinfeld, 2002, JGR, and references therein)
! fuchs_sutugin = (0.75*accom*(1. + knudsen)) /
! (knudsen*(1.0 + knudsen + 0.283*accom) + 0.75*accom)
        fuchs_sutugin = (0.4875_r8*(1._r8 + knudsen)) / &
                        (knudsen*(1.184_r8 + knudsen) + 0.4875_r8)

! instantaneous uptake rate
        uptkrate = 12.56637_r8*radxnum_cd*gasdiffus*fuchs_sutugin



   end subroutine cldaero_uptakerate

end module cam_cldaero
