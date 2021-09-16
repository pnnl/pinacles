module parcel_model_wrapper
    use cpmMD
    use constant
    use aerospecMD

    use chemMD
  
    use iso_c_binding, only: c_double, c_int, C_CHAR, C_NULL_CHAR, c_ptr, c_f_pointer
contains
    

! inputs from the LES side which are needed:
! temperature, pressure, water vapor mr
! concentration per bin
! total mass per bin (aerosol + liquid)
! aerosol mass per bin (not sure I need this)
subroutine c_parcel_model_init(ichem, nbins)bind(C)
! subroutine c_parcel_model_init(ichem, nbins, na0, pmass_total_aero_v)bind(C)
    implicit none
    integer(c_int), value, intent(in) :: ichem, nbins
    ! real(c_double), dimension(nbins), intent(out) :: pmass_total_aero_v
    ! real(c_double), intent(in) :: na0

    if(nbins .NE. 20)then
        print*, "nbins inconsistent with aerobins defined in parcel mode code"
    end if
    call chem(ichem)
    call aerospec_setbins()
    ! Here we are just setting pmass in a 1-D array in bin space
    ! Assign to 3d space on LES side
    ! call aerospec_setdist(na0,pmass_total_aero_v)

end subroutine c_parcel_model_init

! Pass in variables in m-kg-s units and convert to cm-g-s for the parcel model here
subroutine c_parcel_model_main(npts, nbins, dt_les, dt_cpm, temp, press,&
     & qv, ql, density,pmass_total_aero_m, pmass_total_m)bind(C)
    implicit none

    real(c_double), value, intent(in) :: dt_les, dt_cpm
    integer(c_int), value, intent(in):: npts,nbins
    real(c_double), dimension(npts), intent(inout) :: temp, press, qv, ql, density ! fluid properties
    real(c_double), dimension(npts,nbins), intent(inout) :: pmass_total_aero_m, pmass_total_m
     
    integer(c_int)  :: ipt, print_flag
    real(c_double) :: press_cgs, density_cgs
    real(c_double) :: temp_in



    ! iaero should be zero as the aerosol distribution is already initialized
    ! Then the aerospec subroutine will just give the aerosol bin sizes

    
    do ipt = 1, npts
        print_flag = ipt
        density_cgs = density(ipt) * 1.0e-3
        press_cgs = press(ipt) * 10.0
        temp_in = temp(ipt)

       
    
        call set_thermo_state(temp(ipt),press_cgs,qv(ipt),density_cgs)

        call set_particle_state(pmass_total_m(ipt,:),pmass_total_aero_m(ipt,:))
    
        call cpm(dt_les,dt_cpm,pmass_total_m(ipt,:),temp(ipt),qv(ipt),ql(ipt),print_flag) 

    
        
    end do

    end subroutine

end module parcel_model_wrapper


