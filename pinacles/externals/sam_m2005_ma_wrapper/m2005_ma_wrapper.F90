module m2005_ma_wrapper
    use module_mp_m2005_ma
    use iso_c_binding, only: c_double, c_int, C_CHAR, C_NULL_CHAR, c_ptr, c_f_pointer
contains

subroutine c_m2005_ma_init(dir_path_c, dir_path_len, nCat, &
                     aero_inv_rm1, aero_sig1, aero_nanew1, &
                     aero_inv_rm2, aero_sig2, aero_nanew2, &
                     nccnst_in) bind(C)


    character(kind=c_char),  intent(in)  :: dir_path_c(100)
    character(256) :: dir_path = ''
    integer, value, intent(in) :: dir_path_len
    integer, value, intent(in) :: nCat
    real, value, intent(in) :: aero_inv_rm1, aero_sig1, aero_nanew1
    real, value, intent(in) :: aero_inv_rm2, aero_sig2, aero_nanew2
    real, value, intent(in) :: nccnst_in

    integer :: stat
    character(8) :: model = 'PINACLES'
    logical :: abort_on_error = .True.


    ! TODO need to fill with values from the character array
    do i = 1, dir_path_len
        dir_path(i:i) = dir_path_c(i)
    end do

    ! Call the fortran subroutine
    call m2005_ma_init(dir_path , nCat, model, stat, abort_on_error, &
                aero_inv_rm1, aero_sig1, aero_nanew1, &
                aero_inv_rm2, aero_sig2, aero_nanew2, nccnst_in)

end subroutine

subroutine c_m2005_ma_main(nx, ny, nz, nmicrofields, nx_gl, ny_gl, my_rank, nsubdomains_x, nsubdomains_y, &
    th_3d, s_3d, w, microfield_4d, n_diag_3d, diag_3d, &
    z, p0, rho0, tabs0, zi, rhow, dx, dz, &
    nrainy, nrmn, ncmn, total_water_prec, &
    tlat, tlatqi, precflux, qpfall, &
    fluxbq, fluxtq, u10arr, precsfc, prec_xy, &
    dt, time, itimestep, LCOND, LSUB, CPD, RGAS, RV, G) bind(c)

! Index bounds
integer(c_int), value, intent(in):: nx, ny, nz, nmicrofields, nx_gl, ny_gl, my_rank, nsubdomains_x, nsubdomains_y, n_diag_3d

!Input output arrays
real(c_double), dimension(nx, ny, nz), intent(inout):: th_3d, s_3d
real(c_double), dimension(nx, ny, nz), intent(in) :: w

real(c_double), dimension(nx, ny), intent(inout) :: precsfc, prec_xy
real(c_double), dimension(nx, ny), intent(in) :: fluxbq, fluxtq, u10arr

real(c_double), dimension(nz+1), intent(inout) :: tlat, tlatqi, precflux, qpfall
real(c_double), dimension(nz), intent(in) :: z, p0, rho0, tabs0
real(c_double), dimension(nz+1), intent(in) :: zi, rhow

real(c_double), intent(inout) :: nrainy, nrmn, ncmn, total_water_prec
real(c_double), value, intent(in) :: dx, dz, dt, time, LCOND, LSUB, CPD, RGAS, RV, G
integer(c_int), value, intent(in) :: itimestep

real(c_double), dimension(nx, ny, nz, n_diag_3d), intent(inout) :: diag_3d

real(c_double), dimension(nx, ny, nz, nmicrofields), intent(inout) :: microfield_4d

call mp_m2005_ma_wrapper_sam(nx, ny, nz, nmicrofields, nx_gl, ny_gl, my_rank, nsubdomains_x, nsubdomains_y, &
                            th_3d, s_3d, w, microfield_4d, n_diag_3d, diag_3d,      &
                            z, p0, rho0, tabs0, zi, rhow, dx, dz,                   &
                            nrainy, nrmn, ncmn, total_water_prec,                   &
                            tlat, tlatqi, precflux, qpfall,                         &
                            fluxbq, fluxtq, u10arr, precsfc, prec_xy,               &
                            dt, time, itimestep, LCOND, LSUB, CPD, RGAS, RV, G)

end subroutine

end module m2005_ma_wrapper