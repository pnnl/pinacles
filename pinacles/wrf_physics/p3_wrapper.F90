module p3_wrapper
    use module_mp_p3, only: p3_init, mp_p3_wrapper_wrf
    use iso_c_binding, only: c_double, c_int, C_CHAR, C_NULL_CHAR, c_ptr, c_f_pointer
contains

subroutine c_p3_init(dir_path_c, dir_path_len, nCat, &
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
    call p3_init(dir_path , nCat, model, stat, abort_on_error, &
                aero_inv_rm1, aero_sig1, aero_nanew1, &
                aero_inv_rm2, aero_sig2, aero_nanew2, nccnst_in)

end subroutine

subroutine c_p3_main(ids, ide, jds, jde, kds, kde, &
    ims, ime, jms, jme, kms, kme, &
    its, ite, jts, jte, kts, kte, &
    th_3d,qv_3d,qc_3d,qr_3d,   &
    qnr_3d,diag_zdbz_3d,diag_effc_3d,diag_effi_3d,diag_vmi_3d,diag_di_3d,  &
    diag_rhopo_3d,th_old_3d,qv_old_3d, &
    qi1_3d,qni1_3d,qir1_3d, qib1_3d, &
    n_diag_3d, diag_3d, &
    LIQUID_SEDIMENTATION, ICE_SEDIMENTATION, nc_3d, &
    pii, p, dz, w, &
    RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV, &
    dt, itimestep, n_iceCat) bind(c)

! Index bounds
integer(c_int), value, intent(in)::  ids, ide, jds, jde, kds, kde , &
    ims, ime, jms, jme, kms, kme , &
    its, ite, jts, jte, kts, kte


!Input output arrays
real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: th_3d,qv_3d,qc_3d,qr_3d,   &
    qnr_3d,diag_zdbz_3d,diag_effc_3d,diag_effi_3d,diag_vmi_3d,diag_di_3d,  &
    diag_rhopo_3d,th_old_3d,qv_old_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: qi1_3d,qni1_3d,qir1_3d,    &
qib1_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: LIQUID_SEDIMENTATION, ICE_SEDIMENTATION  

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: nc_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(in) :: pii, p, dz, w

real(c_double), dimension(ims:ime, jms:jme), intent(inout) :: RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV
real(c_double), value, intent(in)    :: dt
integer(c_int), value, intent(in) :: itimestep
integer(c_int), value, intent(in) :: n_iceCat

integer(c_int), value, intent(in) :: n_diag_3d
real(c_double), dimension(ims:ime, kms:kme, jms:jme, n_diag_3d), intent(inout) :: diag_3d

call mp_p3_wrapper_wrf(th_3d,qv_3d,qc_3d,qr_3d,qnr_3d,                            &
                              th_old_3d,qv_old_3d,                                       &
                              pii,p,dz,w,dt,itimestep,                                   &
                              rainnc,rainncv,sr,snownc,snowncv,n_iceCat,                 &
                              ids, ide, jds, jde, kds, kde ,                             &
                              ims, ime, jms, jme, kms, kme ,                             &
                              its, ite, jts, jte, kts, kte ,                             &
                              diag_zdbz_3d,diag_effc_3d,diag_effi_3d,                    &
                              diag_vmi_3d,diag_di_3d,diag_rhopo_3d,                      &
                              qi1_3d,qni1_3d,qir1_3d,qib1_3d,                            &
                              n_diag_3d,diag_3d, LIQUID_SEDIMENTATION, ICE_SEDIMENTATION, nc_3d)

end subroutine


subroutine c_p3_main_1mom(ids, ide, jds, jde, kds, kde, &
    ims, ime, jms, jme, kms, kme, &
    its, ite, jts, jte, kts, kte, &
    th_3d,qv_3d,qc_3d,qr_3d,   &
    qnr_3d,diag_zdbz_3d,diag_effc_3d,diag_effi_3d,diag_vmi_3d,diag_di_3d,  &
    diag_rhopo_3d,th_old_3d,qv_old_3d, &
    qi1_3d,qni1_3d,qir1_3d, qib1_3d, &
    n_diag_3d, diag_3d, &
    LIQUID_SEDIMENTATION, ICE_SEDIMENTATION, &
    pii, p, dz, w, &
    RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV, &
    dt, itimestep, n_iceCat) bind(c)

! Index bounds
integer(c_int), value, intent(in)::  ids, ide, jds, jde, kds, kde , &
    ims, ime, jms, jme, kms, kme , &
    its, ite, jts, jte, kts, kte


!Input output arrays
real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: th_3d,qv_3d,qc_3d,qr_3d,   &
    qnr_3d,diag_zdbz_3d,diag_effc_3d,diag_effi_3d,diag_vmi_3d,diag_di_3d,  &
    diag_rhopo_3d,th_old_3d,qv_old_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: qi1_3d,qni1_3d,qir1_3d,    &
qib1_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: LIQUID_SEDIMENTATION, ICE_SEDIMENTATION  

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(in) :: pii, p, dz, w

real(c_double), dimension(ims:ime, jms:jme), intent(inout) :: RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV
real(c_double), value, intent(in)    :: dt
integer(c_int), value, intent(in) :: itimestep
integer(c_int), value, intent(in) :: n_iceCat

integer(c_int), value, intent(in) :: n_diag_3d
real(c_double), dimension(ims:ime, kms:kme, jms:jme, n_diag_3d), intent(inout) :: diag_3d


call mp_p3_wrapper_wrf(th_3d,qv_3d,qc_3d,qr_3d,qnr_3d,                            &
                              th_old_3d,qv_old_3d,                                       &
                              pii,p,dz,w,dt,itimestep,                                   &
                              rainnc,rainncv,sr,snownc,snowncv,n_iceCat,                 &
                              ids, ide, jds, jde, kds, kde ,                             &
                              ims, ime, jms, jme, kms, kme ,                             &
                              its, ite, jts, jte, kts, kte ,                             &
                              diag_zdbz_3d,diag_effc_3d,diag_effi_3d,                    &
                              diag_vmi_3d,diag_di_3d,diag_rhopo_3d,                      &
                              qi1_3d,qni1_3d,qir1_3d,qib1_3d,                            &
                              n_diag_3d,diag_3d,                                         &
                              LIQUID_SEDIMENTATION, ICE_SEDIMENTATION)

end subroutine

end module p3_wrapper