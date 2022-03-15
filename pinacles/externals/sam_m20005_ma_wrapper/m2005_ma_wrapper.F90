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

subroutine c_m2005_ma_main(ids, ide, jds, jde, kds, kde, &
    ims, ime, jms, jme, kms, kme, &
    its, ite, jts, jte, kts, kte, &
    th_3d, s_3d, microfield_4d, th_old_3d, qv_old_3d, &
    n_diag_3d, diag_3d, &
    LIQUID_SEDIMENTATION, ICE_SEDIMENTATION, &
    Nacc_sct_3d, Nait_sct_3d, Nait2a_ct_3d, Mait2a_ct_3d, &
    relhum_3d, diag_effc_3d, diag_effr_3d, diag_effi_3d, diag_effs_3d, &
    pii, p, dz, w, &
    RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV, &
    dt, itimestep, n_iceCat, nmicrofields) bind(c)

! Index bounds
integer(c_int), value, intent(in)::  ids, ide, jds, jde, kds, kde , &
    ims, ime, jms, jme, kms, kme , &
    its, ite, jts, jte, kts, kte


!Input output arrays
real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: th_3d,s_3d,th_old_3d,qv_old_3d,   &
    Nacc_sct_3d, Nait_sct_3d, Nait2a_ct_3d, Mait2a_ct_3d, &
    relhum_3d, diag_effc_3d, diag_effr_3d, diag_effi_3d, diag_effs_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(inout):: LIQUID_SEDIMENTATION, ICE_SEDIMENTATION  

real(c_double), dimension(ims:ime, kms:kme, jms:jme), intent(in) :: pii, p, dz, w

real(c_double), dimension(ims:ime, jms:jme), intent(inout) :: RAINNC,RAINNCV,SR,SNOWNC,SNOWNCV
real(c_double), value, intent(in)    :: dt
integer(c_int), value, intent(in) :: itimestep
integer(c_int), value, intent(in) :: n_iceCat

integer(c_int), value, intent(in) :: n_diag_3d, nmicrofields
real(c_double), dimension(ims:ime, kms:kme, jms:jme, n_diag_3d), intent(inout) :: diag_3d

real(c_double), dimension(ims:ime, kms:kme, jms:jme, nmicrofields), intent(inout) :: microfield_4d

call mp_m2005_ma_wrapper_sam(th_3d, s_3d, microfield_4d,                                         &
                              th_old_3d, qv_old_3d,                                              &
                              pii, p, dz, w, dt, itimestep,                                      &
                              rainnc, rainncv, sr, snownc, snowncv, n_iceCat, nmicrofields       &
                              ids, ide, jds, jde, kds, kde ,                                     &
                              ims, ime, jms, jme, kms, kme ,                                     &
                              its, ite, jts, jte, kts, kte ,                                     &
                              Nacc_sct_3d, Nait_sct_3d, Nait2a_ct_3d, Mait2a_ct_3d,              &
                              relhum_3d, diag_effc_3d, diag_effr_3d, diag_effi_3d, diag_effs_3d, &
                              n_diag_3d, diag_3d, LIQUID_SEDIMENTATION, ICE_SEDIMENTATION)

end subroutine

end module m2005_ma_wrapper