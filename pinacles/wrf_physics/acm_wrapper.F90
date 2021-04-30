module acm_wrapper

    use iso_c_binding, only: c_double, c_int, C_CHAR, C_NULL_CHAR, c_ptr, c_f_pointer
    use module_bl_acm, only: ACMPBL

    contains

    
    subroutine c_acmpbl(XTIME,    DTPBL,    U3D,   V3D,                  &
        PP3D,     DZ8W,     TH3D,  T3D,                  &
        QV3D,     QC3D,     QI3D,  RR3D,                 &
#if (WRF_CHEM == 1)
        CHEM3D,   VD3D,     NCHEM,                       &  ! For WRF-Chem
        KDVEL, NDVEL, NUM_VERT_MIX,                      &  ! For WRF-Chem
#endif
        UST,      HFX,      QFX,   TSK,                  &
        PSFC,     EP1,      G,                           &
        ROVCP,    RD,       CPD,                         &
        PBLH,     KPBL2D,   EXCH_H, REGIME,              &
        GZ1OZ0,   WSPD,     PSIM, MUT, RMOL,             &
        RUBLTEN,  RVBLTEN,  RTHBLTEN,                    &
        RQVBLTEN, RQCBLTEN, RQIBLTEN,                    &
        ids,ide, jds,jde, kds,kde,                       &
        ims,ime, jms,jme, kms,kme,                       &
        its,ite, jts,jte, kts,kte) bind(c)


    !.......Arguments
    ! DECLARATIONS - INTEGER
        INTEGER(c_int), value,  INTENT(IN   )   ::      ids,ide, jds,jde, kds,kde, &
                                          ims,ime, jms,jme, kms,kme, &
                                          its,ite, jts,jte, kts,kte, XTIME
    
    ! DECLARATIONS - REAL
        REAL(c_double), value,                             INTENT(IN)  ::  DTPBL, EP1,   &
                                                            G, ROVCP, RD, CPD
    
        REAL(c_double),    DIMENSION( ims:ime, kms:kme, jms:jme ),                         &
                 INTENT(IN) ::                              U3D, V3D,            &
                                                            PP3D, DZ8W, T3D,     &
                                                            QV3D, QC3D, QI3D,    &
                                                            RR3D, TH3D
    
        REAL(c_double),    DIMENSION( ims:ime, jms:jme ), INTENT(IN) :: PSIM, GZ1OZ0,     &
                                                              HFX, QFX, TSK,    &
                                                              PSFC, WSPD, MUT
    
        REAL(c_double),    DIMENSION( ims:ime, jms:jme ), INTENT(INOUT) ::  PBLH, REGIME, & 
                                                                  UST, RMOL
    
        REAL(c_double),    DIMENSION( ims:ime, kms:kme, jms:jme ),                         &
                 INTENT(INOUT)   ::                         RUBLTEN, RVBLTEN,    &
                                                            RTHBLTEN, RQVBLTEN,  &
                                                            RQCBLTEN, RQIBLTEN
    
        real(c_double),     dimension( ims:ime, kms:kme, jms:jme ),                         &
                 intent(inout)   ::                         exch_h
    
        INTEGER(c_int),  DIMENSION( ims:ime, jms:jme ), INTENT(OUT  ) ::  KPBL2D
     
#if (WRF_CHEM == 1)
    !... Chem
        INTEGER(c_int), value, INTENT(IN   )   ::   nchem, kdvel, ndvel, num_vert_mix
        REAL(c_double),    DIMENSION( ims:ime, kms:kme, jms:jme, nchem ), INTENT(INOUT) :: CHEM3D
        REAL(c_double),    DIMENSION( ims:ime, kdvel, jms:jme, ndvel ), INTENT(IN) :: VD3D
#endif
    
#if (WRF_CHEM == 1)
    !... Chem
        REAL(c_double),    DIMENSION( ims:ime, kms:kme, nchem ) :: CHEM2D
        REAL(c_double),    DIMENSION( ims:ime, kdvel, ndvel ) :: VD2D
#endif
    
    call ACMPBL(XTIME,    DTPBL,    U3D,   V3D,                  &
    PP3D,     DZ8W,     TH3D,  T3D,                  &
    QV3D,     QC3D,     QI3D,  RR3D,                 &
#if (WRF_CHEM == 1)
    CHEM3D,   VD3D,     NCHEM,                       &  ! For WRF-Chem
    KDVEL, NDVEL, NUM_VERT_MIX,                      &  ! For WRF-Chem
#endif
    UST,      HFX,      QFX,   TSK,                  &
    PSFC,     EP1,      G,                           &
    ROVCP,    RD,       CPD,                         &
    PBLH,     KPBL2D,   EXCH_H, REGIME,              &
    GZ1OZ0,   WSPD,     PSIM, MUT, RMOL,             &
    RUBLTEN,  RVBLTEN,  RTHBLTEN,                    &
    RQVBLTEN, RQCBLTEN, RQIBLTEN,                    &
    ids,ide, jds,jde, kds,kde,                       &
    ims,ime, jms,jme, kms,kme,                       &
    its,ite, jts,jte, kts,kte)


    end subroutine c_acmpbl

end module acm_wrapper
