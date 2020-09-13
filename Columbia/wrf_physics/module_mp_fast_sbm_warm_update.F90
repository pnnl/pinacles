!#if( BUILD_SBM_FAST != 1)
!      MODULE module_mp_fast_sbm
!      CONTAINS
!      SUBROUTINE SBM_fast
!         REAL :: dummy
!         dummy = 1
!      END SUBROUTINE SBM_fast
!      END MODULE module_mp_fast_sb
!#else

! +-----------------------------------------------------------------------------+
! +-----------------------------------------------------------------------------+

! This is the spectral-bin microphysics scheme based on the Hebrew University
! Cloud Model (HUCM), originally formulated and coded by Alexander Khain
! (email: Alexander.Khain@mail.huji.ac.il);
! The WRF bin microphysics scheme (Fast SBM or FSBM) solves equations for four
! size distribution functions: aerosols, drop (including rain drops), snow and
! graupel/hail (from which mass mixing ratio qna, qc, qr, qs, qg/qh and
! their number concentrations are calculated).

! The scheme is generally written in CGS units. In the updated scheme (FSBM-2)
! the users can choose either graupel or hail to describe dense particles
! (see the 'hail_opt' switch). By default, the 'hail_opt = 1' is used.
! Hail particles have larger terminal velocity than graupel per mass bin.
! 'hail_opt' is recommended to be used in simulations of continental clouds
! systems. The Graupel option may lead to better results in simulations of
! maritime convection.

! The aerosol spectrum in FSBM-2 is approximated by 3-lognormal size distribution
! representing smallest aerosols (nucleation mode), intermediate-size
! (accumulation mode) and largest aerosols (coarse mode). The BC/IC for aerosols
! ,as well as aerosols vertical distribution profile -- are set from within the
! FSBM-2 scheme (see the 'DOMAIN_ID' parameter). The DOMAIN_ID forces BC to be applied 
! for the parent domain only.

! The user can set the liquid water content threshold (LWC) in which rimed snow
! is being transferred to hail/graupel (see 'ALCR' parameter).
! The default value is ALCR = 0.5 [g/m3]. Increasing this value will result
! in an increase of snow mass content, and a decrease in hail/graupel mass
! contents.

! We thank and acknowledge contribution from Jiwen Fan (PNNL), Alexander Rhyzkov
! (CIMMS/NSSL), Jeffery Snyder (CIMMS/NSSL), Jimy Dudhia (NCAR) and Dave Gill! (NCAR).

! The previous WRF FSBM version (FSBM-1) was coded by Barry Lynn (email:
! Barry.H.Lynn@gmail.com); This updated WRF SBM version (FSBM-2) was coded and
! is maintained by Jacob Shpund (email: kobby.shpund@mail.huji.ac.il).
! Please feel free to reachout with questions about the scheme.

! Useful references:
! -------------------
!   Khain, A. P., and I. Sednev, 1996: Simulation of precipitation formation in
! the Eastern Mediterranean coastal zone using a spectral microphysics cloud
! ensemble model. Atmospheric Research, 43: 77-110;
!   Khain, A. P., A. Pokrovsky and M. Pinsky, A. Seifert, and V. Phillips, 2004:
! Effects of atmospheric aerosols on deep convective clouds as seen from
! simulations using a spectral microphysics mixed-phase cumulus cloud model
! Part 1: Model description. J. Atmos. Sci 61, 2963-2982);
!   Khain A. P. and M. Pinsky, 2018: Physical Processes in Clouds and Cloud
! modeling. Cambridge University Press. 642 pp
!   Shpund, J., A. Khain, and D. Rosenfeld, 2019: Effects of Sea Spray on the
! Dynamics and Microphysics of an Idealized Tropical Cyclone. J. Atmos. Sci., 0,
! https://doi.org/10.1175/JAS-D-18-0270.1 (A preliminary description of the
! updated FSBM-2 scheme)

! When using the FSBM-2 version please cite:
! -------------------------------------------
!   Shpund, J., Khain, A., Lynn, B., Fan, J., Han, B., Ryzhkov, A., Snyder, J., 
! Dudhia, J. and Gill, D., 2019. Simulating a Mesoscale Convective System Using WRF 
! With a New Spectral Bin Microphysics: 1: Hail vs Graupel. 
! Journal of Geophysical Research: Atmospheres.

! +---------------------------------------------------------------------------- +
! +-----------------------------------------------------------------------------+
!---- Note by Jiwen Fan
! (1) The main subroutine is FAST_SBM where all the microphysics processes are
!     called 
! (2) For aerosol setup, seach "Aerosol setup", where one can set up aerosol SD,
! composition information (molecular weight, ions, and density). For SD, there
! is a choice for a lognormal distribution, or read from an observed SD.
! (3) My postdoc Yuwei Zhang has added cloud related diagnostics (mainly process
! rates) and added an option to read in the observed SD. Observed SD data should be processed following! a format of the file "CCN_size_33bin.dat" which is in  size (cm), dN (# cm-3), and dNdlogD
! for 33bins

#define SBM_DIAG  !turn on the diagnostics
!#undef SBM_DIAG  !turn off the diagnostics 
 ! +----------------------------------------------------------------------------+
 ! +----------------------------------------------------------------------------+
  MODULE module_mp_warm_sbm
  
  !USE module_state_description,ONLY:  p_ff1i01, p_ff1i33, p_ff8i01, p_ff8i33,   &
  !                                    p_ff8in01,p_ff8in33, & 
  !                                    p_ff1i01_bfcc,p_ff1i33_bfcc, &
  !                                    p_nc_autoconv,p_qc_autoconv,p_qr_autoconv,p_nr_autoconv,p_qv_autoconv,p_t_autoconv,p_w_autoconv, &                                      
  !                                    p_auto_cldmsink_b,p_auto_cldnsink_b, &                                      
  !                                    p_accr_cldmsink_b,p_accr_cldnsink_b, &
  !                                    p_selfc_rainnchng_b

 PRIVATE

 PUBLIC WARM_SBM,WARM_HUCMINIT,falfluxhucm_z,ckern_z,coal_bott_new_warm,breakinit_ks,ecoalmass,&
 ecoaldiam,ecoallowlist,ecoalochs,vtbeard,coll_xxx_lwf,coll_breakup_ks,courant_bott_ks,polysvp,&
 jersupsat_ks,jerdfun_ks,jernewf_ks,jerdfun_new_ks,relaxation_time,ccn_regeneration,jernucl01_ks,&
 water_nucleation,cloud_base_super,lognormal_modes_aerosol,coll_xxx_bott,coll_xxx_bott_mod1,&
 coll_xxx_bott_mod2,supmax_coeff,jertimesc_ks,jerrate_ks,collenergy,onecond1,kernals_ks


! Kind paramater
 INTEGER, PARAMETER, PRIVATE:: R8SIZE = 8
 INTEGER, PARAMETER, PRIVATE:: R16SIZE = 16 
 INTEGER, PARAMETER, PRIVATE:: R4SIZE = 4

! JacobS: Hard coding the bin-wise indices for the NMM core
! INTEGER, PRIVATE,PARAMETER ::  p_ff1i01=2, p_ff1i33=34,p_ff5i01=35,p_ff5i33=67,p_ff6i01=68,&
!                                p_ff6i33=100,p_ff8i01=101,p_ff8i43=143

! JacobS: Hard coding for the polarimetric operator output array 
  INTEGER, PRIVATE,PARAMETER :: r_p_ff1i01=2, r_p_ff1i06=07,r_p_ff2i01=08,r_p_ff2i06=13,r_p_ff3i01=14,&
                                r_p_ff3i06=19,r_p_ff4i01=20,r_p_ff4i06=25,r_p_ff5i01=26,r_p_ff5i06=31,r_p_ff6i01=32,r_p_ff6i06=37,&
                                r_p_ff7i01=38,r_p_ff7i06=43,r_p_ff8i01=44,r_p_ff8i06=49,r_p_ff9i01=50,r_p_ff9i06=55

  INTEGER, PRIVATE,PARAMETER :: p_ff1i01=1,p_ff1i33=33,p_ff5i01=1,p_ff5i33=1, &
                                p_ff6i01=1,p_ff6i33=1,p_ff8i01=34,p_ff8i33=66,p_ff8in01=67,p_ff8in33=99,&
                                p_ff1i01_bfcc= 1,  &
                                p_ff1i33_bfcc= 33, &
                                p_nc_autoconv= 34, &
                                p_qc_autoconv= 35 , &
                                p_qr_autoconv= 36 , &
                                p_nr_autoconv= 37 , &
                                p_qv_autoconv= 38 , &
                                p_t_autoconv=  39 , &
                                p_w_autoconv=  40 , &                                     
                                p_auto_cldmsink_b= 41,&
                                p_auto_cldnsink_b= 42,&
                                p_accr_cldmsink_b= 43,&
                                p_accr_cldnsink_b= 44, &
                                p_selfc_rainnchng_b= 45 


 INTEGER,PARAMETER :: IBREAKUP = 1
 INTEGER,PARAMETER :: Snow_BreakUp_On = 1
 INTEGER,PARAMETER :: Spont_Rain_BreakUp_On = 1
 LOGICAL,PARAMETER :: CONSERV = .TRUE.
 INTEGER,PARAMETER :: JIWEN_FAN_MELT = 1
 LOGICAL,PARAMETER :: IPolar_HUCM = .FALSE. ! CK
 INTEGER,PARAMETER :: hail_opt = 1

 double precision,PARAMETER :: DX_BOUND = 1433
 double precision, PARAMETER ::  SCAL = 1.d0
 INTEGER,PARAMETER :: ICEPROCS = 0 ! CK
 INTEGER,PARAMETER :: ICETURB = 0, LIQTURB = 0

 INTEGER,PARAMETER :: icempl=1,ICEMAX=3,NCD=33,NHYDR=5,NHYDRO=7                &
             					,K0_LL=8,KRMIN_LL=1,KRMAX_LL=19,L0_LL=6                  &
             					,IEPS_400=1,IEPS_800=0,IEPS_1600=0                       &
             					,K0L_GL=16,K0G_GL=16                                     &
             					,KRMINL_GL=1,KRMAXL_GL=24                                &
             					,KRMING_GL=1,KRMAXG_GL=33,kr_icempl=9                    &
             					,KRBREAK=17,KRICE=18                                     & ! KRDROP=Bin 15 --> 50um
             					!,NKR=43,JMAX=43,NRG=2,JBREAK=28,BR_MAX=43,KRMIN_BREAKUP=31,NKR_aerosol=43   ! 43 bins
             					,NKR=33,JMAX=33,NRG=2,JBREAK=18,BR_MAX=33,KRMIN_BREAKUP=31,NKR_aerosol=33    ! 33 bins

 double precision :: dt_coll
 double precision,PARAMETER :: C1_MEY=0.00033,C2_MEY=0.0,COL=0.23105, &
                   p1=1000000.0,p2=750000.0,p3=500000.0,  &
                   ALCR = 0.5, &
                   ALCR_G = 100.0 ! ... [KS] forcing no transition from graupel to hail in this version
 INTEGER :: NCOND, NCOLL

integer,parameter :: kp_flux_max = 33
double precision, parameter :: G_LIM = 1.0D-16 ! [g/cm^3]
integer,parameter :: kr_sgs_max = 20 ! rg(20)=218.88 mkm

INTEGER,PARAMETER :: ISIGN_KO_1 = 0, ISIGN_KO_2 = 0,  ISIGN_3POINT = 1,  &
                      IDebug_Print_DebugModule = 1
 DOUBLE PRECISION,PARAMETER::COEFF_REMAPING = 0.0066667D0
 DOUBLE PRECISION,PARAMETER::VENTPL_MAX = 5.0D0

 DOUBLE PRECISION,PARAMETER::RW_PW_MIN = 1.0D-10
 DOUBLE PRECISION,PARAMETER::RI_PI_MIN = 1.0D-10
 DOUBLE PRECISION,PARAMETER::RW_PW_RI_PI_MIN = 1.0D-10
 DOUBLE PRECISION,PARAMETER::RATIO_ICEW_MIN = 1.0D-4

 INTEGER,PARAMETER :: Use_cloud_base_nuc = 0
	double precision,PARAMETER::T_NUCL_DROP_MIN = -80.0D0
	double precision,PARAMETER::T_NUCL_ICE_MIN = -37.0D0
! Ice nucleation method
! using MEYERS method : ice_nucl_method == 0
! using DE_MOTT method : ice_nucl_method == 1
	INTEGER,PARAMETER :: ice_nucl_method = 0
	INTEGER,PARAMETER :: ISIGN_TQ_ICENUCL = 1
! DELSUPICE_MAX=59%
  DOUBLE PRECISION,PARAMETER::DELSUPICE_MAX = 59.0D0
  

 double precision :: &
 					 RADXX(NKR,NHYDR-1),MASSXX(NKR,NHYDR-1),DENXX(NKR,NHYDR-1) &
 					,MASSXXO(NKR,NHYDRO),DENXXO(NKR,NHYDRO),VRI(NKR)           &
          ,XX(nkr),ROCCN(nkr),FCCNR_MIX(NKR),FCCNR(NKR)

 double precision,DIMENSION (NKR) :: FF1R_D,XL_D,VR1_D &
 							,FF3R_D,XS_D,VR3_D,VTS_D,FLIQFR_SD,RO3BL_D &
 							,FF4R_D,XG_D,VR4_D,VTG_D,FLIQFR_GD,RO4BL_D &
 							,FF5R_D,XH_D,VR5_D,VTH_D,FLIQFR_HD,RO5BL_D &
 							,XS_MELT_D,XG_MELT_D,XH_MELT_D,VR_TEST,FRIMFR_SD,RF3R

 ! ... SBMRADAR VARIABLES
 double precision,DIMENSION (nkr,icemax) :: XI_MELT_D &
							,FF2R_D,XI_D,VR2_D,VTC_D,FLIQFR_ID,RO2BL_D
 double precision :: T_NEW_D,rhocgs_D,pcgs_D,DT_D,qv_old_D,qv_d

 double precision,private :: C2,C3,C4
 double precision,private ::  &
 	            xl_mg(nkr),xs_mg(nkr),xg_mg(nkr),xh_mg(nkr) &
             ,xi1_mg(nkr),xi2_mg(nkr),xi3_mg(nkr)

 ! ----------------------------------------------------------------------------------+
 ! ... WARM-SBM-Init
 ! ... Holding Lookup tables and memory arrays for the FAST_SBM module
         double precision, ALLOCATABLE, DIMENSION(:)::                             &
                                          bin_mass,tab_colum,tab_dendr,tab_snow,bin_log
         double precision, ALLOCATABLE, DIMENSION(:) ::                            &
                                          RLEC,RSEC,RGEC,RHEC,XL,XS,XG,XH,VR1,VR3,VR4,VR5
         double precision, ALLOCATABLE, DIMENSION(:,:)::                           &
                                          RIEC,XI,VR2
         double precision, ALLOCATABLE ::                              &
                                          COEFIN(:),SLIC(:,:),TLIC(:,:), &
                                          YWLL_1000MB(:,:),YWLL_750MB(:,:),YWLL_500MB(:,:)
         double precision, ALLOCATABLE ::                                                   &
                                         RO1BL(:), RO2BL(:,:), RO3BL(:), RO4BL(:), RO5BL(:),  &
                                         RADXXO(:,:)

         INTEGER,ALLOCATABLE ::              ima(:,:)
         double precision, ALLOCATABLE ::  chucm(:,:)

         double precision, ALLOCATABLE ::  BRKWEIGHT(:),ECOALMASSM(:,:), Prob(:),Gain_Var_New(:,:),NND(:,:)
         double precision, ALLOCATABLE ::  DROPRADII(:),PKIJ(:,:,:),QKJ(:,:)
         INTEGER ::          ikr_spon_break

         double precision, ALLOCATABLE ::  cwll(:,:)
                                             
         double precision,ALLOCATABLE ::  FCCNR_MAR(:),FCCNR_CON(:)
         double precision,ALLOCATABLE ::  FCCNR_OBS(:),CCNR(:)
         double precision,ALLOCATABLE :: Scale_CCN_Factor,XCCN(:),RCCN(:),FCCN(:),FCCN_nucl(:)
 ! ... WARM-SBM-Init
 ! --------------------------------------------------------------------------------+

 INTEGER :: icloud

! ----Aerosol setup (Jiwen Fan)
! Aerosol size distribution (SD)
 INTEGER,PARAMETER :: ILogNormal_modes_Aerosol = 1 !Follow lognormal
! distribution
! integer,parameter :: ILogNormal_modes_Aerosol = 0 ! read in a SD file from observation. Currently the file name for the observed SD is "CCN_size_33bin.dat", whcih is from the July 18 2017 ENA case.  
 integer,parameter :: do_Aero_BC = 0
 integer,parameter :: ICCN_reg = 1
 ! Aerosol composition
 double precision, parameter :: mwaero = 22.9 + 35.5 ! sea salt
 !double precision,parameter :: mwaero = 115.0
 integer,parameter :: ions = 2        	! sea salt
 !integer,parameter  :: ions = 3         ! ammonium-sulfate
 double precision,parameter :: RO_SOLUTE = 2.16   	! sea salt
 !double precision,parameter ::  RO_SOLUTE = 1.79  	! ammonium-sulfate
! for diagnostic CCN for places where sources exist (Added by Jiwen Fan on April
! 25, 2020)
logical,parameter :: diagCCN = .false.
double precision, allocatable :: fccnorig(:), fccnd(:) ! for diagCCN
 ! ----Aerosol setup (end)

 double precision :: FR_LIM(NKR), FRH_LIM(NKR),lh_ce_1, lh_ce_2, lh_ce_3,  &
                      lh_frz, lh_mlt, lh_rime, lh_homo, ce_bf, ce_af, ds_bf, &
                      ds_af, mlt_bf, mlt_af, frz_af, frz_bf, cldnucl_af,     &
                      cldnucl_bf, icenucl_af, icenucl_bf, lh_ice_nucl, del_cldnucl_sum, &
                      del_icenucl_sum, del_ce_sum, del_ds_sum,Del_CCNreg, CCN_reg

 double precision :: auto_cld_nsink_b,auto_cld_msink_b, &  
                       accr_cld_nsink_b,accr_cld_msink_b,  &
                       selfc_rain_nchng_b,dbl_orhocgs,dbl_odt 
!---YZ2020-----------------------------------------------
double precision ttdiffl, automass_ch, autonum_ch, nrautonum 
!--------------------------------------------------------

   CONTAINS
 
  SUBROUTINE WARM_SBM (w,u,v,th_old,                                  &
  &                      chem_new,n_chem,                             &
  &                      itimestep,DT,DX,DY,                          &
  &                      dz8w,rho_phy,p_phy,pi_phy,th_phy,            &
  &                      xland,domain_id, &
  &                      QV,QC,QR,QV_OLD,                             &
  &                      QNC,QNR,QNA,QNA_nucl,                        &
  &                      ids,ide, jds,jde, kds,kde,		        	      &
  &                      ims,ime, jms,jme, kms,kme,		        	      &
  &                      its,ite, jts,jte, kts,kte,                   &
  &                      diagflag,KRDROP,      	                      &
  &                      sbmradar,num_sbmradar,                       &
  &                      RAINNC,RAINNCV,SR,                           &
  &                      MA,LH_rate,CE_rate,CldNucl_rate,             &
  &                      n_reg_ccn,                                   &
  &                      num_sbm_output_container,sbm_output_container, &  
  &                      difful_tend,   &  !liquid mass change rate due to droplet diffusional growth (kg/kg/s)
  &                      diffur_tend,   &  !rain mass change rate due to droplet diffusional growth (kg/kg/s)
  &                      tempdiffl      &  !latent heat rate due to droplet diffusional growth (K/s)                         
                                        )

  IMPLICIT NONE
 !-----------------------------------------------------------------------
 	INTEGER :: KR,IKL,ICE

 	INTEGER,INTENT(IN) :: IDS,IDE,JDS,JDE,KDS,KDE                     &
 	&                     ,IMS,IME,JMS,JME,KMS,KME                    &
 	&                     ,ITS,ITE,JTS,JTE,KTS,KTE                    &
  &                     ,ITIMESTEP,N_CHEM,NUM_SBMRADAR,domain_id,KRDROP &
  &                     ,num_sbm_output_container 

 	double precision, INTENT(IN) 	    :: DT,DX,DY
 	double precision,  DIMENSION( ims:ime , kms:kme , jms:jme ), &
 	INTENT(IN   ) ::                                 &
 							  U, &
 							  V, &
 							  W

 	double precision    ,DIMENSION(ims:ime,kms:kme,jms:jme,n_chem),INTENT(INOUT)   :: chem_new
  double precision    ,DIMENSION(ims:ime,kms:kme,jms:jme,num_sbmradar),INTENT(INOUT)   :: sbmradar
  double precision    ,DIMENSION(ims:ime,kms:kme,jms:jme,num_sbm_output_container),INTENT(INOUT)   :: sbm_output_container
 	double precision    ,DIMENSION( ims:ime , kms:kme , jms:jme ),               &
 		      INTENT(INOUT) ::                                          &
 						  qv, 		      &
 						  qv_old, 	    &
 						  th_old, 	    &
 						  qc, 		      &
 						  qr, 		      &
 						  qnc, 		      &
 						  qnr, 		      &
              qna,qna_nucl, &
              MA,LH_rate,CE_rate,CldNucl_rate,n_reg_ccn

       double precision , DIMENSION( ims:ime , jms:jme ) , INTENT(IN)   :: XLAND
       LOGICAL, OPTIONAL, INTENT(IN) :: diagflag

       !INTEGER, DIMENSION( ims:ime , jms:jme ), INTENT(IN)::   IVGTYP
       !double precision, DIMENSION( ims:ime, jms:jme ), INTENT(IN   )    :: XLAT, XLONG
       double precision, INTENT(IN),     DIMENSION(ims:ime, kms:kme, jms:jme)::      &
      &                      dz8w,p_phy,pi_phy,rho_phy
       double precision, INTENT(INOUT),  DIMENSION(ims:ime, kms:kme, jms:jme)::      &
      &                      th_phy
       double precision, INTENT(INOUT),  DIMENSION(ims:ime,jms:jme), OPTIONAL ::     &
      &      RAINNC,RAINNCV,SR
!-----YZ2020:Define arrays for diagnostics------------------------@
#ifdef SBM_DIAG
      double precision, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT)::  &
      difful_tend,diffur_tend,tempdiffl
#endif
!-----------------------------------------------------------------@

 !-----------------------------------------------------------------------
 !     LOCAL VARS
 !-----------------------------------------------------------------------

       double precision,  DIMENSION(its-1:ite+1, kts:kte, jts-1:jte+1)::  &
                                                  t_new,t_old,zcgs,rhocgs,pcgs

       INTEGER :: I,J,K,KFLIP
       INTEGER :: KRFREEZ

       double precision,PARAMETER :: Z0IN=2.0E5,ZMIN=2.0E5

       double precision :: EPSF2D, &
      &        TAUR1,TAUR2,EPS_R1,EPS_R2,ANC1IN, &
      &        PEPL,PEPI,PERL,PERI,ANC1,ANC2,PARSP, &
      &        AFREEZMY,BFREEZMY,BFREEZMAX, &
      &        TCRIT,TTCOAL, &
      &        EPSF1,EPSF3,EPSF4, &
      &        SUP2_OLD, DSUPICEXZ,TFREEZ_OLD,DTFREEZXZ, &
      &        AA1_MY,BB1_MY,AA2_MY,BB2_MY, &
      &        DTIME,DTCOND,DTNEW,DTCOLL, &
      &        A1_MYN, BB1_MYN, A2_MYN, BB2_MYN
      DATA A1_MYN, BB1_MYN, A2_MYN, BB2_MYN  &
      &      /2.53,5.42,3.41E1,6.13/
      DATA AA1_MY,BB1_MY,AA2_MY,BB2_MY/2.53E12,5.42E3,3.41E13,6.13E3/
             !QSUM,ISUM,QSUM1,QSUM2,CCNSUM1,CCNSUM2
      DATA KRFREEZ,BFREEZMAX,ANC1,ANC2,PARSP,PEPL,PEPI,PERL,PERI, &
      &  TAUR1,TAUR2,EPS_R1,EPS_R2,TTCOAL,AFREEZMY,&
      &  BFREEZMY,EPSF1,EPSF3,EPSF4,TCRIT/21,&
      &  0.6600E00, &
      &  1.0000E02,1.0000E02,0.9000E02, &
      &  0.6000E00,0.6000E00,1.0000E-03,1.0000E-03, &
      &  0.5000E00,0.8000E00,0.1500E09,0.1500E09, &
      &  2.3315E02,0.3333E-04,0.6600E00, &
      &  0.1000E-02,0.1000E-05,0.1000E-05, &
      &  2.7015E02/

      double precision,DIMENSION (nkr) :: FF1IN,FF3IN,FF4IN,FF5IN,&
      &              FF1R,FF3R,FF4R,FF5R,FLIQFR_S,FRIMFR_S,FLIQFR_G,FLIQFR_H, &
      &              FF1R_NEW,FF3R_NEW,FF4R_NEW,FF5R_NEW
      double precision,DIMENSION (nkr) :: FL3R,FL4R,FL5R,FL3R_NEW,FL4R_NEW,FL5R_NEW

      double precision,DIMENSION (nkr,icemax) :: FF2IN,FF2R,FLIQFR_I

      double precision :: XI_MELT(NKR,ICEMAX),XS_MELT(NKR),XG_MELT(NKR),XH_MELT(NKR)
 !!!! NOTE: ZCGS AND OTHER VARIABLES ARE ALSO DIMENSIONED IN FALFLUXHUCM
      double precision :: DEL1NR,DEL2NR,DEL12R,DEL12RD,ES1N,ES2N,EW1N,EW1PN
      double precision :: DELSUP1,DELSUP2,DELDIV1,DELDIV2
      double precision :: TT,QQ,TTA,QQA,PP,DPSA,DELTATEMP,DELTAQ
      double precision :: DIV1,DIV2,DIV3,DIV4,DEL1IN,DEL2IN,DEL1AD,DEL2AD
      double precision :: DEL_BB,DEL_BBN,DEL_BBR, TTA_r
      double precision :: FACTZ,CONCCCN_XZ,CONCDROP
      double precision :: SUPICE(KTE),AR1,AR2, &
      					& DERIVT_X,DERIVT_Y,DERIVT_Z,DERIVS_X,DERIVS_Y,DERIVS_Z, &
      					& ES2NPLSX,ES2NPLSY,EW1NPLSX,EW1NPLSY,UX,VX, &
      					& DEL2INPLSX,DEL2INPLSY,DZZ(KTE)
 	 INTEGER KRR,I_START,I_END,J_START,J_END
      double precision :: DTFREEZ_XYZ(ITE,KTE,JTE),DSUPICE_XYZ(ITE,KTE,JTE)

      double precision :: DXHUCM,DYHUCM
      double precision :: FMAX1,FMAX2(ICEMAX),FMAX3,FMAX4,FMAX5
 	INTEGER ISYM1,ISYM2(ICEMAX),ISYM3,ISYM4,ISYM5
 	INTEGER DIFFU
 	double precision :: DELTAW
 	double precision :: zcgs_z(kts:kte),pcgs_z(kts:kte),rhocgs_z(kts:kte),ffx_z(kts:kte,nkr)
 	double precision :: z_full
 	double precision :: VRX(kts:kte,NKR)

 	double precision :: VR1_Z(NKR,KTS:KTE), FACTOR_P, VR1_Z3D(NKR,ITS:ITE,KTS:KTE,JTS:JTE)

 	double precision, PARAMETER :: RON=8.E6, GON=5.E7,PI=3.14159265359
 	double precision :: EFF_N,EFF_D
     double precision :: EFF_NI(its:ite,kts:kte,jts:jte),eff_di(its:ite,kts:kte,jts:jte)
 	double precision :: EFF_NQIC,eff_DQIC
 	double precision :: EFF_NQIP,eff_DQIP
 	double precision :: EFF_NQID,eff_DQID
 	double precision :: lambda,chi0,xi1,xi2,xi3,xi4,xi5,r_e,chi_3,f1,f2,volume,surface_area,xi6,ft,chi_e,ft_bin
 	double precision, DIMENSION(kts:kte)::                            &
 						  qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, t1d, p1d
   double precision, DIMENSION(kts:kte):: dBZ
   
 	double precision :: raddumb(nkr),massdumb(nkr)
   double precision :: hydrosum
   ! added for diagCCN
   double precision :: ndrop, subtot,tot0  ! for diagnostic CCN (diagCCN)

 	integer imax,kmax,jmax
   double precision :: gmax,tmax,qmax,divmax,rainmax,qnmax,inmax,knmax, &
                         hydro,difmax, tdif, tt_old, w_stag, w_stag_my, qq_old,teten,es
 	integer  print_int
 	parameter (print_int=300)

 	integer t_print,i_print,j_print,k_print
 	double precision, DIMENSION(kts:kte):: zmks_1d
 	double precision :: dx_dbl, dy_dbl
 	INTEGER,DIMENSION (nkr) :: melt_snow,melt_graupel,melt_hail,melt_ice
 	!DOUBLE PRECISION,DIMENSION (nkr) :: dmelt_snow,dmelt_graupel,dmelt_hail,dmelt_ice
 	INTEGER ihucm_flag
 	double precision :: NSNOW_ADD

 	! ... Polar-HUCM
 	INTEGER,PARAMETER :: n_radar = 10
 	integer :: ijk, Mod_Flag
 	double precision,PARAMETER :: wavelength = 11.0D0 ! ### (KS) - Rhyzkov uses this wavelength (NEXRAD)
 	INTEGER :: IWL
 	double precision :: DIST_SING
 	double precision :: BKDEN_Snow(NKR)
 	double precision ::  DISTANCE,FL1_FD(NKR),BULK(NKR), BulkDens_Snow(NKR)
 	double precision ::  FF1_FD(NKR),FFL_FD(NKR),OUT1(n_radar),OUT2(n_radar),OUT3(n_radar),OUT4(n_radar),OUT5(n_radar), &
 						   OUT6(n_radar),OUT7(n_radar),OUT8(n_radar),OUT9(n_radar), FL1R_FD(NKR)
 	double precision :: rate_shed_per_grau_grampersec(NKR), rate_shed_per_hail_grampersec(NKR), rhoair_max

 	integer :: count_H, count_G, count_S_l, count_S_r

 	double precision :: RMin_G
 	integer :: KR_GRAUP_MAX_BLAHAK, KR_G_TO_H

 	! ... Cloud Base .........................................................
 	double precision ::	SUP_WATER, ES1N_KS, ES1N_dummy, ES2N_dummy
 	logical :: K_found
 	integer ::	KZ_Cloud_Base(its:ite,jts:jte), IS_THIS_CLOUDBASE,KR_Small_Ice
 	! ........................................................................
 	double precision :: qna0(its:ite,kts:kte,jts:jte), fr_hom, w_stagm, CollEff_out, FACT
 	double precision :: FACTZ_new(KMS:KME,NKR), TT_r, n_reg_ccn_bf, n_reg_ccn_af, Win
 ! ### (KS) ............................................................................................
 	INTEGER :: NZ,NZZ,II,JJ
!---YZ2020:Arrays for process rate calculation---------------------@
   double precision totlbf_diffu, totlaf_diffu, totrbf_diffu, totraf_diffu, del_difful_sum, del_diffur_sum
!------------------------------------------------------------------@
  XS_d = XS

  if (itimestep == 1)then
    !if (iceprocs == 1) call wrf_message(" FAST SBM: ICE PROCESES ACTIVE ")
    !if (iceprocs == 0) call wrf_message(" FAST SBM: LIQUID PROCESES ONLY")
  end if

  NCOND = 3
  NCOLL = 1
  DTCOND = DT/DBLE(NCOND)
  DTCOLL = DT/DBLE(NCOLL)
  dt_coll = DTCOLL

  DEL_BB=BB2_MY-BB1_MY
  DEL_BBN=BB2_MYN-BB1_MYN
  DEL_BBR=BB1_MYN/DEL_BBN
 
  I_START=MAX(1,ITS-1)
  J_START=MAX(1,JTS-1)
  I_END= IDE!MIN(IDE-1,ITE+1)
  J_END= JDE!MIN(JDE-1,JTE+1)

  if (itimestep == 1)then
    DO j = j_start,j_end
        DO k = kts,kte
          DO i = i_start,i_end
            th_old(i,k,j)=th_phy(i,k,j)
            qv_old(i,k,j)=qv(i,k,j)
          END DO
        END DO
    END DO
  end if

  DO j = j_start,j_end
    DO k = kts,kte
        DO i = i_start,i_end
          t_new(i,k,j) = th_phy(i,k,j)*pi_phy(i,k,j)
          !tempc(i,k,j)= t_new(i,k,j)-273.16
          t_old(i,k,j) = th_old(i,k,j)*pi_phy(i,k,j)
        END DO
    END DO
  END DO

  DO j = jts,jte
    DO i = its,ite
      DO k = kts,kte

        rhocgs(I,K,J)=rho_phy(I,K,J)*0.001
    ! ... Drops
        KRR=0
        DO KR=p_ff1i01,p_ff1i33
          KRR=KRR+1
          chem_new(I,K,J,KR)=chem_new(I,K,J,KR)*RHOCGS(I,K,J)/COL/XL(KRR)/XL(KRR)/3.0
        END DO
    ! ... Aerosols
        KRR=0
        DO KR=p_ff8i01,p_ff8i33
            KRR=KRR+1
            chem_new(I,K,J,KR) = chem_new(I,K,J,KR)*RHOCGS(I,K,J)/1000.0
                                ! chem_new (input) is #/kg
        END DO
    ! ... Nucleated Aerosols
        KRR=0
        DO KR=p_ff8in01,p_ff8in33
            KRR=KRR+1
            chem_new(I,K,J,KR) = chem_new(I,K,J,KR)*RHOCGS(I,K,J)/1000.0                                   
        END DO   

      END DO 
    END DO	
  END DO 

  DXHUCM=100.*DX
  DYHUCM=100.*DY

   DO j = j_start,j_end
      DO i = i_start,i_end
         z_full=0.
         DO k = kts,kte
            pcgs(I,K,J)=P_PHY(I,K,J)*10.
            rhocgs(I,K,J)=rho_phy(I,K,J)*0.001
            zcgs(I,K,J)=z_full+0.5*dz8w(I,K,J)*100
            !height(i,k,j) = 1.0e-2*zcgs(i,k,j) ! in [m]
            z_full=z_full+dz8w(i,k,j)*100.
         ENDDO
      ENDDO
   ENDDO

 ! +---------------------------------------+
 ! ... Initial Aerosol distribution
 ! +---------------------------------------+
    if (itimestep == 1)then
      FACTZ_new = 0.0
      DO j = jts,jte
        DO i = its,ite
          DO k = kts,kte
              rhoair_max = rhocgs(i,1,j) ! [g/cm3]
              IF (zcgs(I,K,J) .LE. ZMIN)THEN
                  FACTZ = 1.0
              ELSE
                  FACTZ=EXP(-(zcgs(I,K,J)-ZMIN)/Z0IN)
              END IF
              if(ILogNormal_modes_Aerosol == 1)then
                ! ... Generic CCN
                KRR = 0
                DO KR = p_ff8i01,p_ff8i33
                  KRR = KRR + 1
                  if (xland(i,j) == 1)then
                    ! chem_new(I,K,J,KR)=FCCNR_CON(KRR)*FACTZ
                    chem_new(I,K,J,KR) = (FCCNR_CON(KRR)/rhoair_max)*rhocgs(i,k,j) ! ... distributed vertically as [#/g]
                  else
                    ! chem_new(I,K,J,KR)=FCCNR_MAR(KRR)*FACTZ	
                    chem_new(I,K,J,KR) = (FCCNR_MAR(KRR)/rhoair_max)*rhocgs(i,k,j) ! ... distributed vertically as [#/g]
                  endif  
                END DO
              else
                ! ... CCN input from observation
                KRR = 0
                DO KR = p_ff8i01,p_ff8i33
                    KRR = KRR + 1
                    chem_new(I,K,J,KR) = FCCNR_OBS(KRR)*FACTZ
                ENDDO
              endif
          end do
        end do
      end do
    end if

 ! +--------------------------------------------+
 ! ... Aerosols boundary conditions
 !    (for 3D application running with MPI)   
 ! +--------------------------------------------+
#if (defined(DM_PARALLEL))
    if (itimestep > 1 .and. domain_id == 1 .and. do_Aero_BC == 1)then
        DO j = jts,jte
          DO k = kts,kte
            DO i = its,ite
              rhoair_max = rhocgs(i,1,j) ! [g/cm3]
              if (i <= 5 .or. i >= IDE-5 .OR. &
                  & j <= 5 .or. j >= JDE-5)THEN
                  
                    IF (zcgs(I,K,J).LE.ZMIN) THEN
                        FACTZ = 1.0
                    ELSE
                        FACTZ=EXP(-(zcgs(I,K,J)-ZMIN)/Z0IN)
                    END IF

                    ! ... CCN
                    ndrop=0.0   ! for diagCCN  (added by Jiwen Fan)
                    KRR = 0
                    do kr = p_ff8i01,p_ff8i33
                      KRR = KRR + 1
                      if(ILogNormal_modes_Aerosol == 1)then
                          if (xland(i,j) == 1)then
                            chem_new(I,K,J,KR)= FCCNR_CON(KRR)*FACTZ
                            fccnorig(krr)=FCCNR_CON(KRR)*FACTZ ! for diagCCN at boundaries
                          else
                            chem_new(I,K,J,KR) = FCCNR_MAR(KRR)*FACTZ
                            fccnorig(krr)=FCCNR_MAR(KRR)*FACTZ ! for diagCCN at boundaries
                          endif
                      else
                          chem_new(I,K,J,KR) = FCCNR_OBS(KRR)*FACTZ
                          fccnorig(krr) = FCCNR_OBS(KRR)*FACTZ ! for diagCCN at boundaries
                      end if
                    enddo

                    if (diagCCN) then  ! for diagCCN
                      krr = 0
                      do kr = p_ff1i01,p_ff1i33  
                        krr = krr + 1
                        ndrop = ndrop + chem_new(I,K,J,KR)*xl(krr)*col ! in [#/cm3]   
                      enddo  ! end DO kr

                      FCCND(:) = fccnorig(:)
                      tot0 = sum(fccnorig(:))*COL                      ! in [#/cm3]
                      if (ndrop >= tot0)  FCCND(:) = 0.0
          
                      if (ndrop < tot0) then
                        subtot = 0.0
                        do krr = nkr, 1, -1
                          subtot = subtot + fccnorig(krr)*COL
                          FCCND(krr) = 0.0
                          if (subtot >= ndrop) then
                            FCCND(krr) = (subtot-ndrop)/COL
                            exit
                          endif
                        enddo
                      endif

                      KRR = 0
                      DO kr = p_ff8i01,p_ff8i33
                        KRR = KRR + 1
                      chem_new(I,K,J,KR) = FCCND(KRR) 
                      ENDDO   
                    end if ! end here for if (diagCCN) then
                  
              end if
            end do
          end do
        end do
      end if
#endif

!---YZ2020:Initialization at each timestep--------------@
#ifdef SBM_DIAG 

    do j = jts,jte
    do i = its,ite
    do k = kts,kte
      difful_tend(I,K,J) = 0.0
      diffur_tend(I,K,J) = 0.0
      tempdiffl(I,K,J) = 0.0
      n_reg_ccn(I,K,J) = 0.0
      sbm_output_container(i,k,j,:) = 0.0
    end do
    end do
    end do

#endif
  !-------------------------------------------------------@

    do j = jts,jte
      do k = kts,kte
          do i = its,ite

            ! ... correcting Look-up-table Terminal velocities
            FACTOR_P = DSQRT(1.0D6/PCGS(I,K,J))
            VR1_Z(1:nkr,K) =  VR1(1:nkr)*FACTOR_P
            VR1_Z3D(1:nkr,I,K,J) = VR1(1:nkr)*FACTOR_P

     			! ... Droplet / Drops
     			  KRR = 0
     			  DO kr = p_ff1i01,p_ff1i33
     				 KRR = KRR + 1
     				 FF1R(KRR) = chem_new(I,K,J,KR)
     				 IF (FF1R(KRR) < 0.0)FF1R(KRR) = 0.0
     			  END DO
     			! ... CCN
     			  KRR = 0
     			  DO kr=p_ff8i01,p_ff8i33
     				 KRR = KRR + 1
     				 FCCN(KRR) = chem_new(I,K,J,KR)
     				 if (fccn(krr) < 0.0)fccn(krr) = 0.0
     			  END DO
            
          ! ... Nucleated CCN
            KRR = 0
            DO kr=p_ff8in01,p_ff8in33
              KRR = KRR + 1
              FCCN_nucl(KRR) = chem_new(I,K,J,KR)
              if (fccn_nucl(krr) < 0.0)fccn_nucl(krr) = 0.0
            END DO  

            lh_ce_1 = 0.0;
            ce_bf = 0.0; ce_af = 0.0; cldnucl_af = 0.0; cldnucl_bf = 0.0; 
            del_cldnucl_sum = 0.0; del_ce_sum = 0.0; del_ds_sum=0.0;
            
            auto_cld_nsink_b = 0.0; auto_cld_msink_b = 0.0;  
            accr_cld_nsink_b = 0.0; accr_cld_msink_b = 0.0;
            selfc_rain_nchng_b = 0.0
      
!--- JacobS: Initialization at each timestep per grid point --------------@
#ifdef SBM_DIAG 
            totlbf_diffu =0.
            totlaf_diffu =0.
            totrbf_diffu =0.
            totraf_diffu =0.
            ttdiffl = 0.
            del_difful_sum = 0.
            del_diffur_sum = 0.
            n_reg_ccn_bf = 0.0
            n_reg_ccn_af = 0.0
#endif
! +---------------------------------------------+
! Neucliation, Condensation, Collisions
! +---------------------------------------------+
          CCN_reg = 0.0
          Del_CCNreg = 0.0  
          IF (T_OLD(I,K,J) > 273.0d0)THEN
             TT=T_OLD(I,K,J)
             QQ=QV_OLD(I,K,J)
             IF(QQ.LE.0.0) QQ = 1.D-10
             PP=pcgs(I,K,J)
             TTA=T_NEW(I,K,J)
             QQA=QV(I,K,J)

             !IF (QQA.LE.0) call wrf_message("WARNING: FAST SBM, QQA < 0")
             IF (QQA.LE.0) print*,'I,J,K,Told,Tnew,QQA = ',I,J,K,TT,TTA,QQA
             IF (QQA.LE.0) QQA = 1.0D-10

             ES1N = AA1_MY*DEXP(-BB1_MY/TT)
             ES2N = AA2_MY*DEXP(-BB2_MY/TT)
             EW1N=QQ*PP/(0.622+0.378*QQ)
             DIV1=EW1N/ES1N
             DEL1IN=EW1N/ES1N-1.
             DIV2=EW1N/ES2N
             DEL2IN=EW1N/ES2N-1.
             ES1N=AA1_MY*DEXP(-BB1_MY/TTA)
             ES2N=AA2_MY*DEXP(-BB2_MY/TTA)
             EW1N=QQA*PP/(0.622+0.378*QQA)
             DIV3=EW1N/ES1N
             DEL1AD=EW1N/ES1N-1.
             DIV4=EW1N/ES2N
             DEL2AD=EW1N/ES2N-1.
             SUP2_OLD=DEL2IN

              IF(del1ad > 0.0 .or. (sum(FF1R)) > 0.0)THEN
! JacobS: commented for this version                
!                CALL Relaxation_Time(TT,QQ,PP,rhocgs(I,K,J),DEL1IN,DEL2IN, &
!                                      XL,VR1_Z(:,K),FF1R,RLEC,RO1BL, &
!                                      XI,VR2_Z,FF2R,RIEC,RO2BL, &
!                                      XS,VR3_Z(:,K),FF3R,RSEC,RO3BL, &
!                                      XG,VR4_Z(:,K),FF4R,RGEC,RO4BL, &
!                                      XH,VR5_Z(:,k),FF5R,RHEC,RO5BL, &
!                                      NKR,ICEMAX,COL,DT,NCOND,DTCOND)

                DELSUP1=(DEL1AD-DEL1IN)/NCOND
                DELSUP2=(DEL2AD-DEL2IN)/NCOND
                DELDIV1=(DIV3-DIV1)/NCOND
                DELDIV2=(DIV4-DIV2)/NCOND
                
                DIFFU=1
                IF (DIV1.EQ.DIV3) DIFFU = 0
                IF (DIV2.EQ.DIV4) DIFFU = 0 

                DTNEW = 0.0
                DO IKL=1,NCOND
                  DTCOND = min(DT-DTNEW,DTCOND)
                  DTNEW = DTNEW + DTCOND

                  IF (DIFFU == 1)THEN
                    IF (DIFFU == 1)THEN
                        DEL1IN = DEL1IN+DELSUP1
                        DEL2IN = DEL2IN+DELSUP2
                        DIV1 = DIV1+DELDIV1
                        DIV2 = DIV2+DELDIV2
                    END IF
                    IF (DIV1.GT.DIV2.AND.TT.LE.265)THEN
                      DIFFU = 0
                    END IF
                    IF (DIFFU == 1)THEN
                      DEL1NR=A1_MYN*(100.*DIV1)
                      DEL2NR=A2_MYN*(100.*DIV2)
                      IF (DEL2NR.EQ.0)print*,'ikl = ',ikl
                      IF (DEL2NR.EQ.0)print*,'div1,div2 = ',div1,div2
                      IF (DEL2NR.EQ.0)print*,'i,j,k = ',i,j,k
                      !IF (DEL2NR.EQ.0)call wrf_error_fatal("fatal error in module_mp_fast_sbm (DEL2NR.EQ.0) , model stop ")
                      DEL12R=DEL1NR/DEL2NR
                      DEL12RD=DEL12R**DEL_BBR
                      EW1PN=AA1_MY*100.*DIV1*DEL12RD/100.
                      TT=-DEL_BB/DLOG(DEL12R)
                      QQ=0.622*EW1PN/(PP-0.378*EW1PN)


                      IF(DEL1IN .GT. 0.0 .OR. DEL2IN .GT. 0.0)THEN
! +------------------------------------------+
! Droplet nucleation :
! +------------------------------------------+
                          FF1IN(:) = FF1R(:)
                          
                          cldnucl_bf = 3.0*col*( sum(ff1in*(xl**2.0)) )/rhocgs(I,K,J)
                          Is_This_CloudBase = 0
                          W_Stag_My = 0.0d0
                          
                          CALL JERNUCL01_KS(FF1IN,FCCN,FCCN_nucl 		        &
                                            ,XL,TT,QQ       					      &
                                            ,rhocgs(I,K,J),pcgs(I,K,J) 			&
                                            ,DEL1IN,DEL2IN     			        &
                                            ,COL 								            &
                                            ,RCCN,DROPRADII,NKR,NKR_aerosol &
                                            ,W_Stag_My,Is_This_CloudBase,RO_SOLUTE,IONS,MWAERO &
                                            ,I,J,K)

                          cldnucl_af = 3.0*col*( sum(ff1in*(xl**2.0)) )/rhocgs(I,K,J)
                          del_cldnucl_sum =  del_cldnucl_sum + (cldnucl_af - cldnucl_bf)

                          FF1R(:) = FF1IN(:)
                      END IF

                      FMAX1=0.0
                      DO KR=1,NKR
                        FF1IN(KR)=FF1R(KR)
                        FMAX1=AMAX1(FF1R(KR),FMAX1)
                      END DO
                      ISYM1 = 0
                      IF(FMAX1 > 0)ISYM1 = 1

!---YZ2020 / JacobS -------------  @
#ifdef SBM_DIAG
                       totlbf_diffu = sum(3.0*col*xl(1:krdrop-1)*xl(1:krdrop-1)*ff1r(1:krdrop-1)/rhocgs(i,k,j))
                       totrbf_diffu = sum(3.0*col*xl(krdrop:nkr)*xl(krdrop:nkr)*ff1r(krdrop:nkr)/rhocgs(i,k,j))
#endif
!----------------------------------@

                      ce_bf = 3.0*col*( sum(ff1r*(xl**2.0)) )/rhocgs(I,K,J)                          
                      IF(ISYM1==1 .AND. ((TT-273.15)>-0.187 .OR.(sum(ISYM2)==0 .AND. &
                          ISYM3==0 .AND. ISYM4==0 .AND. ISYM5==0)))THEN
                          ! ... only warm phase
                          CALL ONECOND1(TT,QQ,PP,rhocgs(I,K,J) &
                                        ,VR1_Z(:,K),pcgs(I,K,J) &
                                        ,DEL1IN,DEL2IN,DIV1,DIV2 &
                                        ,FF1R,FF1IN,XL,RLEC,RO1BL &
                                        ,AA1_MY,BB1_MY,AA2_MY,BB2_MY &
                                        ,C1_MEY,C2_MEY &
                                        ,COL,DTCOND,ICEMAX,NKR,ISYM1 &
                                        ,ISYM2,ISYM3,ISYM4,ISYM5,I,J,K,W(i,k,j),DX,Itimestep,CCN_reg)
                      END IF
                      ce_af = 3.0*col*( sum(ff1r*(xl**2.0)) )/rhocgs(I,K,J)               
                      del_ce_sum = del_ce_sum + (ce_af - ce_bf)
!---YZ2020 / JacobS :accumalate the rates diff at every ncond step ---------@
#ifdef SBM_DIAG
                      totlaf_diffu = sum(3.0*col*xl(1:krdrop-1)*xl(1:krdrop-1)*ff1r(1:krdrop-1)/rhocgs(i,k,j))
                      totraf_diffu = sum(3.0*xl(krdrop:nkr)*xl(krdrop:nkr)*ff1r(krdrop:nkr)/rhocgs(i,k,j))
                      del_difful_sum = del_difful_sum + (totlaf_diffu - totlbf_diffu)
                      del_diffur_sum = del_diffur_sum + (totraf_diffu - totrbf_diffu)
#endif
!----------------------------------------------------------@
                      END IF ! DIFF.NE.0
                  END IF 	! DIFFU.NE.0
                END DO ! NCOND - end of NCOND loop

                !... CCN_regenaration
                n_reg_ccn_bf = col*sum(FCCN)
                if(ICCN_reg == 1 .and. CCN_reg > 0.0)then
                  call CCN_regeneration(NKR,COL,CCN_reg,FCCN,FCCN_nucl,Del_CCNreg,1,I,J,K)
                endif
                n_reg_ccn_af = col*sum(FCCN)
! +----------------------------------+
! Collision-Coalescence
! +----------------------------------+
! [JS]: Here we save several state fields to be used in the Autoconv - DNN work.
!       Almost the MP state can be derived from the DSD, we save the DSD as well 

                sbm_output_container(i,k,j,p_nc_autoconv) = 3.0*col*sum(ff1r(1:krdrop)*xl(1:krdrop))/rhocgs(I,K,J)*1.0e3                      ! NC [#/kg]
                sbm_output_container(i,k,j,p_qc_autoconv) = 3.0*col*sum(ff1r(1:krdrop)*xl(1:krdrop)*xl(1:krdrop))/rhocgs(I,K,J)               ! QC [kg/kg]
                sbm_output_container(i,k,j,p_qr_autoconv) = 3.0*col*sum(ff1r(krdrop+1:nkr)*xl(krdrop+1:nkr) & 
                  *xl(krdrop+1:nkr))/rhocgs(I,K,J)   ! QR [kg/kg]
                sbm_output_container(i,k,j,p_nr_autoconv) = 3.0*col*sum(ff1r(krdrop+1:nkr)*xl(krdrop+1:nkr))/rhocgs(I,K,J)*1.0e3              ! NR [#/kg]
                sbm_output_container(i,k,j,p_qv_autoconv) = QQ                                                                                ! Qv [kg/kg]
                sbm_output_container(i,k,j,p_t_autoconv) = TT/pi_phy(i,k,j)                                                                   ! Potential Temp [K]
                if (k < kte) then
                  sbm_output_container(i,k,j,p_w_autoconv) = 0.5*(W(i,k,j) + W(i,k+1,j))                                                      ! W [m/s]
                endif
                krr = 0
                do kr = p_ff1i01_bfcc,p_ff1i33_bfcc
                  krr = krr + 1  
                  sbm_output_container(i,k,j,kr) = 3.0*col*ff1r(krr)*xl(krr)*xl(krr)/rhocgs(I,K,J)                                            ! Size distribution [#/kg]                        
                enddo

                  DO IKL = 1,NCOLL
                    IF ( TT >= 273.15 ) THEN
                      CALL COAL_BOTT_NEW_WARM (FF1R,TT,QQ,PP, 					                &
                                              rhocgs(I,K,J),dt_coll,TCRIT,TTCOAL, 	    &           
                                              DEL1IN, DEL2IN, 			        	          &
                                              I,J,K,Itimestep,KRDROP)
                    END IF
                  END DO ! NCOLL - end of NCOLL loop

!---YZ2020:divided by dt for process rate-----------------@
#ifdef SBM_DIAG
                  difful_tend(i,k,j)    = difful_tend(i,k,j)  + del_difful_sum/dt !g/g/s
                  diffur_tend(i,k,j)    = diffur_tend(i,k,j)  + del_diffur_sum/dt !g/g/s            
                  tempdiffl(i,k,j) = ttdiffl/dt  ! K s-1
                  n_reg_ccn(i,k,j) = (n_reg_ccn_af - n_reg_ccn_bf)/DT ! in [#/cm3/s]
#endif
!---------------------------------------------------------@

                  T_new(i,k,j) = TT
                  qv(i,k,j) = QQ
                   
            ! in case Sw,mass  
            ENDIF 
        ! in case T_OLD(I,K,J) > 0.0d0
        END IF
     
        ! ... Process rate (integrated)
        LH_rate(i,k,j) = LH_rate(i,k,j) +  lh_ce_1/dt
        CE_rate(i,k,j) = CE_rate(i,k,j) +    del_ce_sum/dt
        CldNucl_rate(i,k,j) = CldNucl_rate(i,k,j) + del_cldnucl_sum/dt

! [JS]: Seperate output container for process rates output from CC: Autoconv, Accretion, Rain selfcollect
        dbl_orhocgs = 1.0d0/dble(rhocgs(I,K,J))
        dbl_odt = 1.0d0/dble(dt)         
        sbm_output_container(i,k,j,p_auto_cldmsink_b)     = auto_cld_msink_b   * dbl_orhocgs*dbl_odt             ! g/g/s
        sbm_output_container(i,k,j,p_auto_cldnsink_b)     = auto_cld_nsink_b   * dbl_orhocgs*1.0d3*dbl_odt       ! #/kg/s
        sbm_output_container(i,k,j,p_accr_cldmsink_b)     = accr_cld_msink_b   * dbl_orhocgs*dbl_odt             ! g/g/s
        sbm_output_container(i,k,j,p_accr_cldnsink_b)     = accr_cld_nsink_b   * dbl_orhocgs*1.0d3*dbl_odt       ! #/kg/s     
        sbm_output_container(i,k,j,p_selfc_rainnchng_b)   = selfc_rain_nchng_b *dbl_orhocgs*1.0d3*dbl_odt        ! #/kg/s
        

        ! Update temperature at the end of MP
        th_phy(i,k,j) = t_new(i,k,j)/pi_phy(i,k,j) 

        ! ... Drops
        KRR = 0
        DO kr = p_ff1i01,p_ff1i33
        KRR = KRR+1
        chem_new(I,K,J,KR) = FF1R(KRR)
        END DO
        ! ... CCN
        KRR = 0
        DO kr=p_ff8i01,p_ff8i33
          KRR=KRR+1
          chem_new(I,K,J,KR)=FCCN(KRR)
        END DO
        ! ... Nucleated CCN
        KRR = 0
        DO kr=p_ff8in01,p_ff8in33
            KRR=KRR+1
            chem_new(I,K,J,KR) = FCCN_nucl(KRR)
        END DO

       END DO
    END DO
  END DO
  
  do j=jts,jte
    do k=kts,kte
      do i=its,ite
       th_old(i,k,j)=th_phy(i,k,j)
       qv_old(i,k,j)=qv(i,k,j)
      end do
    end do
  end do

! +-----------------------------+
! Hydrometeor Sedimentation
! +-----------------------------+ 
    do j = jts,jte
      do i = its,ite
! ... Drops ...
        do k = kts,kte
          rhocgs_z(k)=rhocgs(i,k,j)
          pcgs_z(k)=pcgs(i,k,j)
          zcgs_z(k)=zcgs(i,k,j)
          vrx(k,:)=vr1_z3D(:,i,k,j)
          krr = 0
          do kr=p_ff1i01,p_ff1i33
            krr=krr+1
            ffx_z(k,krr)=chem_new(i,k,j,kr)/rhocgs(i,k,j)
          end do
        end do
        call FALFLUXHUCM_Z(ffx_z,VRX,RHOCGS_z,PCGS_z,ZCGS_z,DT,kts,kte,nkr)
        do k = kts,kte
          krr = 0
          do kr=p_ff1i01,p_ff1i33
            krr=krr+1
            chem_new(i,k,j,kr)=ffx_z(k,krr)*rhocgs(i,k,j)
          end do
        end do
      end do
    end do

  ! ... Output block
    DO j = jts,jte
      DO k = kts,kte
        DO i = its,ite
          QC(I,K,J) = 0.0
          QR(I,K,J) = 0.0
          QNC(I,K,J) = 0.0
          QNR(I,K,J) = 0.0
          QNA(I,K,J) = 0.0
          QNA_nucl(I,K,J) = 0.0

          tt= th_phy(i,k,j)*pi_phy(i,k,j)

          ! ... Drop output
          KRR = 0
          DO KR = p_ff1i01,p_ff1i33
            KRR=KRR+1
            IF (KRR < KRDROP)THEN
              QC(I,K,J) = QC(I,K,J) &
              + (1./RHOCGS(I,K,J))*COL*chem_new(I,K,J,KR)*XL(KRR)*XL(KRR)*3
              QNC(I,K,J) = QNC(I,K,J) &
              + COL*chem_new(I,K,J,KR)*XL(KRR)*3.0/rhocgs(I,K,J)*1000.0 ! #/kg
            ELSE
              QR(I,K,J) = QR(I,K,J) &
              + (1./RHOCGS(I,K,J))*COL*chem_new(I,K,J,KR)*XL(KRR)*XL(KRR)*3.0
              QNR(I,K,J) = QNR(I,K,J) &
              + COL*chem_new(I,K,J,KR)*XL(KRR)*3.0/rhocgs(I,K,J)*1000.0 ! #/kg
            END IF
          END DO

          ! ... Aerosols output 
          KRR = 0
          DO  KR = p_ff8i01,p_ff8i33
            KRR = KRR + 1
            QNA(I,K,J) = QNA(I,K,J) &
                    + COL*chem_new(I,K,J,KR)/rhocgs(I,K,J)*1000.0   ! #/kg
          END DO

          ! ... Nucleated aerosols output
          KRR = 0
          DO  KR = p_ff8in01,p_ff8in33
              KRR = KRR + 1
              QNA_nucl(I,K,J) = QNA_nucl(I,K,J) &
                  + COL*chem_new(I,K,J,KR)/rhocgs(I,K,J)*1000.0   ! #/kg
          END DO
 		    END DO
      END DO
    END DO

 998   format(' ',10(f10.1,1x))

  DO j = jts,jte
    DO i = its,ite
      RAINNCV(I,J) = 0.0
      krr = 0
      DO KR=p_ff1i01,p_ff1i33
        krr=krr+1
        DELTAW = VR1_Z3D(KRR,I,1,J)
        RAINNC(I,J) = RAINNC(I,J) &
          +10.0*(3./RO1BL(KRR))*COL*DT*DELTAW* &
          chem_new(I,1,J,KR)*XL(KRR)*XL(KRR)
        RAINNCV(I,J) = RAINNCV(I,J) &
          +10.0*(3./RO1BL(KRR))*COL*DT*DELTAW* &
          chem_new(I,1,J,KR)*XL(KRR)*XL(KRR)
      END DO
     
! ..........................................
! ... Polarimetric Forward Radar Operator
! ..........................................
      if ( PRESENT (diagflag) ) then
        if( diagflag .and. IPolar_HUCM ) then

          !*** In the future we can adopt the Polarimetric simulator to this warm phase ***!

        ! diagflag .and. IPolar_HUCM
        endif
      ! PRESENT(diagflag)
      endif

   ! cycle by I
   END DO
 ! cycle by J
 END DO

   if (conserv)then
 		  DO j = jts,jte
 		     DO i = its,ite
             DO k = kts,kte
              
         		  rhocgs(I,K,J)=rho_phy(I,K,J)*0.001
              ! ... Drops  
              krr=0
         		  DO KR=p_ff1i01,p_ff1i33
         			    krr=krr+1
         		       chem_new(I,K,J,KR)=chem_new(I,K,J,KR)/RHOCGS(I,K,J)*COL*XL(KRR)*XL(KRR)*3.0
         		        if (qc(i,k,j)+qr(i,k,j).lt.1.e-13)chem_new(I,K,J,KR)=0.0
               END DO
         		  ! ... CCN
         		  KRR=0
         		  DO KR=p_ff8i01,p_ff8i33
         		   KRR=KRR+1
         		   chem_new(I,K,J,KR)=chem_new(I,K,J,KR)/RHOCGS(I,K,J)*1000.0
               END DO
              ! ... Nucleated CCN
               KRR=0
               DO KR=p_ff8in01,p_ff8in33
                KRR=KRR+1
                chem_new(I,K,J,KR)=chem_new(I,K,J,KR)/RHOCGS(I,K,J)*1000.0
               END DO 

            END DO
         END DO
      END DO
    endif

   RETURN
   END SUBROUTINE WARM_SBM
 ! +-------------------------------------------------------------+
   SUBROUTINE FALFLUXHUCM_Z(chem_new,VR1,RHOCGS,PCGS,ZCGS,DT, &
   						               kts,kte,nkr)

     IMPLICIT NONE

 	   integer,intent(in) :: kts,kte,nkr
 	   double precision,intent(inout) :: chem_new(:,:)
 	   double precision,intent(in) :: rhocgs(:),pcgs(:),zcgs(:),VR1(:,:),DT

 	  ! ... Locals
 	  integer :: I,J,K,KR
    double precision :: TFALL,DTFALL,VFALL(KTE),DWFLUX(KTE)
    integer :: IFALL,N,NSUB

 ! FALLING FLUXES FOR EACH KIND OF CLOUD PARTICLES: C.G.S. UNIT
 ! ADAPTED FROM GSFC CODE FOR HUCM
 !  The flux at k=1 is assumed to be the ground so FLUX(1) is the
 ! flux into the ground. DWFLUX(1) is at the lowest half level where
 ! Q(1) etc are defined. The formula for FLUX(1) uses Q(1) etc which
 ! is actually half a grid level above it. This is what is meant by
 ! an upstream method. Upstream in this case is above because the
 ! velocity is downwards.
 ! USE UPSTREAM METHOD (VFALL IS POSITIVE)

       DO KR=1,NKR
        IFALL=0
        DO k = kts,kte
           IF(chem_new(K,KR).GE.1.E-20)IFALL=1
        END DO
        IF (IFALL.EQ.1)THEN
         TFALL=1.E10
         DO K=kts,kte
          ! [KS] VFALL(K) = VR1(K,KR)*SQRT(1.E6/PCGS(K))
 		       VFALL(K) = VR1(K,KR) ! ... [KS] : The pressure effect is taken into account at the beggining of the calculations
           TFALL=AMIN1(TFALL,ZCGS(K)/(VFALL(K)+1.E-20))
         END DO
         IF(TFALL.GE.1.E10)STOP
         NSUB=(INT(2.0*DT/TFALL)+1)
         DTFALL=DT/NSUB

         DO N=1,NSUB
           DO K=KTS,KTE-1
             DWFLUX(K)=-(RHOCGS(K)*VFALL(K)*chem_new(k,kr)- &
             RHOCGS(K+1)* &
             VFALL(K+1)*chem_new(K+1,KR))/(RHOCGS(K)*(ZCGS(K+1)- &
             ZCGS(K)))
           END DO
 ! NO Z ABOVE TOP, SO USE THE SAME DELTAZ
           DWFLUX(KTE)=-(RHOCGS(KTE)*VFALL(KTE)* &
      &                 chem_new(kte,kr))/(RHOCGS(KTE)*(ZCGS(KTE)-ZCGS(KTE-1)))
           DO K=kts,kte
            chem_new(k,kr)=chem_new(k,kr)+DWFLUX(K)*DTFALL
           END DO
         END DO
        END IF
       END DO

       RETURN
       END SUBROUTINE FALFLUXHUCM_Z
 ! +----------------------------------+
   SUBROUTINE WARM_HUCMINIT(DT, ccncon1,radius_mean1,sig1, &
    ccncon2,radius_mean2,sig2, &
    ccncon3,radius_mean3,sig3)
 !	  USE module_domain
 !	  USE module_dm

 	  IMPLICIT NONE

    double precision,intent(in) :: DT
    double precision,intent(in) :: ccncon1,radius_mean1,sig1
    double precision,intent(in) :: ccncon2,radius_mean2,sig2
    double precision,intent(in) :: ccncon3,radius_mean3,sig3

    LOGICAL , EXTERNAL      :: wrf_dm_on_monitor
    LOGICAL :: opened
    CHARACTER*80 errmess
    integer :: I,J,KR,IType,HUJISBM_UNIT1
    double precision :: dlnr,ax,deg01,CONCCCNIN,CONTCCNIN

 	  character(len=256),parameter :: dir_43 = "SBM_input_43", dir_33 = "SBM_input_33"
 	  character(len=256) :: input_dir,Fname

 	 if(nkr == 33) input_dir = trim(dir_33)
 	 if(nkr == 43) input_dir = trim(dir_43)

     !call wrf_message(" FAST SBM: INITIALIZING WRF_HUJISBM ")
    ! call wrf_message(" FAST SBM: ****** WRF_HUJISBM ******* ")

 ! LookUpTable #1
 ! +-------------------------------------------------------+
 	if (.NOT. ALLOCATED(bin_mass)) ALLOCATE(bin_mass(nkr))
 	if (.NOT. ALLOCATED(tab_colum)) ALLOCATE(tab_colum(nkr))
 	if (.NOT. ALLOCATED(tab_dendr)) ALLOCATE(tab_dendr(nkr))
 	if (.NOT. ALLOCATED(tab_snow)) ALLOCATE(tab_snow(nkr))
 	if (.NOT. ALLOCATED(bin_log)) ALLOCATE(bin_log(nkr))

 	dlnr=dlog(2.d0)/(3.d0)

 	hujisbm_unit1 = -1
 	IF ( .TRUE.) THEN
 		DO i = 20,99
 			INQUIRE ( i , OPENED = opened )
 			IF ( .NOT. opened ) THEN
 				hujisbm_unit1 = i
 				GOTO 2060
 			ENDIF
 		ENDDO
 	2060  CONTINUE
 	ENDIF

#if (defined(DM_PARALLEL))
 		!CALL wrf_dm_bcast_bytes( hujisbm_unit1 , IWORDSIZE )
#endif

 	IF ( hujisbm_unit1 < 0 ) THEN
     	!CALL wrf_error_fatal ( 'module_mp_FAST-SBM: Table-1 -- FAST_SBM_INIT: '// 			&
 			!				              'Can not find unused fortran unit to read in lookup table, model stop' )
 	ENDIF

 	IF ( .TRUE. ) THEN
 			WRITE(errmess, '(A,I2)') 'module_mp_FAST-SBM : Table-1 -- opening "BLKD_SDC.dat" on unit',hujisbm_unit1
 			!CALL wrf_debug(150, errmess)
 			OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/BLKD_SDC.dat",FORM="FORMATTED",STATUS="OLD",ERR=2070)
 			DO kr=1,NKR
 				READ(hujisbm_unit1,*) bin_mass(kr),tab_colum(kr),tab_dendr(kr),tab_snow(kr)
 				bin_log(kr) = log10(bin_mass(kr))
 			ENDDO
 	ENDIF
#define DM_BCAST_MACRO_R4(A) CALL wrf_dm_bcast_bytes(A, size(A)*R4SIZE)
#define DM_BCAST_MACRO_R8(A) CALL wrf_dm_bcast_bytes(A, size(A)*R8SIZE)
#define DM_BCAST_MACRO_R16(A) CALL wrf_dm_bcast_bytes(A, size(A)*R16SIZE)

#if (defined(DM_PARALLEL))
    DM_BCAST_MACRO_R8(bin_mass)
 	  DM_BCAST_MACRO_R8(tab_colum)
 	  DM_BCAST_MACRO_R8(tab_dendr)
 	  DM_BCAST_MACRO_R8(tab_snow)
 	  DM_BCAST_MACRO_R8(bin_log)
#endif

 print *, 'FAST_SBM_INIT : succesfull reading Table-1'
  !CALL wrf_debug(000, errmess)
 ! +-----------------------------------------------------------------------+

 ! LookUpTable #2
 ! +----------------------------------------------+
     if (.NOT. ALLOCATED(RLEC)) ALLOCATE(RLEC(nkr))
     if (.NOT. ALLOCATED(RIEC)) ALLOCATE(RIEC(nkr,icemax))
     if (.NOT. ALLOCATED(RSEC)) ALLOCATE(RSEC(nkr))
     if (.NOT. ALLOCATED(RGEC)) ALLOCATE(RGEC(nkr))
     if (.NOT. ALLOCATED(RHEC)) ALLOCATE(RHEC(nkr))

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
         DO i = 31,99
             INQUIRE ( i , OPENED = opened )
             IF ( .NOT. opened ) THEN
                 hujisbm_unit1 = i
                 GOTO 2061
             ENDIF
         ENDDO
     2061  CONTINUE
     ENDIF

#if (defined(DM_PARALLEL))
 	!CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )
#endif
     IF ( hujisbm_unit1 < 0 ) THEN
         !CALL wrf_error_fatal ( 'module_mp_FAST-SBM: Table-2 -- FAST_SBM_INIT: '// 			&
         !                      'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF

 IF (.TRUE. ) THEN
 	print *, 'module_mp_FAST-SBM : Table-2 -- opening capacity.asc on unit',hujisbm_unit1
 	!CALL wrf_debug(150, errmess)
 	OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/capacity33.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
 	!OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/capacity43.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
 	900	FORMAT(6E13.5)
 	READ(hujisbm_unit1,900) RLEC,RIEC,RSEC,RGEC,RHEC
 END IF

#if (defined(DM_PARALLEL))
     DM_BCAST_MACRO_R4(RLEC)
     DM_BCAST_MACRO_R4(RIEC)
     DM_BCAST_MACRO_R4(RSEC)
     DM_BCAST_MACRO_R4(RGEC)
     DM_BCAST_MACRO_R4(RHEC)
#endif

     print *, 'FAST_SBM_INIT : succesfull reading Table-2'
     !CALL wrf_debug(000, errmess)
 ! +----------------------------------------------------------------------+

 ! LookUpTable #3
 ! +-----------------------------------------------+
     if (.NOT. ALLOCATED(XL)) ALLOCATE(XL(nkr))
     if (.NOT. ALLOCATED(XI)) ALLOCATE(XI(nkr,icemax))
     if (.NOT. ALLOCATED(XS)) ALLOCATE(XS(nkr))
     if (.NOT. ALLOCATED(XG)) ALLOCATE(XG(nkr))
     if (.NOT. ALLOCATED(XH)) ALLOCATE(XH(nkr))

     hujisbm_unit1 = -1
     IF (.TRUE. ) THEN
       DO i = 31,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2062
         ENDIF
       ENDDO
     2062 CONTINUE
     ENDIF

#if (defined(DM_PARALLEL))
    ! CALL wrf_dm_bcast_bytes ( hujisbm_unit1, IWORDSIZE )
#endif

     IF ( hujisbm_unit1 < 0 ) THEN
         !CALL wrf_error_fatal ( 'module_mp_FAST_SBM: Table-3 -- FAST_SBM_INIT: '// 		&
         !                     'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF
     IF ( .TRUE. ) THEN
         print *, 'module_mp_FAST_SBM : Table-3 -- opening masses.asc on unit ',hujisbm_unit1
         !CALL wrf_debug(150, errmess)
         OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/masses33.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         !OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/masses43.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         READ(hujisbm_unit1,900) XL,XI,XS,XG,XH
         CLOSE(hujisbm_unit1)
     ENDIF

#if (defined(DM_PARALLEL))
   	DM_BCAST_MACRO_R4(XL)
     DM_BCAST_MACRO_R4(XI)
     DM_BCAST_MACRO_R4(XS)
     DM_BCAST_MACRO_R4(XG)
     DM_BCAST_MACRO_R4(XH)
#endif

      print *, 'FAST_SBM_INIT : succesfull reading Table-3'
      !CALL wrf_debug(000, errmess)
 ! +-------------------------------------------------------------------------+

 ! LookUpTable #4
 ! TERMINAL VELOSITY :
 ! +---------------------------------------------------+
     if (.NOT. ALLOCATED(VR1)) ALLOCATE(VR1(nkr))
     if (.NOT. ALLOCATED(VR2)) ALLOCATE(VR2(nkr,icemax))
     if (.NOT. ALLOCATED(VR3)) ALLOCATE(VR3(nkr))
     if (.NOT. ALLOCATED(VR4)) ALLOCATE(VR4(nkr))
     if (.NOT. ALLOCATED(VR5)) ALLOCATE(VR5(nkr))

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
       DO i = 31,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2063
         ENDIF
       ENDDO
     2063   CONTINUE
     ENDIF

#if (defined(DM_PARALLEL))
    ! CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )
#endif
     IF ( hujisbm_unit1 < 0 ) THEN
        ! CALL wrf_error_fatal ( 'module_mp_FAST_SBM: Table-4 -- FAST_SBM_INIT: '// 										&
         !                        'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF

     IF ( .TRUE. ) THEN
         print *, 'module_mp_FAST_SBM : Table-4 -- opening termvels.asc on unit ',hujisbm_unit1
         !CALL wrf_debug(150, errmess)
         OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/termvels33_corrected.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         !OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/termvels43_corrected.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         READ(hujisbm_unit1,900) VR1,VR2,VR3,VR4,VR5
        CLOSE(hujisbm_unit1)
     ENDIF

#if (defined(DM_PARALLEL))
 	  DM_BCAST_MACRO_R4(VR1)
    DM_BCAST_MACRO_R4(VR2)
    DM_BCAST_MACRO_R4(VR3)
    DM_BCAST_MACRO_R4(VR4)
    DM_BCAST_MACRO_R4(VR5)
#endif
     print *, 'FAST_SBM_INIT : succesfull reading Table-4'
     !CALL wrf_debug(000, errmess)
 ! +----------------------------------------------------------------------+


 ! LookUpTable #5
 ! CONSTANTS :
 ! +---------------------------------------------------+
    
     !*** Do not need it here ***!

 ! +----------------------------------------------------------------------+

 ! LookUpTable #6
 ! KERNELS DEPENDING ON PRESSURE :
 ! +------------------------------------------------------------------+
     if (.NOT. ALLOCATED(YWLL_1000MB)) ALLOCATE(YWLL_1000MB(nkr,nkr))
     if (.NOT. ALLOCATED(YWLL_750MB)) ALLOCATE(YWLL_750MB(nkr,nkr))
     if (.NOT. ALLOCATED(YWLL_500MB)) ALLOCATE(YWLL_500MB(nkr,nkr))

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
       DO i = 31,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2066
         ENDIF
       ENDDO
       hujisbm_unit1 = -1
     2066     CONTINUE
     ENDIF

#if (defined(DM_PARALLEL))
 	!	CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )
#endif
     IF ( hujisbm_unit1 < 0 ) THEN
        ! CALL wrf_error_fatal ( 'module_mp_FAST_SBM: Table-6 -- FAST_SBM_INIT: '// 			&
        !                         'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF
     IF ( .TRUE. ) THEN
         print *, 'module_mp_FAST_SBM : Table-6 -- opening kernels_z.asc on unit  ',hujisbm_unit1
         !CALL wrf_debug(150, errmess)
         Fname = trim(input_dir)//'/kernLL_z33.asc'
         !Fname = trim(input_dir)//'/kernLL_z43.asc'
         OPEN(UNIT=hujisbm_unit1,FILE=Fname,FORM="FORMATTED",STATUS="OLD",ERR=2070)
         READ(hujisbm_unit1,900) YWLL_1000MB,YWLL_750MB,YWLL_500MB
         CLOSE(hujisbm_unit1)
     END IF

   	DO I=1,NKR
   		DO J=1,NKR
   			IF(I > 33 .OR. J > 33) THEN
   				YWLL_1000MB(I,J) = 0.0
   				YWLL_750MB(I,J) =  0.0
   				YWLL_500MB(I,J) =  0.0
   			ENDIF
   		ENDDO
   	ENDDO

#if (defined(DM_PARALLEL))
    DM_BCAST_MACRO_R4(YWLL_1000MB)
    DM_BCAST_MACRO_R4(YWLL_750MB)
    DM_BCAST_MACRO_R4(YWLL_500MB)
#endif

     print *, 'FAST_SBM_INIT : succesfull reading Table-6'
     !CALL wrf_debug(000, errmess)
 ! +-----------------------------------------------------------------------+

 ! LookUpTable #7
 ! COLLISIONS KERNELS :
 ! +-----------------------------------------------------------------------+

     ! No ice-ice / ice-liquid here


     print *,'FAST_SBM_INIT : succesfull reading Table-7'
     !CALL wrf_debug(000, errmess)
 ! +-----------------------------------------------------------------------+

 ! LookUpTable #8
 ! BULKDENSITY:
 ! +--------------------------------------------------------------+
     if (.NOT. ALLOCATED(RO1BL)) ALLOCATE(RO1BL(nkr))
     if (.NOT. ALLOCATED(RO2BL)) ALLOCATE(RO2BL(nkr,icemax))
     if (.NOT. ALLOCATED(RO3BL)) ALLOCATE(RO3BL(nkr))
     if (.NOT. ALLOCATED(RO4BL)) ALLOCATE(RO4BL(nkr))
     if (.NOT. ALLOCATED(RO5BL)) ALLOCATE(RO5BL(nkr))

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
       DO i = 31,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2068
         ENDIF
       ENDDO
     2068     CONTINUE
     ENDIF

#if (defined(DM_PARALLEL))
  !   CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )
#endif
     IF ( hujisbm_unit1 < 0 ) THEN
        ! CALL wrf_error_fatal ( 'module_mp_FAST_SBM: Table-8 -- FAST_SBM_INIT: '// 			&
        !                         'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF
     IF ( .TRUE. ) THEN
         print *,'module_mp_WRFsbm : Table-8 -- opening bulkdens.asc on unit ',hujisbm_unit1
         !CALL wrf_debug(150, errmess)
         OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/bulkdens33.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         !OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/bulkdens43.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         READ(hujisbm_unit1,900) RO1BL,RO2BL,RO3BL,RO4BL,RO5BL
         CLOSE(hujisbm_unit1)
     END IF

#if (defined(DM_PARALLEL))
 	    DM_BCAST_MACRO_R4(RO1BL)
      DM_BCAST_MACRO_R4(RO2BL)
      DM_BCAST_MACRO_R4(RO3BL)
      DM_BCAST_MACRO_R4(RO4BL)
      DM_BCAST_MACRO_R4(RO5BL)
#endif
     print *, 'FAST_SBM_INIT : succesfull reading Table-8'
     !CALL wrf_debug(000, errmess)
 ! +----------------------------------------------------------------------+

 ! LookUpTable #9
 ! BULKRADII:
 ! +-----------------------------------------------------------+
     if (.NOT. ALLOCATED(RADXXO)) ALLOCATE(RADXXO(nkr,nhydro))
     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
       DO i = 31,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2069
         ENDIF
       ENDDO
     2069     CONTINUE
     ENDIF
#if (defined(DM_PARALLEL))
 	!	CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )
#endif
     IF ( hujisbm_unit1 < 0 ) THEN
      !CALL wrf_error_fatal ( 'module_mp_FAST_SBM: Table-9 -- FAST_SBM_INIT: '// 			&
      !                           'Can not find unused fortran unit to read in lookup table,model stop' )
     ENDIF
     IF ( .True. ) THEN
         print *, 'module_mp_WARM_SBM : Table-9 -- opening bulkradii.asc on unit',hujisbm_unit1
         !CALL wrf_debug(150, errmess)
         OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/bulkradii33.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         !OPEN(UNIT=hujisbm_unit1,FILE=trim(input_dir)//"/bulkradii43.asc",FORM="FORMATTED",STATUS="OLD",ERR=2070)
         READ(hujisbm_unit1,*) RADXXO
         CLOSE(hujisbm_unit1)
     END IF

#if (defined(DM_PARALLEL))
       DM_BCAST_MACRO_R4(RADXXO)
#endif
     print *, 'FAST_SBM_INIT : succesfull reading Table-9'
     !CALL wrf_debug(000, errmess)
 ! +-----------------------------------------------------------------------+

 ! LookUpTable #10
 ! Polar-HUCM Scattering Amplitudes Look-up table :
 ! +-----------------------------------------------------------------------+
 
 !--CK comments out--!    
 !    CALL LOAD_TABLES(NKR)  ! (KS) - Loading the scattering look-up-table

 ! ... (KS) - Broadcating Liquid drops
!#if (defined(DM_PARALLEL))
! Commented out for now (CK / JacobS)
!   	DM_BCAST_MACRO_R16 ( FAF1 )
!   	DM_BCAST_MACRO_R16 ( FBF1 )
!   	DM_BCAST_MACRO_R16 ( FAB1 )
!   	DM_BCAST_MACRO_R16 ( FBB1 )
   ! ... (KS) - Broadcating Snow
!   	DM_BCAST_MACRO_R16 ( FAF3 )
!   	DM_BCAST_MACRO_R16 ( FBF3 )
!   	DM_BCAST_MACRO_R16 ( FAB3 )
!   	DM_BCAST_MACRO_R16 ( FBB3 )
   ! ... (KS) - Broadcating Graupel
!   	DM_BCAST_MACRO_R16 ( FAF4 )
!   	DM_BCAST_MACRO_R16 ( FBF4 )
!   	DM_BCAST_MACRO_R16 ( FAB4 )
!   	DM_BCAST_MACRO_R16 ( FBB4 )
   ! ### (KS) - Broadcating Hail
!   	DM_BCAST_MACRO_R16 ( FAF5 )
!   	DM_BCAST_MACRO_R16 ( FBF5 )
!   	DM_BCAST_MACRO_R16 ( FAB5 )
!   	DM_BCAST_MACRO_R16 ( FBB5 )
 ! ### (KS) - Broadcating Temperature intervals
!   	CALL wrf_dm_bcast_integer ( temps_water , size ( temps_water ) )
!   	CALL wrf_dm_bcast_integer ( temps_fd , size ( temps_fd ) )
!   	CALL wrf_dm_bcast_integer ( temps_crystals , size ( temps_crystals ) )
!   	CALL wrf_dm_bcast_integer ( temps_snow , size ( temps_snow ) )
!   	CALL wrf_dm_bcast_integer ( temps_graupel , size ( temps_graupel ) )
!   	CALL wrf_dm_bcast_integer ( temps_hail , size ( temps_hail ) )
 ! ### (KS) - Broadcating Liquid fraction intervals
!   	DM_BCAST_MACRO_R4 ( fws_fd )
!   	DM_BCAST_MACRO_R4 ( fws_crystals )
!   	DM_BCAST_MACRO_R4 ( fws_snow )
!   	DM_BCAST_MACRO_R4 ( fws_graupel )
!   	DM_BCAST_MACRO_R4 ( fws_hail )
 ! ### (KS) - Broadcating Usetables array
! 	  CALL wrf_dm_bcast_integer ( usetables , size ( usetables ) * IWORDSIZE )
!#endif
  print *,'FAST_SBM_INIT : succesfull reading Table-10'
  !call wrf_message(errmess)
 ! +-----------------------------------------------------------------------+

 ! calculation of the mass(in mg) for categories boundaries :
   ax=2.d0**(1.0)

   do i=1,nkr
   	 xl_mg(i) = xl(i)*1.e3
      xs_mg(i) = xs(i)*1.e3
      xg_mg(i) = xg(i)*1.e3
      xh_mg(i) = xh(i)*1.e3
      xi1_mg(i) = xi(i,1)*1.e3
      xi2_mg(i) = xi(i,2)*1.e3
      xi3_mg(i) = xi(i,3)*1.e3
   enddo

   if (.NOT. ALLOCATED(IMA)) ALLOCATE(IMA(nkr,nkr))
   if (.NOT. ALLOCATED(CHUCM)) ALLOCATE(CHUCM(nkr,nkr))
   chucm  = 0.0d0
   ima = 0
   CALL courant_bott_KS(xl, nkr, chucm, ima, scal) ! ### (KS) : New courant_bott_KS (without XL_MG(0:nkr))
   print *, 'FAST_SBM_INIT : succesfull reading "courant_bott_KS" '
   !CALL wrf_debug(000, errmess)

  DEG01=1./3.
  CONCCCNIN=0.
  CONTCCNIN=0.
  if (.NOT. ALLOCATED(DROPRADII)) ALLOCATE(DROPRADII(NKR))
  DO KR=1,NKR
  DROPRADII(KR)=(3.0*XL(KR)/4.0/3.141593/1.0)**DEG01
  ENDDO

 ! +-------------------------------------------------------------+
 ! Allocating Aerosols Array
 ! +-------------------------+
 if (.NOT. ALLOCATED(FCCNR_MAR)) ALLOCATE(FCCNR_MAR(NKR_aerosol))
 if (.NOT. ALLOCATED(FCCNR_CON)) ALLOCATE(FCCNR_CON(NKR_aerosol))
 if (.NOT. ALLOCATED(XCCN)) ALLOCATE(XCCN(NKR_aerosol))
 if (.NOT. ALLOCATED(RCCN)) ALLOCATE(RCCN(NKR_aerosol))
 if (.NOT. ALLOCATED(Scale_CCN_Factor)) ALLOCATE(Scale_CCN_Factor)
 if (.NOT. ALLOCATED(FCCN)) ALLOCATE(FCCN(NKR_aerosol))
 if (.NOT. ALLOCATED(FCCN_nucl)) ALLOCATE(FCCN_nucl(NKR_aerosol))
 if (.NOT. ALLOCATED(FCCNR_obs)) ALLOCATE(FCCNR_obs(NKR_aerosol))
 if (.NOT. ALLOCATED(CCNR)) ALLOCATE(CCNR(NKR_aerosol))

! ... Initializing the FCCNR_MAR and FCCNR_CON
  FCCNR_CON = 0.0
  FCCNR_MAR = 0.0
  FCCNR_obs = 0.0
  FCCN_nucl = 0.0
  FCCN = 0.0
  Scale_CCN_Factor = 1.0
  XCCN = 0.0
  RCCN = 0.0

  IF(ILogNormal_modes_Aerosol == 1)THEN
    CALL LogNormal_modes_Aerosol(FCCNR_CON,FCCNR_MAR,NKR_aerosol,COL,XL,XCCN,RCCN,RO_SOLUTE,Scale_CCN_Factor,1, &
    ccncon1,radius_mean1,sig1, &
    ccncon2,radius_mean2,sig2, &
    ccncon3,radius_mean3,sig3)
    CALL LogNormal_modes_Aerosol(FCCNR_CON,FCCNR_MAR,NKR_aerosol,COL,XL,XCCN,RCCN,RO_SOLUTE,Scale_CCN_Factor,2, &
    ccncon1,radius_mean1,sig1, &
    ccncon2,radius_mean2,sig2, &
    ccncon3,radius_mean3,sig3)
 	  print *, 'FAST_SBM_INIT : succesfull reading "LogNormal_modes_Aerosol" '
    !CALL wrf_debug(000, errmess)
    !---YZ2020Mar:read aerosol size distribution from observation----@
  ELSE ! read an observed SD with a format of aerosol size (cm), dN (#cm-3) and dNdlogD for 33bins (Jinwe Fan)
    OPEN(UNIT=hujisbm_unit1,FILE="CCN_size_33bin.dat",FORM="FORMATTED",STATUS="OLD",ERR=2070)
    do KR=1,NKR
       READ(hujisbm_unit1,*) RCCN(KR),CCNR(KR),FCCNR_OBS(KR) !---aerosol size (cm), dN (# cm-3) and dNdlogD for 33bins
    end do
    CLOSE(hujisbm_unit1)
    !call wrf_message("FAST_SBM_INIT: succesfull reading aerosol SD from observation")
  ENDIF
 ! +-------------------------------------------------------------+

 	 if (.NOT. ALLOCATED(PKIJ)) ALLOCATE(PKIJ(JBREAK,JBREAK,JBREAK))
 	 if (.NOT. ALLOCATED(QKJ)) ALLOCATE(QKJ(JBREAK,JBREAK))
 	 if (.NOT. ALLOCATED(ECOALMASSM)) ALLOCATE(ECOALMASSM(NKR,NKR))
 	 if (.NOT. ALLOCATED(BRKWEIGHT)) ALLOCATE(BRKWEIGHT(JBREAK))
    PKIJ = 0.0e0
    QKJ = 0.0e0
    ECOALMASSM = 0.0d0
    BRKWEIGHT = 0.0d0
 	 CALL BREAKINIT_KS(PKIJ,QKJ,ECOALMASSM,BRKWEIGHT,XL,DROPRADII,BR_MAX,JBREAK,JMAX,NKR,VR1) ! Rain Spontanous Breakup
#if (defined(DM_PARALLEL))
 	 	DM_BCAST_MACRO_R4 (PKIJ)
    DM_BCAST_MACRO_R4 (QKJ)
#endif
 	  print *, 'FAST_SBM_INIT : succesfull reading BREAKINIT_KS" '
    !CALL wrf_debug(000, errmess)
  ! +--------------------------------------------------------------------------------------------------------------------+

    if (.NOT. ALLOCATED(cwll)) ALLOCATE(cwll(nkr,nkr))
    cwll(:,:) = 0.0e0

    call Kernals_KS(dt,nkr,7.6D6)


   100	FORMAT(10I4)
   101   FORMAT(3X,F7.5,E13.5)
   102	FORMAT(4E12.4)
   105	FORMAT(A48)
   106	FORMAT(A80)
   123	FORMAT(3E12.4,3I4)
   200	FORMAT(6E13.5)
   201   FORMAT(6D13.5)
   300	FORMAT(8E14.6)
   301   FORMAT(3X,F8.3,3X,E13.5)
   302   FORMAT(5E13.5)

  return
  2070  continue

      print *,                                         &
                 'module_mp_FAST_SBM_INIT: error opening hujisbm_DATA on unit,model stop ' &
                 &, hujisbm_unit1
      !CALL wrf_error_fatal(errmess)

  END SUBROUTINE WARM_HUCMINIT
 ! -----------------------------------------------------------------+
  subroutine Kernals_KS(dtime_coal,nkr,p_z)

  implicit none

  integer :: nkr
  double precision,intent(in) :: dtime_coal,p_z

  ! ### Locals
  integer :: i,j
  double precision,parameter :: p1=1.0e6,p2=0.75e6,p3=0.50e6,p4=0.3e6
  double precision :: dlnr, scal, dtimelnr, pdm, p_1, p_2, p_3, ckern_1, ckern_2, &
  					          ckern_3

 ! p1=1.00D6 dynes/cm^2 = 1000.0 mb
 ! p2=0.75D6 dynes/cm^2 =  750.0 mb
 ! p3=0.50D6 dynes/cm^2 =  500.0 mb
 ! p4=0.30D6 dynes/cm^2 =  300.0 mb

  scal = 1.0
 	dlnr = dlog(2.0d0)/(3.0d0*scal)
 	dtimelnr = dtime_coal*dlnr

 	p_1=p1
 	p_2=p2
 	p_3=p3
 	do i=1,nkr
 		do j=1,nkr
 			! 1. water - water
 			ckern_1 = YWLL_1000mb(i,j)
 			ckern_2 = YWLL_750mb(i,j)
 			ckern_3 = YWLL_500mb(i,j)
 			cwll(i,j) = ckern_z(p_z,p_1,p_2,p_3,ckern_1,ckern_2,ckern_3)*dtime_coal*dlnr
 		end do
 	end do

 	! ... ECOALMASSM is from "BreakIniit_KS"
 	DO I=1,NKR
 	 DO J=1,NKR
 		CWLL(I,J) = ECOALMASSM(I,J)*CWLL(I,J)
 	 END DO
  END DO

  return
  end subroutine Kernals_KS
 ! ------------------------------------------------------------+
  double precision function ckern_z (p_z,p_1,p_2,p_3,ckern_1,ckern_2,ckern_3)

 	implicit none

 	double precision,intent(in) :: p_z,p_1,p_2,p_3,ckern_1, &
 									                ckern_2,ckern_3

 	if(p_z>=p_1) ckern_z = ckern_1
 	!if(p_z==p_2) ckern_z=ckern_2
 	if(p_z<=p_3) ckern_z = ckern_3
 	if(p_z<p_1 .and. p_z>=p_2) ckern_z = ckern_2 + (ckern_1-ckern_2)*(p_z-p_2)/(p_1-p_2)
 	if(p_z<p_2 .and. p_z>p_3) ckern_z = ckern_3 + (ckern_2-ckern_3)*(p_z-p_3)/(p_2-p_3)

  return
  end function ckern_z
 ! +----------------------------------------------------------------------------+
   SUBROUTINE ONECOND1 &
				 & (TT,QQ,PP,ROR &
				 & ,VR1,PSINGLE &
				 & ,DEL1N,DEL2N,DIV1,DIV2 &
				 & ,FF1,PSI1,R1,RLEC,RO1BL &
				 & ,AA1_MY,BB1_MY,AA2_MY,BB2_MY &
				 & ,C1_MEY,C2_MEY &
				 & ,COL,DTCOND,ICEMAX,NKR,ISYM1 &
				   ,ISYM2,ISYM3,ISYM4,ISYM5,Iin,Jin,Kin,W_in,DX_in,Itimestep,CCN_reg)

        IMPLICIT NONE


       INTEGER NKR,ICEMAX, ISYM1, ISYM2(ICEMAX),ISYM3,ISYM4,ISYM5, Iin, Jin, Kin, &
 	  		  sea_spray_no_temp_change_per_grid, Itimestep
       double precision    COL,VR1(NKR),PSINGLE &
      &       ,AA1_MY,BB1_MY,AA2_MY,BB2_MY &
      &       ,DTCOND, W_in,DX_in,CCN_reg

       double precision C1_MEY,C2_MEY
       INTEGER I_ABERGERON,I_BERGERON, &
      & KR,ICE,ITIME,KCOND,NR,NRM, &
      & KLIMIT, &
      & KM,KLIMITL
       double precision AL1,AL2,D,GAM,POD, &
      & RV_MY,CF_MY,D_MYIN,AL1_MY,AL2_MY,ALC,DT0LREF,DTLREF, &
      & A1_MYN, BB1_MYN, A2_MYN, BB2_MYN,DT,DTT,XRAD, &
      & TPC1, TPC2, TPC3, TPC4, TPC5, &
      & EPSDEL, EPSDEL2,DT0L, DT0I,&
      & ROR, &
      & CWHUCM,B6,B8L,B8I, &
      & DEL1,DEL2,DEL1S,DEL2S, &
      & TIMENEW,TIMEREV,SFN11,SFN12, &
      & SFNL,SFNI,B5L,B5I,B7L,B7I,DOPL,DOPI,RW,RI,QW,PW, &
      & PI,QI,DEL1N0,DEL2N0,D1N0,D2N0,DTNEWL,DTNEWL1,D1N,D2N, &
      & DEL_R1,DT0L0,DT0I0, &
      & DTNEWL0, &
      & DTNEWL2
        double precision DT_WATER_COND,DT_WATER_EVAP

        INTEGER K

       double precision  FF1_OLD(NKR),SUPINTW(NKR)
       DOUBLE PRECISION DSUPINTW(NKR),DD1N,DB11_MY,DAL1,DAL2
       DOUBLE PRECISION COL3,RORI,TPN,TPS,QPN,QPS,TOLD,QOLD &
      &                  ,FI1_K,FI2_K,FI3_K,FI4_K,FI5_K &
      &                  ,R1_K,R2_K,R3_K,R4_K,R5_K &
      &                  ,FI1R1,FI2R2,FI3R3,FI4R4,FI5R5 &
      &                  ,RMASSLAA,RMASSLBB,RMASSIAA,RMASSIBB &
      &                  ,ES1N,ES2N,EW1N,ARGEXP &
      &                  ,TT,QQ,PP &
      &                  ,DEL1N,DEL2N,DIV1,DIV2 &
      &                  ,OPER2,OPER3,AR1,AR2

        DOUBLE PRECISION DELMASSL1

 ! DROPLETS

         double precision R1(NKR) &
      &           ,RLEC(NKR),RO1BL(NKR) &
      &           ,FI1(NKR),FF1(NKR),PSI1(NKR) &
      &           ,B11_MY(NKR),B12_MY(NKR)

 ! WORK ARRAYS

 ! NEW ALGORITHM OF MIXED PHASE FOR EVAPORATION


 	double precision DTIMEO(NKR),DTIMEL(NKR) &
      &           ,TIMESTEPD(NKR)

 ! NEW ALGORITHM (NO TYPE OF ICE)

 	double precision :: FL1(NKR),sfndummy(3),R1N(NKR),totccn_before, totccn_after
 	INTEGER :: IDROP

 	DOUBLE PRECISION :: R1D(NKR),R1ND(NKR)

 	OPER2(AR1)=0.622/(0.622+0.378*AR1)/AR1
 	OPER3(AR1,AR2)=AR1*AR2/(0.622+0.378*AR1)

 	DATA AL1 /2500./, AL2 /2834./, D /0.211/ &
      &      ,GAM /1.E-4/, POD /10./

 	DATA RV_MY,CF_MY,D_MYIN,AL1_MY,AL2_MY &
      &      /461.5,0.24E-1,0.211E-4,2.5E6,2.834E6/

 	DATA A1_MYN, BB1_MYN, A2_MYN, BB2_MYN &
      &      /2.53,5.42,3.41E1,6.13/

 	DATA TPC1, TPC2, TPC3, TPC4, TPC5 &
      &      /-4.0,-8.1,-12.7,-17.8,-22.4/


 	DATA EPSDEL, EPSDEL2 /0.1E-03,0.1E-03/

 	DATA DT0L, DT0I /1.E20,1.E20/

 	DOUBLE PRECISION :: DEL1_d , DEL2_d, RW_d , PW_d, RI_d, PI_d, D1N_d, D2N_d, &
 						VR1_d(NKR)

 sfndummy = 0.0
 B12_MY = 0.0
 B11_MY = 0.0

  I_ABERGERON=0
  I_BERGERON=0
  COL3=3.0*COL
 ITIME=0
 KCOND=0
 DT_WATER_COND=0.4
 DT_WATER_EVAP=0.4
 ITIME=0
 KCOND=0
 DT0LREF=0.2
 DTLREF=0.4

 NR=NKR
 NRM=NKR-1
 DT=DTCOND
 DTT=DTCOND
 XRAD=0.

  CWHUCM=0.
 XRAD=0.
 B6=CWHUCM*GAM-XRAD
 B8L=1./ROR
 B8I=1./ROR
 RORI=1./ROR

 ! ... CCN_regeneration
totccn_before = 0.0
totccn_before = sum(psi1(1:nkr)*r1(1:nkr))*3.0*col
 DO KR=1,NKR
    FF1_OLD(KR)=FF1(KR)
    SUPINTW(KR)=0.0
    DSUPINTW(KR)=0.0
 ENDDO

 TPN=TT
 QPN=QQ
 DO KR=1,NKR
     FI1(KR)=FF1(KR)
 END DO

 ! WARM MP (CONDENSATION OR EVAPORATION) (BEGIN)
 TIMENEW=0.
 ITIME=0

 TOLD = TPN
 QOLD = QPN
 R1D = R1
 R1ND = R1D
 SFNL = 0.0
 SFN11 = 0.0

 56  ITIME = ITIME+1
 TIMEREV = DT-TIMENEW
 TIMEREV = DT-TIMENEW
 DEL1 = DEL1N
 DEL2 = DEL2N
 DEL1S = DEL1N
 DEL2S = DEL2N
 TPS = TPN
 QPS = QPN

 IF(ISYM1 == 1)THEN
 	FL1 = 0.0
 	VR1_d = VR1
 	CALL JERRATE_KS &
 				(R1D,TPS,PP,VR1_d,RLEC,RO1BL,B11_MY,1,1,fl1,NKR,ICEMAX)
 	sfndummy(1)=SFN11
 	CALL JERTIMESC_KS(FI1,R1D,SFNDUMMY,B11_MY,B8L,1,NKR,ICEMAX,COL)
 	SFN11 = sfndummy(1)
 ENDIF

 SFN12 = 0.0
 SFNL = SFN11 + SFN12
 SFNI = 0.

 B5L=BB1_MY/TPS/TPS
 B5I=BB2_MY/TPS/TPS
 B7L=B5L*B6
 B7I=B5I*B6
 DOPL=1.+DEL1S
 DOPI=1.+DEL2S
 RW=(OPER2(QPS)+B5L*AL1)*DOPL*SFNL
 RI=(OPER2(QPS)+B5L*AL2)*DOPL*SFNI
 QW=B7L*DOPL
 PW=(OPER2(QPS)+B5I*AL1)*DOPI*SFNL
 PI=(OPER2(QPS)+B5I*AL2)*DOPI*SFNI
 QI=B7I*DOPI

 IF(RW.NE.RW .or. PW.NE.PW)THEN
    print*, 'NaN In ONECOND1'
    !call wrf_error_fatal("fatal error in ONECOND1 (RW or PW are NaN), model stop")
 ENDIF

 KCOND=10
 IF(DEL1N >= 0.0D0) KCOND=11

   IF(KCOND == 11) THEN
   	  DTNEWL = DT
      DTNEWL = DT
      DTNEWL = AMIN1(DTNEWL,TIMEREV)
      TIMENEW = TIMENEW + DTNEWL
      DTT = DTNEWL

   	  !IF (DTT < 0.0) call wrf_error_fatal("fatal error in ONECOND1-DEL1N>0:(DTT<0), model stop")

     	DEL1_d = DEL1
     	DEL2_d = DEL2
     	RW_d = RW
     	PW_d = PW
     	RI_d = RI
     	PI_d = PI

 	    CALL JERSUPSAT_KS(DEL1_d,DEL2_d,DEL1N,DEL2N, &
             					  RW_d,PW_d,RI_d,PI_d, &
             					  DTT,D1N_d,D2N_d,0.0d0,0.0d0, &
             					  ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)
     	DEL1 = DEL1_d
     	DEL2 = DEL2_d
     	RW = RW_d
     	PW = PW_d
     	RI = RI_d
     	PI = PI_d
     	D1N = D1N_d
     	D2N = D2N_d

     	IF(ISYM1 == 1)THEN
     		IDROP = ISYM1
     		CALL JERDFUN_KS(R1D, R1ND, B11_MY, FI1, PSI1, fl1, D1N, &
     						        ISYM1, 1, 1, TPN, IDROP, FR_LIM, FRH_LIM, ICEMAX, NKR, COL, 1, Iin, Jin ,Kin, Itimestep)
     	ENDIF

     	IF((DEL1.GT.0.AND.DEL1N.LT.0) &
       		&.AND.ABS(DEL1N).GT.EPSDEL) THEN
             		!call wrf_error_fatal ("fatal error in ONECOND1-1 (DEL1.GT.0.AND.DEL1N.LT.0), model stop")
     	ENDIF

    ! IN CASE : KCOND.EQ.11
    ELSE

 	    ! EVAPORATION - ONLY WATER
 	    ! IN CASE : KCOND.NE.11
    	DTIMEO = DT
      DTNEWL = DT
      DTNEWL = AMIN1(DTNEWL,TIMEREV)
      TIMENEW = TIMENEW + DTNEWL
      DTT = DTNEWL

 	    !IF (DTT < 0.0) call wrf_error_fatal("fatal error in ONECOND1-DEL1N<0:(DTT<0), model stop")

 	    DEL1_d = DEL1
 	    DEL2_d = DEL2
 	    RW_d = RW
 	    PW_d = PW
 	    RI_d = RI
 	    PI_d = PI
 	    CALL JERSUPSAT_KS(DEL1_d,DEL2_d,DEL1N,DEL2N, &
 					  RW_d,PW_d,RI_d,PI_d, &
 					  DTT,D1N_d,D2N_d,0.0d0,0.0d0, &
 					  ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)
     	DEL1 = DEL1_d
     	DEL2 = DEL2_d
     	RW = RW_d
     	PW = PW_d
     	RI = RI_d
     	PI = PI_d
     	D1N = D1N_d
     	D2N = D2N_d

      IF(ISYM1 == 1)THEN
 	      IDROP = ISYM1
 	      CALL JERDFUN_KS(R1D, R1ND, B11_MY, &
				              FI1, PSI1, fl1, D1N, &
 					            ISYM1, 1, 1, TPN, IDROP, FR_LIM, FRH_LIM, ICEMAX, NKR, COL, 1, Iin, Jin ,Kin, Itimestep)
      ENDIF

      IF((DEL1.LT.0.AND.DEL1N.GT.0) &
        .AND.ABS(DEL1N).GT.EPSDEL) THEN
         !call wrf_error_fatal ("fatal error in ONECOND1-2 (DEL1.LT.0.AND.DEL1N.GT.0), model stop")
      ENDIF

    ENDIF


 RMASSLBB=0.
 RMASSLAA=0.

 ! ... before JERNEWF (ONLY WATER)
 DO K=1,NKR
  FI1_K = FI1(K)
  R1_K = R1(K)
  FI1R1 = FI1_K*R1_K*R1_K
  RMASSLBB = RMASSLBB+FI1R1
 ENDDO
 RMASSLBB = RMASSLBB*COL3*RORI
 IF(RMASSLBB.LE.0.) RMASSLBB=0.
 ! ... after JERNEWF (ONLY WATER)
 DO K=1,NKR
  FI1_K=PSI1(K)
  R1_K=R1(K)
  FI1R1=FI1_K*R1_K*R1_K
  RMASSLAA=RMASSLAA+FI1R1
 END DO
 RMASSLAA=RMASSLAA*COL3*RORI
 IF(RMASSLAA.LE.0.) RMASSLAA=0.

 DELMASSL1 = RMASSLAA - RMASSLBB
 QPN = QPS - DELMASSL1
 DAL1 = AL1
 TPN = TPS + DAL1*DELMASSL1

 IF(ABS(DAL1*DELMASSL1) > 3.0 )THEN
 	print*,"ONECOND1-in(start)"
	print*,"I=",Iin,"J=",Jin,"Kin",Kin,"W",w_in,"DX",dx_in
 	print*,"DELMASSL1",DELMASSL1,"DT",DTT
 	print*,"DEL1N,DEL2N,DEL1,DEL2,D1N,D2N,RW,PW,RI,PI,DT"
 	print*,DEL1N,DEL2N,DEL1,DEL2,D1N,D2N,RW,PW,RI,PI,DTT
 	print*,"TPS",TPS,"QPS",QPS
	print*,'FI1 before',FI1,'PSI1 after',PSI1
 	print*,"ONECOND1-in(end)"
 	!call wrf_error_fatal ("fatal error in ONECOND1-in (ABS(DAL1*DELMASSL1) > 3.0), model stop")
 ENDIF

 ! ... SUPERSATURATION (ONLY WATER)
 ARGEXP=-BB1_MY/TPN
 ES1N=AA1_MY*DEXP(ARGEXP)
 ARGEXP=-BB2_MY/TPN
 ES2N=AA2_MY*DEXP(ARGEXP)
 EW1N=OPER3(QPN,PP)
 IF(ES1N == 0.0D0)THEN
          DEL1N=0.5
          DIV1=1.5
 ELSE
          DIV1 = EW1N/ES1N
          DEL1N = EW1N/ES1N-1.
 END IF
 IF(ES2N == 0.0D0)THEN
          DEL2N=0.5
          DIV2=1.5
 ELSE
          DEL2N = EW1N/ES2N-1.
          DIV2 = EW1N/ES2N
 END IF
 IF(ISYM1 == 1) THEN
 	DO KR=1,NKR
           SUPINTW(KR)=SUPINTW(KR)+B11_MY(KR)*D1N
           DD1N=D1N
           DB11_MY=B11_MY(KR)
           DSUPINTW(KR)=DSUPINTW(KR)+DB11_MY*DD1N
 	ENDDO
 ENDIF

 ! ... REPEATE TIME STEP (ONLY WATER: CONDENSATION OR EVAPORATION)
 IF(TIMENEW.LT.DT) GOTO 56

 57  CONTINUE

 IF(ISYM1 == 1) THEN
    CALL JERDFUN_NEW_KS (R1D,R1ND,SUPINTW, &
 					FF1_OLD,PSI1, &
 					TPN,IDROP,FR_LIM, NKR, COL,1,Iin,Jin,Kin,Itimestep)
 ENDIF ! in case ISYM1/=0

 RMASSLAA=0.0
 RMASSLBB=0.0

 DO K=1,NKR
  FI1_K=FF1_OLD(K)
  R1_K=R1(K)
  FI1R1=FI1_K*R1_K*R1_K
  RMASSLBB=RMASSLBB+FI1R1
 ENDDO
 RMASSLBB=RMASSLBB*COL3*RORI
 IF(RMASSLBB.LT.0.0) RMASSLBB=0.0

 DO K=1,NKR
  FI1_K=PSI1(K)
  R1_K=R1(K)
  FI1R1=FI1_K*R1_K*R1_K
  RMASSLAA=RMASSLAA+FI1R1
 ENDDO
 RMASSLAA=RMASSLAA*COL3*RORI
 IF(RMASSLAA.LT.0.0) RMASSLAA=0.0
 DELMASSL1 = RMASSLAA-RMASSLBB

 QPN = QOLD - DELMASSL1
 DAL1 = AL1
 TPN = TOLD + DAL1*DELMASSL1

 lh_ce_1 = lh_ce_1 + DAL1*DELMASSL1
!---YZ2020---------------------------------------@
#ifdef SBM_DIAG
  ttdiffl= ttdiffl+DAL1*DELMASSL1
#endif
!------------------------------------------------@

! ... CCN regeneration
totccn_after = 0.0
totccn_after = sum(psi1(1:nkr)*r1(1:nkr))*3.0*col ! [cm-3]
CCN_reg = CCN_reg + max((totccn_before - totccn_after),0.0)
 IF(ABS(DAL1*DELMASSL1) > 5.0 )THEN
 	print*,"ONECOND1-out (start)"
 	print*,"I=",Iin,"J=",Jin,"Kin",Kin,"W",w_in,"DX",dx_in
 	print*,"DEL1N,DEL2N,D1N,D2N,RW,PW,RI,PI,DT"
 	print*,DEL1N,DEL2N,D1N,D2N,RW,PW,RI,PI,DTT
 	print*,"I=",Iin,"J=",Jin,"Kin",Kin
 	print*,"TPS=",TPS,"QPS=",QPS,"delmassl1",delmassl1
 	print*,"DAL1=",DAL1
 	print*,RMASSLBB,RMASSLAA
 	print*,"FI1",FI1
 	print*,"PSI1",PSI1
 	print*,"ONECOND1-out (end)"
 	IF(ABS(DAL1*DELMASSL1) > 5.0 )THEN
 	!	call wrf_error_fatal ("fatal error in ONECOND1-out (ABS(DAL1*DELMASSL1) > 5.0), model stop")
   ENDIF
  ENDIF

 ! ... SUPERSATURATION
 ARGEXP=-BB1_MY/TPN
 ES1N=AA1_MY*DEXP(ARGEXP)
 ARGEXP=-BB2_MY/TPN
 ES2N=AA2_MY*DEXP(ARGEXP)
 EW1N=OPER3(QPN,PP)
 IF(ES1N == 0.0D0)THEN
  	DEL1N=0.5
  	DIV1=1.5
 	!call wrf_error_fatal ("fatal error in ONECOND1 (ES1N.EQ.0), model stop")
 ELSE
    DIV1=EW1N/ES1N
    DEL1N=EW1N/ES1N-1.
 END IF
 IF(ES2N.EQ.0)THEN
    DEL2N=0.5
    DIV2=1.5
   !call wrf_error_fatal ("fatal error in ONECOND1 (ES2N.EQ.0), model stop")
 ELSE
    DEL2N=EW1N/ES2N-1.
    DIV2=EW1N/ES2N
 END IF

 TT=TPN
 QQ=QPN
 DO KR=1,NKR
  FF1(KR)=PSI1(KR)
 ENDDO

 RETURN
 END SUBROUTINE ONECOND1
 ! +------------------------------------------------------------------------+
 	SUBROUTINE COAL_BOTT_NEW_WARM(FF1R,TT,QQ,PP,RHO,dt_coll,TCRIT,TTCOAL,     &
				                        DEL1in, DEL2in,                             &
  		                          Iin,Jin,Kin,Itimestep,krdrop)
 								                     

  implicit none

  integer,intent(in) :: Iin,Jin,Kin,Itimestep,krdrop
  double precision,intent(in) :: tcrit,ttcoal,dt_coll
  double precision,intent(inout) :: ff1r(:) 
  double precision,intent(inout) :: del1in,del2in,tt,qq
  double precision,intent(in) :: pp

  integer :: KR,ICE,icol_drop,icol_snow,icol_graupel,icol_hail,               &
            icol_column,icol_plate,icol_dendrite,icol_drop_brk
  double precision :: g1(nkr),g2(nkr,icemax),g3(nkr),g4(nkr),g5(nkr),        &
                      gdumb(JMAX),gdumb_bf_breakup(JMAX),xl_dumb(JMAX),       &
                      g_orig(nkr),g2_1(nkr),g2_2(nkr),g2_3(nkr)
  double precision :: cont_fin_drop,dconc,conc_icempl,deldrop,t_new,         &
                      delt_new,cont_fin_ice,conc_old,conc_new,cont_init_ice,  &
                      cont_init_drop,ALWC,T_new_real,PP_r,rho,ES1N,ES2N,EW1N
  double precision,parameter :: tt_no_coll=273.16

  integer :: I, J, IT, NDIV, It_is_rain1, It_is_rain2, It_is_cloud
  double precision :: break_drop_bef,break_drop_aft,dtbreakup,break_drop_per,  &
                      prdkrn,fl1(nkr),rf1(nkr),rf3(nkr),fl3(nkr),               &
                      fl4(nkr),fl5(nkr),fl2_1(nkr),fl2_2(nkr),fl2_3(nkr),       &
                      rf2(nkr),rf4(nkr),rf5(nkr),conc_drop_old, conc_drop_new,  &
                      dconc_drop, dm_rime(nkr), conc_plate_icempl,              &
                      col3, cont_coll_drop,cld_dsd_nbf,cld_dsd_naf,cld_dsd_mbf, & 
                      cld_dsd_maf,total_cld_nsink,total_cld_msink,rain_dsd_nbf, &
                      rain_dsd_naf,rain_dsd_maf,rain_dsd_mbf,total_rain_nchng,dsd_hlp(nkr), &
                      rlt_m_iblnc,sink_mass,m_iblnc,gain_mass,mass_flux(nkr),tot_mbf,tot_maf

  double precision,parameter :: prdkrn1 = 1.0d0
  double precision,parameter :: prdkrn1_r = 1.0
  integer,parameter :: icempl = 1
  double precision,parameter :: t_ice_mpl = 270.15D0 ! for ice multiplication in temp > 268.15
  double precision,PARAMETER :: g_lim = 1.0D-19*1.0D3,AA1_MY = 2.53E12,  &
                                BB1_MY = 5.42E3, AA2_MY = 3.41E13 ,BB2_MY = 6.13E3


    icol_drop_brk = 0
    icol_drop = 0
    
    t_new = tt
    PP_r = PP

    call Kernals_KS(dt_coll,nkr,PP_r)

    DO KR = 1,NKR
      G1(KR) = FF1R(KR)*3.0*XL(KR)*XL(KR)*1.0E3
      if(kr > KRMIN_BREAKUP .and. g1(kr) > g_lim) icol_drop_brk = 1
      IF (IBREAKUP == 0) icol_drop_brk = 0
      if(g1(kr) > g_lim) icol_drop=1
    END DO

  if (icol_drop == 1)then
! ... Drop-Drop collisions

! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! --- [JS]: Here we Calculate the Autoconversion rate using a DUMMY call to coll_xxx_lwf();     
! --- [JS]  Evaluating autoconv. from the cloud-mode spectra
    
    dsd_hlp = g1
    !dsd_hlp(krdrop+1:nkr) = 0.0d0
    rain_dsd_nbf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3                                          ! in [#/cm3]
    rain_dsd_mbf = col*sum(dsd_hlp(krdrop+1:nkr))*1.e-3                                                           ! in [g/cm3]
    cld_dsd_mbf  = col*sum(dsd_hlp(1:krdrop))*1.e-3                                                               ! in [g/cm3]
    cld_dsd_nbf  = col*sum(dsd_hlp(1:krdrop)/xl(1:krdrop))*1.e-3                                                  ! in [#/cm3]
    mass_flux = 0.0                                                  
    call coll_xxx_Bott_mod1(dsd_hlp,1,krdrop,1,krdrop,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)
    rain_dsd_naf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3                                          ! in [#/cm3]
    rain_dsd_maf = col*sum(dsd_hlp(krdrop+1:nkr))*1.e-3                                                           ! in [g/cm3]
    cld_dsd_maf  = col*sum(dsd_hlp(1:krdrop))*1.e-3                                                               ! in [g/cm3]
    cld_dsd_naf  = col*sum(dsd_hlp(1:krdrop)/xl(1:krdrop))*1.e-3                                                  ! in [#/cm3]
    It_is_cloud = 0
    if(cld_dsd_mbf > 0.01*1.0e-6) It_is_cloud = 1 
    if ( It_is_cloud == 1 ) then
      auto_cld_msink_b  = rain_dsd_maf - rain_dsd_mbf                                                              ! [+]
      auto_cld_nsink_b  = cld_dsd_nbf  - cld_dsd_naf                                                               ! [+]  
    endif

! --- [JS] Evaluating accretion from the full spectra
    dsd_hlp = g1
    cld_dsd_nbf  = col*sum(dsd_hlp(1:krdrop)/xl(1:krdrop))*1.e-3                                                   ! in [#/cm3]
    cld_dsd_mbf  = col*sum(dsd_hlp(1:krdrop))*1.e-3                                                                ! in [g/cm3]
    rain_dsd_nbf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3                                           ! in [#/cm3]
    rain_dsd_mbf = col*sum(dsd_hlp(krdrop+1:nkr))*1.e-3                                                            ! in [g/cm3]
    mass_flux = 0.0
    call coll_xxx_Bott_mod2(dsd_hlp,1,krdrop,krdrop+1,nkr,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)
    cld_dsd_naf  = col*sum(dsd_hlp(1:krdrop)/xl(1:krdrop))*1.e-3                                                   ! in [#/cm3]
    cld_dsd_maf  = col*sum(dsd_hlp(1:krdrop))*1.e-3                                                                ! in [g/cm3]
    rain_dsd_naf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3                                           ! in [#/cm3]
    rain_dsd_maf = col*sum(dsd_hlp(krdrop+1:nkr))*1.e-3                                                            ! in [g/cm3]
    It_is_cloud = 0
    if(cld_dsd_mbf > 0.01*1.0e-6) It_is_cloud = 1
    if ( It_is_cloud == 1 ) then
      accr_cld_nsink_b  = cld_dsd_nbf - cld_dsd_naf                                                                ! [+]   
      accr_cld_msink_b  = cld_dsd_mbf - cld_dsd_maf                                                                ! [+]
    endif

! --- [JS] Evaluating rain self-collection from the rain spectra
    dsd_hlp = g1
    !dsd_hlp(1:krdrop) = 0.0d0 
    rain_dsd_mbf = col*sum(dsd_hlp(krdrop+1:nkr))*1.e-3
    rain_dsd_nbf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3
    mass_flux = 0.0                                         
    It_is_rain1 = 0 
    call coll_xxx_Bott_mod1(dsd_hlp,krdrop+1,nkr,krdrop+1,nkr,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)
    rain_dsd_naf = col*sum(dsd_hlp(krdrop+1:nkr)/xl(krdrop+1:nkr))*1.e-3                                           ! in [#/cm3]
    if(rain_dsd_mbf > 1.0d-50) It_is_rain1 = 1
    if ( It_is_rain1 == 1 ) then
      selfc_rain_nchng_b = max(0.0d0,rain_dsd_nbf - rain_dsd_naf)
    endif
  
  ! --- [JS]: Here we call the standard 'coll_xxx()' routine with the full spectrum (DSD+RSD);                                                                        ! in [g/cm3]
    fl1 = 1.0
    mass_flux = 0.d0
    call coll_xxx_Bott(g1,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)                                        
  ! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

! ... Breakup
  if(icol_drop_brk == 1)then
    ndiv = 1
    10     	continue
    do it = 1,ndiv
      dtbreakup = dt_coll/ndiv
      if (it == 1)then
        do kr=1,JMAX
          gdumb(kr)= g1(kr)*1.D-3
          gdumb_bf_breakup(kr) =  g1(kr)*1.D-3
          xl_dumb(kr)=xl_mg(KR)*1.D-3
        end do
        break_drop_bef=0.d0
        do kr=1,JMAX
          break_drop_bef = break_drop_bef+g1(kr)*1.D-3
        end do
      end if

      call coll_breakup_KS(gdumb, xl_dumb, JMAX, dtbreakup, JBREAK, PKIJ, QKJ, NKR, NKR ,IT)
    end do

    do KR=1,NKR
      FF1R(KR) = (1.0d3*GDUMB(KR))/(3.0*XL(KR)*XL(KR)*1.E3)
      if(FF1R(KR) < 0.0)then
        if(ndiv < 8)then
          ndiv = 2*ndiv
          go to 10
        else
          !print*,"noBreakUp",Iin,Jin,Kin,Itimestep,ndiv
          go to 11
          !!call wrf_error_fatal ("in coal_bott af-coll_breakup - FF1R/GDUMB < 0.0")
        endif
      endif
      if(FF1R(kr) .ne. FF1R(kr)) then
        print*,kr,GDUMB(kr),GDUMB_BF_BREAKUP(kr),XL(kr)
        print*,IT,NDIV, DTBREAKUP
        print*,GDUMB
        print*,GDUMB_BF_BREAKUP
        !call wrf_error_fatal ("in coal_bott af-coll_breakup - FF1R NaN, model stop")
      endif
    enddo

    break_drop_aft=0.0d0
    do kr=1,JMAX
      break_drop_aft=break_drop_aft+gdumb(kr)
    end do
    break_drop_per=break_drop_aft/break_drop_bef
    if (break_drop_per > 1.001)then
      ndiv=ndiv*2
      GO TO 10
    else
      do kr=1,JMAX
        g1(kr) = gdumb(kr)*1.D3
      end do
    end if
  ! if icol_drop_brk.eq.1
  end if
! if icol_drop.eq.1
end if

11   continue
 
 ! recalculation of density function f1,f3,f4,f5 in  units [1/(g*cm**3)] :
     DO KR=1,NKR
        FF1R(KR)=G1(KR)/(3.*XL(KR)*XL(KR)*1.E3)
        if((FF1R(kr) .ne. FF1R(kr)) .or. FF1R(kr) < 0.0)then
	 	       print*,"G1",G1
 		 	     !call wrf_error_fatal ("stop at end coal_bott - FF1R NaN or FF1R < 0.0, model stop")
	      endif
 		END DO

 	if (abs(tt-t_new).gt.5.0) then
 		!call wrf_error_fatal ("fatal error in module_mp_WARM_sbm Del_T 5 K, model stop")
 	endif

  tt = t_new

 	RETURN
 	END SUBROUTINE COAL_BOTT_NEW_WARM
 ! ..................................................................................................
     SUBROUTINE BREAKINIT_KS(PKIJ,QKJ,ECOALMASSM,BRKWEIGHT,XL_r,DROPRADII,BR_MAX,JBREAK,JMAX,NKR,VR1)

  !   USE module_domain
  !   USE module_dm

     IMPLICIT NONE

 ! ... Interface
     integer,intent(in) :: br_max, JBREAK, NKR, JMAX
     double precision,intent(inout) :: ECOALMASSM(:,:),BRKWEIGHT(:)
     double precision,intent(in) :: XL_r(:), DROPRADII(:), VR1(:)
     double precision,intent(inout) :: PKIJ(:,:,:),QKJ(:,:)
 ! ... Interface

     !REAL :: XL_r(size(NKR))
     INTEGER :: hujisbm_unit1
     LOGICAL, PARAMETER :: PRINT_diag=.FALSE.
     LOGICAL :: opened
     LOGICAL , EXTERNAL :: wrf_dm_on_monitor
     CHARACTER*80 errmess

 !.....INPUT VARIABLES
 !
 !     GT    : MASS DISTRIBUTION FUNCTION
 !     XT_MG : MASS OF BIN IN MG
 !     JMAX  : NUMBER OF BINS

 !.....LOCAL VARIABLES

     DOUBLE PRECISION :: XL_d(NKR), DROPRADII_d(NKR), VR1_d(NKR)
     INTEGER :: IE,JE,KE
     INTEGER,PARAMETER :: AP = 1
     INTEGER :: I,J,K,JDIFF
     double precision :: RPKIJ(JBREAK,JBREAK,JBREAK),RQKJ(JBREAK,JBREAK)
     double precision :: PI,D0,HLP
     DOUBLE PRECISION :: M(0:JBREAK),ALM
     double precision :: DBREAK(JBREAK),GAIN,LOSS

 !.....DECLARATIONS FOR INIT
     INTEGER :: IP,KP,JP,KQ,JQ
     double precision :: XTJ

     CHARACTER*256 FILENAME_P,FILENAME_Q, file_p, file_q

     xl_d = xl_r

     IE = JBREAK
     JE = JBREAK
     KE = JBREAK

     if(nkr == 43) file_p = 'SBM_input_43/'//'coeff_p43.dat'
     if(nkr == 43) file_q = 'SBM_input_43/'//'coeff_q43.dat'
     if(nkr == 33) file_p = 'SBM_input_33/'//'coeff_p_new_33.dat' ! new Version 33 (taken from 43bins)
     if(nkr == 33) file_q = 'SBM_input_33/'//'coeff_q_new_33.dat' ! new Version 33   (taken from 43 bins)

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
         DO i = 20,99
             INQUIRE ( i , OPENED = opened )
             IF ( .NOT. opened ) THEN
                 hujisbm_unit1 = i
                 GOTO 2061
             ENDIF
         ENDDO
         2061     CONTINUE
     ENDIF

   !  CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )

     IF ( hujisbm_unit1 < 0 ) THEN
       !call wrf_error_fatal  ( 'Can not find unused fortran unit to read in BREAKINIT_KS lookup table, model stop' )
     ENDIF

     IF ( .TRUE. ) THEN
       OPEN(UNIT=hujisbm_unit1,FILE=trim(file_p),         &
       !OPEN(UNIT=hujisbm_unit1,FILE="coeff_p.asc",       &
            FORM="FORMATTED",STATUS="OLD",ERR=2070)

         DO K=1,KE
             DO I=1,IE
                 DO J=1,I
                 READ(hujisbm_unit1,'(3I6,1E16.8)') KP,IP,JP,PKIJ(KP,IP,JP) ! PKIJ=[g^3*cm^3/s]
                 ENDDO
             ENDDO
         ENDDO
         CLOSE(hujisbm_unit1)
     END IF

     hujisbm_unit1 = -1
     IF ( .TRUE. ) THEN
       DO i = 20,99
         INQUIRE ( i , OPENED = opened )
         IF ( .NOT. opened ) THEN
           hujisbm_unit1 = i
           GOTO 2062
         ENDIF
       ENDDO
       2062     CONTINUE
     ENDIF

    ! CALL wrf_dm_bcast_bytes ( hujisbm_unit1 , IWORDSIZE )

     IF ( hujisbm_unit1 < 0 ) THEN
       !call wrf_error_fatal  ( 'Can not find unused fortran unit to read in BREAKINIT_KS lookup table, model stop' )
     ENDIF

     IF ( .TRUE. ) THEN
      OPEN(UNIT=hujisbm_unit1,FILE=trim(file_q),    &
           FORM="FORMATTED",STATUS="OLD",ERR=2070)
          DO K=1,KE
             DO J=1,JE
                READ(hujisbm_unit1,'(2I6,1E16.8)') KQ,JQ,QKJ(KQ,JQ)
             ENDDO
          ENDDO
      CLOSE(hujisbm_unit1)
     END IF

     DROPRADII_d = DROPRADII
     vr1_d = vr1
     DO J=1,NKR
         DO I=1,NKR
             ECOALMASSM(I,J)=ECOALMASS(xl_d(I), xl_d(J), DROPRADII_d, vr1_d, NKR)
          ENDDO
     ENDDO
 ! ... Correction of coalescence efficiencies for drop collision kernels
     DO J=25,31
         ECOALMASSM(NKR,J)=0.1D-29
     ENDDO

      RETURN

 2070  continue
       
      WRITE( errmess , '(A,I4)' )                                               &
            'module_FAST_SBM: error opening hujisbm_DATA on unit, model stop'   &
            , hujisbm_unit1
      !call wrf_error_fatal (errmess)
       
    END SUBROUTINE BREAKINIT_KS

 !coalescence efficiency as function of masses
 !----------------------------------------------------------------------------+
     double precision FUNCTION ecoalmass(x1, x2, DROPRADII, VR1_BREAKUP, NKR)

     implicit none
     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(NKR), VR1_BREAKUP(NKR), x1, x2

     double precision,PARAMETER :: zero=0.0d0,one=1.0d0,eps=1.0d-10
     double precision :: rho, PI, akPI, Deta, Dksi

     rho=1.0d0             ! [rho]=g/cm^3

     PI=3.1415927d0
     akPI=6.0d0/PI

     Deta = (akPI*x1/rho)**(1.0d0/3.0d0)
     Dksi = (akPI*x2/rho)**(1.0d0/3.0d0)

     ecoalmass = ecoaldiam(Deta, Dksi, DROPRADII, VR1_BREAKUP, NKR)

     RETURN
     END FUNCTION ecoalmass
 !coalescence efficiency as function of diameters
 !---------------------------------------------------------------------------+
     double precision FUNCTION ecoaldiam(Deta,Dksi,DROPRADII,VR1_BREAKUP,NKR)

     implicit none
     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(nkr), VR1_BREAKUP(nkr),Deta,Dksi

     double precision :: Dgr, Dkl, Rgr, RKl, q, qmin, qmax, e, x, e1, e2, sin1, cos1
     double precision,PARAMETER :: zero=0.0d0,one=1.0d0,eps=1.0d-30,PI=3.1415927d0

     Dgr=dmax1(Deta,Dksi)
     Dkl=dmin1(Deta,Dksi)

     Rgr=0.5d0*Dgr
     Rkl=0.5d0*Dkl

     q=0.5d0*(Rkl+Rgr)

     qmin=250.0d-4
     qmax=500.0d-4

     if(Dkl<100.0d-4) then

         e=1.0d0

          elseif (q<qmin) then

          e = ecoalOchs(Dgr,Dkl,DROPRADII, VR1_BREAKUP, NKR)

     elseif(q>=qmin.and.q<qmax) then

         x=(q-qmin)/(qmax-qmin)

         sin1=dsin(PI/2.0d0*x)
         cos1=dcos(PI/2.0d0*x)

         e1=ecoalOchs(Dgr, Dkl, DROPRADII, VR1_BREAKUP, NKR)
         e2=ecoalLowList(Dgr, Dkl, DROPRADII, VR1_BREAKUP, NKR)

         e=cos1**2*e1+sin1**2*e2

     elseif(q>=qmax) then

         e=ecoalLowList(Dgr, Dkl, DROPRADII, VR1_BREAKUP, NKR)

     else

         e=0.999d0

     endif

     ecoaldiam=dmax1(dmin1(one,e),eps)

 RETURN
 END FUNCTION ecoaldiam
 !coalescence efficiency (Low & List)
 !----------------------------------------------------------------------------+
     double precision FUNCTION ecoalLowList(Dgr,Dkl,DROPRADII,VR1_BREAKUP,NKR)

     implicit none

     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(NKR), VR1_BREAKUP(NKR)
     double precision,intent(inout) :: Dgr, Dkl

     double precision :: sigma, aka, akb, dSTSc, ST, Sc, ET, CKE, qq0, qq1, qq2, Ecl, W1, W2, DC
     double precision,PARAMETER :: epsi=1.d-20

 ! 1 J = 10^7 g cm^2/s^2

     sigma=72.8d0    ! Surface Tension,[sigma]=g/s^2 (7.28E-2 N/m)
     aka=0.778d0      ! Empirical Constant
     akb=2.61d-4      ! Empirical Constant,[b]=2.61E6 m^2/J^2

     CALL collenergy(Dgr,Dkl,CKE,ST,Sc,W1,W2,Dc,DROPRADII,VR1_BREAKUP,NKR)

     dSTSc=ST-Sc         ! Diff. of Surf. Energies   [dSTSc] = g*cm^2/s^2
     ET=CKE+dSTSc        ! Coal. Energy,             [ET]    =     "

     IF(ET<50.0d0) THEN    ! ET < 5 uJ (= 50 g*cm^2/s^2)

         qq0=1.0d0+(Dkl/Dgr)
         qq1=aka/qq0**2
         qq2=akb*sigma*(ET**2)/(Sc+epsi)
         Ecl=qq1*dexp(-qq2)

     !if(i_breakup==24.and.j_breakup==25) then
     !print*, 'IF(ET<50.0d0) THEN'
     !print*, 'Ecl=qq1*dexp(-qq2)'
     !print*, 'qq1,qq2,Ecl'
     !print*,  qq1,qq2,Ecl
     !endif

     ELSE

         Ecl=0.0d0

     ENDIF

     ecoalLowList=Ecl

     RETURN
     END FUNCTION ecoalLowList

 !coalescence efficiency (Beard and Ochs)
 !---------------------------------------------------------------------------+
     double precision FUNCTION ecoalOchs(D_l,D_s,DROPRADII, VR1_BREAKUP,NKR)

     implicit none

     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(NKR), VR1_BREAKUP(NKR), D_l, D_s

     double precision :: PI, sigma, R_s, R_l, p, vTl, vTs, dv, Weber_number, pa1, pa2, pa3, g, x, e
     double precision,PARAMETER :: epsf=1.d-30 , FPMIN=1.d-30

     PI=3.1415927d0
     sigma=72.8d0       ! Surface Tension [sigma] = g/s^2 (7.28E-2 N/m)
                    ! Alles in CGS (1 J = 10^7 g cm^2/s^2)
     R_s=0.5d0*D_s
     R_l=0.5d0*D_l
     p=R_s/R_l

     vTl=vTBeard(D_l,DROPRADII, VR1_BREAKUP,NKR)

     vTs=vTBeard(D_s,DROPRADII, VR1_BREAKUP,NKR)

     dv=dabs(vTl-vTs)

     if(dv<FPMIN) dv=FPMIN

     Weber_number=R_s*dv**2/sigma

     pa1=1.0d0+p
     pa2=1.0d0+p**2
     pa3=1.0d0+p**3

     g=2**(3.0d0/2.0d0)/(6.0d0*PI)*p**4*pa1/(pa2*pa3)
     x=Weber_number**(0.5d0)*g

     e=0.767d0-10.14d0*x

     ecoalOchs=e

     RETURN
     END FUNCTION ecoalOchs
 !ecoalOchs
 !Calculating the Collision Energy
 !------------------------------------------------------------------------------+
     SUBROUTINE COLLENERGY(Dgr,Dkl,CKE,ST,Sc,W1,W2,Dc,DROPRADII,VR1_BREAKUP,NKR)


     implicit none
     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(NKR), VR1_BREAKUP(NKR)
     double precision,intent(inout) :: Dgr, Dkl, CKE, ST, Sc, W1, W2, Dc

     double precision :: PI, rho, sigma, ak10, Dgka2, Dgka3, v1, v2, dv, Dgkb3
     double precision,PARAMETER :: epsf = 1.d-30, FPMIN = 1.d-30

     !EXTERNAL vTBeard

     PI=3.1415927d0
     rho=1.0d0            ! Water Density,[rho]=g/cm^3
     sigma=72.8d0         ! Surf. Tension,(H2O,20°C)=7.28d-2 N/m
                      ! [sigma]=g/s^2
     ak10=rho*PI/12.0d0

     Dgr=dmax1(Dgr,epsf)
     Dkl=dmax1(Dkl,epsf)

     Dgka2=(Dgr**2)+(Dkl**2)

     Dgka3=(Dgr**3)+(Dkl**3)

     if(Dgr/=Dkl) then

         v1=vTBeard(Dgr,DROPRADII, VR1_BREAKUP,NKR)
         v2=vTBeard(Dkl,DROPRADII, VR1_BREAKUP,NKR)
         dv=(v1-v2)
         if(dv<FPMIN) dv=FPMIN
         dv=dv**2
         if(dv<FPMIN) dv=FPMIN
         Dgkb3=(Dgr**3)*(Dkl**3)
         CKE=ak10*dv*Dgkb3/Dgka3            ! Collision Energy [CKE]=g*cm^2/s^2

 !if(i_breakup==24.and.j_breakup==25) then
 !print*, 'Dgr,Dkl'
 !print*,  Dgr,Dkl
 !print*, 'Dgkb3,Dgka2,Dgka3,ak10'
 !print*,  Dgkb3,Dgka2,Dgka3,ak10
 !print*, 'v1,v2,dv,CKE'
 !print*,  v1,v2,dv,CKE
 !endif

     else

         CKE = 0.0d0

     endif

     ST=PI*sigma*Dgka2                 ! Surf.Energy (Parent Drop)
     Sc=PI*sigma*Dgka3**(2.0d0/3.0d0)  ! Surf.Energy (coal.System)

     W1=CKE/(Sc+epsf)                  ! Weber Number 1
     W2=CKE/(ST+epsf)                  ! Weber Number 2

     Dc=Dgka3**(1.0d0/3.0d0)           ! Diam. of coal. System

 !if(i_breakup==24.and.j_breakup==25) then
 !print*, 'ST,Sc,W1,W2,dc'
 !print*,  ST,Sc,W1,W2,dc
 !endif

     RETURN
     END SUBROUTINE COLLENERGY
 !COLLENERGY
 !Calculating Terminal Velocity (Beard-Formula)
 !------------------------------------------------------------------------+
 ! new change 23.07.07                                         (start)
     double precision FUNCTION vTBeard(diam,DROPRADII, VR1_BREAKUP, NKR)

     implicit none

     integer,intent(in) :: NKR
     double precision,intent(in) :: DROPRADII(NKR), VR1_BREAKUP(NKR), diam

     integer :: kr
     double precision :: aa

     aa   = diam/2.0d0           ! Radius in cm

     IF(aa <= DROPRADII(1)) vTBeard=VR1_BREAKUP(1)
     IF(aa > DROPRADII(NKR)) vTBeard=VR1_BREAKUP(NKR)

     DO KR=1,NKR-1
         IF(aa>DROPRADII(KR).and.aa<=DROPRADII(KR+1)) then
             vTBeard=VR1_BREAKUP(KR+1)
         ENDIF
     ENDDO

     RETURN
     END FUNCTION vTBeard
     !vTBeard
 ! new change 23.07.07                                           (end)
 ! +-------------------------------------------------------------------+
     subroutine coll_xxx_lwf(g,fl,ckxx,x,c,ima,prdkrn,nkr,output_flux)
   
       implicit none
   
       integer,intent(in) :: nkr
       double precision,intent(inout) :: g(:),fl(:)
       double precision,intent(in) ::	ckxx(:,:),x(:), c(:,:)
       integer,intent(in) :: ima(:,:)
       double precision,intent(in) :: prdkrn
       double precision,intent(inout) :: output_flux(:)
   
   ! ... Locals
      double precision:: gmin,x01,x02,x03,gsi,gsj,gsk,gsi_w,gsj_w,gsk_w,gk, &
                          gk_w,fl_gk,fl_gsk,flux,x1,flux_w,g_k_w,g_kp_old,g_kp_w
      integer :: i,ix0,ix1,j,k,kp
   ! ... Locals
   
     gmin=g_lim*1.0d3
   
   ! ix0 - lower limit of integration by i
     do i=1,nkr-1
      ix0=i
      if(g(i).gt.gmin) goto 2000
     enddo
     2000   continue
     if(ix0.eq.nkr-1) return
   
   ! ix1 - upper limit of integration by i
     do i=nkr-1,1,-1
      ix1=i
      if(g(i).gt.gmin) goto 2010
     enddo
     2010   continue
   
   ! ... collisions
         do i=ix0,ix1
            if(g(i).le.gmin) goto 2020
            do j=i,ix1
               if(g(j).le.gmin) goto 2021
               k=ima(i,j)
               kp=k+1
               x01=ckxx(i,j)*g(i)*g(j)*prdkrn
               x02=dmin1(x01,g(i)*x(j))
               if(j.ne.k) x03=dmin1(x02,g(j)*x(i))
               if(j.eq.k) x03=x02
               gsi=x03/x(j)
               gsj=x03/x(i)
               gsk=gsi+gsj
               if(gsk.le.gmin) goto 2021
               gsi_w=gsi*fl(i)
               gsj_w=gsj*fl(j)
               gsk_w=gsi_w+gsj_w
               gsk_w=dmin1(gsk_w,gsk)
               g(i)=g(i)-gsi
               g(i)=dmax1(g(i),0.0d0)
               g(j)=g(j)-gsj
     ! new change of 23.01.11                                      (start)
               if(j.ne.k) g(j)=dmax1(g(j),0.0d0)
     ! new change of 23.01.11                                        (end)
               gk=g(k)+gsk
   
               if(g(j).lt.0.d0.and.gk.le.gmin) then
                 g(j)=0.d0
                 g(k)=g(k)+gsi
                 goto 2021
             endif
   
               if(gk.le.gmin) goto 2021
   
               gk_w=g(k)*fl(k)+gsk_w
               gk_w=dmin1(gk_w,gk)
   
               fl_gk=gk_w/gk
               fl_gsk=gsk_w/gsk
               flux=0.d0
               x1=dlog(g(kp)/gk+1.d-15)
               flux=gsk/x1*(dexp(0.5d0*x1)-dexp(x1*(0.5d0-c(i,j))))
               flux=dmin1(flux,gsk)
               flux=dmin1(flux,gk)
! --- [JS] changed to >= and corrected to bin #33               
               if(kp.ge.kp_flux_max) flux=0.5d0*flux
               flux_w=flux*fl_gsk
               flux_w=dmin1(flux_w,gsk_w)
               flux_w=dmin1(flux_w,gk_w)
               g(k)=gk-flux
               g(k)=dmax1(g(k),gmin)
               g_k_w=gk_w-flux_w
               g_k_w=dmin1(g_k_w,g(k))
               g_k_w=dmax1(g_k_w,0.0d0)
               fl(k)=g_k_w/g(k)
               g_kp_old=g(kp)
               g(kp)=g(kp)+flux
! --- [JS] output flux - for autoconv.
               output_flux(kp) = output_flux(kp) + flux
               g(kp)=dmax1(g(kp),gmin)
               g_kp_w=g_kp_old*fl(kp)+flux_w
               g_kp_w=dmin1(g_kp_w,g(kp))
               fl(kp)=g_kp_w/g(kp)
   
               if(fl(k).gt.1.0d0.and.fl(k).le.1.0001d0) &
                   fl(k)=1.0d0
   
               if(fl(kp).gt.1.0d0.and.fl(kp).le.1.0001d0) &
                   fl(kp)=1.0d0
   
               if(fl(k).gt.1.0001d0.or.fl(kp).gt.1.0001d0 &
                  .or.fl(k).lt.0.0d0.or.fl(kp).lt.0.0d0) then
   
                 print*,    'in subroutine coll_xxx_lwf'
                 print*,    'snow - snow = snow'
   
                 if(fl(k).gt.1.0001d0)  print*, 'fl(k).gt.1.0001d0'
                 if(fl(kp).gt.1.0001d0) print*, 'fl(kp).gt.1.0001d0'
   
                 if(fl(k).lt.0.0d0)  print*, 'fl(k).lt.0.0d0'
                 if(fl(kp).lt.0.0d0) print*, 'fl(kp).lt.0.0d0'
   
                 print*,    'i,j,k,kp'
                 print*,     i,j,k,kp
                 print*,    'ix0,ix1'
                 print*,     ix0,ix1
   
                 print*,   'ckxx(i,j),x01,x02,x03'
                   print 204, ckxx(i,j),x01,x02,x03
   
                 print*,   'gsi,gsj,gsk'
                   print 203, gsi,gsj,gsk
   
                 print*,   'gsi_w,gsj_w,gsk_w'
                   print 203, gsi_w,gsj_w,gsk_w
   
                 print*,   'gk,gk_w'
                   print 202, gk,gk_w
   
                 print*,   'fl_gk,fl_gsk'
                   print 202, fl_gk,fl_gsk
   
                 print*,   'x1,c(i,j)'
                   print 202, x1,c(i,j)
   
                 print*,   'flux'
                   print 201, flux
   
                 print*,   'flux_w'
                   print 201, flux_w
   
                 print*,   'g_k_w'
                   print 201, g_k_w
   
                   print *,  'g_kp_w'
                   print 201, g_kp_w
   
                 if(fl(k).lt.0.0d0) print*, &
                    'stop 2022: in subroutine coll_xxx_lwf, fl(k) < 0'
   
                 if(fl(kp).lt.0.0d0) print*, &
                    'stop 2022: in subroutine coll_xxx_lwf, fl(kp) < 0'
   
                 if(fl(k).gt.1.0001d0) print*, &
                    'stop 2022: in sub. coll_xxx_lwf, fl(k) > 1.0001'
   
                 if(fl(kp).gt.1.0001d0) print*, &
                    'stop 2022: in sub. coll_xxx_lwf, fl(kp) > 1.0001'
                       !call wrf_error_fatal ("in coal_bott sub. coll_xxx_lwf, model stop")
                 endif
   2021     continue
          enddo
   ! cycle by j
   2020    continue
      enddo
   ! cycle by i
   
   201    format(1x,d13.5)
   202    format(1x,2d13.5)
   203    format(1x,3d13.5)
   204    format(1x,4d13.5)
   
    return
    end subroutine coll_xxx_lwf
! +-----------------------------------------------------------+
  subroutine coll_breakup_KS (gt_mg, xt_mg, jmax, dt, jbreak, &
                              PKIJ, QKJ, NKRinput, NKR, ITin)

    implicit none
  ! ... Interface
    integer,intent(in) :: jmax, jbreak, NKRInput, NKR, ITin
    double precision,intent(in) :: xt_mg(:), dt
    double precision,intent(in) :: pkij(:,:,:),qkj(:,:)
    double precision,intent(inout) :: gt_mg(:)
  ! ... Interface

  ! ... Locals
  ! ke = jbreak
  integer,parameter :: ia=1, ja=1, ka=1
  integer :: ie, je, ke, nkrdiff, jdiff, k, i, j
  double precision,parameter :: eps = 1.0d-20
  double precision :: gt(jmax), xt(jmax+1), ft(jmax), fa(jmax), dg(jmax), df(jmax), dbreak(jbreak) &
                    ,amweight(jbreak), gain, aloss
  ! ... Locals

  ie=jbreak
  je=jbreak
  ke=jbreak

  !input variables

  ! gt_mg : mass distribution function of Bott
  ! xt_mg : mass of bin in mg
  ! jmax  : number of bins
  ! dt    : timestep in s

  !in CGS

  nkrdiff = nkrinput-nkr
  do j=1,jmax
  xt(j)=xt_mg(j)
  gt(j)=gt_mg(j)
  ft(j)=gt(j)/xt(j)/xt(j)
  enddo

  !shift between coagulation and breakup grid
  jdiff=jmax-jbreak

  !initialization
  !shift to breakup grid
  fa = 0.0
  do k=1,ke-nkrdiff
    fa(k)=ft(k+jdiff+nkrdiff)
  enddo

  !breakup: bleck's first order method
  !pkij: gain coefficients
  !qkj : loss coefficients

  xt(jmax+1)=xt(jmax)*2.0d0

  amweight = 0.0
  dbreak = 0.0
  do k=1,ke-nkrdiff
    gain=0.0d0
    do i=1,ie-nkrdiff
      do j=1,i
        gain=gain+fa(i)*fa(j)*pkij(k,i,j)
      enddo
    enddo
    aloss=0.0d0
    do j=1,je-nkrdiff
      aloss=aloss+fa(j)*qkj(k,j)
    enddo
    j=jmax-jbreak+k+nkrdiff
    amweight(k)=2.0/(xt(j+1)**2.0-xt(j)**2.0)
    dbreak(k)=amweight(k)*(gain-fa(k)*aloss)

    if(dbreak(k) .ne. dbreak(k)) then
      print*,dbreak(k),amweight(k),gain,fa(k),aloss
      print*,"-"
      print*,dbreak
      print*,"-"
      print*,amweight
      print*,"-"
      print*,j,jmax,jbreak,k,nkrdiff
      print*,"-"
      print*,fa
      print*,"-"
      print*,xt
      print*,"-"
      print*,gt
      !call wrf_error_fatal (" inside coll_breakup, NaN, model stop")
    endif
  enddo

  !shift rate to coagulation grid
  df = 0.0d0
  do j=1,jdiff+nkrdiff
    df(j)=0.0d0
  enddo

  do j=1,ke-nkrdiff
    df(j+jdiff)=dbreak(j)
  enddo

  !transformation to mass distribution function g(ln x)
  do j=1,jmax
    dg(j)=df(j)*xt(j)*xt(j)
  enddo

  !time integration

  do j=1,jmax
    gt(j)=gt(j)+dg(j)*dt
  !	if(gt(j) < 0.0 .and. ITin == 8) then
  !  print*, 'gt(j) < 0'
  !  print*, 'j',j
  !  print*, 'dg(j),dt,gt(j)'
  !  print*,  dg(j),dt,gt(j)
  !  hlp=dmin1(gt(j),hlp)
  !	gt(j) = eps
  !	print*,'kr',j
  !	print*,'gt',gt
  !	print*,'dg',dg
  !	print*,'gt_mg',gt_mg
  ! stop "in coll_breakup_ks gt(kr) < 0.0 "
  !  endif
  enddo

  gt_mg = gt

  return
  end subroutine coll_breakup_KS
     ! +----------------------------------------------------+
  subroutine courant_bott_KS(xl, nkr, chucm, ima, scal)

    implicit none

    integer,intent(in) :: nkr
    double precision,intent(in) :: xl(:)
    double precision,intent(inout) :: chucm(:,:)
    integer,intent(inout) :: ima(:,:)
    double precision,intent(in) :: scal

    ! ... Locals
    integer :: k, kk, j, i
    double precision :: x0, xl_mg(nkr), dlnr
    ! ... Locals

    ! ima(i,j) - k-category number,
    ! chucm(i,j)   - courant number :
    ! logarithmic grid distance(dlnr) :

      !xl_mg(0)=xl_mg(1)/2
      xl_mg(1:nkr) = xl(1:nkr)*1.0D3

      dlnr=dlog(2.0d0)/(3.0d0*scal)

      do i = 1,nkr
        do j = i,nkr
            x0 = xl_mg(i) + xl_mg(j)
            do k = j,nkr
              !if(k == 1) goto 1000 ! ### (KS)
              kk = k
              if(k == 1) goto 1000
              if(xl_mg(k) >= x0 .and. xl_mg(k-1) < x0) then
                chucm(i,j) = dlog(x0/xl_mg(k-1))/(3.d0*dlnr)
                if(chucm(i,j) > 1.0d0-1.d-08) then
                  chucm(i,j) = 0.0d0
                  kk = kk + 1
                endif
                ima(i,j) = min(nkr-1,kk-1)
                !if (ima(i,j) == 0) then
                !	print*,"ima==0"
                !endif
                goto 2000
              endif
              1000 continue
            enddo
            2000  continue
            !if(i.eq.nkr.or.j.eq.nkr) ima(i,j)=nkr
            chucm(j,i) = chucm(i,j)
            ima(j,i) = ima(i,j)
        enddo
      enddo

      return
      end subroutine courant_bott_KS
! +--------------------------------------+
! +----------------------------------------------------------+
  double precision FUNCTION POLYSVP (TT,ITYPE)

  implicit none

  double precision,intent(in) :: TT
  integer,intent(in) :: ITYPE

  double precision,parameter :: C1 = -9.09718E0, C2 = -3.56654E0, C3 = 0.876793E0, C4 = 0.78583503E0, &
                      AA1_MY = 2.53E12, BB1_MY = 5.42E3, AA2_MY = 3.41E13, BB2_MY = 6.13E3
  double precision :: ES1N, ES2N

  method_select: SELECT CASE(ITYPE)

  ! liquid
  Case(0)
    ES1N = AA1_MY*EXP(-BB1_MY/TT)
    POLYSVP = ES1N ! [dyn/cm2] to [mb]

  ! ice  
  Case(1)
    ES2N = AA2_MY*EXP(-BB2_MY/TT)
    POLYSVP = ES2N ! [dyn/cm2] to [mb]

  END SELECT method_select

  return
  end function POLYSVP
! + -------------------------------------------------------- +
      SUBROUTINE JERRATE_KS (xlS, &
                            TP,PP, &
                            Vxl,RIEC,RO1BL, &
                            B11_MY, &
                            ID,IN,fl1,NKR,ICEMAX)

    IMPLICIT NONE
! ... Interface
    INTEGER,INTENT(IN) :: ID, IN, NKR, ICEMAX
    double precision,INTENT(IN) :: RO1BL(NKR,ID),RIEC(NKR,ID),FL1(NKR)
    double precision,INTENT(INOUT) :: B11_MY(NKR,ID)
    double precision,INTENT(IN) :: PP, TP, xlS(NKR,ID),Vxl(NKR,ID)
! ... Interface
! ... Locals
    INTEGER :: KR, nskin(nkr), ICE
    double precision :: VENTPLM(NKR), FD1(NKR,ICEMAX),FK1(NKR,ICEMAX), xl_MY1(NKR,ICEMAX), &
                          AL1_MY(2),ESAT1(2), TPreal
    double precision :: PZERO, TZERO, CONST, D_MY, COEFF_VISCOUS, SHMIDT_NUMBER,     &
                          A, B, RVT, SHMIDT_NUMBER03, XLS_KR_ICE, RO1BL_KR_ICE, VXL_KR_ICE, REINOLDS_NUMBER, &
                          RESHM, VENTPL, CONSTL, DETL

    double precision :: deg01,deg03

! A1L_MY - CONSTANTS FOR "MAXWELL": MKS
    double precision,parameter:: RV_MY=461.5D4, CF_MY=2.4D3, D_MYIN=0.211D0

! CGS :

! RV_MY, CM*CM/SEC/SEC/KELVIN - INDIVIDUAL GAS CONSTANT
!                               FOR WATER VAPOUR
  !RV_MY=461.5D4

! D_MYIN, CM*CM/SEC - COEFFICIENT OF DIFFUSION OF WATER VAPOUR

  !D_MYIN=0.211D0

! PZERO, DYNES/CM/CM - REFERENCE PRESSURE

  PZERO=1.013D6

! TZERO, KELVIN - REFERENCE TEMPERATURE

  TZERO=273.15D0

do kr=1,nkr
  if (in==2 .and. fl1(kr)==0.0 .or. in==6 .or. in==3 .and. tp<273.15) then
      nskin(kr) = 2
  else !in==1 or in==6 or lef/=0
      nskin(kr) = 1
  endif
enddo

! CONSTANTS FOR CLAUSIUS-CLAPEYRON EQUATION :

! A1_MY(1),G/SEC/SEC/CM

!	A1_MY(1)=2.53D12

! A1_MY(2),G/SEC/SEC/CM

!	A1_MY(2)=3.41D13

! BB1_MY(1), KELVIN

!	BB1_MY(1)=5.42D3

! BB1_MY(2), KELVIN

!	BB1_MY(2)=6.13D3

! AL1_MY(1), CM*CM/SEC/SEC - LATENT HEAT OF VAPORIZATION

  AL1_MY(1)=2.5D10

! AL1_MY(2), CM*CM/SEC/SEC - LATENT HEAT OF SUBLIMATION

  AL1_MY(2)=2.834D10

! CF_MY, G*CM/SEC/SEC/SEC/KELVIN - COEFFICIENT OF
!                                  THERMAL CONDUCTIVITY OF AIR
  !CF_MY=2.4D3

  DEG01=1.0/3.0
  DEG03=1.0/3.0

  CONST=12.566372D0

! coefficient of diffusion

  D_MY=D_MYIN*(PZERO/PP)*(TP/TZERO)**1.94D0

! coefficient of viscousity

! COEFF_VISCOUS=0.13 cm*cm/sec

        COEFF_VISCOUS=0.13D0

! Shmidt number

        SHMIDT_NUMBER=COEFF_VISCOUS/D_MY

! Constants used for calculation of Reinolds number

        A=2.0D0*(3.0D0/4.0D0/3.141593D0)**DEG01
        B=A/COEFF_VISCOUS

  RVT=RV_MY*TP
  !	ESAT1(IN)=A1_MY(IN)*DEXP(-BB1_MY(IN)/TP)
  !	if (IN==1) then
  !            ESAT1(IN)=EW(TP)
  !	ELSE
  !            ESAT1(IN)=EI(TP)
  !	endif

    ! ... (KS) - update the saturation vapor pressure
    !ESAT1(1)=EW(TP)
    !ESAT1(2)=EI(TP)
    TPreal = TP
    ESAT1(1) = POLYSVP(TPreal,0)
    ESAT1(2) = POLYSVP(TPreal,1)

      DO KR=1,NKR
        VENTPLM(KR)=0.0D0
    ENDDO

    SHMIDT_NUMBER03=SHMIDT_NUMBER**DEG03

      DO ICE=1,ID
        DO KR=1,NKR

          xlS_KR_ICE=xlS(KR,ICE)
          RO1BL_KR_ICE=RO1BL(KR,ICE)
          Vxl_KR_ICE=Vxl(KR,ICE)
! Reynolds numbers
          REINOLDS_NUMBER= &
              B*Vxl_KR_ICE*(xlS_KR_ICE/RO1BL_KR_ICE)**DEG03
          RESHM=DSQRT(REINOLDS_NUMBER)*SHMIDT_NUMBER03

          IF(REINOLDS_NUMBER<2.5D0) THEN
            VENTPL=1.0D0+0.108D0*RESHM*RESHM
            VENTPLM(KR)=VENTPL
          ELSE
            VENTPL=0.78D0+0.308D0*RESHM
            VENTPLM(KR)=VENTPL
          ENDIF

        ENDDO
! cycle by KR

! VENTPL_MAX is given in MICRO.PRM include file

        DO KR=1,NKR

        VENTPL=VENTPLM(KR)

        IF(VENTPL>VENTPL_MAX) THEN
          VENTPL=VENTPL_MAX
          VENTPLM(KR)=VENTPL
        ENDIF

        CONSTL=CONST*RIEC(KR,ICE)

        FD1(KR,ICE)=RVT/D_MY/ESAT1(nskin(kr))
        FK1(KR,ICE)=(AL1_MY(nskin(kr))/RVT-1.0D0)*AL1_MY(nskin(kr))/CF_MY/TP

        xl_MY1(KR,ICE)=VENTPL*CONSTL
        ! growth rate
        DETL=FK1(KR,ICE)+FD1(KR,ICE)
        B11_MY(KR,ICE)=xl_MY1(KR,ICE)/DETL

        ENDDO
! cycle by KR

      ENDDO
! cycle by ICE

  RETURN
  END SUBROUTINE JERRATE_KS
        
! SUBROUTINE JERRATE
! ................................................................................
  SUBROUTINE JERTIMESC_KS (FI1,X1,SFN11, &
                            B11_MY,CF,ID,NKR,ICEMAX,COL)

  IMPLICIT NONE

! ... Interface
  INTEGER,INTENT(IN) :: ID,NKR,ICEMAX
  double precision,INTENT(in) :: B11_MY(NKR,ID), FI1(NKR,ID), COL, CF
  double precision,INTENT(in) :: X1(NKR,ID)
  double precision,INTENT(out) :: SFN11(ID)
! ... Interface

! ... Locals
  INTEGER :: ICE, KR
  double precision :: SFN11S, FK, DELM, FUN, B11
! ... Locals

  DO ICE=1,ID
      SFN11S=0.0D0
      SFN11(ICE)=CF*SFN11S
    DO KR=1,NKR
! value of size distribution functions
        FK=FI1(KR,ICE)
! delta-m
        DELM=X1(KR,ICE)*3.0D0*COL
! integral's expression
          FUN=FK*DELM
! values of integrals
          B11=B11_MY(KR,ICE)
        SFN11S=SFN11S+FUN*B11
  ENDDO
! cycle by kr
! correction
    SFN11(ICE)=CF*SFN11S
  ENDDO

! cycle by ice

  RETURN
  END SUBROUTINE JERTIMESC_KS
! +--------------------------------------------------------+
  SUBROUTINE JERSUPSAT_KS (DEL1,DEL2,DEL1N,DEL2N,         &
                            RW,PW,RI,PI,                   &
                            DT,DEL1INT,DEL2INT,DYN1,DYN2,  &
                            ISYM1,ISYM2,ISYM3,ISYM4,ISYM5)

    IMPLICIT NONE
! ... Interface
    INTEGER,INTENT(INOUT) :: 		ISYM1, ISYM2(:), ISYM3, ISYM4, ISYM5
    double precision,INTENT(IN) ::   DT, DYN1, DYN2
    double precision,INTENT(IN) :: 	DEL1, DEL2
    double precision,INTENT(INOUT) :: DEL1N,DEL2N,DEL1INT,DEL2INT,RW, PW, RI, PI
! ... Interface
! ... Locals
      INTEGER :: I, ISYMICE, IRW, IPW, IRI, IPI
      double precision :: X, EXPM1, DETER, EXPR, EXPP, A, ALFA, BETA, GAMA, G31, G32, G2, EXPB, EXPG, &
                C11, C21, C12, C22, A1DEL1N, A2DEL1N, A3DEL1N, A4DEL1N, A1DEL1INT, A2DEL1INT, &
              A3DEL1INT, A4DEL1INT, A1DEL2N, A2DEL2N, A3DEL2N , A4DEL2N, A1DEL2INT, A2DEL2INT, &
              A3DEL2INT, A4DEL2INT, A5DEL2INT
! ... Locals

    EXPM1(x)=x+x*x/2.0D0+x*x*x/6.0D0+x*x*x*x/24.0D0+ &
                  x*x*x*x*x/120.0D0

  ISYMICE = sum(ISYM2) + ISYM3 + ISYM4 + ISYM5
  IRW = 1
  IPW = 1
  IRI = 1
  IPI = 1  
  IF(AMAX1(RW,PW,RI,PI)<=RW_PW_RI_PI_MIN) THEN
    RW = 0.0
    IRW = 0
    PW = 0.0
    IPW = 0
    RI = 0.0
    IRI = 0
    PI = 0.0
    IPI = 0
    ISYM1 = 0
    ISYMICE = 0

  ELSE

    IF(DMAX1(RW,PW)>RW_PW_MIN) THEN

      ! ... (KS) - A zero can pass through, assign a minimum value
      IF(RW < RW_PW_MIN*RW_PW_MIN) THEN
        RW = 1.0D-20
        IRW = 0
      ENDIF
      IF(PW < RW_PW_MIN*RW_PW_MIN)THEN
        PW = 1.0D-20
        IPW = 0
      ENDIF

      IF(DMAX1(PI/PW,RI/RW)<=RATIO_ICEW_MIN) THEN
        ! ... only water
        RI = 0.0
        IRI = 0
        PI = 0.0
        IPI = 0
        ISYMICE = 0
      ENDIF

      IF(DMIN1(PI/PW,RI/RW)>1.0D0/RATIO_ICEW_MIN) THEN
        ! ... only ice
        RW = 0.0
        IRW = 0
        PW = 0.0
        IPW = 0
        ISYM1 = 0
      ENDIF

    ELSE
      ! only ice
      RW = 0.0
      IRW = 0
      PW = 0.0
      IPW = 0
      ISYM1 = 0

    ENDIF
    ENDIF

  IF(ISYMICE == 0)THEN
    ISYM2 = 0
    ISYM3 = 0
    ISYM4 = 0
    ISYM5 = 0
  ENDIF

    DETER=RW*PI-PW*RI


    IF(IRW == 0 .AND. IRI == 0) THEN

          DEL1N=DEL1+DYN1*DT
          DEL2N=DEL2+DYN2*DT
          DEL1INT=DEL1*DT+DYN1*DT*DT/2.0D0
          DEL2INT=DEL2*DT+DYN2*DT*DT/2.0D0

          GOTO 100

    ENDIF

! solution of equation for supersaturation with
! different DETER values

    IF(IRI == 0) THEN
! ... only water                                                     (start)

      EXPR=EXP(-RW*DT)
      IF(ABS(RW*DT)>1.0E-6) THEN
        DEL1N=DEL1*EXPR+(DYN1/RW)*(1.0D0-EXPR)
        DEL2N=PW*DEL1*EXPR/RW-PW*DYN1*DT/RW- &
              PW*DYN1*EXPR/(RW*RW)+DYN2*DT+ &
              DEL2-PW*DEL1/RW+PW*DYN1/(RW*RW)
        DEL1INT=-DEL1*EXPR/RW+DYN1*DT/RW+ &
                  DYN1*EXPR/(RW*RW)+DEL1/RW-DYN1/(RW*RW)
        DEL2INT=PW*DEL1*EXPR/(-RW*RW)-PW*DYN1*DT*DT/(2.0D0*RW)+ &
                PW*DYN1*EXPR/(RW*RW*RW)+DYN2*DT*DT/2.0D0+ &
                DEL2*DT-PW*DEL1*DT/RW+PW*DYN1*DT/(RW*RW)+ &
                PW*DEL1/(RW*RW)-PW*DYN1/(RW*RW*RW)
        GOTO 100
! in case DABS(RW*DT)>1.0D-6
        ELSE

! in case DABS(RW*DT)<=1.0D-6

          EXPR=EXPM1(-RW*DT)
          DEL1N=DEL1+DEL1*EXPR+(DYN1/RW)*(0.0D0-EXPR)
          DEL2N=PW*DEL1*EXPR/RW-PW*DYN1*DT/RW- &
                    PW*DYN1*EXPR/(RW*RW)+DYN2*DT+DEL2
          DEL1INT=-DEL1*EXPR/RW+DYN1*DT/RW+DYN1*EXPR/(RW*RW)
          DEL2INT=PW*DEL1*EXPR/(-RW*RW)-PW*DYN1*DT*DT/(2.0D0*RW)+ &
                      PW*DYN1*EXPR/(RW*RW*RW)+DYN2*DT*DT/2.0D0+ &
                      DEL2*DT-PW*DEL1*DT/RW+PW*DYN1*DT/(RW*RW)
          GOTO 100

          ENDIF
! ... only water                                                    (end)

! in case RI==0.0D0
    ENDIF

    IF(IRW  ==  0) THEN
! ... only ice                                                    (start)

      EXPP=EXP(-PI*DT)

      IF(ABS(PI*DT)>1.0E-6) THEN

        DEL2N = DEL2*EXPP+(DYN2/PI)*(1.0D0-EXPP)
        DEL2INT = -DEL2*EXPP/PI+DYN2*DT/PI+ &
                    DYN2*EXPP/(PI*PI)+DEL2/PI-DYN2/(PI*PI)
        DEL1N = +RI*DEL2*EXPP/PI-RI*DYN2*DT/PI- &
                  RI*DYN2*EXPP/(PI*PI)+DYN1*DT+ &
                  DEL1-RI*DEL2/PI+RI*DYN2/(PI*PI)
        DEL1INT = -RI*DEL2*EXPP/(PI*PI)-RI*DYN2*DT*DT/(2.0D0*PI)+ &
                    RI*DYN2*EXPP/(PI*PI*PI)+DYN1*DT*DT/2.0D0+ &
                    DEL1*DT-RI*DEL2*DT/PI+RI*DYN2*DT/(PI*PI)+ &
                    RI*DEL2/(PI*PI)-RI*DYN2/(PI*PI*PI)
        GOTO 100
! in case DABS(PI*DT)>1.0D-6
      ELSE

! in case DABS(PI*DT)<=1.0D-6

          EXPP=EXPM1(-PI*DT)
          DEL2N=DEL2+DEL2*EXPP-EXPP*DYN2/PI
          DEL2INT=-DEL2*EXPP/PI+DYN2*DT/PI+DYN2*EXPP/(PI*PI)
          DEL1N=+RI*DEL2*EXPP/PI-RI*DYN2*DT/PI- &
                    RI*DYN2*EXPP/(PI*PI)+DYN1*DT+DEL1
          DEL1INT=-RI*DEL2*EXPP/(PI*PI)-RI*DYN2*DT*DT/(2.0D0*PI)+ &
                      RI*DYN2*EXPP/(PI*PI*PI)+DYN1*DT*DT/2.0D0+ &
                      DEL1*DT-RI*DEL2*DT/PI+RI*DYN2*DT/(PI*PI)
          GOTO 100

      ENDIF
! ... only ice                                                      (end)

! in case RW==0.0D0
    ENDIF

    IF(IRW == 1 .AND. IRI == 1) THEN

      A=(RW-PI)*(RW-PI)+4.0E0*PW*RI

        IF(A < 0.0) THEN
            PRINT*,   'IN SUBROUTINE JERSUPSAT: A < 0'
            PRINT*,   'DETER'
            PRINT 201, DETER
            PRINT*,   'RW,PW,RI,PI'
            PRINT 204, RW,PW,RI,PI
            PRINT*,   'DT,DYN1,DYN2'
            PRINT 203, DT,DYN1,DYN2
            PRINT*,   'DEL1,DEL2'
            PRINT 202, DEL1,DEL2
            PRINT*,   'STOP 1905:A < 0'
            !call wrf_error_fatal ("fatal error: STOP 1905:A < 0, model stop")
        ENDIF
! ... water and ice                                               (start)
        ALFA=DSQRT((RW-PI)*(RW-PI)+4.0D0*PW*RI)

! 5/8/04 Nir, Beta is negative to the simple solution so it will decay

        BETA=0.5D0*(ALFA+RW+PI)
        GAMA=0.5D0*(ALFA-RW-PI)
        G31=PI*DYN1-RI*DYN2
        G32=-PW*DYN1+RW*DYN2
        G2=RW*PI-RI*PW
        IF (G2 < 1.0d-20) G2 = 1.0004d-11*1.0003d-11-1.0002d-11*1.0001e-11 ! ... (KS) - 24th,May,2016
        EXPB=DEXP(-BETA*DT)
        EXPG=DEXP(GAMA*DT)

        IF(DABS(GAMA*DT)>1.0E-6) THEN
          C11=(BETA*DEL1-RW*DEL1-RI*DEL2-BETA*G31/G2+DYN1)/ALFA
          C21=(GAMA*DEL1+RW*DEL1+RI*DEL2-GAMA*G31/G2-DYN1)/ALFA
          C12=(BETA*DEL2-PW*DEL1-PI*DEL2-BETA*G32/G2+DYN2)/ALFA
          C22=(GAMA*DEL2+PW*DEL1+PI*DEL2-GAMA*G32/G2-DYN2)/ALFA
          DEL1N=C11*EXPG+C21*EXPB+G31/G2
          DEL1INT=C11*EXPG/GAMA-C21*EXPB/BETA+(C21/BETA-C11/GAMA) &
                  +G31*DT/G2
          DEL2N=C12*EXPG+C22*EXPB+G32/G2
          DEL2INT=C12*EXPG/GAMA-C22*EXPB/BETA+(C22/BETA-C12/GAMA) &
                  +G32*DT/G2
            GOTO 100
! in case DABS(GAMA*DT)>1.0D-6
          ELSE
! in case DABS(GAMA*DT)<=1.0D-6
            IF(ABS(RI/RW)>1.0E-12) THEN
              IF(ABS(RW/RI)>1.0E-12) THEN
                ALFA=DSQRT((RW-PI)*(RW-PI)+4.0D0*PW*RI)
                BETA=0.5D0*(ALFA+RW+PI)
                GAMA=0.5D0*(ALFA-RW-PI)
                IF (GAMA < 0.5*2.0d-10) GAMA=0.5D0*(2.002d-10-2.001d-10) ! ... (KS) - 24th,May,2016
                EXPG=EXPM1(GAMA*DT)
                EXPB=DEXP(-BETA*DT)

! beta/alfa could be very close to 1 that why I transform it
! remember alfa-beta=gama

                C11=(BETA*DEL1-RW*DEL1-RI*DEL2+DYN1)/ALFA
                C21=(GAMA*DEL1+RW*DEL1+RI*DEL2-GAMA*G31/G2-DYN1)/ALFA
                C12=(BETA*DEL2-PW*DEL1-PI*DEL2+DYN2)/ALFA
                C22=(GAMA*DEL2+PW*DEL1+PI*DEL2-GAMA*G32/G2-DYN2)/ALFA

                A1DEL1N=C11
                A2DEL1N=C11*EXPG
                A3DEL1N=C21*EXPB
                A4DEL1N=G31/G2*(GAMA/ALFA+(GAMA/ALFA-1.0D0)*EXPG)

                DEL1N=A1DEL1N+A2DEL1N+A3DEL1N+A4DEL1N

                A1DEL1INT=C11*EXPG/GAMA
                A2DEL1INT=-C21*EXPB/BETA
                A3DEL1INT=C21/BETA
                A4DEL1INT=G31/G2*DT*(GAMA/ALFA)

                DEL1INT=A1DEL1INT+A2DEL1INT+A3DEL1INT+A4DEL1INT

                A1DEL2N=C12
                A2DEL2N=C12*EXPG
                A3DEL2N=C22*EXPB
                A4DEL2N=G32/G2*(GAMA/ALFA+ &
                        (GAMA/ALFA-1.0D0)* &
                        (GAMA*DT+GAMA*GAMA*DT*DT/2.0D0))

                DEL2N=A1DEL2N+A2DEL2N+A3DEL2N+A4DEL2N

                A1DEL2INT=C12*EXPG/GAMA
                A2DEL2INT=-C22*EXPB/BETA
                A3DEL2INT=C22/BETA
                A4DEL2INT=G32/G2*DT*(GAMA/ALFA)
                A5DEL2INT=G32/G2*(GAMA/ALFA-1.0D0)* &
                                  (GAMA*DT*DT/2.0D0)

                DEL2INT=A1DEL2INT+A2DEL2INT+A3DEL2INT+A4DEL2INT+ &
                        A5DEL2INT

! in case DABS(RW/RI)>1D-12
              ELSE

! in case DABS(RW/RI)<=1D-12

                X=-2.0D0*RW*PI+RW*RW+4.0D0*PW*RI

                ALFA=PI*(1+(X/PI)/2.0D0-(X/PI)*(X/PI)/8.0D0)
                BETA=PI+(X/PI)/4.0D0-(X/PI)*(X/PI)/16.0D0+RW/2.0D0
                GAMA=(X/PI)/4.0D0-(X/PI)*(X/PI)/16.0D0-RW/2.0D0

                EXPG=EXPM1(GAMA*DT)
                EXPB=DEXP(-BETA*DT)

                C11=(BETA*DEL1-RW*DEL1-RI*DEL2+DYN1)/ALFA
                C21=(GAMA*DEL1+RW*DEL1+RI*DEL2-GAMA*G31/G2-DYN1)/ALFA
                C12=(BETA*DEL2-PW*DEL1-PI*DEL2+DYN2)/ALFA
                C22=(GAMA*DEL2+PW*DEL1+PI*DEL2-GAMA*G32/G2-DYN2)/ALFA

                DEL1N=C11+C11*EXPG+C21*EXPB+ &
                          G31/G2*(GAMA/ALFA+(GAMA/ALFA-1)*EXPG)
                DEL1INT=C11*EXPG/GAMA-C21*EXPB/BETA+(C21/BETA)+ &
                            G31/G2*DT*(GAMA/ALFA)
                DEL2N=C12+C12*EXPG+C22*EXPB+G32/G2*(GAMA/ALFA+ &
                        (GAMA/ALFA-1.0D0)* &
                        (GAMA*DT+GAMA*GAMA*DT*DT/2.0D0))
                DEL2INT=C12*EXPG/GAMA-C22*EXPB/BETA+ &
                  (C22/BETA)+G32/G2*DT*(GAMA/ALFA)+ &
                  G32/G2*(GAMA/ALFA-1.0D0)*(GAMA*DT*DT/2.0D0)

! in case DABS(RW/RI)<=1D-12
            ENDIF
! alfa/beta 2
! in case DABS(RI/RW)>1D-12

            ELSE

! in case DABS(RI/RW)<=1D-12

              X=-2.0D0*RW*PI+PI*PI+4.0D0*PW*RI

              ALFA=RW*(1.0D0+(X/RW)/2.0D0-(X/RW)*(X/RW)/8.0D0)
              BETA=RW+(X/RW)/4.0D0-(X/RW)*(X/RW)/16.0D0+PI/2.0D0
              GAMA=(X/RW)/4.0D0-(X/RW)*(X/RW)/16.0D0-PI/2.0D0

              EXPG=EXPM1(GAMA*DT)
              EXPB=DEXP(-BETA*DT)

              C11=(BETA*DEL1-RW*DEL1-RI*DEL2+DYN1)/ALFA
              C21=(GAMA*DEL1+RW*DEL1+RI*DEL2-GAMA*G31/G2-DYN1)/ALFA
              C12=(BETA*DEL2-PW*DEL1-PI*DEL2+DYN2)/ALFA
              C22=(GAMA*DEL2+PW*DEL1+PI*DEL2-GAMA*G32/G2-DYN2)/ALFA

              DEL1N=C11+C11*EXPG+C21*EXPB+ &
                    G31/G2*(GAMA/ALFA+(GAMA/ALFA-1.0D0)*EXPG)
              DEL1INT=C11*EXPG/GAMA-C21*EXPB/BETA+(C21/BETA)+ &
                      G31/G2*DT*(GAMA/ALFA)
              DEL2N=C12+C12*EXPG+C22*EXPB+G32/G2* &
                    (GAMA/ALFA+ &
                    (GAMA/ALFA-1.0D0)*(GAMA*DT+GAMA*GAMA*DT*DT/2.0D0))
              DEL2INT=C12*EXPG/GAMA-C22*EXPB/BETA+C22/BETA+ &
                G32/G2*DT*(GAMA/ALFA)+ &
                G32/G2*(GAMA/ALFA-1.0D0)*(GAMA*DT*DT/2.0D0)
! alfa/beta
! in case DABS(RI/RW)<=1D-12
      ENDIF
! in case DABS(GAMA*DT)<=1D-6
    ENDIF

! water and ice                                                 (end)

! in case ISYM1/=0.AND.ISYM2/=0

        ENDIF

  100    CONTINUE

  201	FORMAT(1X,D13.5)
  202	FORMAT(1X,2D13.5)
  203	FORMAT(1X,3D13.5)
  204	FORMAT(1X,4D13.5)

        RETURN
        END SUBROUTINE JERSUPSAT_KS
        
! SUBROUTINE JERSUPSAT
! ....................................................................
  SUBROUTINE JERDFUN_KS (xi,xiN,B21_MY, &
                          FI2,PSI2,fl2,DEL2N, &
                          ISYM2,IND,ITYPE,TPN,IDROP, &
                          FR_LIM,FRH_LIM,ICEMAX,NKR,COL,Ihydro,Iin,Jin,Kin,Itimestep)

  IMPLICIT NONE
! ... Interface
  INTEGER,INTENT(IN) :: ISYM2, IND, ITYPE, NKR, ICEMAX, Ihydro, Iin, Jin ,Kin, Itimestep
  INTEGER,INTENT(INOUT) :: IDROP
  double precision,INTENT(IN) :: B21_MY(:), FI2(:), FR_LIM(:), FRH_LIM(:), &
                      DEL2N, COL
  double precision,INTENT(IN) :: TPN, xi(:)
  double precision,INTENT(INOUT) :: xiN(:)
  double precision,INTENT(INOUT) :: PSI2(:), FL2(:)
! ... Interface

! ... Locals
  INTEGER :: ITYP, KR, NR, ICE, K, IDSD_Negative
  double precision :: FL2_NEW(NKR), FI2R(NKR), PSI2R(NKR), C, DEGREE1, DEGREE2, DEGREE3, D, RATEXI, &
                        B, A, xiR(NKR),xiNR(NKR), FR_LIM_KR
! ... Locals


  C = 2.0D0/3.0D0

  DEGREE1 = 1.0D0/3.0D0
  DEGREE2 = C
  DEGREE3 = 3.0D0/2.0D0

  IF(IND > 1) THEN
    ITYP = ITYPE
  ELSE
    ITYP = 1
  ENDIF

  DO KR=1,NKR
      PSI2R(KR) = FI2(KR)
      FI2R(KR) = FI2(KR)
  ENDDO

  NR=NKR

! new size distribution functions                             (start)

  IF(ISYM2 == 1) THEN
    IF(IND==1 .AND. ITYPE==1) THEN
! drop diffusional growth
      DO KR=1,NKR
          D=xi(KR)**DEGREE1
          RATExi=C*DEL2N*B21_MY(KR)/D
          B=xi(KR)**DEGREE2
          A=B+RATExi
          IF(A<0.0D0) THEN
            xiN(KR)=1.0D-50
          ELSE
            xiN(KR)=A**DEGREE3
          ENDIF
      ENDDO
! in case IND==1.AND.ITYPE==1
    ELSE
! in case IND/=1.OR.ITYPE/=1
            DO KR=1,NKR
              RATExi = DEL2N*B21_MY(KR)
              xiN(KR) = xi(KR) + RATExi
            ENDDO
    ENDIF

! recalculation of size distribution functions                (start)

      DO KR=1,NKR
        xiR(KR) = xi(KR)
        xiNR(KR) = xiN(KR)
        FI2R(KR) = FI2(KR)
      END DO

        IDSD_Negative = 0
      CALL JERNEWF_KS &
            (NR,xiR,FI2R,PSI2R,xiNR,ISIGN_3POINT,TPN,IDROP,NKR,COL,IDSD_Negative,Ihydro,Iin,Jin,Kin,Itimestep)
      IF(IDSD_Negative == 1)THEN
        IF(ISIGN_KO_1 == 1) THEN
          ! ... (KS) - we do not use Kovatch-Ouland as separate method
          !	CALL JERNEWF_KO_KS &
          !					(NR,xiR,FI2R,PSI2R,xiNR,NKR,COL)
        ENDIF
      ENDIF

        DO KR=1,NKR
          IF(ITYPE==5) THEN
                FR_LIM_KR=FRH_LIM(KR)
          ELSE
                FR_LIM_KR=FR_LIM(KR)
          ENDIF
          IF(PSI2R(KR)<0.0D0) THEN
            PRINT*,    'STOP 1506 : PSI2R(KR)<0.0D0, in JERDFUN_KS'
            !call wrf_error_fatal ("fatal error in PSI2R(KR)<0.0D0, in JERDFUN_KS, model stop")
          ENDIF
          PSI2(KR) = PSI2R(KR)
      ENDDO
! cycle by ICE
! recalculation of size distribution functions                  (end)
! in case ISYM2/=0
  ENDIF
! new size distribution functions                               (end)

  201	FORMAT(1X,D13.5)
  304   FORMAT(1X,I2,2X,4D13.5)

  RETURN
  END SUBROUTINE JERDFUN_KS
        ! +----------------------------------------------------------------------------+
    SUBROUTINE JERNEWF_KS &
                (NRX,RR,FI,PSI,RN,I3POINT,TPN,IDROP,NKR,COL,IDSD_Negative,Ihydro, &
              Iin,Jin,Kin,Itimestep)

        IMPLICIT NONE
! ... Interface
    INTEGER,INTENT(IN) :: NRX, I3POINT, NKR, Ihydro, Iin, Jin, Kin, Itimestep
    INTEGER,INTENT(INOUT) :: IDROP, IDSD_Negative
    double precision,INTENT(IN) :: TPN
    double precision,INTENT(IN) :: COL
    double precision,INTENT(INOUT) :: PSI(:), RN(:), FI(:), RR(:)
! ... Interface

! ... Locals
    INTEGER :: KMAX, KR, I, K , NRXP, ISIGN_DIFFUSIONAL_GROWTH, NRX1,  &
              I3POINT_CONDEVAP, IEvap
    double precision :: RNTMP,RRTMP,RRP,RRM,RNTMP2,RRTMP2,RRP2,RRM2, GN1,GN2, &
                GN3,GN1P,GMAT,GMAT2, &
                CDROP(NRX),DELTA_CDROP(NRX),RRS(NRX+1),PSINEW(NRX+1), &
                PSI_IM,PSI_I,PSI_IP, AOLDCON, ANEWCON, AOLDMASS, ANEWMASS

    INTEGER,PARAMETER :: KRDROP_REMAPING_MIN = 6, KRDROP_REMAPING_MAX = 12
! ... Locals

! >> [KS] 22ndMay19	IF(TPN .LT. 273.15-5.0D0) IDROP=0

! INITIAL VALUES FOR SOME VARIABLES

    NRXP = NRX + 1
!   NRX1 = 24
!   NRX1 = 35
    NRX1 = NKR

    DO I=1,NRX
! RN(I), g - new masses after condensation or evaporation
      IF(RN(I) < 0.0D0) THEN
          RN(I) = 1.0D-50
          FI(I) = 0.0D0
      ENDIF
    ENDDO

! new change 26.10.09                                         (start)
  DO K=1,NRX
      RRS(K)=RR(K)
  ENDDO
! new change 26.10.09                                           (end)

  I3POINT_CONDEVAP = I3POINT

  IEvap = 0
  IF(RN(1) < RRS(1)) THEN
! evaporation
    I3POINT_CONDEVAP = 0
! new change 26.10.09                                         (start)
    IDROP = 0
! new change 26.10.09                                           (end)
    NRX1 = NRX
    IEvap = 1
  ENDIF

  IF(IDROP == 0) I3POINT_CONDEVAP = 0

! new change 26.10.09                                         (start)

  DO K=1,NRX
      PSI(K)=0.0D0
      CDROP(K)=0.0D0
      DELTA_CDROP(K)=0.0D0
      PSINEW(K)=0.0D0
  ENDDO

  RRS(NRXP)=RRS(NRX)*1024.0D0

  PSINEW(NRXP) = 0.0D0

! new change 26.10.09                                           (end)

  ISIGN_DIFFUSIONAL_GROWTH = 0

  DO K=1,NRX
      IF(RN(K).NE.RR(K)) THEN
      ISIGN_DIFFUSIONAL_GROWTH = 1
      GOTO 2000
      ENDIF
  ENDDO

  2000   CONTINUE

  IF(ISIGN_DIFFUSIONAL_GROWTH == 1) THEN

! Kovetz-Olund method                                         (start)

! new change 26.10.09                                         (start)
    DO K=1,NRX1 ! ... [KS] >> NRX1-1
! new change 26.10.09                                           (end)

      IF(FI(K) > 0.0) THEN
        IF(DABS(RN(K)-RR(K)) < 1.0D-16) THEN
            PSINEW(K) = FI(K)*RR(K)
            CYCLE
        ENDIF

        I = 1
        DO WHILE (.NOT.(RRS(I) <= RN(K) .AND. RRS(I+1) >= RN(K)) &
                  .AND.I.LT.NRX1) ! [KS] >> was NRX1-1
                  I = I + 1
        ENDDO

        IF(RN(K).LT.RRS(1)) THEN
          RNTMP=RN(K)
          RRTMP=0.0D0
          RRP=RRS(1)
          GMAT2=(RNTMP-RRTMP)/(RRP-RRTMP)
          PSINEW(1)=PSINEW(1)+FI(K)*RR(K)*GMAT2
        ELSE

        RNTMP=RN(K)
        RRTMP=RRS(I)
        RRP=RRS(I+1)
        GMAT2=(RNTMP-RRTMP)/(RRP-RRTMP)
        GMAT=(RRP-RNTMP)/(RRP-RRTMP)
        PSINEW(I)=PSINEW(I)+FI(K)*RR(K)*GMAT
        PSINEW(I+1)=PSINEW(I+1)+FI(K)*RR(K)*GMAT2
        ENDIF
! in case FI(K).NE.0.0D0
      ENDIF

  3000    CONTINUE

    ENDDO
! cycle by K

    DO KR=1,NRX1
        PSI(KR)=PSINEW(KR)
    ENDDO

    DO KR=NRX1+1,NRX
        PSI(KR)=FI(KR)
    ENDDO
! Kovetz-Olund method                                           (end)

! calculation both new total drop concentrations(after KO) and new
! total drop masses (after KO)

! 3point method	                                              (start)
    IF(I3POINT_CONDEVAP == 1) THEN
      DO K=1,NRX1-1
        IF(FI(K) > 0.0) THEN
          IF(DABS(RN(K)-RR(K)).LT.1.0D-16) THEN
              PSI(K) = FI(K)*RR(K)
              GOTO 3001
            ENDIF

          IF(RRS(2).LT.RN(K)) THEN
              I = 2
              DO WHILE &
                      (.NOT.(RRS(I) <= RN(K) .AND. RRS(I+1) >= RN(K)) &
                      .AND.I.LT.NRX1-1)
                    I = I + 1
              ENDDO
              RNTMP=RN(K)

              RRTMP=RRS(I)
              RRP=RRS(I+1)
              RRM=RRS(I-1)

              RNTMP2=RN(K+1)

              RRTMP2=RRS(I+1)
              RRP2=RRS(I+2)
              RRM2=RRS(I)

              GN1=(RRP-RNTMP)*(RRTMP-RNTMP)/(RRP-RRM)/ &
                  (RRTMP-RRM)

              GN1P=(RRP2-RNTMP2)*(RRTMP2-RNTMP2)/ &
                    (RRP2-RRM2)/(RRTMP2-RRM2)

              GN2=(RRP-RNTMP)*(RNTMP-RRM)/(RRP-RRTMP)/ &
                    (RRTMP-RRM)

              GMAT=(RRP-RNTMP)/(RRP-RRTMP)

              GN3=(RRTMP-RNTMP)*(RRM-RNTMP)/(RRP-RRM)/ &
                                            (RRP-RRTMP)
              GMAT2=(RNTMP-RRTMP)/(RRP-RRTMP)

              PSI_IM = PSI(I-1)+GN1*FI(K)*RR(K)

              PSI_I = PSI(I)+GN1P*FI(K+1)*RR(K+1)+&
                    (GN2-GMAT)*FI(K)*RR(K)

              PSI_IP = PSI(I+1)+(GN3-GMAT2)*FI(K)*RR(K)

              IF(PSI_IM > 0.0D0) THEN

                IF(PSI_IP > 0.0D0) THEN

                  IF(I > 2) THEN
! smoothing criteria
                    IF(PSI_IM > PSI(I-2) .AND. PSI_IM < PSI_I &
                      .AND. PSI(I-2) < PSI(I) .OR. PSI(I-2) >= PSI(I)) THEN

                      PSI(I-1) = PSI_IM

                      PSI(I) = PSI(I) + FI(K)*RR(K)*(GN2-GMAT)

                      PSI(I+1) = PSI_IP
! in case smoothing criteria
                    ENDIF
! in case I.GT.2
                  ENDIF

! in case PSI_IP.GT.0.0D0
              ELSE
                      EXIT
              ENDIF
! in case PSI_IM.GT.0.0D0
          ELSE
                EXIT
          ENDIF
! in case I.LT.NRX1-2
!         ENDIF

! in case RRS(2).LT.RN(K)
        ENDIF

! in case FI(K).NE.0.0D0
      ENDIF

  3001 CONTINUE

      ENDDO
        ! cycle by K

      ! in case I3POINT_CONDEVAP.NE.0
    ENDIF
! 3 point method                                                (end)

! PSI(K) - new hydrometeor size distribution function

    DO K=1,NRX1
        PSI(K)=PSI(K)/RR(K)
    ENDDO

    DO K=NRX1+1,NRX
      PSI(K)=FI(K)
    ENDDO

    IF(IDROP == 1) THEN
        DO K=KRDROP_REMAPING_MIN,KRDROP_REMAPING_MAX
          CDROP(K)=3.0D0*COL*PSI(K)*RR(K)
        ENDDO
          ! KMAX - right boundary spectrum of drop sdf
          !(KRDROP_REMAP_MIN =< KMAX =< KRDROP_REMAP_MAX)
        DO K=KRDROP_REMAPING_MAX,KRDROP_REMAPING_MIN,-1
            KMAX=K
            IF(PSI(K).GT.0.0D0) GOTO 2011
        ENDDO

    2011  CONTINUE
  ! Andrei's new change 28.04.10                                (start)
        DO K=KMAX-1,KRDROP_REMAPING_MIN,-1
  ! Andrei's new change 28.04.10                                  (end)
          IF(CDROP(K).GT.0.0D0) THEN
            DELTA_CDROP(K)=CDROP(K+1)/CDROP(K)
              IF(DELTA_CDROP(K).LT.COEFF_REMAPING) THEN
                CDROP(K)=CDROP(K)+CDROP(K+1)
                CDROP(K+1)=0.0D0
              ENDIF
          ENDIF
        ENDDO

        DO K=KRDROP_REMAPING_MIN,KMAX
          PSI(K)=CDROP(K)/(3.0D0*COL*RR(K))
        ENDDO

  ! in case IDROP.NE.0
      ENDIF

! new change 26.10.09                                           (end)

! in case ISIGN_DIFFUSIONAL_GROWTH.NE.0
        ELSE
! in case ISIGN_DIFFUSIONAL_GROWTH.EQ.0
          DO K=1,NRX
              PSI(K)=FI(K)
          ENDDO
        ENDIF

    DO KR=1,NRX
          IF(PSI(KR) < 0.0) THEN ! ... (KS)
          IDSD_Negative = 1
          print*, "IDSD_Negative=",IDSD_Negative,"kr",kr
          PRINT*,    'IN SUBROUTINE JERNEWF'
          PRINT*,		'PSI(KR)<0'
          PRINT*,    'BEFORE EXIT'
          PRINT*,    'ISIGN_DIFFUSIONAL_GROWTH'
          PRINT*,     ISIGN_DIFFUSIONAL_GROWTH
          PRINT*,    'I3POINT_CONDEVAP'
          PRINT*,     I3POINT_CONDEVAP
          PRINT*,    'K,RR(K),RN(K),K=1,NRX'
          PRINT*,    (K,RR(K),RN(K),K=1,NRX)
          PRINT*,    'K,RR(K),RN(K),FI(K),PSI(K),K=1,NRX'
          PRINT 304, (K,RR(K),RN(K),FI(K),PSI(K),K=1,NRX)
          PRINT*,		IDROP,Ihydro,Iin,Jin,Kin,Itimestep
          !call wrf_error_fatal ("fatal error in SUBROUTINE JERNEWF PSI(KR)<0, < min, model stop")
      ENDIF
    ENDDO

  304   FORMAT(1X,I2,2X,4D13.5)

        RETURN
        END SUBROUTINE JERNEWF_KS
! +------------------------------------------------------------------+
  SUBROUTINE JERDFUN_NEW_KS &
                (xi,xiN,B21_MY, &
              FI2,PSI2, &
              TPN,IDROP,FR_LIM,NKR,COL,Ihydro,Iin,Jin,Kin,Itimestep)

  IMPLICIT NONE

! ... Interface
  INTEGER,INTENT(INOUT) :: IDROP, NKR
  INTEGER,INTENT(IN) :: Ihydro,Iin,Jin,Kin,Itimestep
  double precision,intent(IN) :: FI2(:), B21_MY(:), FR_LIM(:), COL
  double precision, INTENT(IN) :: TPN, xi(:)
  double precision,INTENT(INOUT) :: PSI2(:)
  double precision,INTENT(INOUT) :: xiN(:)
! ... Interface

! ... Locals
  INTEGER :: NR, KR, IDSD_Negative
  double precision :: C, DEGREE1, DEGREE2, DEGREE3, D, RATEXI, B, A, &
                        xiR(NKR),FI2R(NKR),PSI2R(NKR),xiNR(NKR)
! ... Locals

  C=2.0D0/3.0D0

  DEGREE1=C/2.0D0
  DEGREE2=C
  DEGREE3=3.0D0/2.0D0

  NR=NKR

  xiR = xi
  FI2R = FI2
  PSI2R = PSI2
  xiNR = xiN

! new drop size distribution functions                             (start)

! drop diffusional growth

  DO KR=1,NKR
      D = xiR(KR)**DEGREE1
! Andrei's new change of 3.09.10                              (start)
!	   RATExi=C*DEL2N*B21_MY(KR)/D
      RATExi = C*B21_MY(KR)/D
! Andrei's new change of 3.09.10                                (end)
      B = xiR(KR)**DEGREE2
      A = B+RATExi
      IF(A<0.0D0) THEN
        xiNR(KR) = 1.0D-50
      ELSE
        xiNR(KR) = A**DEGREE3
      ENDIF
  ENDDO

! recalculation of size distribution functions                (start)

  IDSD_Negative = 0
  CALL JERNEWF_KS &
      (NR,xiR,FI2R,PSI2R,xiNR,ISIGN_3POINT,TPN,IDROP,NKR,COL,IDSD_Negative,Ihydro,Iin,Jin,Kin,Itimestep)
  IF(IDSD_Negative == 1)THEN
    IF(ISIGN_KO_2 == 1) THEN
      ! ... (KS) - we do not use Kovatch-Ouland as separate method
      !	CALL JERNEWF_KO_KS &
      !  				(NR,xiR,FI2R,PSI2R,xiNR,NKR,COL)
    ENDIF
  ENDIF

  PSI2 = PSI2R

! recalculation of drop size distribution functions                  (end)
! new drop size distribution functions                          (end)

  201	FORMAT(1X,D13.5)

  RETURN
  END SUBROUTINE JERDFUN_NEW_KS
! +---------------------------------------------------------+
  SUBROUTINE Relaxation_Time(TPS,QPS,PP,ROR,DEL1S,DEL2S, &
                              R1,VR1,FF1in,RLEC,RO1BL, &
                              R2,VR2,FF2in,RIEC,RO2BL, &
                              R3,VR3,FF3in,RSEC,RO3BL, &
                              R4,VR4,FF4in,RGEC,RO4BL, &
                              R5,VR5,FF5in,RHEC,RO5BL, &
                              NKR,ICEMAX,COL,DTdyn,NCOND,DTCOND)

  implicit none
! ... Interface
  integer,intent(in) :: NKR,ICEMAX
  integer,intent(out) :: NCOND
  double precision,intent(in) :: R1(:),FF1in(:),RLEC(:),RO1BL(:), &
              R2(:,:),FF2in(:,:),RIEC(:,:),RO2BL(:,:), &
              R3(NKR),FF3in(:),RSEC(:),RO3BL(:), &
              R4(NKR),FF4in(:),RGEC(:),RO4BL(:), &
              R5(NKR),FF5in(:),RHEC(:),RO5BL(:), &
              ROR,COL,DTdyn,VR1(:),VR2(:,:),VR3(:),VR4(:),VR5(:)
  double precision,intent(in) :: TPS,QPS,PP,DEL1S,DEL2S
  double precision,intent(out) :: DTCOND
! ... Interface
! ... Local
  integer :: ISYM1, ISYM2(ICEMAX), ISYM3, ISYM4, ISYM5, ISYM_SUM, ICM
  double precision,parameter :: AA1_MY = 2.53D12, BB1_MY = 5.42D3, AA2_MY = 3.41D13, &
                                  BB2_MY = 6.13E3, AL1 = 2500.0, AL2 = 2834.0
  double precision,parameter :: TAU_Min = 0.1 ! [s]
  double precision :: OPER2, AR1, TAU_RELAX, B5L, B5I, &
                        R1D(NKR), R2D(NKR,ICEMAX), R3D(NKR), R4D(NKR), R5D(NKR), &
                        VR1_d(nkr),VR2_d(nkr,icemax),VR3_d(nkr),VR4_d(nkr),VR5_d(nkr)
  double precision :: B11_MY(NKR), B21_MY(NKR,ICEMAX), B31_MY(NKR), &
                        B41_MY(NKR), B51_MY(NKR), FL1(NKR), FL3(NKR), FL4(NKR), FL5(NKR), &
                        SFNDUMMY(3), SFN11, SFNI1(ICEMAX), SFNII1, SFN21, SFN31, SFN41, SFN51, SFNI, SFNL, B8L, B8I, RI, PW, &
                        DOPL, DOPI, TAU_w, TAU_i, phi, RW, PI
! ... Local

    OPER2(AR1)=0.622/(0.622+0.378*AR1)/AR1
    VR1_d = VR1
    VR2_d = VR2
    VR3_d = VR3
    VR4_d = VR4
    VR5_d = VR5


    ISYM1 = 0
    ISYM2 = 0
    ISYM3 = 0
    ISYM4 = 0
    ISYM5 = 0
    IF(sum(FF1in) > 0.0) ISYM1 = 1
    IF(sum(FF2in(:,1)) > 1.0D-10) ISYM2(1) = 1
    IF(sum(FF2in(:,2)) > 1.0D-10) ISYM2(2) = 1
    IF(sum(FF2in(:,3)) > 1.0D-10) ISYM2(3) = 1
    IF(sum(FF3in) > 1.0D-10) ISYM3 = 1
    IF(sum(FF4in) > 1.0D-10) ISYM4 = 1
    IF(sum(FF5in) > 1.0D-10) ISYM5 = 1

    ISYM_SUM = ISYM1 + sum(ISYM2) + ISYM3 + ISYM4  + ISYM5
    IF(ISYM_SUM == 0)THEN
      TAU_RELAX = DTdyn
      NCOND = nint(DTdyn/TAU_RELAX)
        DTCOND = TAU_RELAX
        RETURN
    ENDIF

    R1D = R1
    R2D = R2
    R3D = R3
    R4D = R4
    R5D = R5
    B8L=1./ROR
      B8I=1./ROR
    ICM = ICEMAX
    SFN11 = 0.0
    SFNI1 = 0.0
    SFN31 = 0.0
    SFN41 = 0.0
    SFN51 = 0.0
    B11_MY = 0.0
    B21_MY = 0.0
    B31_MY = 0.0
    B41_MY = 0.0
    B51_MY = 0.0


      ! ... Drops
      IF(ISYM1 == 1)THEN
        FL1 = 0.0
        CALL JERRATE_KS(R1D,TPS,PP,VR1_d,RLEC,RO1BL,B11_MY,1,1,fl1,NKR,ICEMAX)
        sfndummy(1) = SFN11
        CALL JERTIMESC_KS(FF1in,R1D,SFNDUMMY,B11_MY,B8I,1,NKR,ICEMAX,COL)
        SFN11 = sfndummy(1)
      ENDIF
      ! ... IC
      !IF(sum(ISYM2) > 0) THEN
      !	FL1 = 0.0
      !	! ... ice crystals
      !	CALL JERRATE_KS (R2D,TPS,PP,VR2_d,RIEC,RO2BL,B21_MY,3,2,fl1,NKR,ICEMAX)
      !	CALL JERTIMESC_KS (FF2in,R2D,SFNI1,B21_MY,B8I,ICM,NKR,ICEMAX,COL)
      !ENDIF
      ! ... Snow
      IF(ISYM3 == 1) THEN
        FL3 = 0.0
        ! ... snow
        CALL JERRATE_KS (R3D,TPS,PP,VR3_d,RSEC,RO3BL,B31_MY,1,3,fl3,NKR,ICEMAX)
        sfndummy(1) = SFN31
        CALL JERTIMESC_KS(FF3in,R3D,SFNDUMMY,B31_MY,B8I,1,NKR,ICEMAX,COL)
          SFN31 = sfndummy(1)
        ENDIF
      ! ... Graupel
      IF(ISYM4 == 1) THEN
        FL4 = 0.0
        ! ... graupel
        CALL JERRATE_KS(R4D,TPS,PP,VR4_d,RGEC,RO4BL,B41_MY,1,2,fl4,NKR,ICEMAX)

        sfndummy(1) = SFN41
        CALL JERTIMESC_KS(FF4in,R4D,SFNDUMMY,B41_MY,B8I,1,NKR,ICEMAX,COL)
          SFN41 = sfndummy(1)
      ENDIF
      ! ... Hail
      IF(ISYM5 == 1) THEN
        FL5 = 0.0
        ! ... hail
        CALL JERRATE_KS(R5D,TPS,PP,VR5_d,RHEC,RO5BL,B51_MY,1,2,fl5,NKR,ICEMAX)

        sfndummy(1) = SFN51
        CALL JERTIMESC_KS(FF5in,R5D,SFNDUMMY,B51_MY,B8I,1,NKR,ICEMAX,COL)
        SFN51 = sfndummy(1)
      ENDIF

      SFNII1 = 0.0
      SFN21 = 0.0
      SFNL = 0.0
      SFNI = 0.0
      RI = 0.0
      PW = 0.0
      SFNII1 = SFNI1(1)+SFNI1(2)+SFNI1(3)
      SFN21 = SFNII1 + SFN31 + SFN41 + SFN51
      SFNL = SFN11  ! Liquid
      SFNI = SFN21 	! Total Ice

      B5L=BB1_MY/TPS/TPS
      B5I=BB2_MY/TPS/TPS
      DOPL=1.+ DEL1S
      DOPI=1.+ DEL2S
      RW=(OPER2(QPS)+B5L*AL1)*DOPL*SFNL
      RI=(OPER2(QPS)+B5L*AL2)*DOPL*SFNI
      PW=(OPER2(QPS)+B5I*AL1)*DOPI*SFNL
      PI=(OPER2(QPS)+B5I*AL2)*DOPI*SFNI

      TAU_w = DTdyn
      TAU_i = DTdyn
      phi = (1.0 + DEL2S)/(1.0 + DEL1S)
      if(PW > 0.0 .or. PI > 0.0) TAU_w = (PW + phi*PI)**(-1.0)
      if(RW > 0.0 .or. RI > 0.0) TAU_i =  phi/(RW + RI*phi)
      TAU_RELAX = DTdyn
      IF(PW > 0.0 .or. RI > 0.0) TAU_RELAX = (PW + RI)**(-1.0)/3.0
      IF(PW > 0.0 .and. RI > 0.0) TAU_RELAX = min(TAU_w,TAU_i)/3.0

      if(TAU_RELAX > DTdyn) TAU_RELAX = DTdyn/3.0
      if(TAU_RELAX < TAU_Min) TAU_RELAX = TAU_Min
      IF(PW <= 0.0 .and. RI <= 0.0) TAU_RELAX = DTdyn

      !if(TAU_RELAX < DTdyn .and. IDebug_Print_DebugModule==1)then
      !		print*,"in Relaxation_Time,TAU_RELAX < DTdyn"
      !  	print*,TAU_RELAX
      !endif

      !NCOND = nint(DTdyn/TAU_RELAX)
      NCOND = ceiling(DTdyn/TAU_RELAX)
      DTCOND = TAU_RELAX

  RETURN
  END SUBROUTINE Relaxation_Time
! +------------------------------+
! +------------------------------------------------------------------------------------------+
subroutine CCN_regeneration(NKR,COL,Evap_tot,FCCN,FCCN_nucl,Del_CCNreg,Imethod,Iin,Jin,Kin)

  implicit none

  integer,intent(in) :: NKR,Imethod,Iin,Jin,Kin
  double precision,intent(in) :: COL,Evap_tot
  double precision,intent(inout) :: FCCN(:),FCCN_nucl(:),Del_CCNreg

  integer :: kr,kr_max,kr_min,Ifound,krr_max
  double precision :: FCCN_diff(nkr),Norm_f,ret_conc,conc_bin_min,Evap_tot_tmp,ret_ccn

  select case (Imethod)
  case(1)

    conc_bin_min = 1.0e-30/NKR
    !FCCN_diff = 0.0
    !do kr = 1,nkr
    !    FCCN_diff(kr) = max(FCCN_passive(kr) - FCCN_nucl(kr),conc_bin_min)
    !end do
    ! This can be negative

    kr_min = -1
    kr_max = -1
    do kr = 1,nkr
        if(FCCN_nucl(kr) > conc_bin_min)then
            kr_min = kr
            exit
        endif
    end do
    do kr = nkr,1,-1
        if(FCCN_nucl(kr) > conc_bin_min)then
            kr_max = kr
            exit
        endif
    end do
    if(kr_max == -1 .and. kr_min == -1) return

    !FCCN_diff(1:nkr) = FCCN_nucl(1:nkr) - FCCN(1:nkr)
    !Norm_f = Evap_tot / sum(FCCN_diff(kr_min:kr_max))*col ! in [#/cm3]
    !if(Norm_f > 1.0) Norm_f = 1.0

    Del_CCNreg = 0.0
    do kr = kr_min,kr_max
        !if(FCCN_diff(kr) < 0.0) !call wrf_error_fatal ("fatal error in CCN_regenaration (FCCN_diff < 0.0), model stop")
        !if(FCCN_diff(kr) < 0.0) stop "CCN_reg"
        !FCCN(kr) = FCCN(kr) + Norm_f*max(FCCN_nucl(kr),0.0)
        !FCCN_nucl(kr) = FCCN_nucl(kr) - Norm_f*max(FCCN_nucl(kr),0.0)
        !FCCN(kr) = FCCN(kr) + Norm_f*FCCN_diff(kr)
        ret_ccn = 0.0
        if(sum(FCCN_nucl) > 1.0e-30) ret_ccn = (Evap_tot/col)*(FCCN_nucl(kr)/sum(FCCN_nucl))
        !FCCN(kr) = FCCN(kr) + Evap_tot/col/(kr_max-kr_min+1)
        FCCN(kr) = FCCN(kr) + ret_ccn ! in cm-3/bin
        Del_CCNreg = Del_CCNreg + ret_ccn
        !FCCN_nucl(kr) = FCCN_nucl(kr) - min(Evap_tot/col/(kr_max-kr_min+1),FCCN_diff(kr))
    end do

  case(2)

    conc_bin_min = 1.0e-30/NKR
    !FCCN_diff = 0.0
    !do kr = 1,nkr
    !    FCCN_diff(kr) = max(FCCN_passive(kr) - FCCN_nucl(kr),conc_bin_min)
    !end do
    ! This can be negative

    kr_min = -1
    kr_max = -1
    do kr = 1,nkr
        if(FCCN_nucl(kr) > conc_bin_min)then
            kr_min = kr
            exit
        endif
    end do
    do kr = nkr,1,-1
        if(FCCN_nucl(kr) > conc_bin_min)then
            kr_max = kr
            exit
        endif
    end do
    if(kr_max == -1 .and. kr_min == -1) return

    Evap_tot_tmp = Evap_tot
    print*,'1 -','Evap_tot_tmp',Evap_tot_tmp
    Del_CCNreg = 0.0
    do kr = kr_min,kr_max
        if( FCCN_nucl(kr) > 1.0e-30 .and. col*FCCN_nucl(kr) <= Evap_tot_tmp )then
            print*,'2 -','col*FCCN(kr)',kr,col*FCCN(kr)
            print*,'3 -','col*FCCN_nucl(kr)',kr,col*FCCN_nucl(kr)
            FCCN(kr) = FCCN(kr) + FCCN_nucl(kr)
            Del_CCNreg = Del_CCNreg + col*FCCN_nucl(kr)
            Evap_tot_tmp = Evap_tot_tmp - col*FCCN_nucl(kr)
            !FCCN_nucl(kr) = FCCN_nucl(kr) - min(FCCN_nucl(kr),FCCN_nucl(kr))
            cycle
        else if( FCCN_nucl(kr) > 1.0e-30 .and. col*FCCN_nucl(kr) > Evap_tot_tmp )then
            print*,'4 -','col*FCCN_nucl(kr)',kr,col*FCCN_nucl(kr)
            print*,'5 -','Evap_tot_tmp',kr,Evap_tot_tmp
            FCCN(kr) = FCCN(kr) + Evap_tot_tmp/col
            Del_CCNreg = Del_CCNreg + Evap_tot_tmp
            !FCCN_nucl(kr) = FCCN_nucl(kr) - min(Evap_tot_tmp/col,FCCN_nucl(kr))
            Evap_tot_tmp = 0.0
            exit
        endif
    end do

  end select

  return
  end subroutine CCN_regeneration
! +------------------------------------------+
! +-----------------------------------------------------------------------------+
  SUBROUTINE JERNUCL01_KS(PSI1_r,FCCNR_r,FCCNR_nucl_r,                    &
                          XL_r,TT,QQ, 			                              &
                          ROR_r,PP_r, 				                            &
                          SUP1,SUP2,      			  		                    &
                          COL_r, 							                            &
                          RCCN_r,DROPRADII_r,NKR,NKR_aerosol,             &
                          Win_r,Is_This_CloudBase,RO_SOLUTE,IONS,MWAERO,  &
                          Iin,Jin,Kin)

implicit none

integer,intent(in) :: 	 Kin,Jin,Iin,NKR,NKR_aerosol,IONS,Is_This_CloudBase
double precision,intent(in) ::     XL_r(:),ROR_r,PP_r,COL_r,RCCN_r(:),DROPRADII_r(:)
double precision,intent(in) ::	 	  MWAERO,RO_SOLUTE,Win_r
double precision,intent(inout) :: 	PSI1_r(:),FCCNR_r(:),FCCNR_nucl_r(:)
double precision,intent(inout) ::  TT,QQ,SUP1,SUP2

! ... Locals
integer :: KR, ICE, K
double precision :: DROPCONCN(NKR), ARG_1, COL3, RORI, TPN, QPN, TPC, AR1, AR2, OPER3,           &
             DEL2N, Win
double precision,parameter :: AL2 = 2834.0D0
double precision :: PSI1(NKR),FCCNR(NKR_aerosol),FCCNR_nucl(NKR_aerosol),ROR,XL(NKR),PP,COL, &
             RCCN(NKR_aerosol),DROPRADII(NKR)
double precision :: TPNreal
! ... Locals

OPER3(AR1,AR2) = AR1*AR2/(0.622D0+0.378D0*AR1)

! ... Adjust the Imput
PSI1 = PSI1_r
FCCNR = FCCNR_r
FCCNR_nucl = FCCNR_nucl_r
XL = XL_r
ROR = ROR_r
PP = PP_r
COL = COL_r
RCCN = RCCN_r
DROPRADII = DROPRADII_r
Win = Win_r

COL3 = 3.0D0*COL
RORI = 1.0D0/ROR

! ... Drop Nucleation (start)
TPN = TT
QPN = QQ

TPC = TT - 273.15D0

IF(SUP1>0.0D0 .AND. TPC > T_NUCL_DROP_MIN) THEN
if(sum(FCCNR) > 0.0)then
DROPCONCN = 0.0D0
CALL WATER_NUCLEATION (COL, NKR_aerosol, PSI1, FCCNR, FCCNR_nucl, xl, TT, QQ, ROR, SUP1, DROPCONCN, &
                   PP, Is_This_CloudBase, Win, RO_SOLUTE, RCCN, IONS,MWAERO)
endif
ENDIF
! ... Drop nucleation (end)


! ... Nucleation of crystals (end)

! ... Output
PSI1_r = PSI1
FCCNR_r = FCCNR
FCCNR_nucl_r = FCCNR_nucl

RETURN
END SUBROUTINE JERNUCL01_KS
! +-------------------------------------------------------------------------------------------------------------------------+
SUBROUTINE WATER_NUCLEATION (COL, NKR, PSI1, FCCNR, FCCNR_nucl, xl, TT, QQ, ROR, SUP1,     &
                            DROPCONCN, PP, Is_This_CloudBase, Win, RO_SOLUTE, &
                            RCCN, IONS, MWAERO)

!===================================================================!
!                                                                   !
! DROP NUCLEATION SCHEME                                            !
!                                                                   !
! Authors: Khain A.P. & Pokrovsky A.G. July 2002 at Huji, Israel    !
!                                                                   !
!===================================================================!
implicit none

! PSI1(KR), 1/g/cm3 - non conservative drop size distribution function
! FCCNR(KR), 1/cm^3 - aerosol(CCN) non conservative, size distribution function
! xl((KR), g        - drop bin masses

integer,intent(in) :: 			Is_This_CloudBase, NKR, IONS
double precision,intent(in) :: 	xl(:), ROR, PP, Win, RCCN(:), COL
double precision,intent(inout) :: FCCNR(:), FCCNR_nucl(:), PSI1(:), DROPCONCN(:), QQ, TT, SUP1
double precision,intent(in) :: 	 RO_SOLUTE, MWAERO

! ... Locals
integer :: 			IMAX, I, NCRITI, KR
double precision :: DX,AR2,RCRITI,DEG01,RORI,CCNCONC(NKR),AKOE,BKOE, AR1, OPER3, RCCN_MINIMUM, &
               DLN1, DLN2, RMASSL_NUCL, ES1N, EW1N
double precision,parameter :: AL1 = 2500.0D0
double precision :: TTreal
! ... Locals

OPER3(AR1,AR2)=AR1*AR2/(0.622D0+0.378D0*AR1)

DROPCONCN(:) = 0.0D0

DEG01 = 1.0D0/3.0D0
RORI=1.0/ROR

!RO_SOLUTE=2.16D0

! imax - right CCN spectrum boundary
IMAX = NKR
DO I=IMAX,1,-1
IF(FCCNR(I) > 0.0D0) THEN
IMAX = I
exit
ENDIF
ENDDO

NCRITI=0
! every iteration we will nucleate one bin, then we will check the new supersaturation
! and new Rcriti.
do while (IMAX>=NCRITI)
CCNCONC = 0.0

! akoe & bkoe - constants in Koehler equation
AKOE=3.3D-05/TT
!BKOE=2.0D0*4.3D0/(22.9D0+35.5D0)
BKOE = ions*4.3/mwaero
BKOE=BKOE*(4.0D0/3.0D0)*3.141593D0*RO_SOLUTE

if(Use_cloud_base_nuc == 1) then
if(Is_This_CloudBase == 1) then
    CALL Cloud_Base_Super (FCCNR, RCCN, TT, PP, Win, NKR, RCRITI, RO_SOLUTE, IONS, MWAERO, COL)
else
     ! rcriti, cm - critical radius of "dry" aerosol
    RCRITI = (AKOE/3.0D0)*(4.0D0/BKOE/SUP1/SUP1)**DEG01
endif
else ! ismax_cloud_base==0
  ! rcriti, cm - critical radius of "dry" aerosol
  RCRITI=(AKOE/3.0D0)*(4.0D0/BKOE/SUP1/SUP1)**DEG01
endif

IF(RCRITI >= RCCN(IMAX)) EXIT ! nothing to nucleate

! find the minimum bin to nucleate
NCRITI = IMAX
do while (RCRITI<=RCCN(NCRITI) .and. NCRITI>1)
  NCRITI=NCRITI-1
enddo

! rccn_minimum - minimum aerosol(ccn) radius
RCCN_MINIMUM = RCCN(1)/10000.0D0
! calculation of ccnconc(ii)=fccnr(ii)*col - aerosol(ccn) bin
!                                            concentrations,
!                                            ii=imin,...,imax
! determination of ncriti   - number bin in which is located rcriti
! calculation of ccnconc(ncriti)=fccnr(ncriti)*dln1/(dln1+dln2),
! where,
! dln1=Ln(rcriti)-Ln(rccn_minimum)
! dln2=Ln(rccn(1)-Ln(rcriti)
! calculation of new value of fccnr(ncriti)

! each iteration we nucleate the last bin
IF (NCRITI==IMAX-1) then
  if (NCRITI>1) then
     DLN1=DLOG(RCRITI)-DLOG(RCCN(IMAX-1))
     DLN2=COL-DLN1
     CCNCONC(IMAX)=DLN2*FCCNR(IMAX)
     FCCNR_nucl(IMAX) = FCCNR_nucl(IMAX) + FCCNR(IMAX)*(1.0 - DLN1/COL)
     FCCNR(IMAX)=FCCNR(IMAX)*DLN1/COL
  else ! NCRITI==1
     DLN1=DLOG(RCRITI)-DLOG(RCCN_MINIMUM)
     DLN2=DLOG(RCCN(1))-DLOG(RCRITI)
     CCNCONC(IMAX)=DLN2*FCCNR(IMAX)
     FCCNR_nucl(IMAX) = FCCNR_nucl(IMAX) + FCCNR(IMAX)*(1.0 - (DLN1/(DLN1+DLN2)))
     FCCNR(IMAX)=FCCNR(IMAX)*DLN1/(DLN1+DLN2)
  endif
else
   CCNCONC(IMAX) = COL*FCCNR(IMAX)
   FCCNR_nucl(IMAX) = FCCNR_nucl(IMAX) + FCCNR(IMAX)
   FCCNR(IMAX) = 0.0D0
endif

! calculate the mass change due to nucleation
RMASSL_NUCL=0.0D0
if (IMAX <= NKR-7) then ! we pass it to drops mass grid
   DROPCONCN(1) = DROPCONCN(1)+CCNCONC(IMAX)
   RMASSL_NUCL = RMASSL_NUCL+CCNCONC(IMAX)*XL(1)*XL(1)
else
   DROPCONCN(8-(NKR-IMAX)) = DROPCONCN(8-(NKR-IMAX))+CCNCONC(IMAX)
   RMASSL_NUCL = RMASSL_NUCL + CCNCONC(IMAX)*XL(8-(NKR-IMAX))*XL(8-(NKR-IMAX))
endif
RMASSL_NUCL = RMASSL_NUCL*COL*3.0*RORI

! prepering to check if we need to nucleate the next bin
IMAX = IMAX-1

! cycle IMAX>=NCRITI
end do

! ... Intergarting for including the previous nucleated drops
IF(sum(DROPCONCN) > 0.0)THEN
DO KR = 1,8
DX = 3.0D0*COL*xl(KR)
PSI1(KR) = PSI1(KR)+DROPCONCN(KR)/DX
ENDDO
ENDIF

RETURN
END SUBROUTINE WATER_NUCLEATION
! +----------------------------------------------------------------------------------------------+
SUBROUTINE Cloud_Base_Super (FCCNR, RCCN, TT, PP, Wbase, NKR, RCRITI, RO_SOLUTE, IONS, MWAERO, &
                            COL)

implicit none

! RCCN(NKR),  cm- aerosol's radius

! FCCNR(KR), 1/cm^3 - aerosol(CCN) non conservative, size
!                     distribution function in point with X,Z
!                     coordinates, KR=1,...,NKR
integer,intent(in) :: 				   NKR, IONS
double precision,intent(in) ::  TT, PP, Wbase, RCCN(:), COL
double precision,intent(inout) :: 	FCCNR(:), RCRITI
double precision,intent(in) ::  MWAERO, RO_SOLUTE

! ... Locals
integer :: NR, NN, KR
double precision :: PL(NKR), supmax(NKR), AKOE, BKOE, C3, PR, CCNCONACT, DL1, DL2, &
                TPC
! ... Locals

CALL supmax_COEFF(AKOE,BKOE,C3,PP,TT,RO_SOLUTE,IONS,MWAERO)

! supmax calculation

! 'Analytical estimation of droplet concentration at cloud base', eq.21, 2012
! calculation of right side hand of equation for S_MAX
! while wbase>0, calculation PR

PR = C3*wbase**(0.75D0)

! calculation supersaturation in cloud base

SupMax = 999.0
PL = 0.0
NN = -1
DO NR=2,NKR
supmax(NR)=DSQRT((4.0D0*AKOE**3.0D0)/(27.0D0*RCCN(NR)**3.0D0*BKOE))
! calculation CCNCONACT- the concentration of ccn that were activated
! following nucleation
! CCNCONACT=N from the paper
! 'Analytical estimation of droplet concentration at cloud base', eq.19, 2012
! CCNCONACT, 1/cm^3- concentration of activated CCN = new droplet concentration
! CCNCONACT=FCCNR(KR)*COL
! COL=Ln2/3

CCNCONACT=0.0D0

! NR represents the number of bin in which rcriti is located
! from NR bin to NKR bin goes to droplets

DO KR=NR,NKR
CCNCONACT = CCNCONACT + COL*FCCNR(KR)
ENDDO

! calculate LHS of equation for S_MAX
! when PL<PR ccn will activate

PL(NR)=supmax(NR)*(DSQRT(CCNCONACT))
IF(PL(NR).LE.PR) THEN
NN = NR
EXIT
ENDIF

END DO ! NR

if (nn == -1) then
print*,"PR, Wbase [cm/s], C3",PR,wbase,C3
print*,"PL",PL
!call wrf_error_fatal  ( 'NN is not defined in cloud base routine, model stop' )
endif

! linear interpolation- finding radius criti of aerosol between
! bin number (nn-1) to (nn)
! 1) finding the difference between pl and pr in the left and right over the
! final bin.

DL1 = dabs(PL(NN-1)-PR) ! left side in the final bin
DL2 = dabs(PL(NN)-PR)   ! right side in the final bin

! 2) fining the left part of bin that will not activate
!	DLN1=COL*DL1/(DL2+DL1)
! 3)finding the right part of bin that activate
!	DLN2=COL-DLN1
! 4)finding radius criti of aerosol- RCRITI

RCRITI = RCCN(NN-1)*dexp(COL*DL1/(DL1+DL2))

! end linear interpolation

RETURN
END SUBROUTINE Cloud_Base_Super
! +-------------------------------------------------------------------+
SUBROUTINE supmax_COEFF (AKOE,BKOE,C3,PP,TT,RO_SOLUTE,IONS,MWAERO)

implicit none

! akoe, cm- constant in Koehler equation
! bkoe    - constant in Koehler equation
! F, cm^-2*s-  from Koehler equation
! C3 - coefficient depends on thermodynamical parameters
! PP, (DYNES/CM/CM)- PRESSURE
! TT, (K)- temperature

integer,intent(in) :: IONS
double precision ,intent(in) :: 	PP, TT
double precision ,intent(out) :: AKOE, BKOE, C3
double precision,intent(in) :: 				MWAERO, RO_SOLUTE

! ... Local
double precision ,parameter :: RV_MY = 461.5D4, CP = 1005.0D4, G = 9.8D2, RD_MY = 287.0D4, & ![cgs]
                          PI = 3.141593D0
double precision :: PZERO,TZERO,ALW1,SW,RO_W,HC,EW,RO_V,DV,RO_A,FL,FR,F,TPC,QV,A1,A2, &
               C1,C2,DEG01,DEG02
! ... Local

TPC = TT-273.15d0

! CGS :
! RV_MY, CM*CM/SEC/SEC/KELVIN - INDIVIDUAL GAS CONSTANT
!                               FOR WATER VAPOUR
!RV_MY=461.5D4

! CP,  CM*CM/SEC/SEC/KELVIN- SPECIFIC HEAT CAPACITY OF
!	                            MOIST AIR AT CONSTANT PRESSURE
!CP=1005.0D4

! G,  CM/SEC/SEC- ACCELERATION OF GRAVITY
!G=9.8D2

! RD_MY, CM*CM/SEC/SEC/KELVIN - INDIVIDUAL GAS CONSTANT
!                               FOR DRY AIR
!RD_MY=287.0D4

! AL2_MY, CM*CM/SEC/SEC - LATENT HEAT OF SUBLIMATION

!	AL2_MY=2.834D10

! PZERO, DYNES/CM/CM - REFERENCE PRESSURE
PZERO=1.01325D6

! TZERO, KELVIN - REFERENCE TEMPERATURE
TZERO=273.15D0

! AL1_MY, CM*CM/SEC/SEC - LATENT HEAT OF VAPORIZATION
! ALW1=AL1_MY - ALW1 depends on temperature
! ALW1, [m^2/sec^2] -latent heat of vaporization-

ALW1 = -6.143419998D-2*tpc**(3.0D0)+1.58927D0*tpc**(2.0D0) &
-2.36418D3*tpc+2.50079D6
! ALW1, [cm^2/sec^2]

ALW1 = ALW1*10.0D3

! Sw, [N*m^-1] - surface tension of water-air interface

IF(tpc.LT.-5.5D0) THEN
Sw = 5.285D-11*tpc**(6.0D0)+6.283D-9*tpc**(5.0D0)+ &
2.933D-7*tpc**(4.0D0)+6.511D-6*tpc**(3.0D0)+ &
6.818D-5*tpc**(2.0D0)+1.15D-4*tpc+7.593D-2
ELSE
Sw = -1.55D-4*tpc+7.566165D-2
ENDIF

! Sw, [g/sec^2]
Sw = Sw*10.0D2

! RO_W, [kg/m^3] - density of liquid water
IF (tpc.LT.0.0D0) THEN
  RO_W= -7.497D-9*tpc**(6.0D0)-3.6449D-7*tpc**(5.0D0) &
        -6.9987D-6*tpc**(4.0D0)+1.518D-4*tpc**(3.0D0) &
        -8.486D-3*tpc**(2.0D0)+6.69D-2*tpc+9.9986D2

ELSE

  RO_W=(-3.932952D-10*tpc**(5.0D0)+1.497562D-7*tpc**(4.0D0) &
       -5.544846D-5*tpc**(3.0D0)-7.92221D-3*tpc**(2.0D0)+ &
       1.8224944D1*tpc+9.998396D2)/(1.0D0+1.8159725D-2*tpc)
ENDIF

! RO_W, [g/cm^3]
RO_W=RO_W*1.0D-3

! HC, [kg*m/kelvin*sec^3] - coefficient of air heat conductivity
HC=7.1128D-5*tpc+2.380696D-2

! HC, [g*cm/kelvin*sec^3]
HC=HC*10.0D4

! ew, water vapor pressure ! ... KS (kg/m2/sec)

ew = 6.38780966D-9*tpc**(6.0D0)+2.03886313D-6*tpc**(5.0D0)+ &
 3.02246994D-4*tpc**(4.0D0)+2.65027242D-2*tpc**(3.0D0)+ &
 1.43053301D0*tpc**(2.0D0)+4.43986062D1*tpc+6.1117675D2

! ew, [g/cm*sec^2]

ew=ew*10.0D0

! akoe & bkoe - constants in Koehler equation

!RO_SOLUTE=2.16D0
AKOE=2.0D0*Sw/(RV_MY*RO_W*(tpc+TZERO))
!BKOE=2.0D0*4.3D0/(22.9D0+35.5D0)
BKOE = ions*4.3/mwaero
BKOE=BKOE*(4.0D0/3.0D0)*pi*RO_SOLUTE

! RO_V, g/cm^3 - density of water vapor
!                calculate from equation of state for water vapor
RO_V = ew/(RV_MY*(tpc+TZERO))

! DV,  [cm^2/sec] - coefficient of diffusion

! 'Pruppacher, H.R., Klett, J.D., 1997. Microphysics of Clouds and Precipitation'
! 'page num 503, eq. 13-3'
DV = 0.211D0*(PZERO/PP)*((tpc+TZERO)/TZERO)**(1.94D0)

! QV,  g/g- water vapor mixing ratio
! ro_a, g/cm^3 - density of air, from equation of state
RO_A=PZERO/((tpc+TZERO)*RD_MY)

! F, s/m^2 - coefficient depending on thermodynamics parameters
!            such as temperature, thermal conductivity
!            of air, etc
! left side of F equation
FL=(RO_W*ALW1**(2.0D0))/(HC*RV_MY*(tpc+TZERO)**(2.0D0))

! right side of F equation
FR = RO_W*RV_MY*(tpc+TZERO)/(ew*DV)
F = FL + FR

! QV, g/g - water vapor mixing ratio
QV=RO_V/RO_A

! A1,A2 -  terms from equation describing changes of
!          supersaturation in an adiabatic cloud air
!	   parcel
! A1, [cm^-1] - constant
! A2, [-]     - constant

A1=(G*ALW1/(CP*RV_MY*(tpc+TZERO)**(2.0D0)))-(G/(RD_MY*(tpc+TZERO)))
A2=(1.0D0/QV)+(ALW1**(2.0D0))/(CP*RV_MY*(tpc+TZERO)**(2.0D0))

! C1,C2,C3,C4- constant parameters

C1=1.058D0
C2=1.904D0
DEG01=1.0D0/3.0D0
DEG02=1.0D0/6.0D0
C3=C1*(F*A1/3.0D0)**(0.75D0)*DSQRT(3.0D0*RO_A/(4.0D0*pi*RO_W*A2))
!C4=(C2-C1)**(DEG01)*(F*A1/3.0D0)**(0.25D0)*RO_A**(DEG02)* &
!      DSQRT(3.0D0/(4.0D0*pi*RO_W*A2))

RETURN
END SUBROUTINE SupMax_COEFF
! +----------------------------------------------------------------------------------------------------+
! +----------------------------------------------------------------------------------------------------+
SUBROUTINE LogNormal_modes_Aerosol(FCCNR_CON,FCCNR_MAR,NKR_local,COL,XL,XCCN,RCCN,RO_SOLUTE,Scale_Fa,IType, &
  ccncon1,radius_mean1,sig1, &
  ccncon2,radius_mean2,sig2, &
  ccncon3,radius_mean3,sig3)

implicit none
! ... Interface
  integer,intent(in) :: NKR_local, Itype
  double precision ,intent(in) :: XL(:), COL, RO_SOLUTE, Scale_Fa
  double precision ,intent(out) :: FCCNR_CON(:), FCCNR_MAR(:)
  double precision ,intent(out) :: XCCN(:),RCCN(:)
  double precision, intent(in) :: ccncon1,radius_mean1,sig1
  double precision, intent(in) :: ccncon2,radius_mean2,sig2
  double precision, intent(in) :: ccncon3,radius_mean3,sig3
! ... Interface
! ... Local
  integer :: mode_num, KR
  integer,parameter :: modemax = 3
  double precision  :: ccncon(modemax), sig(modemax), radius_mean(modemax)
  double precision  :: CONCCCNIN, FCCNR_tmp(NKR_local), DEG01, X0DROP, &
                        XOCCN, X0, R0, RCCN_MICRON, S_KR, S(NKR_local), X0CCN, ROCCN(NKR_local), &
                        RO_SOLUTE_Ammon, RO_SOLUTE_NaCl,arg11,arg12,arg13,arg21,arg22,arg23, & 
                        arg31,arg32,arg33,dNbydlogR_norm1,dNbydlogR_norm2,dNbydlogR_norm3


  double precision ,PARAMETER :: RCCN_MAX = 0.4D-4         ! [cm]
  double precision ,PARAMETER :: RCCN_MIN = 0.003D-4		! [cm]
  ! ... Minimal radii for dry aerosol for the 3 log normal distribution
   double precision ,PARAMETER :: RCCN_MIN_3LN = 0.00048D-4 ! [cm]
  double precision ,PARAMETER :: PI = 3.14159265D0
              double precision ,PARAMETER :: ROCCN0 = 0.1000E01 !---YZ2020Mar
! ... Local

! ... Calculating the CCN radius grid
!RO_SOLUTE_NaCl = 2.16D0  ! [g/cm3]
!RO_SOLUTE_Ammon = 1.79	 ! [g/cm3]

  ! NOTE: rccn(1) = 1.2 nm
!       rccn(33) = 2.1 um  

  DEG01 = 1.0D0/3.0D0
X0DROP = XL(2)
!X0CCN = X0DROP/(2.0**(NKR_local-1))
X0CCN = X0DROP/(2.0**(NKR_local))
!--YZ2020Mar:change to bin size from old version---@
!	DO KR = NKR_local,1,-1
!	   ROCCN(KR) = RO_SOLUTE
!     !X0 = X0CCN*2.0D0**(KR-1)
!     X0 = X0CCN*2.0D0**(KR)
!	   R0 = (3.0D0*X0/4.0D0/3.141593D0/ROCCN(KR))**DEG01
!	   XCCN(KR) = X0
!	   RCCN(KR) = R0
!	ENDDO
 DO KR=1,NKR_local
    ROCCN(KR)=ROCCN0
    X0=X0CCN*2.**(KR-1)
    R0=(3.*X0/4./3.141593/ROCCN(KR))**DEG01
    XCCN(KR)=X0
    RCCN(KR)=R0
 END DO
!-----------------------------------------------@
IF(IType == 1) THEN ! Maritime regime

  !ccncon1 = 90.0 !340.000
  !radius_mean1 = 0.03d-4 !0.00500D-04
  !sig1 = 1.28!1.60000

  !ccncon2 = 15 !60.0000 
  !radius_mean2 =  0.14d-4!0.03500D-04
  !sig2 = 1.75 !2.00000

  !ccncon3 = 0.0 !3.10000
  !radius_mean3 = 0.31000D-04
  !sig3 = 2.70000

  !ccncon1 = 340.000
  !radius_mean1 = 0.00500D-04
  !sig1 = 1.60000

  !ccncon2 = 60.0000
  !radius_mean2 = 0.03500D-04
  !sig2 = 2.00000

  !ccncon3 = 3.10000
  !radius_mean3 = 0.31000D-04
  !sig3 = 2.70000

ELSE IF(IType == 2) THEN ! Continental regime

  !ccncon1 = 1000.000
  !radius_mean1 = 0.00800D-04
  !sig1 = 1.60000

  !ccncon2 = 800.0000
  !radius_mean2 = 0.03400D-04
  !sig2 = 2.10000

  !ccncon3 = 0.72000
  !radius_mean3 = 0.46000D-04
  !sig3 = 2.20000

ENDIF

FCCNR_tmp = 0.0
CONCCCNIN = 0.0

arg11 = ccncon1/(sqrt(2.0D0*pi)*log(sig1))
arg21 = ccncon2/(sqrt(2.0D0*pi)*log(sig2))
arg31 = ccncon3/(sqrt(2.0D0*pi)*log(sig3))

dNbydlogR_norm1 = 0.0
dNbydlogR_norm2 = 0.0
dNbydlogR_norm3 = 0.0
do kr = NKR_local,1,-1
    if(RCCN(kr) > RCCN_MIN_3LN .and. RCCN(kr) < RCCN_MAX)then
        arg12 = (log(RCCN(kr)/radius_mean1))**2.0
        arg13 = 2.0D0*((log(sig1))**2.0);
        dNbydlogR_norm1 = arg11*exp(-arg12/arg13)*(log(2.0)/3.0)
        arg22 = (log(RCCN(kr)/radius_mean2))**2.0
        arg23 = 2.0D0*((log(sig2))**2.0)
        dNbydlogR_norm2 = dNbydlogR_norm1 + arg21*exp(-arg22/arg23)*(log(2.0)/3.0)
        arg32 = (log(RCCN(kr)/radius_mean3))**2.0
        arg33 = 2.0D0*((log(sig3))**2.0)
        dNbydlogR_norm3 = dNbydlogR_norm2 + arg31*exp(-arg32/arg33)*(log(2.0)/3.0);
        FCCNR_tmp(kr) = dNbydlogR_norm3/col
    endif
enddo

CONCCCNIN = col * sum(FCCNR_tmp(:))
print*,'CONCCCNIN',CONCCCNIN
if(IType == 1) FCCNR_MAR = Scale_Fa*FCCNR_tmp
if(IType == 2) FCCNR_CON = Scale_Fa*FCCNR_tmp

RETURN
END SUBROUTINE LogNormal_modes_Aerosol
! +---------------------------------------+
! +---------------------------------------------------------------+
subroutine coll_xxx_Bott(g,ckxx,x,c,ima,prdkrn,nkr,output_flux)

  implicit none

  integer,intent(in) :: nkr
  double precision,intent(inout) :: g(:)
  double precision,intent(in) ::	ckxx(:,:),x(:), c(:,:)
  integer,intent(in) :: ima(:,:)
  double precision,intent(in) :: prdkrn
  double precision,intent(inout) :: output_flux(:)

! ... Locals
 double precision:: gmin,x01,x02,x03,gsi,gsj,gsk,gsi_w,gsj_w,gsk_w,gk, &
                     gk_w,fl_gk,fl_gsk,flux,x1,flux_w,g_k_w,g_kp_old,g_kp_w
 integer :: i,ix0,ix1,j,k,kp
 integer :: kp_flux_max
! ... Locals

gmin = 1.0d-60 
kp_flux_max = nkr

! ix0 - lower limit of integration by i
do i=1,nkr-1
 ix0=i
 if(g(i).gt.gmin) goto 2000
enddo
2000   continue
if(ix0.eq.nkr-1) return

! ix1 - upper limit of integration by i
do i=nkr-1,1,-1
 ix1=i
 if(g(i).gt.gmin) goto 2010
enddo
2010   continue

if(ix1 == nkr) ix1 = nkr - 1

! ... collisions
    do i=ix0,ix1
       if(g(i).le.gmin) goto 2020
       do j=i,ix1
          if(g(j).le.gmin) goto 2021
          k=ima(i,j)
          kp=k+1
          x01=ckxx(i,j)*g(i)*g(j)*prdkrn
          x02=dmin1(x01,g(i)*x(j))
          if(j.ne.k) x03=dmin1(x02,g(j)*x(i))
          if(j.eq.k) x03=x02
          gsi=x03/x(j)
          gsj=x03/x(i)
          gsk=gsi+gsj
          g(i)=g(i)-gsi  ! This needs to be limited (for all the hydro)
          g(j)=g(j)-gsj
          gk=g(k)+gsk    ! When j=/k needs to be limited (only for different hydro)

          flux=0.d0

          if (gk.gt.gmin) then
				      x1=dlog(g(kp)/gk+gmin)
	!           x2=dexp(0.5*x1)-dexp(-0.5*x1)
	!           flux=gsk/x2*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
				      flux=gsk/x1*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
				      flux=min(flux,gk,gsk)
              g(k)=gk-flux
              g(kp)=g(kp)+flux
! --- [JS] output flux - for autoconv.
              output_flux(kp) = output_flux(kp) + flux
			    endif

          if(g(i) < 0.0 .or. g(j) < 0.0 .or. g(k) < 0.0 .or. g(kp) < 0.0) then 

            print*,    'i,j,k,kp'
            print*,     i,j,k,kp
            print*,    'ix0,ix1'
            print*,     ix0,ix1

            print*,   'g(i),g(j),g(k),g(kp)'
            print 203, g(i),g(j),g(k),g(kp)

            stop 'stop in collisions'
          end if

2021     continue
     enddo
! cycle by j
2020    continue
 enddo
! cycle by i

201    format(1x,d13.5)
202    format(1x,2d13.5)
203    format(1x,3d13.5)
204    format(1x,4d13.5)

return
end subroutine coll_xxx_Bott
! +-----------------------------------------------------------------------------+
subroutine coll_xxx_Bott_mod1(g,Is,Ie,Js,Je,ckxx,x,c,ima,prdkrn,nkr,output_flux)

!coll_xxx_Bott_mod(dsd_hlp,Is,Ie,Js,Je,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)

  implicit none

  integer,intent(in) :: nkr,Is,Js,Ie,Je
  double precision,intent(inout) :: g(:)
  double precision,intent(in) ::	ckxx(:,:),x(:), c(:,:)
  integer,intent(in) :: ima(:,:)
  double precision,intent(in) :: prdkrn
  double precision,intent(inout) :: output_flux(:)

! ... Locals
 double precision:: gmin,x01,x02,x03,gsi,gsj,gsk,gsi_w,gsj_w,gsk_w,gk, &
                     gk_w,fl_gk,fl_gsk,flux,x1,flux_w,g_k_w,g_kp_old,g_kp_w
 integer :: i,ix0,ix1,j,k,kp,Iee,Jee
 integer :: kp_flux_max
! ... Locals

gmin = 1.0d-60 
kp_flux_max = nkr

! ix0 - lower limit of integration by i
do i=1,nkr-1
 ix0=i
 if(g(i).gt.gmin) goto 2000
enddo
2000   continue
if(ix0.eq.nkr-1) return

! ix1 - upper limit of integration by i
do i=nkr-1,1,-1
 ix1=i
 if(g(i).gt.gmin) goto 2010
enddo
2010   continue

Iee = Ie; Jee = Je
if (Iee == nkr) Iee = nkr-1
if (Jee == nkr) Jee = nkr-1

! ... collisions
    do i=Is,Iee
       if(g(i).le.gmin) goto 2020
       do j=i,Jee
          if(g(j).le.gmin) goto 2021
          k=ima(i,j)
          kp=k+1

          x01=ckxx(i,j)*g(i)*g(j)*prdkrn
          x02=dmin1(x01,g(i)*x(j))
          if(j.ne.k) x03=dmin1(x02,g(j)*x(i))
          if(j.eq.k) x03=x02
          gsi=x03/x(j)
          gsj=x03/x(i)
          gsk=gsi+gsj
          g(i)=g(i)-gsi
          ! ... [JS]: added this limit
          !g(i)=dmax1(g(i),0.d0)
          g(j)=g(j)-gsj
          ! ... [JS]: added this limit
          !g(j)=dmax1(g(j),0.d0)
          gk=g(k)+gsk

          flux=0.d0

          if (gk.gt.gmin) then
				      x1=dlog(g(kp)/gk+gmin)
	!           x2=dexp(0.5*x1)-dexp(-0.5*x1)
	!           flux=gsk/x2*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
				      flux=gsk/x1*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
              flux=min(flux,gk,gsk)
              g(k)=gk-flux
              ! ... [JS]: added this limit
              !g(k)=dmax1(g(k),0.d0)
              g(kp)=g(kp)+flux
! --- [JS] output flux - for autoconv.
              output_flux(kp) = output_flux(kp) + flux
			    endif

          if(g(i) < 0.0 .or. g(j) < 0.0 .or. g(k) < 0.0 .or. g(kp) < 0.0) then 

            print*,    'i,j,k,kp'
            print*,     i,j,k,kp
            print*,    'ix0,ix1'
            print*,     ix0,ix1

            print*,   'g(i),g(j),g(k),g(kp)'
            print 203, g(i),g(j),g(k),g(kp)

            stop 'stop in collisions'
          end if

2021     continue
     enddo
! cycle by j
2020    continue
 enddo
! cycle by i

201    format(1x,d13.5)
202    format(1x,2d13.5)
203    format(1x,3d13.5)
204    format(1x,4d13.5)

return
end subroutine coll_xxx_Bott_mod1
! +------------------------------+
 ! +-----------------------------------------------------------------------------+
subroutine coll_xxx_Bott_mod2(g,Is,Ie,Js,Je,ckxx,x,c,ima,prdkrn,nkr,output_flux)

  !coll_xxx_Bott_mod(dsd_hlp,Is,Ie,Js,Je,cwll,xl_mg,chucm,ima,1.0d0,nkr,mass_flux)
  
    implicit none
  
    integer,intent(in) :: nkr,Is,Js,Ie,Je
    double precision,intent(inout) :: g(:)
    double precision,intent(in) ::	ckxx(:,:),x(:), c(:,:)
    integer,intent(in) :: ima(:,:)
    double precision,intent(in) :: prdkrn
    double precision,intent(inout) :: output_flux(:)
  
  ! ... Locals
   double precision:: gmin,x01,x02,x03,gsi,gsj,gsk,gsi_w,gsj_w,gsk_w,gk, &
                       gk_w,fl_gk,fl_gsk,flux,x1,flux_w,g_k_w,g_kp_old,g_kp_w
   integer :: i,ix0,ix1,j,k,kp,Iee,Jee
   integer :: kp_flux_max
  ! ... Locals
  
  gmin = 1.0d-60 
  kp_flux_max = nkr
  
  ! ix0 - lower limit of integration by i
  do i=1,nkr-1
   ix0=i
   if(g(i).gt.gmin) goto 2000
  enddo
  2000   continue
  if(ix0.eq.nkr-1) return
  
  ! ix1 - upper limit of integration by i
  do i=nkr-1,1,-1
   ix1=i
   if(g(i).gt.gmin) goto 2010
  enddo
  2010   continue
  
  Iee = Ie; Jee = Je
  if (Iee == nkr) Iee = nkr-1
  if (Jee == nkr) Jee = nkr-1

  ! ... collisions
      do i=Is,Iee
         if(g(i).le.gmin) goto 2020
         do j=Js,Jee
            if(g(j).le.gmin) goto 2021
            k=ima(i,j)
            kp=k+1
            x01=ckxx(i,j)*g(i)*g(j)*prdkrn
            x02=dmin1(x01,g(i)*x(j))
            if(j.ne.k) x03=dmin1(x02,g(j)*x(i))
            if(j.eq.k) x03=x02
            gsi=x03/x(j)
            gsj=x03/x(i)
            gsk=gsi+gsj
            g(i)=g(i)-gsi
            ! ... [JS]: added this limit
            !g(i)=dmax1(g(i),0.d0)
            g(j)=g(j)-gsj
            ! ... [JS]: added this limit
            !g(j)=dmax1(g(j),0.d0)
            gk=g(k)+gsk
  
            flux=0.d0
  
            if (gk.gt.gmin) then
                x1=dlog(g(kp)/gk+gmin)
    !           x2=dexp(0.5*x1)-dexp(-0.5*x1)
    !           flux=gsk/x2*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
                flux=gsk/x1*(dexp(0.5*x1)-dexp(x1*(0.5-c(i,j))))
                flux=min(flux,gk,gsk)
                g(k)=gk-flux
                ! ... [JS]: added this limit
                !g(k)=dmax1(g(k),0.d0)
                g(kp)=g(kp)+flux
  ! --- [JS] output flux - for autoconv.
                output_flux(kp) = output_flux(kp) + flux
            endif
  
            if(g(i) < 0.0 .or. g(j) < 0.0 .or. g(k) < 0.0 .or. g(kp) < 0.0) then 
  
              print*,    'i,j,k,kp'
              print*,     i,j,k,kp
              print*,    'ix0,ix1'
              print*,     ix0,ix1
  
              print*,   'g(i),g(j),g(k),g(kp)'
              print 203, g(i),g(j),g(k),g(kp)
  
              stop 'stop in collisions'
            end if
  
  2021     continue
       enddo
  ! cycle by j
  2020    continue
   enddo
  ! cycle by i
  
  201    format(1x,d13.5)
  202    format(1x,2d13.5)
  203    format(1x,3d13.5)
  204    format(1x,4d13.5)
  
  return
  end subroutine coll_xxx_Bott_mod2
! +---------------------------------+             
END MODULE module_mp_warm_sbm
!#endif
