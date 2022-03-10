MODULE src_scavenging
!------------------------------------------------------------------------------
!
! Description:
!
!   Module to compute aerosol scavenging in and below clouds.
!
!
! Current Code Owner: UW, Andreas Muhlbauer
!  phone:  +1 206 543 9208
!  fax:    
!  email:  andreasm@atmos.washington.edu
!
! History:
! Version    Date       Name
! ---------- ---------- ----
! 1.0        2011/04/15 Andreas Muhlbauer
!  Initial release
!
! @VERSION@    @DATE@     <Your name>
!  <Modification comments>         
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!==============================================================================
  
  use grid, only: masterproc

  IMPLICIT NONE

  ! Precision
  INTEGER, PARAMETER :: dp=selected_real_kind(9,99)
  
  ! Numerical constants
  REAL(KIND=dp), PARAMETER ::      & 
       pi       = 3.141592654_dp,  & ! Pi
       twopi    = 6.283185307_dp,  & ! 2*Pi
       threepi  = 9.424777961_dp,  & ! 3*Pi
       pio4     = 0.785398163_dp,  & ! Pi/4
       sqrt2    = 1.414213562_dp,  & ! SQRT(2)
       sqrtpi   = 1.772453851_dp,  & ! SQRT(pi)
       sqrtpi_r = 0.564189584_dp,  & ! 1/SQRT(pi)
       three_r  = 0.333333333_dp,  & ! 1/3
       six_r    = 0.166666667_dp     ! 1/6
       
  ! Physical constants
  REAL(KIND=dp), PARAMETER ::      & 
       gravi    = 9.80665_dp,      & ! Gravitational acceleration  ( m s-2)
       kboltz   = 1.3806503E-23_dp,& ! Boltzmann constant          (m2 kg s-2 K-1)
       r_gas    = 8.314_dp,        & ! Universal gas constant      ( J kg-1 mol-1)
       r_d      = 287.058_dp,      & ! Gas constant dry   air      ( J kg-1 K-1)
       r_v      = 461.5_dp,        & ! Gas constant moist air      ( J kg-1 K-1)
       l_i      = 2.830E+6_dp,     & ! Latent heat of sublimation  ( J kg-1 )
       m_a      = 28.97E-3_dp,     & ! Molar mass of air           (kg mol-1)
       m_w      = 18.0E-3_dp,      & ! Molar mass of water         (kg mol-1)
       mu_w     = 1.787E-3_dp,     & ! Dynamic viscosity of water  (kg m-1 s-1)
       rho_w    = 0.997E3_dp,      & ! Density of water            (kg m-3)
       rho_i    = 0.5E3_dp,        & ! Density of ice              (kg m-3)
       rho_s    = 0.1E3_dp,        & ! Density of snow             (kg m-3)
       rho_g    = 0.4E3_dp,        & ! Density of graupel          (kg m-3)
       ec       = 42.2E-4_dp,      & ! Kin. energy diss. rate (conv. clouds)
! should be replaced by true values...
!       rho_a    = 1.292_dp,        & ! Density of air   [kg/m3]
       k_p      = 6.7_dp,          & ! Thermal cond. of NaCl (see S&P, p. 481)
!       rho_p    = 2.165E3_dp         ! Density of aerosol          (kg m-3)
       rho_p    = 1.769E3_dp         ! Density of aerosol          (kg m-3)
! brnr check thermal conductivity value above, look up value for SO2
  ! Controls parameters (numerical accuracy, lookup table dimensions)
  INTEGER, PARAMETER :: &
       na       = 150,  & ! Number of bins for aerosols
       nc       = 150,  & ! Number of bins for cloud droplets
       nr       = 400, & ! Number of bins for rain drops
       ni       = 40,  & ! Number of bins for ice
       ns       = 40, & ! Number of bins for snow
       ng       = 40, & ! Number of bins for graupel
       npr      = 400, & ! Number of bins for sedimentation flux (precipitation rate)
       nt       = 6,    & ! Number of bins for air temperature
       np       = 6,    & ! Number of bins for air pressure
       nrhw     = 13,   & ! Number of bins for relative humidity w.r.t liquid water
       nrhi     = 9,   & ! Number of bins for relative humidity w.r.t ice
       ngl      = 40,   & ! Number of Gauss-Laguerre integration points
       ngh      = 20      ! Number of Gauss-Hermite  integration points
  
  REAL(KIND=dp), DIMENSION(:), ALLOCATABLE ::  &
       xda,             & ! Array with aerosol diameters         (m)
       xdc,             & ! Array with cloud droplet diameters   (m)
       xdr,             & ! Array with rain drop diameters       (m)
       xdi,             & ! Array with ice diameters             (m)
       xds,             & ! Array with snow diameters            (m)
       xdg,             & ! Array with graupel diameters         (m)
       xprci,           & ! Array with precip. rates (cloud,ice) (kg m-2 s-1)
       xprrsg,          & ! Array with precip. rates (rain,snow,grpl.) (mm h-1)
       xt,              & ! Array with air temperature (K)
       xp,              & ! Array with air pressure (Pa)
       xrhw,            & ! Array with relative humidity w.r.t liquid water
       xrhi,            & ! Array with relative humidity w.r.t ice
       u_c,             & ! Array with cloud droplet fall speeds (m s-1)
       u_r,             & ! Array with rain drop fall speeds     (m s-1)
       u_i,             & ! Array with ice fall speeds           (m s-1)
       u_s,             & ! Array with snow fall speeds          (m s-1)
       u_g                ! Array with graupel fall speeds       (m s-1)
  
  REAL(KIND=dp), DIMENSION(:), ALLOCATABLE :: &
       xgl_c,             & ! Array with Gauss-Laguerre points  for cloud
       wgl_c,             & ! Array with Gauss-Laguerre weights for cloud
       xgl_r,             & ! Array with Gauss-Laguerre points  for rain
       wgl_r,             & ! Array with Gauss-Laguerre weights for rain
       xgl_i,             & ! Array with Gauss-Laguerre points  for ice
       wgl_i,             & ! Array with Gauss-Laguerre weights for ice
       xgl_s,             & ! Array with Gauss-Laguerre points  for snow
       wgl_s,             & ! Array with Gauss-Laguerre weights for snow
       xgl_g,             & ! Array with Gauss-Laguerre points  for graupel
       wgl_g,             & ! Array with Gauss-Laguerre weights for graupel
       xgh,               & ! Array with Gauss-Laguerre points
       wgh                  ! Array with Gauss-Laguerre weights
  
  REAL(KIND=dp), DIMENSION(:,:,:,:,:), ALLOCATABLE :: &
       e_c,               & ! Lookup table for collision eff. aerosol-cloud
       e_r,               & ! Lookup table for collision eff. aerosol-rain
       e_i,               & ! Lookup table for collision eff. aerosol-ice
       e_s,               & ! Lookup table for collision eff. aerosol-snow
       e_g                  ! Lookup table for collision eff. aerosol-graupel
  
  REAL(KIND=dp), DIMENSION(:,:,:,:,:), ALLOCATABLE :: &
       g_c,               & ! Lookup table for scavenging coeff. aerosol-cloud   (s-1)
       g_r,               & ! Lookup table for scavenging coeff. aerosol-rain    (s-1)
       g_i,               & ! Lookup table for scavenging coeff. aerosol-ice     (s-1)
       g_s,               & ! Lookup table for scavenging coeff. aerosol-snow    (s-1)
       g_g                  ! Lookup table for scavenging coeff. aerosol-graupel (s-1)
       
  REAL(KIND=dp), DIMENSION(1) :: &
       t0,                       & ! Standard temperature        (  K   )
       p0                          ! Standard pressure           (  Pa  )

  TYPE HYDROMETEOR
     CHARACTER(LEN=20) :: name   ! Name of hydrometeor class
     REAL(KIND=dp)     :: nu     ! Shape parameter of gamma distribution
!!$     REAL(KIND=dp)     :: alpha  !
!!$     REAL(KIND=dp)     :: beta   !
!!$     REAL(KIND=dp)     :: gamma  !
!!$     REAL(KIND=dp)     :: sigma  !
     REAL(KIND=dp)     :: avelo  ! A coefficient of fall velocity power law (U=a*D^b)
     REAL(KIND=dp)     :: bvelo  ! B coefficien of ftall velocity power law (U=a*D^b) 
     REAL(KIND=dp)     :: amass  ! A coefficient of mass-diameter relation  (M=a*D^b)
     REAL(KIND=dp)     :: bmass  ! B coefficient of mass-diameter relation  (M=a*D^b)
  END TYPE HYDROMETEOR

  TYPE (HYDROMETEOR) :: cloud_morrison = HYDROMETEOR( & 
       &  'cloud',        & ! Name of hydrometeor class
       &  1.0_dp,         & ! Shape parameter of gamma distribution
       &  3.E7_dp,        & ! A coefficient of fall velocity power law (U=a*D^b)
       &  2.0_dp,         & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_w/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR) :: rain_kessler = HYDROMETEOR( & 
       &  'rain',         & ! Name of hydrometeor class
       &  0.0_dp,         & ! Shape parameter of gamma distribution
       &  130.0_dp,       & ! A coefficient of fall velocity power law (U=a*D^b)
       &  0.5_dp,         & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_w/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR) :: rain_morrison = HYDROMETEOR( & 
       &  'rain',         & ! Name of hydrometeor class
       &  0.0_dp,         & ! Shape parameter of gamma distribution
       &  841.99667_dp,   & ! A coefficient of fall velocity power law (U=a*D^b)
       &  0.8_dp,         & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_w/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR) :: ice_morrison = HYDROMETEOR( & 
       &  'ice',          & ! Name of hydrometeor class
       &  0.0_dp,         & ! Shape parameter of gamma distribution
       &  700.0_dp,       & ! A coefficient of fall velocity power law (U=a*D^b)
       &  1.0_dp,         & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_i/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR) :: snow_morrison = HYDROMETEOR( &
       &  'snow',         & ! Name of hydrometeor class
       &  0.0_dp,         & ! Shape parameter of gamma distribution
       &  11.72_dp,       & ! A coefficient of fall velocity power law (U=a*D^b)
       &  0.41_dp,        & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_s/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR) :: graupel_morrison = HYDROMETEOR( & 
       &  'graupel',      & ! Name of hydrometeor class
       &  0.0_dp,         & ! Shape parameter of gamma distribution
       &  19.3_dp,        & ! A coefficient of fall velocity power law (U=a*D^b)
       &  0.37_dp,        & ! B coefficient of fall velocity power law (U=a*D^b)
       &  pi*rho_g/6._dp, & ! A coefficient of mass-diameter relation  (M=a*D^b)
       &  3._dp)            ! B coefficient of mass-diameter relation  (M=a*D^b)

  TYPE (HYDROMETEOR), SAVE :: cloud, rain, ice, snow, graupel

  ! Control switches
  LOGICAL, PARAMETER :: luse_standard_t_p = .false.

CONTAINS


  SUBROUTINE scav_cloud_2m(qna,qma,cmd,sigma, &
       & t,p,                                 &
       & qc,                                  &
       & qnc,                                 &
       & nu,                                  &
       & ec3d,                                &
       & ldissip3d,                           &
       & ni,nj,nk,nmod,dt)                         

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: ni, nj, nk, nmod
    LOGICAL, INTENT(IN) :: ldissip3d
    REAL(KIND=dp), INTENT(IN) :: dt
    REAL(KIND=dp), DIMENSION(nmod), INTENT(IN) :: sigma
    REAL(KIND=dp), DIMENSION(ni,nj,nk), INTENT(IN) ::  qc, qnc, nu, ec3d, t, p
    REAL(KIND=dp), DIMENSION(ni,nk,nk,nmod), INTENT(INOUT):: qna,qma,cmd

    ! Local variables
    INTEGER :: i, j, k, imod
    REAL(KIND=dp), PARAMETER :: eps=1.E-20_dp
    REAL(KIND=dp), PARAMETER :: qxeps=1.E-8_dp
    REAL(KIND=dp) :: zt, zp
    REAL(KIND=dp) :: lam, n0, const0, const1, const2, const3
    REAL(KIND=dp) :: rho_a, mu_a, lambda_a, k_a
    REAL(KIND=dp) :: alpha, beta, delta, epsilon, lkt, dum
    REAL(KIND=dp) :: gamma1, gamma2, gamma3
    REAL(KIND=dp) :: m1, m3, muneg2, mu0, mu1, mu3
    REAL(KIND=dp) :: qtend, ntend
    REAL(KIND=dp), DIMENSION(ni,nj,nk) :: ec3dloc 
    REAL(KIND=dp), DIMENSION(ni,nj,nk,nmod) :: dndt,dmdt
   

    if (ldissip3d.ne..TRUE.) then
       ec3dloc(:,:,:) = ec
    else
       ec3dloc = ec3d
    end if

    imod = nmod

    DO k=1,nk
       DO j=1,nj
          DO i=1,ni
             if ((qc(i,j,k).GT.qxeps).AND.&
                  (qma(i,j,k,nmod).GT.(0._dp+eps))) then
                const0=nu(i,j,k)+1._dp
                const1=(nu(i,j,k)+3._dp)*(nu(i,j,k)+2._dp)*const0
                const2=1._dp/EXP(gammln(const0))
                
    
                ! Slope parameter (m-1)
                lam=(pi * rho_w * qnc(i,j,k) * const1 / &
                     (6._dp*(qc(i,j,k)+eps)))**three_r
                
                ! Intercept parameter (m-3 m-nu)
                n0=const2 * qnc(i,j,k) * lam**const0
                
                
                zt = t(i,j,k)
                zp = p(i,j,k)
                
                ! Air density (replace with reference density in future)
                rho_a = zp/(r_d*zt)
                
                ! Dynamic viscosity of air (match with morrison formulation in future)
                mu_a=1.E-5_dp*(1.718_dp+4.9E-3_dp* & 
                     (zt-273.15_dp)-1.2E-5_dp*(zt-273.15_dp)**2._dp) 
                
                ! Mean free path in air
                lambda_a=2._dp*mu_a/(zp*SQRT(8._dp*m_a/(pi*r_gas*zt)))
                
                ! Thermal conductivity of air
                !k_a = 4.184E-3_dp*(5.69_dp+0.017_dp*(zt-273.15_dp))
                
                ! Vapor diffusivity of air
                !d_v = 0.211E-4_dp*(zt/273.15_dp)**1.94_dp*(101325.0_dp/zp)   
                
                ! Latent heat of condensation/evaporation
                !l_v = 2.5E6_dp*(273.15_dp/zt)**(0.167_dp+3.67E-4_dp*zt)          
                
                ! Saturation vapor pressure w.r.t water
                !e_sw = saturation_water(zt)     
                !g_w=( rho_w*r_v*zt/e_sw/d_v + rho_w*l_v/k_a/zt * &
                !     ((l_v/r_v/zt)-1._dp) )**(-1._dp)
                
                ! Supersaturation w.r.t. water
                !s_w=rh(i,j,k)-1._dp                      
                
                
                ! Aerosol Cunningham slip correction factor
                !C_p = 1._dp + 1.26_dp*(2._dp*lambda_a/da)* &
                !     (101325.0_dp/p)*(t/273.15_dp)
                !C_p = 1._dp + 2._dp*lambda_a/da* &     
                !     (1.257_dp+0.4_dp*EXP(-1.1_dp*da/2._dp/lambda_a))
                
                ! Aerosol Knudson number
                !Kn=2._dp*lambda_a/da  
                
                ! B term for thermophoresis, see PK p. 725
                !B_p=0.4_dp*C_p*(k_a+2.5_dp*k_p*Kn)/((1._dp+3._dp*Kn)* & 
                !     (k_p+2._dp*k_a+5._dp*k_p*Kn))
                
                lkt=1.657_dp*lambda_a*kboltz*zt
                alpha=4._dp*lkt/(3._dp*mu_a)
                beta=1.3_dp*(rho_a*ec3dloc(i,j,k)/mu_a)**0.5_dp
                !delta=-2._dp*pi*l_v*rho_w*s_w*g_w*B_p/zp
                !epsilon=2.4_dp*pi*rho_w*s_w*g_w/rho_a
                
                !m0=moment_gamma(n0(i,j,k),lam(i,j,k),nu,0._dp)
                m1=moment_gamma(n0,lam,nu(i,j,k),1._dp)
                !m2=moment_gamma(n0(i,j,k),lam(i,j,k),nu,2._dp)
                m3=moment_gamma(n0,lam,nu(i,j,k),3._dp)    
             
                muneg2 = moment_lognorm(qna(i,j,k,imod),cmd(i,j,k,imod),sigma(imod),-2._dp)
                mu0 = qna(i,j,k,imod)
                mu1 = moment_lognorm(qna(i,j,k,imod),cmd(i,j,k,imod),sigma(imod),1._dp)
                mu3 = moment_lognorm(qna(i,i,k,imod),cmd(i,j,k,imod),sigma(imod),2._dp)
                !gamma1=alpha/da**2._dp*m1
                !gamma2=beta*(da**3._dp*m0+3._dp*da**2._dp*m1+3._dp*da*m2+m3)
                !gamma3=m1*(delta+epsilon)
                !gamma(i,j,k)=gamma1+gamma2+MAX(gamma3,0._dp)
                
                dndt(i,j,k,imod) = -1._dp*(alpha*m1*muneg2+beta*m3*mu0)
                dmdt(i,j,k,imod) = -1._dp*pi*rho_p*six_r*(alpha*m1*mu1+beta*m3*mu3)
             else
                dndt(i,j,k,imod) = 0._dp
                dmdt(i,j,k,imod) = 0._dp
             end if
          ENDDO
       ENDDO
    ENDDO

   
    ! Add tendencies
    DO imod=1,nmod
       ! Convert diameter of average mass to count median diameter
       const3=EXP(-1.5_dp*LOG(sigma(imod))**2._dp)
       DO k=1,nk
          DO j=1,nj
             DO i=1,ni
                qna(i,j,k,imod) = qna(i,j,k,imod) + dndt(i,j,k,imod)*dt
                qna(i,j,k,imod)=MAX(qna(i,j,k,imod),0._dp)
                qma(i,j,k,imod) = qma(i,j,k,imod) + dmdt(i,j,k,imod)*dt
                qma(i,j,k,imod)=MAX(qma(i,j,k,imod),0._dp)
                if (qna(i,j,k,imod).gt.0.) then
                   cmd(i,j,k,imod)=const3*(6._dp*qma(i,j,k,imod)/ &
                        (pi*rho_p*qna(i,j,k,imod)))**three_r
                else 
                   cmd(i,j,k,imod)=0.
                end if
             ENDDO
          ENDDO
       ENDDO
    ENDDO


  END SUBROUTINE scav_cloud_2m


  SUBROUTINE memory(cflag)
    ! Subroutine to allocate/deallocate memory used in the
    ! aerosol scavenging parameterization

    IMPLICIT NONE

    CHARACTER(LEN=*), INTENT(IN) :: cflag
    

    IF (TRIM(cflag) == 'allocate') THEN
       
       ALLOCATE(xda(na)); xda=0._dp
       ALLOCATE(xdc(nc)); xdc=0._dp
       ALLOCATE(xdr(nr)); xdr=0._dp
      ! ALLOCATE(xdi(ni)); xdi=0._dp
      ! ALLOCATE(xds(ns)); xds=0._dp
      ! ALLOCATE(xdg(ng)); xdg=0._dp
       ALLOCATE(xprci(npr));  xprci=0._dp
       ALLOCATE(xprrsg(npr)); xprrsg=0._dp
       ALLOCATE(xt(nt)); xt=0._dp
       ALLOCATE(xp(np)); xp=0._dp
       ALLOCATE(xrhw(nrhw)); xrhw=0._dp
      ! ALLOCATE(xrhi(nrhi)); xrhi=0._dp

       ALLOCATE(u_c(nc)); u_c=0._dp
       ALLOCATE(u_r(nr)); u_r=0._dp
      ! ALLOCATE(u_i(ni)); u_i=0._dp
      ! ALLOCATE(u_s(ns)); u_s=0._dp
      ! ALLOCATE(u_g(ng)); u_g=0._dp
       
       ALLOCATE(xgl_c(ngl),wgl_c(ngl)); xgl_c=0._dp; wgl_c=0._dp
       ALLOCATE(xgl_r(ngl),wgl_r(ngl)); xgl_r=0._dp; wgl_r=0._dp
      ! ALLOCATE(xgl_i(ngl),wgl_i(ngl)); xgl_i=0._dp; wgl_i=0._dp
      ! ALLOCATE(xgl_s(ngl),wgl_s(ngl)); xgl_s=0._dp; wgl_s=0._dp
      ! ALLOCATE(xgl_g(ngl),wgl_g(ngl)); xgl_g=0._dp; wgl_g=0._dp

       ALLOCATE(xgh(ngh),wgh(ngh)); xgh=0._dp; wgh=0._dp

       IF (luse_standard_t_p) THEN
          ALLOCATE(e_c(na,nc,1,1,nrhw));    e_c=0._dp
          ALLOCATE(e_r(na,nr,1,1,1));       e_r=0._dp
         ! ALLOCATE(e_i(na,ni,1,1,nrhi));    e_i=0._dp
         ! ALLOCATE(e_s(na,ns,1,1,1));       e_s=0._dp
         ! ALLOCATE(e_g(na,ng,1,1,1));       e_g=0._dp

          ALLOCATE(g_c(na,npr,1,1,nrhw));   g_c=0._dp
          ALLOCATE(g_r(na,npr,1,1,1));      g_r=0._dp
         ! ALLOCATE(g_i(na,npr,1,1,nrhi));   g_i=0._dp
         ! ALLOCATE(g_s(na,npr,1,1,1));      g_s=0._dp
         ! ALLOCATE(g_g(na,npr,1,1,1));      g_g=0._dp
       ELSE
          ALLOCATE(e_c(na,nc,nt,np,nrhw));  e_c=0._dp
          ALLOCATE(e_r(na,nr,nt,np,1));     e_r=0._dp
         ! ALLOCATE(e_i(na,ni,nt,np,nrhi));  e_i=0._dp
         ! ALLOCATE(e_s(na,ns,nt,np,1));     e_s=0._dp
         ! ALLOCATE(e_g(na,ng,nt,np,1));     e_g=0._dp

          ALLOCATE(g_c(na,npr,nt,np,nrhw)); g_c=0._dp
          ALLOCATE(g_r(na,npr,nt,np,1));    g_r=0._dp
         ! ALLOCATE(g_i(na,npr,nt,np,nrhi)); g_i=0._dp
         ! ALLOCATE(g_s(na,npr,nt,np,1));    g_s=0._dp
         ! ALLOCATE(g_g(na,npr,nt,np,1));    g_g=0._dp
       ENDIF

    ELSEIF (TRIM(cflag) == 'deallocate') THEN
       
       IF (ALLOCATED(xda))   DEALLOCATE(xda)
       IF (ALLOCATED(xdc))   DEALLOCATE(xdc)
       IF (ALLOCATED(xdr))   DEALLOCATE(xdr)
       IF (ALLOCATED(xdi))   DEALLOCATE(xdi)
       IF (ALLOCATED(xds))   DEALLOCATE(xds)
       IF (ALLOCATED(xdg))   DEALLOCATE(xdg)
       IF (ALLOCATED(xprci)) DEALLOCATE(xprci)
       IF (ALLOCATED(xprrsg))DEALLOCATE(xprrsg)
       IF (ALLOCATED(xt))    DEALLOCATE(xt)
       IF (ALLOCATED(xp))    DEALLOCATE(xp)
       IF (ALLOCATED(xrhw))  DEALLOCATE(xrhw)
       IF (ALLOCATED(xrhi))  DEALLOCATE(xrhi)

       IF (ALLOCATED(u_c))   DEALLOCATE(u_c)
       IF (ALLOCATED(u_r))   DEALLOCATE(u_r)
       IF (ALLOCATED(u_i))   DEALLOCATE(u_i)
       IF (ALLOCATED(u_s))   DEALLOCATE(u_s)
       IF (ALLOCATED(u_g))   DEALLOCATE(u_g)

       IF (ALLOCATED(xgl_c)) DEALLOCATE(xgl_c,wgl_c)
       IF (ALLOCATED(xgl_r)) DEALLOCATE(xgl_r,wgl_r)
       IF (ALLOCATED(xgl_i)) DEALLOCATE(xgl_i,wgl_i)
       IF (ALLOCATED(xgl_s)) DEALLOCATE(xgl_s,wgl_s)
       IF (ALLOCATED(xgl_g)) DEALLOCATE(xgl_g,wgl_g)

       IF (ALLOCATED(xgh))   DEALLOCATE(xgh,wgh) 

       IF (ALLOCATED(e_c))   DEALLOCATE(e_c)
       IF (ALLOCATED(e_r))   DEALLOCATE(e_r)
       IF (ALLOCATED(e_i))   DEALLOCATE(e_i)
       IF (ALLOCATED(e_s))   DEALLOCATE(e_s)
       IF (ALLOCATED(e_g))   DEALLOCATE(e_g)

       IF (ALLOCATED(g_c))   DEALLOCATE(g_c)
       IF (ALLOCATED(g_r))   DEALLOCATE(g_r)
       IF (ALLOCATED(g_i))   DEALLOCATE(g_i)
       IF (ALLOCATED(g_s))   DEALLOCATE(g_s)
       IF (ALLOCATED(g_g))   DEALLOCATE(g_g)
       
    ELSE
       WRITE(*,*) "   Error: cflag must be 'allocate' or 'deallocate'"
    ENDIF

  END SUBROUTINE memory


  SUBROUTINE init_scavenging
    !
    ! Initializations for aerosol scavenging parameterization
    !

    IMPLICIT NONE

    INTEGER :: n,i,j
    REAL(KIND=dp) :: dx, n0mp, lammp, n0, lam
    REAL(KIND=dp) :: const0_c, const1_c, const2_c, const3_c, const4_c
    REAL(KIND=dp) :: const0_i, const1_i, const2_i, const3_i, const4_i
    REAL(KIND=dp), PARAMETER :: ncsm=100.0E6_dp ! Cloud number 100 per cc
    REAL(KIND=dp), PARAMETER :: nism=100.0E3_dp ! Ice   number 100 per l
    
    !bloss: Move this into init_scavenging, since it is not used outside
    REAL(KIND=dp), DIMENSION(2), SAVE ::   &
         d_bnds_aero,     & ! Diameter bounds for aerosol spectrum
         d_bnds_cloud,    & ! Diameter bounds for cloud droplet spectrum
         d_bnds_rain,     & ! Diameter bounds for rain drop spectrum
         d_bnds_ice,      & ! Diameter bounds for ice
         d_bnds_snow,     & ! Diameter bounds for snow
         d_bnds_graupel,  & ! Diameter bounds for graupel
         pr_bnds,         & ! Bounds for precipitation rate
         t_bnds,          & ! Bounds for air temperature
         p_bnds,          & ! Bounds for air pressure
         rhw_bnds,        & ! Bounds for relative humidity w.r.t. liquid water
         rhi_bnds           ! Bounds for relative humidity w.r.t. ice

    ! For debug output
    LOGICAL, PARAMETER :: ldebug=.false.
    REAL(kind=dp) :: x1, x2, x3, x4, x5, x6, x7
    INTEGER :: idx1, idx2, idx3, idx4, idx5, idx6, idx7

    ! Set hydrometeor class
    IF (ldebug) THEN
       WRITE(*,*) "   Set hydrometeor classes"
    ENDIF
    cloud=cloud_morrison
    rain=rain_morrison !changed from rain_kessler
    ice=ice_morrison
    snow=snow_morrison
    graupel=graupel_morrison
  

    ! Set bins for lookup tables
    IF (ldebug) THEN
       WRITE(*,*) "   Set bins for lookup tables"
    ENDIF
    
    ! Aerosols
    d_bnds_aero = (/ LOG(1.E-9_dp), LOG(1.E-5_dp) /)
    dx = (d_bnds_aero(2)-d_bnds_aero(1))/REAL(na-1,dp)
    DO n=1,na
       xda(n)=EXP(d_bnds_aero(1)+REAL(n-1,dp)*dx)
    ENDDO

    ! Cloud
    d_bnds_cloud = (/ LOG(1.E-6_dp), LOG(80.E-6_dp) /)
    dx = (d_bnds_cloud(2)-d_bnds_cloud(1))/REAL(nc-1,dp)
    DO n=1,nc
       xdc(n)=EXP(d_bnds_cloud(1)+REAL(n-1,dp)*dx)
       u_c(n)=cloud%avelo*xdc(n)**cloud%bvelo
    ENDDO

    ! Rain
    d_bnds_rain = (/ LOG(80.E-6_dp), LOG(5.E-3_dp) /)
    dx = (d_bnds_rain(2)-d_bnds_rain(1))/REAL(nr-1,dp)
    DO n=1,nr
       xdr(n)=EXP(d_bnds_rain(1)+REAL(n-1,dp)*dx)
       u_r(n)=rain%avelo*xdr(n)**rain%bvelo
    ENDDO

   ! ! Ice
   ! d_bnds_ice = (/ LOG(10.E-6_dp), LOG(200.E-6_dp) /)
   ! dx = (d_bnds_ice(2)-d_bnds_ice(1))/REAL(ni-1,dp)
   ! DO n=1,ni
   !    xdi(n)=EXP(d_bnds_ice(1)+REAL(n-1,dp)*dx)
   !    u_i(n)=ice%avelo*xdi(n)**ice%bvelo
   ! ENDDO

   ! ! Snow
   ! d_bnds_snow = (/ LOG(100.E-6_dp), LOG(5.E-3_dp) /)
   ! dx = (d_bnds_snow(2)-d_bnds_snow(1))/REAL(ns-1,dp)
   ! DO n=1,ns
   !    xds(n)=EXP(d_bnds_snow(1)+REAL(n-1,dp)*dx)
   !    u_s(n)=snow%avelo*xds(n)**snow%bvelo
   ! ENDDO

   ! ! Graupel
   ! d_bnds_graupel = (/ LOG(100.E-6_dp), LOG(5.E-3_dp) /)
   ! dx = (d_bnds_graupel(2)-d_bnds_graupel(1))/REAL(ng-1,dp)
   ! DO n=1,ng
   !    xdg(n)=EXP(d_bnds_graupel(1)+REAL(n-1,dp)*dx)
   !    u_g(n)=graupel%avelo*xdg(n)**graupel%bvelo
   ! ENDDO

    ! Precipitation rate (kg m-2 s-1) for cloud, ice
    pr_bnds = (/ LOG(1.E-7_dp), LOG(1.E-3_dp) /)
    dx = (pr_bnds(2)-pr_bnds(1))/REAL(npr-1,dp)
    DO n=1,npr
       xprci(n)=EXP(pr_bnds(1)+REAL(n-1,dp)*dx)
    ENDDO

    ! Precipitation rate (mm/h) for rain, snow, graupel
    pr_bnds = (/ LOG(1.E-3_dp), LOG(2.E2_dp) /)
    dx = (pr_bnds(2)-pr_bnds(1))/REAL(npr-1,dp)
    DO n=1,npr
       xprrsg(n)=EXP(pr_bnds(1)+REAL(n-1,dp)*dx)
    ENDDO

    ! Temperature (K)
    t_bnds = (/ 230._dp, 310._dp/)
    dx = (t_bnds(2)-t_bnds(1))/REAL(nt-1,dp)
    DO n=1,nt
       xt(n)=t_bnds(1)+REAL(n-1,dp)*dx
    ENDDO

    ! Pressure (Pa)
    p_bnds = (/ 250.E2_dp, 1050.E2_dp/)
    dx = (p_bnds(2)-p_bnds(1))/REAL(np-1,dp)
    DO n=1,np
       xp(n)=p_bnds(1)+REAL(n-1,dp)*dx
    ENDDO

    ! Relative humidity w.r.t. to water (0-1.xx)
    rhw_bnds = (/ 0.8_dp, 1.05_dp/)
    dx = (rhw_bnds(2)-rhw_bnds(1))/REAL(nrhw-1,dp)
    DO n=1,nrhw
       xrhw(n)=rhw_bnds(1)+REAL(n-1,dp)*dx
    ENDDO

   ! ! Relative humidity w.r.t. to ice (0-1.xx)
   ! rhi_bnds = (/ 0.8_dp, 1.3_dp/)
   ! dx = (rhi_bnds(2)-rhi_bnds(1))/REAL(nrhi-1,dp)
   ! DO n=1,nrhi
   !    xrhi(n)=rhi_bnds(1)+REAL(n-1,dp)*dx
   ! ENDDO

    
    ! Initialization of Gauss-Laguerre and Gauss-Hermite points and weights
    IF (ldebug) THEN
       WRITE(*,*) "   Init quadrature points and weights"
    ENDIF
    
    CALL init_gauss_hermite(ngh,xgh,wgh)
    
    ! Cloud
    CALL init_gauss_laguerre(ngl,cloud%nu,cloud%bvelo,xgl_c,wgl_c)
    ! Rain
    CALL init_gauss_laguerre(ngl,rain%nu,rain%bvelo,xgl_r,wgl_r)
    !! Ice
    !CALL init_gauss_laguerre(ngl,ice%nu,ice%bvelo,xgl_i,wgl_i)
    !! Snow
    !CALL init_gauss_laguerre(ngl,snow%nu,snow%bvelo,xgl_s,wgl_s)
    !! Graupel
    !CALL init_gauss_laguerre(ngl,graupel%nu,graupel%bvelo,xgl_g,wgl_g)

   
    ! Compute lookup tables for collision efficiencies
    if(masterproc) WRITE(*,*) "In src_scavenging.f90: Compute lookup tables for aerosol scavenging"
    if(masterproc) WRITE(*,*) "This could take a minute ..."
    ! Aerosol and cloud
    IF (ldebug) THEN
       WRITE(*,*) "   Compute lookup tables for collision efficiencies (aerosol-cloud)"
    ENDIF
    IF (luse_standard_t_p) THEN
       t0=282.00_dp
       p0=900.00E2_dp
       CALL collision_efficiency_cloud(na,nc,1,1,nrhw,xda,xdc,u_c,t0,p0,xrhw,e_c)
    ELSE
       CALL collision_efficiency_cloud(na,nc,nt,np,nrhw,xda,xdc,u_c,xt,xp,xrhw,e_c)
    ENDIF
    
    ! Aerosol and rain 
    IF (ldebug) THEN
       WRITE(*,*) "   Compute lookup tables for collision efficiencies (aerosol-rain)"
    ENDIF
    IF (luse_standard_t_p) THEN
       t0=282.00_dp
       p0=900.00E2_dp
       CALL collision_efficiency_rain(na,nr,1,1,xda,xdr,u_r,t0,p0,e_r)
    ELSE
       CALL collision_efficiency_rain(na,nr,nt,np,xda,xdr,u_r,xt,xp,e_r)
    ENDIF

    !! Aerosol and ice
    !IF (ldebug) THEN
    !   WRITE(*,*) "   Compute lookup tables for collision efficiencies (aerosol-ice)"
    !ENDIF
    !IF (luse_standard_t_p) THEN
    !   t0=263.15_dp
    !   p0=1013.15E2_dp
    !   CALL collision_efficiency_ice(na,ni,1,1,nrhi,xda,xdi,u_i,t0,p0,xrhi,e_i)
    !ELSE
    !   CALL collision_efficiency_ice(na,ni,nt,np,nrhi,xda,xdi,u_i,xt,xp,xrhi,e_i)
    !ENDIF

    !! Aerosol and snow
    !IF (ldebug) THEN
    !   WRITE(*,*) "   Compute lookup tables for collision efficiencies (aerosol-snow)"
    !ENDIF
    !IF (luse_standard_t_p) THEN
    !   t0=263.15_dp
    !   p0=1013.15E2_dp
    !   CALL collision_efficiency_snow(na,ns,1,1,xda,xds,u_s,t0,p0,e_s)
    !ELSE
    !   CALL collision_efficiency_snow(na,ns,nt,np,xda,xds,u_s,xt,xp,e_s)
    !ENDIF
    
    !! Aerosol and graupel
    !IF (ldebug) THEN
    !   WRITE(*,*) "   Compute lookup tables for collision efficiencies (aerosol-graupel)"
    !ENDIF
    !IF (luse_standard_t_p) THEN
    !   t0=263.15_dp
    !   p0=1013.15E2_dp
    !   CALL collision_efficiency_snow(na,ng,1,1,xda,xdg,u_g,t0,p0,e_g)
    !ELSE
    !   CALL collision_efficiency_snow(na,ng,nt,np,xda,xdg,u_g,xt,xp,e_g)
    !ENDIF
    
    
    ! Compute lookup tables for scavenging coefficients
    ! Aerosol and rain
    IF (ldebug) THEN
       WRITE(*,*) "   Compute lookup tables for scavenging coefficients"
    ENDIF
    ! 
    ! Note: MP distribution for rain (Marshall and Palmer, 1948) is used here
    ! only to compute lookup table for scavenging coefficients 
    ! as a function of rain rate (mm/h).
    ! Later, the real rain rate is determined from the size distribution parameters.
    ! Snow and graupel are treated similarly by assuming MP-type distributions
    ! for snow and graupel following Gunn and Marshall (1958) or
    ! Sekhon and Srivastava (1970).
    ! See also PK, p. 34 and p. 59 for details on MP-type distributions or
    ! Rogers and Yau, p. 171, and p. 180-181.
    !    
    ! For cloud droplets and ice the size distribution parameters n0, lam
    ! are computed from the precipitation fluxes by assuming constant
    ! cloud/ice number concentrations (in analogy to a single-moment scheme).
    ! Again, this is only done here to provide a lookup table as a function
    ! of hydrometeor sedimentation flux. Later, the true precipitation fluxes
    ! are computed from the prognostic hydrometeor number and mass densities.
    !

    const0_c = cloud%nu+1._dp
    const1_c = EXP(gammln(const0_c))
    const2_c = cloud%amass*cloud%avelo*ncsm*EXP(gammln(cloud%bmass+cloud%bvelo+const0_c))
    const3_c = 1._dp/(cloud%bmass+cloud%bvelo)
    const4_c = ncsm/const1_c

    !const0_i = ice%nu+1._dp
    !const1_i = EXP(gammln(const0_i))
    !const2_i = ice%amass*ice%avelo*nism*EXP(gammln(ice%bmass+ice%bvelo+const0_i))
    !const3_i = 1._dp/(ice%bmass+ice%bvelo)
    !const4_i = nism/const1_i
    
    IF (luse_standard_t_p) THEN
       DO n=1,npr

          ! Cloud
          lam=(const2_c/(const1_c*xprci(n)))**const3_c
          n0=const4_c*lam**const0_c
          CALL scavenging_coefficient(na,nc,1,1,nrhw,xda,xdc,e_c, &
               cloud%nu,n0,lam,cloud%avelo,cloud%bvelo,        &
               ngl,xgl_c,wgl_c,g_c(:,n,:,:,:))
          
          ! Rain
          n0mp=8.E6_dp ! m-3 m-1
          lammp=4.1E3_dp*xprrsg(n)**(-.21_dp) ! m-1
          CALL scavenging_coefficient(na,nr,1,1,1,xda,xdr,e_r, &
               rain%nu,n0mp,lammp,rain%avelo,rain%bvelo,       &
               ngl,xgl_r,wgl_r,g_r(:,n,:,:,:))
          
         ! ! Ice
         ! lam=(const2_i/(const1_i*xprci(n)))**const3_i
         ! n0=const4_i*lam**const0_i
         ! CALL scavenging_coefficient(na,ni,1,1,1,xda,xdi,e_i, &
         !      ice%nu,n0,lam,ice%avelo,ice%bvelo,              &
         !      ngl,xgl_i,wgl_i,g_i(:,n,:,:,:))

         ! ! Snow
         ! ! Gunn and Marshall (1958)
         ! !n0mp=3.8E6_dp*xprrsg(n)**(-.87_dp) ! m-3 m-1
         ! !lammp=25.5E2_dp*xprrsg(n)**(-.48_dp) ! m-1
         ! ! Sekhon and Srivastava (1970)
         ! n0mp=2.5E6_dp*xprrsg(n)**(-.94_dp) ! m-3 m-1
         ! lammp=22.9E2_dp*xprrsg(n)**(-.45_dp) ! m-1
         ! CALL scavenging_coefficient(na,ns,1,1,1,xda,xds,e_s, &
         !      snow%nu,n0mp,lammp,snow%avelo,snow%bvelo,       &
         !      ngl,xgl_s,wgl_s,g_s(:,n,:,:,:))
          
         ! ! Graupel
         ! n0mp=2.5E6_dp*xprrsg(n)**(-.94_dp) ! m-3 m-1
         ! lammp=22.9E2_dp*xprrsg(n)**(-.45_dp) ! m-1
         ! CALL scavenging_coefficient(na,ng,1,1,1,xda,xdg,e_g, &
         !      graupel%nu,n0mp,lammp,graupel%avelo,graupel%bvelo, &
         !      ngl,xgl_g,wgl_g,g_g(:,n,:,:,:))

       ENDDO
    ELSE
       DO n=1,npr

          ! Cloud
          lam=(const2_c/(const1_c*xprci(n)))**const3_c
          n0=const4_c*lam**const0_c
          CALL scavenging_coefficient(na,nc,nt,np,nrhw,xda,xdc,e_c, &
               cloud%nu,n0,lam,cloud%avelo,cloud%bvelo,        &
               ngl,xgl_c,wgl_c,g_c(:,n,:,:,:))
          
          ! Rain
          n0mp=8.E6_dp ! m-3 m-1
          lammp=1.E3_dp*4.1_dp*xprrsg(n)**(-.21_dp) ! m-1
          CALL scavenging_coefficient(na,nr,nt,np,1,xda,xdr,e_r, &
               rain%nu,n0mp,lammp,rain%avelo,rain%bvelo,         &
               ngl,xgl_r,wgl_r,g_r(:,n,:,:,:))

         ! ! Ice
         ! lam=(const2_i/(const1_i*xprci(n)))**const3_i
         ! n0=const4_i*lam**const0_i
         ! CALL scavenging_coefficient(na,ni,nt,np,nrhi,xda,xdi,e_i, &
         !      ice%nu,n0,lam,ice%avelo,ice%bvelo,              &
         !      ngl,xgl_i,wgl_i,g_i(:,n,:,:,:))
          
         ! ! Snow
         ! ! Gunn and Marshall (1958)
         ! !n0mp=3.8E6_dp*xprrsg(n)**(-.87_dp) ! m-3 m-1
         ! !lammp=25.5E2_dp*xprrsg(n)**(-.48_dp) ! m-1
         ! ! Sekhon and Srivastava (1970)
         ! n0mp=2.5E6_dp*xprrsg(n)**(-.94_dp) ! m-3 m-1
         ! lammp=22.9E2_dp*xprrsg(n)**(-.45_dp) ! m-1
         ! CALL scavenging_coefficient(na,ns,nt,np,1,xda,xds,e_s, &
         !      snow%nu,n0mp,lammp,snow%avelo,snow%bvelo,         &
         !      ngl,xgl_s,wgl_s,g_s(:,n,:,:,:))
         ! 
         ! ! Graupel
         ! n0mp=2.5E6_dp*xprrsg(n)**(-.94_dp) ! m-3 m-1
         ! lammp=22.9E2_dp*xprrsg(n)**(-.45_dp) ! m-1
         ! CALL scavenging_coefficient(na,nr,nt,np,1,xda,xdg,e_g, &
         !      graupel%nu,n0mp,lammp,graupel%avelo,graupel%bvelo,&
         !      ngl,xgl_g,wgl_g,g_g(:,n,:,:,:))

       ENDDO
    ENDIF

    IF (ldebug) THEN

       ! Write debug output for collision efficiencies
       ! Aerosol-cloud
       x1=40.E-6
       x2=60.E-6
       x3=80.E-6
       x4=282.0_dp
       x5=900.0E2_dp
       x6=1.00_dp

       CALL locate(xdc,nc,x1,idx1)
       CALL locate(xdc,nc,x2,idx2)
       CALL locate(xdc,nc,x3,idx3)
       IF (luse_standard_t_p) THEN
          idx4=1
          idx5=1
       ELSE
          CALL locate(xt,nt,x4,idx4)
          CALL locate(xp,np,x5,idx5)
       ENDIF
       CALL locate(xrhw,nrhw,x6,idx6)

       OPEN(10,file="e_cloud.dat",status="replace",action="write")
       WRITE(10,"(4es20.13)") 0.0_dp, x1, x2, x3
       WRITE(10,"(4es20.13)") (xda(n), e_c(n,idx1,idx4,idx5,idx6), &
            e_c(n,idx2,idx4,idx5,idx6), e_c(n,idx3,idx4,idx5,idx6), n=1,na)
       CLOSE(10,status="keep")

       OPEN(12,file="e_cloud_all.dat",status="replace",action="write",form="unformatted")
       WRITE(12) e_c,xda,xdc,xrhw
       CLOSE(12,status="keep")

       ! Write debug output for collision efficiencies
       ! Aerosol-rain
       x1=0.1E-3
       x2=0.5E-3
       x3=1.0E-3
       x4=282.0_dp
       x5=900.0E2_dp
       
       CALL locate(xdr,nr,x1,idx1)
       CALL locate(xdr,nr,x2,idx2)
       CALL locate(xdr,nr,x3,idx3)
       IF (luse_standard_t_p) THEN
          idx4=1
          idx5=1
       ELSE
          CALL locate(xt,nt,x4,idx4)
          CALL locate(xp,np,x5,idx5)
       ENDIF
       
       OPEN(10,file="e_rain.dat",status="replace",action="write")
       WRITE(10,"(4es20.13)") 0.0, x1, x2, x3
       WRITE(10,"(4es20.13)") (xda(n), e_r(n,idx1,idx4,idx5,1), e_r(n,idx2,idx4,idx5,1), &
            e_r(n,idx3,idx4,idx5,1), n=1,na)
       CLOSE(10,status="keep")

       OPEN(12,file="e_rain_all.dat",status="replace",action="write",form="unformatted")
       WRITE(12) e_r,xda,xdr
       CLOSE(12,status="keep")

      ! ! Write debug output for collision efficiencies
      ! ! Aerosol-ice
      ! x1=100.E-6
      ! x2=150.E-6
      ! x3=200.E-6
      ! x4=263.0_dp
      ! x5=1013.0E2_dp
      ! x6=0.95_dp

      ! CALL locate(xdi,ni,x1,idx1)
      ! CALL locate(xdi,ni,x2,idx2)
      ! CALL locate(xdi,ni,x3,idx3)
      ! IF (luse_standard_t_p) THEN
      !    idx4=1
      !    idx5=1
      ! ELSE
      !    CALL locate(xt,nt,x4,idx4)
      !    CALL locate(xp,np,x5,idx5)
      ! ENDIF
      ! CALL locate(xrhw,nrhw,x6,idx6)

      ! OPEN(10,file="e_ice.dat",status="replace",action="write")
      ! WRITE(10,"(4es20.13)") 0.0_dp, x1, x2, x3
      ! WRITE(10,"(4es20.13)") (xda(n), e_i(n,idx1,idx4,idx5,idx6), &
      !      e_i(n,idx2,idx4,idx5,idx6), e_i(n,idx3,idx4,idx5,idx6), n=1,na)
      ! CLOSE(10,status="keep")

      ! ! Write debug output for collision efficiencies
      ! ! Aerosol-snow
      ! x1=1.0E-3
      ! x2=2.5E-3
      ! x3=4.5E-3
      ! x4=263.0_dp
      ! x5=1013.0E2_dp
       
      ! CALL locate(xds,ns,x1,idx1)
      ! CALL locate(xds,ns,x2,idx2)
      ! CALL locate(xds,ns,x3,idx3)
      ! IF (luse_standard_t_p) THEN
      !    idx4=1
      !    idx5=1
      ! ELSE
      !    CALL locate(xt,nt,x4,idx4)
      !    CALL locate(xp,np,x5,idx5)
      ! ENDIF
       
      ! OPEN(10,file="e_snow.dat",status="replace",action="write")
      ! WRITE(10,"(4es20.13)") 0.0, x1, x2, x3
      ! WRITE(10,"(4es20.13)") (xda(n), e_s(n,idx1,idx4,idx5,1), e_s(n,idx2,idx4,idx5,1), &
      !      e_s(n,idx3,idx4,idx5,1), n=1,na)
      ! CLOSE(10,status="keep")

      ! ! Write debug output for collision efficiencies
      ! ! Aerosol-graupel
      ! x1=1.0E-3
      ! x2=2.5E-3
      ! x3=4.5E-3
      ! x4=263.0_dp
      ! x5=1013.0E2_dp
       
      ! CALL locate(xdg,ng,x1,idx1)
      ! CALL locate(xdg,ng,x2,idx2)
      ! CALL locate(xdg,ng,x3,idx3)
      ! IF (luse_standard_t_p) THEN
      !    idx4=1
      !    idx5=1
      ! ELSE
      !    CALL locate(xt,nt,x4,idx4)
      !    CALL locate(xp,np,x5,idx5)
      ! ENDIF
       
      ! OPEN(10,file="e_graupel.dat",status="replace",action="write")
      ! WRITE(10,"(4es20.13)") 0.0, x1, x2, x3
      ! WRITE(10,"(4es20.13)") (xda(n), e_g(n,idx1,idx4,idx5,1), e_g(n,idx2,idx4,idx5,1), &
      !      e_g(n,idx3,idx4,idx5,1), n=1,na)
      ! CLOSE(10,status="keep")

       ! Write debug output for scavenging coefficients
       ! Aerosol-rain
       x1=0.01
       x2=0.1
       x3=1.0
       x4=10.0
       x5=100.0
       
       CALL locate(xprrsg,npr,x1,idx1)
       CALL locate(xprrsg,npr,x2,idx2)
       CALL locate(xprrsg,npr,x3,idx3)
       CALL locate(xprrsg,npr,x4,idx4)
       CALL locate(xprrsg,npr,x5,idx5)

       IF (luse_standard_t_p) THEN
          idx6=1
          idx7=1
       ELSE
          CALL locate(xt,nt,x6,idx6)
          CALL locate(xp,np,x7,idx7)
       ENDIF
       
       OPEN(11,file="g_rain.dat",status="replace",action="write")
       WRITE(11,"(6es20.13)") 0.0, x1, x2, x3, x4, x5
       WRITE(11,"(6es20.13)") (xda(n), g_r(n,idx1,idx6,idx7,1), g_r(n,idx2,idx6,idx7,1), &
            g_r(n,idx3,idx6,idx7,1), g_r(n,idx4,idx6,idx7,1), g_r(n,idx5,idx6,idx7,1),   &
            n=1,na)
       CLOSE(11,status="keep")

      ! ! Write debug output for scavenging coefficients
      ! ! Aerosol-snow
      ! x1=0.01
      ! x2=0.1
      ! x3=1.0
      ! x4=10.0
      ! x5=100.0
       
      ! CALL locate(xprrsg,npr,x1,idx1)
      ! CALL locate(xprrsg,npr,x2,idx2)
      ! CALL locate(xprrsg,npr,x3,idx3)
      ! CALL locate(xprrsg,npr,x4,idx4)
      ! CALL locate(xprrsg,npr,x5,idx5)

      ! IF (luse_standard_t_p) THEN
      !    idx6=1
      !    idx7=1
      ! ELSE
      !    CALL locate(xt,nt,x6,idx6)
      !    CALL locate(xp,np,x7,idx7)
      ! ENDIF
       
      ! OPEN(11,file="g_snow.dat",status="replace",action="write")
      ! WRITE(11,"(6es20.13)") 0.0, x1, x2, x3, x4, x5
      ! WRITE(11,"(6es20.13)") (xda(n), g_s(n,idx1,idx6,idx7,1), g_s(n,idx2,idx6,idx7,1), &
      !      g_s(n,idx3,idx6,idx7,1), g_s(n,idx4,idx6,idx7,1), g_s(n,idx5,idx6,idx7,1),   &
      !      n=1,na)
      ! CLOSE(11,status="keep")

      ! ! Write debug output for scavenging coefficients
      ! ! Aerosol-graupel
      ! x1=0.01
      ! x2=0.1
      ! x3=1.0
      ! x4=10.0
      ! x5=100.0
       
      ! CALL locate(xprrsg,npr,x1,idx1)
      ! CALL locate(xprrsg,npr,x2,idx2)
      ! CALL locate(xprrsg,npr,x3,idx3)
      ! CALL locate(xprrsg,npr,x4,idx4)
      ! CALL locate(xprrsg,npr,x5,idx5)

      ! IF (luse_standard_t_p) THEN
      !    idx6=1
      !    idx7=1
      ! ELSE
      !    CALL locate(xt,nt,x6,idx6)
      !    CALL locate(xp,np,x7,idx7)
      ! ENDIF
       
      ! OPEN(11,file="g_graupel.dat",status="replace",action="write")
      ! WRITE(11,"(6es20.13)") 0.0, x1, x2, x3, x4, x5
      ! WRITE(11,"(6es20.13)") (xda(n), g_g(n,idx1,idx6,idx7,1), g_g(n,idx2,idx6,idx7,1), &
      !      g_g(n,idx3,idx6,idx7,1), g_g(n,idx4,idx6,idx7,1), g_g(n,idx5,idx6,idx7,1),   &
      !      n=1,na)
      ! CLOSE(11,status="keep")

    ENDIF
   
  END SUBROUTINE init_scavenging


  SUBROUTINE m2011_scavenging(qna,qma,cmd, cmddry, sigma, &
       & t,p,qv,rhw,rhi,                                  &
       & qc,qr,qi,qs,qg,                                  &
       & qnc,qnr,qni,qns,qng,                             &
       & lcloud, lrain, lice, lsnow, lgraupel,            &
       & ie,je,ke,nmod,dt,lprint)
    ! 
    ! Aerosol wet scavenging parameterization
    !

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: &
         ie,               &
         je,               &
         ke,               &
         nmod                ! Number of lognormal aerosol modes
    
    REAL(KIND=dp), DIMENSION(ie,je,ke), INTENT(IN) :: &
         t,                & ! Air Temperature
         p,                & ! Air Pressure
         qv,               & ! Specific humidity
         rhw,              & ! relative humidity over liquid water
         rhi,              & ! relative humidity over ice
         qc,               & ! Cloud liquid water
         qr,               & ! Rain         water
         qi,               & ! Ice          
         qs,               & ! Snow
         qg,               & ! Graupel
         qnc,              & ! Cloud droplet number concentration
         qnr,              & ! Rain  drop    number concentration
         qni,              & ! Ice           number concentration
         qns,              & ! Snow          number concentration
         qng                 ! Graupel       number concentration
    
    REAL(KIND=dp), DIMENSION(ie,je,ke,nmod), INTENT(INOUT) :: &
         qna,              & ! Aerosol number concentrations
         qma,              & ! Aerosol mass   concentrations
         cmd,              &   ! Count median diameter 
         cmddry
    REAL(KIND=dp), DIMENSION(nmod), INTENT(IN) :: sigma              &
         & ! Standard deviation of lognormal aerosol modes
    
    REAL(KIND=dp), INTENT(IN) :: dt

    LOGICAL, INTENT(IN) :: lcloud, lrain, lice, lsnow, lgraupel,&
         & lprint
    

    ! Local vaiables
    INTEGER :: i, j, k, n, imod
    INTEGER :: idx_c, idx_i, idx_r, idx_s, idx_g, idx_t, idx_p, idx_rhw, idx_rhi

    REAL(KIND=dp) :: dd
    REAL(KIND=dp) :: const
    REAL(KIND=dp), PARAMETER :: qxeps=1.E-5_dp

    !REAL(KIND=dp), DIMENSION(ie,je,ke) :: rhw, rhi
    REAL(KIND=dp), DIMENSION(ie,je,ke) :: lam_c, n0_c, pr_c
    REAL(KIND=dp), DIMENSION(ie,je,ke) :: lam_r, n0_r, pr_r
    REAL(KIND=dp), DIMENSION(ie,je,ke) :: lam_i, n0_i, pr_i
    REAL(KIND=dp), DIMENSION(ie,je,ke) :: lam_s, n0_s, pr_s
    REAL(KIND=dp), DIMENSION(ie,je,ke) :: lam_g, n0_g, pr_g
    REAL(KIND=dp), DIMENSION(ie,je,ke,na) :: g3d
    REAL(KIND=dp), DIMENSION(ie,je,ke,nmod) :: dndt, dmdt


    ! Compute size hydrometeor distribution parameters 
    ! and sedimentation fluxes from given hydrometeor
    ! mass and number densities

    ! Cloud
    CALL psdparam(qc,qnc,rho_w,cloud%nu,cloud%avelo,cloud%bvelo, &
         cloud%amass,cloud%bmass,lam_c,n0_c,pr_c,ie,je,ke)
    
    ! Rain
    CALL psdparam(qr,qnr,rho_w,rain%nu,rain%avelo,rain%bvelo, &
         rain%amass,rain%bmass,lam_r,n0_r,pr_r,ie,je,ke)
    
    !! Ice
    !CALL psdparam(qi,qni,rho_i,ice%nu,ice%avelo,ice%bvelo, &
    !     ice%amass,ice%bmass,lam_i,n0_i,pr_i,ie,je,ke)
    
    !! Snow
    !CALL psdparam(qs,qns,rho_s,snow%nu,snow%avelo,snow%bvelo, &
    !     snow%amass,snow%bmass,lam_s,n0_s,pr_s,ie,je,ke)
    
    !! Graupel
    !CALL psdparam(qg,qng,rho_g,graupel%nu,graupel%avelo,graupel%bvelo, &
    !     graupel%amass,graupel%bmass,lam_g,n0_g,pr_g,ie,je,ke)


    ! Compute relative humidity w.r.t. liquid water/ice from t, p, qv
    ! ...
    ! ...
    !rhw(:,:,:) = 0.95_dp
    !rhi(:,:,:) = 0.99_dp
   
    g3d=0._dp
    IF (luse_standard_t_p) THEN
       DO k=1,ke
          DO j=1,je
             DO i=1,ie

                ! Table look up
                ! Index for relative humidity
                ! ... to be done...
                !idx_rhw=1
                !idx_rhi=1

                IF (lcloud) CALL locate(xrhw,nrhw,rhw(i,j,k),idx_rhw) 
                !IF (lice)   CALL locate(xrhi,nrhi,rhi(i,j,k),idx_rhi)

                ! Index for precipitation rate
                ! Cloud
                IF (lcloud .AND. qc(i,j,k) > qxeps) THEN
                   CALL locate(xprci,npr,pr_c(i,j,k),idx_c)
                   DO n=1,na
                      g3d(i,j,k,n) = g3d(i,j,k,n) + g_c(n,idx_c,1,1,idx_rhw)
                   ENDDO
                ENDIF
                
                ! Rain
                IF (lrain .AND. qr(i,j,k) > qxeps) THEN
                   CALL locate(xprrsg,npr,pr_r(i,j,k)*3600.0_dp,idx_r)
                   DO n=1,na
                      g3d(i,j,k,n) = g3d(i,j,k,n) + g_r(n,idx_r,1,1,1)
                   ENDDO
                ENDIF
                
                !! Ice
                !IF (lice .AND. qi(i,j,k) > qxeps) THEN
                !   CALL locate(xprci,npr,pr_i(i,j,k),idx_i)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_i(n,idx_i,1,1,idx_rhi)
                !   ENDDO
                !ENDIF

                !! Snow
                !IF (lsnow .AND. qs(i,j,k) > qxeps) THEN
                !   CALL locate(xprrsg,npr,pr_s(i,j,k)*3600.0_dp,idx_s)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_s(n,idx_s,1,1,1)
                !   ENDDO
                !ENDIF

                !! Graupel
                !IF (lgraupel .AND. qg(i,j,k) > qxeps) THEN
                !   CALL locate(xprrsg,npr,pr_g(i,j,k)*3600.0_dp,idx_g)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_g(n,idx_g,1,1,1)
                !   ENDDO
                !ENDIF
                
             ENDDO
          ENDDO
       ENDDO
    ELSE
       DO k=1,ke
          DO j=1,je
             DO i=1,ie
       
                ! Index for temperature
                CALL locate(xt,nt,t(i,j,k),idx_t)

                ! Index for pressure
                CALL locate(xp,np,p(i,j,k),idx_p)

                ! Index for relative humidity
                ! ... to be done...
                !idx_rhw=1
                !idx_rhi=1
                
                IF (lcloud) CALL locate(xrhw,nrhw,rhw(i,j,k),idx_rhw) 
                !IF (lice)   CALL locate(xrhi,nrhi,rhi(i,j,k),idx_rhi)

                                ! Index for precipitation rate
                ! Cloud
                IF (lcloud .AND. qc(i,j,k) > qxeps) THEN
                   CALL locate(xprci,npr,pr_c(i,j,k),idx_c)
                   DO n=1,na
                      g3d(i,j,k,n) = g3d(i,j,k,n) + g_c(n,idx_c,idx_t,idx_p,idx_rhw)
                   ENDDO
                ENDIF
                
                ! Rain
                IF (lrain .AND. qr(i,j,k) > qxeps) THEN
                   CALL locate(xprrsg,npr,pr_r(i,j,k)*3600.0_dp,idx_r)
                   DO n=1,na
                      g3d(i,j,k,n) = g3d(i,j,k,n) + g_r(n,idx_r,idx_t,idx_p,1)
                   ENDDO
                ENDIF
                
                !! Ice
                !IF (lice .AND. qi(i,j,k) > qxeps) THEN
                !   CALL locate(xprci,npr,pr_i(i,j,k),idx_i)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_i(n,idx_i,idx_t,idx_p,idx_rhi)
                !   ENDDO
                !ENDIF

                !! Snow
                !IF (lsnow .AND. qs(i,j,k) > qxeps) THEN
                !   CALL locate(xprrsg,npr,pr_s(i,j,k)*3600.0_dp,idx_s)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_s(n,idx_s,idx_t,idx_p,1)
                !   ENDDO
                !ENDIF

                !! Graupel
                !IF (lgraupel .AND. qg(i,j,k) > qxeps) THEN
                !   CALL locate(xprrsg,npr,pr_g(i,j,k)*3600.0_dp,idx_g)
                !   DO n=1,na
                !      g3d(i,j,k,n) = g3d(i,j,k,n) + g_g(n,idx_g,idx_t,idx_p,1)
                !   ENDDO
                !ENDIF

             ENDDO
          ENDDO
       ENDDO
    ENDIF

    IF (lprint) THEN
        DO k = 186, 231
           print*, k, 'diam=', cmd(1,1,k,1), 'gamma=',g3d(1,1,k,1), g3d(1,1,k,2), g3d(1,1,k,3), g3d(1,1,k,4), g3d(1,1,k, 5), 'pflux=', pr_c(1,1,k), 'relhum=', rhw(1,1,k)
        ENDDO
    ENDIF
    ! Compute aerosol scavenging by cloud, rain, ice, snow and graupel
    CALL scavenging_num(na,xda,ngh,xgh,wgh, &
         ie,je,ke,nmod,qna,cmd,sigma,g3d,dndt,dmdt,lprint)
     
    WHERE (dndt > 0._dp) dndt=0._dp
    WHERE (dmdt > 0._dp) dmdt=0._dp

    if (lprint) then
        do k=186,231
           print*, k, 'n=',  qna(1,1,k,1), 'dndt=', dndt(1,1,k,1), 'dmdt=', dmdt(1,1,k,1)
        enddo
    endif
   
    ! Add tendencies
    DO imod=1,nmod
       ! Convert diameter of average mass to count median diameter
       const=EXP(-1.5_dp*LOG(sigma(imod))**2._dp)
       DO k=1,ke
          DO j=1,je
             DO i=1,ie
                qna(i,j,k,imod) = qna(i,j,k,imod) + dndt(i,j,k,imod)*dt
                qna(i,j,k,imod)=MAX(qna(i,j,k,imod),0._dp)
                qma(i,j,k,imod) = qma(i,j,k,imod) + dmdt(i,j,k,imod)*dt*(cmddry(i,j,k,imod)/cmd(i,j,k,imod))**3.
                qma(i,j,k,imod)=MAX(qma(i,j,k,imod),0._dp)
                if (qna(i,j,k,imod).gt.0.) then
                   cmd(i,j,k,imod)=const*(6._dp*qma(i,j,k,imod)/ &
                        (pi*rho_p*qna(i,j,k,imod)))**three_r
                else 
                   cmd(i,j,k,imod)=0.
                end if
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    

  END SUBROUTINE m2011_scavenging


  SUBROUTINE psdparam(qx,qnx,rhox,nu,avelo,bvelo,amass,bmass,lam,n0,pr,ni,nj,nk)

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: ni, nj, nk
    REAL(KIND=dp), DIMENSION(ni,nj,nk), INTENT(IN) :: qx, qnx
    REAL(KIND=dp), INTENT(IN) :: rhox, nu, avelo, bvelo, amass, bmass
    REAL(KIND=dp), DIMENSION(ni,nj,nk), INTENT(OUT) :: lam, n0, pr

    ! Local variables
    INTEGER :: i, j, k
    REAL(KIND=dp) :: const0, const1, const2, const3, const4, const5
    REAL(KIND=dp), PARAMETER :: eps=1.E-20_dp

    
    const0=nu+1._dp
    const1=(nu+3._dp)*(nu+2._dp)*const0
    const2=1._dp/EXP(gammln(const0))
    const3=bvelo+bmass+const0
    const4=EXP(gammln(const3))
    const5=avelo*amass*const4
    
    DO k=1,nk
       DO j=1,nj
          DO i=1,ni

             ! Slope parameter (m-1)
             lam(i,j,k)=(pi * rhox * qnx(i,j,k) * const1 / &
                  (6._dp*(qx(i,j,k)+eps)))**three_r

             ! Intercept parameter (m-3 m-nu)
             n0(i,j,k)=const2 * qnx(i,j,k) * lam(i,j,k)**const0

             ! Sedimentation (mass) flux (i.e., precipitation rate) (kg m-2 s-1)
             ! from intercept and slope parameter
             pr(i,j,k)=const5 * n0(i,j,k) / (lam(i,j,k)+eps)**const3

          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE psdparam
             

  SUBROUTINE collision_efficiency_cloud(na,nc,nt,np,nrh,xda,xdc,xu,xt,xp,xrh,e)
    ! Subroutine to calculate the collision efficiencies
    ! for aerosols and cloud droplets.
    !
    ! Input:
    ! ------
    ! na ... Number or bins for aerosols
    ! nc ... Number of bins for cloud droplets
    ! nt ... Number of bins for temperature
    ! np ... Number of bins for pressure
    ! nrh... Number of bins for rel. hum. w.r.t. liquid water
    ! xda ... Aerosol diameters (m)
    ! xdc ... Cloud droplet diameters (m)
    ! xu ... Terminal fall velocity of cloud droplets (m/s)
    ! xt ... Air temperature (K)
    ! xp ... Air pressure (Pa)
    ! rh ... Relative humidity (0-1)
    !
    ! Output:
    ! -------
    ! e  ... Collision efficiency
    !
    IMPLICIT NONE
    
    INTEGER, INTENT(IN) :: na, nc, nt, np, nrh
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: xda
    REAL(KIND=dp), DIMENSION(nc), INTENT(IN) :: xdc, xu
    REAL(KIND=dp), DIMENSION(nt), INTENT(IN) :: xt
    REAL(KIND=dp), DIMENSION(np), INTENT(IN) :: xp
    REAL(KIND=dp), DIMENSION(nrh), INTENT(IN) :: xrh
    REAL(KIND=dp), DIMENSION(na,nc,nt,np,nrh), INTENT(OUT) :: e

    ! Local variables
    REAL(KIND=dp) :: Re, C_c, D_c, vol, C_p, D_p, & 
         Sc, St, tau_a, mu_a, nu_a, u_a, & 
         e1, e2, e3, e4, Kn, b_p, f_v, k_a, & 
         l_v, e_sw, s_w, g_w, f_v_crit, d_v, lambda_a, &
         rho_a, da, dc, u, t, p, rh, const1
    INTEGER :: i, j, k, m, n


    DO n=1,nrh

       rh= xrh(n)
       
       DO m=1,np

          p = xp(m)

          DO k=1,nt

             t = xt(k)

             ! Air density
             rho_a = p/(r_d*t)

             ! Dynamic viscosity of air
             mu_a = 1.E-5_dp*(1.718_dp+4.9E-3_dp* &
                  (t-273.15_dp)-1.2E-5_dp*(t-273.15_dp)**2._dp)
             
             ! Kinematic viscosity of air
             nu_a = mu_a/rho_a 
             const1 = SQRT(ec/nu_a)

             ! Mean free path in air
             lambda_a=2._dp*mu_a/(p*SQRT(8._dp*M_a/(pi*r_gas*t)))

             ! Thermal conductivity of air
             k_a = 4.184E-3_dp*(5.69_dp+0.017_dp*(t-273.15_dp))

             ! Vapor diffusivity of air 
             d_v = 0.211E-4_dp*(t/273.15_dp)**1.94_dp*(101325._dp/p)

             ! Latent heat of condensation/evaporation
             l_v = 2.5E6_dp*(273.15_dp/t)**(0.167_dp+3.67E-4_dp*t)

             ! Saturation vapor pressure w.r.t water
             e_sw = saturation_water(t)
    
             g_w=( rho_w*R_v*T/e_sw/D_v + rho_w*L_v/k_a/T * &
                  ((L_v/R_v/T)-1._dp) )
             g_w=1._dp/g_w
    
             ! Supersaturation w.r.t. water
             s_w=rh-1._dp     
       
             DO j=1,nc
                
                dc=xdc(j)
                u =xu(j) 

                ! Reynolds number of cloud droplets
                Re = dc*u*rho_a/mu_a

                ! Ventilation coefficient (PK, p, 541)
                f_v_crit = 0.71_dp**three_r*Re**0.5_dp
                IF ( f_v_crit < 1.4_dp ) THEN
                   f_v=1.0_dp+0.108_dp*f_v_crit**2._dp
                ELSE
                   f_v=0.78_dp+0.308_dp*f_v_crit
                ENDIF

                ! Cunningham slip correction factor for cloud droplets
                !C_c = 1._dp + 1.26_dp*(2._dp*lambda_a/dc)* &
                !     (101325.0_dp/p)*(T/273.15_dp)
                C_c = 1._dp + 2._dp*lambda_a/dc* &
                     (1.257_dp+0.4_dp*EXP(-1.1_dp*dc/2._dp/lambda_a))

                ! Diffusivity of cloud droplets
                D_c = kboltz*T*C_c/(threepi*mu_a*dc)   
                
                ! Sweep out volume of cloud droplets
                vol = pio4*dc**2._dp*u     
                
                DO i=1,na
                   
                   da=xda(i)
                   
                   ! Cunningham slip correction factor for aerosols
                   !C_p = 1._dp + 1.26_dp*(2._dp*lambda_a/da)* &
                   !     (101325.0_dp/p)*(T/273.15_dp)
                   C_p = 1._dp + 2._dp*lambda_a/da* &
                        (1.257_dp+0.4_dp*EXP(-1.1_dp*da/2._dp/lambda_a))
          
                   ! Diffusivity of aerosols
                   D_p = kboltz*T*C_p/(threepi*mu_a*da)   
          
                   ! Schmidt number of aerosols
                   Sc = mu_a/(rho_a*D_p)          

                   ! Knudson number of aerosols
                   Kn = 2._dp*lambda_a/da       
       
                   ! B term for thermophoresis, see PK p. 725
                   B_p=0.4_dp*C_p*(k_a+2.5_dp*k_p*Kn)/((1._dp+3._dp*Kn)* &
                        (k_p+2._dp*k_a+5._dp*k_p*Kn))

                   ! Relaxation time scale (aerosols)
                   tau_a = rho_p*da**2._dp*C_p/(18._dp*mu_a)

                   ! Terminal fall velocity of aerosols
                   u_a = tau_a*gravi                  
          
                   ! Stokes number
                   St = 2._dp*tau_a*(u-u_a)/dc

                   ! Collision kernel for convective Brownian diffusion
                   e1 = twopi*(D_p+D_c)*(da+dc)
                   ! Thermophoresis kernel
                   e2 = -twopi*B_p*f_v*L_v*rho_w*s_w*g_w*dc/p
                   ! Diffusiophoresis kernel
                   e3 = 2.4_dp*pi*f_v*rho_w*s_w*g_w*dc/rho_a
                   ! Turbulent collision kernel
                   e4 = 1.3_dp*const1*(da+dc)**3._dp
          
                   !e(i,j,k,m,n) = e1 + MAX(e2,0._dp) + MAX(e3,0._dp) + e4
                   e(i,j,k,m,n) = e1 +  e4
                   ! e(i,j,k,m,n) = e1 + MAX((e2+e3),0._dp) + e4
                   e(i,j,k,m,n) = e(i,j,k,m,n) / vol

                   ! Constrain collision efficiency
                   !e(i,j,k,m,n) = MIN(e(i,j,k,m,n),1._dp)       

                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    
  END SUBROUTINE collision_efficiency_cloud


  SUBROUTINE collision_efficiency_rain(na,nr,nt,np,xda,xdr,xu,xt,xp,e)
    ! Subroutine to calculate the collision efficiencies
    ! for aerosols and rain drops after Slinn (1983).
    !
    ! Input:
    ! ------
    ! na ... Number or bins for aerosols
    ! nr ... Number of bins for rain drops
    ! nt ... Number of bins for temperature
    ! np ... Number of bins for pressure
    ! xda ... Aerosol diameters (m)
    ! xdr ... Rain drop diameters (m)
    ! xu ... Terminal fall velocity of rain drops (m/s)
    ! xt  ... Air temperature (K)
    ! xp  ... Air pressure (Pa)
    !
    ! Output:
    ! -------
    ! e  ... Collision efficiency
    !
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: na, nr, nt, np
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: xda
    REAL(KIND=dp), DIMENSION(nr), INTENT(IN) :: xdr, xu
    REAL(KIND=dp), DIMENSION(nt), INTENT(IN) :: xt
    REAL(KIND=dp), DIMENSION(np), INTENT(IN) :: xp
    REAL(KIND=dp), DIMENSION(na,nr,nt,np), INTENT(OUT) :: e

    ! Local variables
    INTEGER :: i, j, k, m
    REAL(KIND=dp) :: da, dr, u
    REAL(KIND=dp) :: Re, C, Dv, Sc, phi, omega_1, St, S, &
         tau_a, u_a, mu_a, rho_a, lambda_a, t, p, e1, e2, e3

    
    DO m=1,np

       p = xp(m)

       DO k=1,nt

          t = xt(k)

          ! Dynamic viscosity of air
          mu_a = 1.E-5_dp*(1.718_dp+4.9E-3_dp* & 
               (t-273.15_dp)-1.2E-5_dp*(t-273.15_dp)**2._dp)

          ! Mean free path in air
          lambda_a = 2._dp*mu_a/(p*SQRT(8._dp*M_a/(pi*r_gas*t)))

          ! Air density
          rho_a = p/(r_d*t)

          ! Viscosity ratio
          omega_1 = mu_a/mu_w

          DO j=1,nr

             dr=xdr(j)
             u =xu(j)

             ! Reynolds number of rain drops
             Re = dr*u*rho_a/(2._dp*mu_a)

             ! Critical Stokes number
             S = (1.2_dp+LOG(1._dp+Re)/12._dp)/(1._dp+LOG(1._dp+Re))
            
             DO i=1,na

                da=xda(i)
                
                ! Cunningham slip correction factor
                ! C = 1. + 1.26*(2*lambda/da)*(101325.0/p)*(T/273.15)  
                C = 1._dp + 2._dp*lambda_a/da*(1.257_dp+0.4_dp*EXP(-1.1_dp*da/2._dp/lambda_a))

                ! Aerosol diffusivity in air
                Dv = kboltz*T*C/(threepi*mu_a*da)

                ! Aerosol Schmidt number
                Sc = mu_a/(rho_a*Dv)

                ! Aerosol relaxation time scale
                tau_a = rho_p*da**2._dp*C/(18._dp*mu_a)

                ! Aerosol terminal fall velocity
                u_a = tau_a*gravi
          
                ! Stokes number
                St = 2._dp*tau_a*(u-u_a)/dr
   
                ! Diameter ratio
                phi = da/dr
    
                ! Collision efficiencies after Slinn (1983)
                ! Brownian motion
                e1 = 4._dp/(Re*Sc) * ( 1._dp + 0.4_dp*Re**0.5_dp*Sc**three_r + & 
                     0.16_dp*Re**0.5_dp*Sc**0.5_dp )
                ! Interception
                e2 = 4._dp*phi*( omega_1 + phi*(1._dp+2._dp*Re**0.5_dp ) )
                ! Impaction
                e3 = ( MAX((St-S),0._dp)/(St-S+(2._dp/3._dp)) )**1.5_dp*(rho_w/rho_p)**0.5_dp
          
                ! Constrain collision efficiency
                e(i,j,k,m) = e1 + e2 + e3
                !e(i,j,k,m) = MIN(e(i,j,k,m),1._dp)

             ENDDO
          ENDDO
       ENDDO
    ENDDO
    
  END SUBROUTINE collision_efficiency_rain


  SUBROUTINE collision_efficiency_snow(na,ns,nt,np,xda,xds,xu,xt,xp,e)
    ! Subroutine to calculate the collision efficiencies
    ! for aerosols and snow flakes after Murakami et al. (1985).
    ! Input:
    ! na ... Number of bins for aerosol size
    ! ns ... Number of bins for snow size
    ! nt ... Number of bins for temperature
    ! np ... Number of bins for pressure
    ! xda ... Aerosol diameters (m)
    ! xds ... Max. snow diameters (m)
    ! xu  ... Terminal fall velocities of snow (m/s)
    ! xt ... Temperature [K]
    ! xp ... Pressure [Pa]
    !
    ! Output:
    ! e ... collision efficiency
    !
    IMPLICIT NONE
    
    INTEGER, INTENT(IN) :: na, ns, nt, np
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: xda
    REAL(KIND=dp), DIMENSION(ns), INTENT(IN) :: xds
    REAL(KIND=dp), DIMENSION(ns), INTENT(IN) :: xu
    REAL(KIND=dp), DIMENSION(nt), INTENT(IN) :: xt
    REAL(KIND=dp), DIMENSION(np), INTENT(IN) :: xp
    REAL(KIND=dp), DIMENSION(na,ns,nt,np), INTENT(OUT) :: e

    ! Local variables
    INTEGER :: i, j, k, m
    REAL(KIND=dp) :: Re, C, Dv, Sc, St, tau_a, mu_a, & 
         u_a, e1, e2, e3, lambda_a, t, p, u, rho_a, ds, da
    REAL(KIND=dp), PARAMETER :: st_crit=0.0625_dp
    REAL(KIND=dp), PARAMETER :: eps=1.E-20_dp


    DO m=1,np

       p = xp(m)
       
       DO k=1,nt

          t = xt(k)
          
          ! Dynamic viscosity of air
          mu_a = 1.E-5_dp*(1.718_dp+4.9E-3_dp* & 
               (t-273.15_dp)-1.2E-5_dp*(t-273.15_dp)**2._dp)

          ! Mean free path in air
          lambda_a = 2._dp*mu_a/(p*SQRT(8._dp*M_a/(pi*r_gas*t)))

          ! Air density
          rho_a = p/(r_d*t)
          
          DO j=1,ns

             ds = xds(j)
             u  = xu(j)

             ! Snow flake Reynolds number
             Re = ds*u*rho_a/(2._dp*mu_a)

             DO i=1,na

                da = xda(i)

                ! Calculate Cunningham slip correction
                ! C = 1._dp + 1.26_dp*(2._dp*lambda_a/da)*(101325.0_dp/p)*(t/273.15_dp)
                C = 1._dp + 2._dp*lambda_a/da* &
                     (1.257_dp+0.4_dp*EXP(-1.1_dp*da/2._dp/lambda_a))

                ! Aerosol diffusivity in air
                Dv = kboltz*t*C/(threepi*mu_a*da)

                ! Aerosol Schmidt number
                Sc = mu_a/(rho_a*Dv) 

                ! Aerosol relaxation time scale
                tau_a = rho_p*da*da*C/(18._dp*mu_a)        

                ! Aerosol terminal fall velocity
                u_a = tau_a*gravi

                ! Stokes number
                St = 2._dp*tau_a*(u-u_a)/ds   

                ! Constrain St > 1/16
                St = MAX(St,St_crit)                   
                
                ! Murakami et al. (1985) collision efficiency
                e1 = (48._dp*Dv/(pi*ds*u))*(0.65_dp+0.44_dp*Re**0.5_dp*Sc**three_r)
                e2 = 28.5_dp*(da/ds)**1.186_dp
                e3 = EXP(-0.11_dp/MAX(SQRT(St)-0.25_dp,eps))
                
                e(i,j,k,m) = e1 + e2 + e3
                
                ! Constrain collision efficiency
                e(i,j,k,m) = MIN(e(i,j,k,m),1._dp)
                           
             ENDDO
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE collision_efficiency_snow


  SUBROUTINE collision_efficiency_ice(na,ni,nt,np,nrh,xda,xdi,xu,xt,xp,xrh,e)
    ! Subroutine to calculate the collision efficiencies
    ! for aerosols and ice crystals after PK
    ! Input:
    ! na ... Number of bins for aerosol size
    ! ni ... Number of bins for ice size
    ! nt ... Number of bins for temperature
    ! np ... Number of bins for pressure
    ! nrh ... Number of bins for relative humidity w.r.t. ice
    ! xda ... Aerosol diameters (m)
    ! xdi ... Max. ice diameters (m)
    ! xu  ... Terminal fall velocities of ice (m/s)
    ! xt ... Temperature (K)
    ! xp ... Pressure (Pa)
    !
    ! Output:
    ! e ... collision efficiency
    !

    IMPLICIT NONE
    
    INTEGER, INTENT(IN) :: na, ni, nt, np, nrh
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: xda
    REAL(KIND=dp), DIMENSION(ni), INTENT(IN) :: xdi, xu
    REAL(KIND=dp), DIMENSION(nt), INTENT(IN) :: xt
    REAL(KIND=dp), DIMENSION(np), INTENT(IN) :: xp
    REAL(KIND=dp), DIMENSION(nrh), INTENT(IN) :: xrh
    REAL(KIND=dp), DIMENSION(na,ni,nt,np,nrh), INTENT(OUT) :: e

    ! Local variables
    REAL(KIND=dp) :: Re, vol, Cap, C_p, D_p, BM_p, & 
         f_v, f_h, phif0, c_th, c_df, & 
         Kn, a_i, f_v_crit, lambda_a, rh, t, p, u, di, da, tmp
    REAL(KIND=dp) :: mu_a, nu_a, k_a, d_v, e_si, g_i, s_i, rho_a
    REAL(KIND=dp), PARAMETER :: eps=1.E-30_dp
    REAL(KIND=dp), PARAMETER :: gamma_i=0.65_dp, sigma_i=1.85_dp
    INTEGER :: i, j, k, m, n

    DO n=1,nrh

       rh = xrh(n)
       
       DO m=1,np

          p = xp(m)
          
          DO k=1,nt

             t = xt(k)

             ! Air density
             rho_a = p/(r_d*t)

             ! Dynamic viscosity of air
             mu_a = 1.E-5_dp*(1.718_dp+4.9E-3_dp* &
                  (t-273.15_dp)-1.2E-5_dp*(t-273.15_dp)**2._dp)
             
             ! Kinematic viscosity of air
             nu_a = mu_a/rho_a 
             
             ! Mean free path in air
             lambda_a=2._dp*mu_a/(p*SQRT(8._dp*M_a/(pi*r_gas*t)))

             ! Thermal conductivity of air
             k_a = 4.184E-3_dp*(5.69_dp+0.017_dp*(t-273.15_dp))

             ! Vapor diffusivity of air 
             d_v = 0.211E-4_dp*(t/273.15_dp)**1.94_dp*(101325._dp/p)

             ! Saturation vapor pressure w.r.t ice
             e_si = saturation_ice(t) 
             
             ! Supersaturation w.r.t. ice
             g_i=( rho_i*R_v*T/e_si/D_v + rho_i*L_i/k_a/T * ((L_i/R_v/T)-1) )**(-1._dp)
             s_i=rh-1._dp
             
             DO j=1,ni

                di = xdi(j)
                u = xu(j)

                ! Ice crystal Reynolds number
                Re = di*u*rho_a/(2._dp*mu_a)

                ! Ventilation coefficient for vapor
                f_v_crit = 0.71_dp**three_r*Re**0.5_dp
                IF ( f_v_crit < 1.4_dp ) THEN
                   f_v=1._dp+0.108_dp*(0.71_dp**three_r*Re**0.5_dp)**2._dp
                ELSE
                   f_v=0.78_dp+0.308_dp*(0.71_dp**three_r*Re**0.5_dp)
                ENDIF

                ! Ice crystal cross section 
                a_i = gamma_i*di**sigma_i

                ! Ice crystal sweep out volume
                vol = a_i*u 
                              
                ! Crystal capac. (hex. plates!)
                ! Attention! This is for hex. plates!!!
                Cap = di/pi                             
                
                DO i=1,na

                   da = xda(i)

                   ! Limit aerosol diameter to < 3 micron
                   da = MIN(da,3.E-6_dp)

                   ! Aerosol Cunningham slip correction factor
                   ! C_p = 1._dp + 1.26_dp*(2._dp*lambda_a/da)*(101325.0_dp/p)*(T/273.15_dp)
                   C_p = 1._dp + 2._dp*lambda_a/da* &
                        (1.257_dp+0.4_dp*EXP(-1.1_dp*da/2._dp/lambda_a))

                   ! Aerosol diffusivity in air
                   D_p = kboltz*T*C_p/(threepi*mu_a*da)            

                   ! Aerosol mobility
                   BM_p = C_p/(threepi*mu_a*da)
              
                   ! Aerosol Knudson number
                   Kn = 2._dp*lambda_a/da

                   ! Thermophoretic effect (Miller and Wang, 1989, Eq. 7)
                   c_th = -6._dp*pi*mu_a*da*(k_a+2.5_dp*k_p*Kn)*k_a/ & 
                        (5._dp*(1._dp+3._dp*Kn)*(k_p+2._dp*k_a+5._dp*k_p*Kn)*p)

                   ! Diffusiophoretic effect (Miller and Wang, 1989, Eq. 9)
                   c_df = threepi*mu_a*da*0.74_dp*D_v*m_a/(C_p*m_w*rho_a)

                   phif0 = -f_v*c_th*L_i*rho_i/k_a*S_i*g_i - &
                        f_v*C_Df*rho_i*S_i*g_i/D_v                  

                   ! Collision kernel for Brownian diffusion, 
                   ! thermophoresis and diffusiophoresis
                   e(i,j,k,m,n) = 4._dp*pi*BM_p*phif0*Cap/ &
                        (EXP(BM_p*phif0/(f_v*D_p))-1._dp+eps)/vol
                   
                   ! Constrain collision efficiency
                   e(i,j,k,m,n) = MIN(MAX(e(i,j,k,m,n),eps),1._dp)

                ENDDO
             ENDDO
          ENDDO
       ENDDO
    ENDDO

  END SUBROUTINE collision_efficiency_ice


  SUBROUTINE fallspeed_kc02(alpha,beta,gamma,sigma,d,rhox,rhoa,g,t,p,iflag,a,b,u)
    ! This routine calculates the terminal fall velocities of 
    ! hydrometeors given their power-law coefficients
    ! alpha, beta, gamma, sigma from table 2 of Khvorostyanov and Curry (2002).
    !
    ! Input:
    ! ------
    ! alpha, beta, gamma, sigma from table
    ! d ... diameter of hydrometeor
    ! rho_x ... bulk density of hydrometeor
    ! rho_a ... air density
    ! g ... gravitational acceleration
    ! t ... temperature
    ! p ... pressure
    ! iflag ... switch ( 2 for graupel, 1 for rain drops, 0 else )
    !
    IMPLICIT NONE

    REAL(KIND=dp), INTENT(IN) :: alpha, beta, gamma, sigma, & 
         d, rhox, rhoa, g, t, p
    INTEGER, INTENT(IN) :: iflag
    REAL(KIND=dp), INTENT(OUT) :: a, b, u

    REAL(KIND=dp) :: d_cgs, rhox_cgs, rhoa_cgs, g_cgs, mua, nua_cgs
    REAL(KIND=dp) :: zeta, alp, mass, area, vol, x, a_Re, b_Re
    REAL(KIND=dp) :: c1
    REAL(KIND=dp) :: c0, d0

    ! MKS to CGS 
    d_cgs=d*1.E2_dp
    rhox_cgs=rhox*1.E-3_dp
    rhoa_cgs=rhoa*1.E-3_dp
    g_cgs=g*1.E2_dp

    ! Parameters
    mua=1.E-5_dp*(1.718_dp+4.9E-3_dp* & 
         (T-273.15_dp)-1.2E-5_dp*(T-273.15_dp)**2._dp)
    mua=mua*10._dp
    nua_cgs=mua/rhoa_cgs
    IF ( iflag .EQ. 2 ) THEN
       c0=0.6_dp
       d0=5.83_dp
    ELSE
       c0=0.29_dp
       d0=9.06_dp 
    ENDIF
    c1=4._dp/(d0**2._dp*c0**0.5_dp)    

    ! Account for oblate spheres
    IF ( iflag .EQ. 1 ) THEN
       zeta=EXP(-d/0.47_dp)+(1._dp-EXP(-d/0.47_dp))*(1._dp/(1._dp+d/0.47_dp))
    ELSE
       zeta=1._dp
    ENDIF
    alp=alpha*zeta

    mass=alp*d_cgs**beta
    area=gamma*d_cgs**sigma
    vol=mass/rhox_cgs

    x=2._dp*vol*(rhox_cgs-rhoa_cgs)*g_cgs*d_cgs**2._dp/(area*rhoa_cgs*nua_cgs**2._dp)
    b_Re=.5_dp*c1*SQRT(x)/((1._dp+c1*SQRT(x))**0.5_dp-1._dp)*(1._dp+c1*SQRT(x))**(-0.5_dp)
    a_Re=d0**2._dp/4._dp*((1._dp+c1*SQRT(x))**0.5_dp-1._dp)**2._dp/x**b_Re

    a=a_Re*nua_cgs**(1._dp-2._dp*b_Re)*(2._dp*alpha*g_cgs/rhoa_cgs/gamma)**b_Re
    b=b_Re*(beta-sigma+2._dp)-1._dp
    u=a*d_cgs**b
    u=u*(1.E5_dp*t/(p*293._dp))**b_Re

    ! CGS to MKS
    a=a*100._dp**(b-1._dp) ! m/s
    a=a*(1.E5_dp*t/(p*293._dp))**b_Re
    u=u*1.E-2_dp         ! m/s

  END SUBROUTINE fallspeed_kc02


  FUNCTION saturation_water(T)
    ! Function to calculate the saturation vapor pressure
    ! with respect to water
    ! Input:
    ! T ... Temperature
    IMPLICIT NONE

    REAL(KIND=dp) :: saturation_water
    REAL(KIND=dp), INTENT(IN) :: T
    REAL(KIND=dp), PARAMETER :: e_3=6.10780000E2_dp
    REAL(KIND=dp), PARAMETER :: T_3=2.7316E2_dp
    REAL(KIND=dp), PARAMETER :: A_w=1.72693882E1_dp
    REAL(KIND=dp), PARAMETER :: B_w=3.58600000E1_dp

    saturation_water=e_3*EXP(A_w*(T-T_3)/(T-B_w))
    
  END FUNCTION saturation_water


  FUNCTION saturation_ice(T)
    ! Function to calculate the saturation vapor pressure
    ! with respect to ice
    ! Input:
    ! T ... Temperature
    IMPLICIT NONE

    REAL(KIND=dp) :: saturation_ice
    REAL(KIND=dp), INTENT(IN) :: T 
    REAL(KIND=dp), PARAMETER :: e_3=6.10780000E2_dp
    REAL(KIND=dp), PARAMETER :: T_3=2.7316E2_dp
    REAL(KIND=dp), PARAMETER :: A_e=2.18745584E1_dp
    REAL(KIND=dp), PARAMETER :: B_e=7.66000000E0_dp

    saturation_ice=e_3*EXP(A_e*(T-T_3)/(T-B_e))

  END FUNCTION saturation_ice


  SUBROUTINE scavenging_coefficient(na,nh,nt,np,nrh,xda,xdh,e,nu,n0,lam,acoef,bcoef, &
       ngl,xgl,wgl,gamma)
    ! This subroutine numerically computes the scavenging coefficients
    ! for different hydrometeors using a Gauss-Laguerre quadrature
    ! with ngl points.
    !
    ! Input:
    ! ------
    ! na ... Number of bins for aerosols
    ! nh ... Number of bins for hydrometeors
    ! nt ... Number of bins for temperature
    ! np ... Number of bins for pressure
    ! nrh... Number of bins for relative humidity
    ! xda... Diameters of aerosols (m)
    ! xdh... Diameters of hydrometeors (m)
    ! e  ... Collision efficiencies
    ! nu ... Shape parameter of hydrometeor size distribution
    ! n0 ... Intercept parameter of hydrometeor size distribution 
    ! lam... Slope parameter of hydrometeor size distribution (m-1)
    ! acoef ... a coefficient of terminal fall velocity power law
    ! bcoef ... b coefficient of terminal fall velocity power law
    ! ngl ... Number of Gauss-Laguerre quadrature points
    ! xgl ... Gauss-Laguerre abscissa values
    ! wgl ... Gauss-Laguerre weights
    !
    ! Output:
    ! -------
    ! gamma ... Scavenging coefficient (s-1)
    !
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: na, nh, nt, np, nrh
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: xda
    REAL(KIND=dp), DIMENSION(nh), INTENT(IN) :: xdh
    REAL(KIND=dp), DIMENSION(na,nh,nt,np,nrh), INTENT(IN) :: e
    REAL(KIND=dp), INTENT(IN) :: nu, n0, lam, acoef, bcoef
    INTEGER, INTENT(IN) :: ngl
    REAL(KIND=dp), DIMENSION(ngl), INTENT(IN) :: xgl, wgl
    REAL(KIND=dp), DIMENSION(na,nt,np,nrh), INTENT(OUT):: gamma

    ! Local variables
    REAL(KIND=dp) :: alf, dr, u, sum, coe, omcoe, eint
    INTEGER :: i, j, k, m, n, idx
    

    DO n=1,nrh
       DO m=1,np
          DO k=1,nt
             DO j=1,na
                
                ! Integration
                sum = 0._dp
                GAUSS_LAGUERRE: DO i=1,ngl
          
                   ! Rain drop diameter from quadrature abscissa values (m)
                   dr = xgl(i)/lam
          
                   ! Terminal fall speed
                   u = acoef*dr**bcoef
          
                   ! Determine collision efficiency from lookup table
                   CALL locate(xdh,nh,dr,idx)
          
                   !          ! Force idx within bounds of lookup table
                   !          idx=MAX(idx,1)
                   !          idx=MIN(idx,nh-1)
                   !          
                   !          ! Linearly interpolate coll. eff. between two adjacent values
                   !          ! to improve accuracy 
                   !          coe=(dr-dh(idx))/(dh(idx+1)-dh(idx))
                   !          omcoe=1._dp-coe
                   !          eint=coe*e(idx,j)+omcoe*e(idx+1,j)
                   eint=e(j,idx,k,m,n)

                   ! Integration
                   sum = sum + wgl(i)*eint

                ENDDO GAUSS_LAGUERRE
    
                ! Scavenging coefficients (s-1)
                gamma(j,k,m,n) = pi*acoef*n0*sum/(4._dp*lam**(3._dp+bcoef+nu))
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    
  END SUBROUTINE scavenging_coefficient


  SUBROUTINE scavenging_num(na,da,ngh,xgh,wgh,ni,nj,nk,nmod,n,d,sigma,gamma,dndt,dmdt, lprint)
    ! Subroutine to numerically calculate aerosol scavenging
    ! using a Gauss-Hermite integration technique.
    !
    ! Input:
    ! ------
    ! na ... Number of aerosol bins
    ! da ... Diameters of aerosols (m)
    ! ngh ... Number of Gauss-Hermite integration points
    ! xgh ... Gauss-Hermite integration points
    ! wgh ... Gauss-Hermite integration weights
    ! ni ... Number of grid points (1. dimension)
    ! nj ... Number of grid points (2. dimension)
    ! nk ... Number of grid points (3. dimension)
    ! nmod ... Number of log-normal aerosol modes
    ! n ... Aerosol number conc. in each mode (m-3)
    ! d ... Count median diameter of each mode (m)
    ! sigma ... Standard deviation of each mode
    ! gamma ... Scavenging coefficients (s-1)
    !
    ! Output:
    ! -------
    ! dndt ... Aerosol number conc. tendency (m-3 s-1)
    ! dmdt ... Aerosol mass density tendency (kg m-3 s-1)
    
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: na
    REAL(KIND=dp), DIMENSION(na), INTENT(IN) :: da
    INTEGER, INTENT(IN) :: ngh
    REAL(KIND=dp), DIMENSION(ngh), INTENT(IN) :: xgh, wgh
    INTEGER, INTENT(in) :: ni, nj, nk, nmod
    REAL(KIND=dp), DIMENSION(ni,nj,nk,nmod), INTENT(IN) :: n, d
    REAL(KIND=dp), DIMENSION(ni,nj,nk,na), INTENT(IN) :: gamma
    REAL(KIND=dp), DIMENSION(nmod), INTENT(IN) :: sigma
    REAL(KIND=dp), DIMENSION(ni,nj,nk,nmod), INTENT(OUT):: dndt, dmdt
    
    ! Local variables
    INTEGER :: i, j, k, imod, jj, kk, idx
    REAL(KIND=dp) :: sumn, summ, dgh, dgh3, coe, omcoe, gamma_int
    REAL(KIND=dp) :: const
  
    LOGICAL :: lprint

    if (lprint) then
       
       do i = 1, na
          print*, 'bin=', i, 'da=', da(i)
       end do

    end if
    
    DO imod=1,nmod
       const=sqrt2*LOG(sigma(imod))
       DO k=1,nk
          DO j=1,nj
             DO i=1,ni
                sumn=0._dp
                summ=0._dp
                DO jj=1,ngh

                   ! Transformed aerosol diameter
                   dgh=d(i,j,k,imod)*EXP(const*xgh(jj))
                   dgh3=dgh*dgh*dgh

                   ! Determine scavenging coef. from lookup table
                   CALL locate(da,na,dgh,idx)

!                   ! Enforce bounds of lookup table
!                   idx=MAX(idx,1)
!                   idx=MIN(idx,na-1)
!
!                   ! Linearly interpolate scavenging coef. between two adjacent
!                   ! values in lookup table to improve accuracy
!                   coe=(dgh-da(idx))/(da(idx+1)-da(idx))
!                   omcoe=1._dp-coe
!                   gamma_int=coe*gamma(i,j,k,idx)+omcoe*gamma(i,j,k,idx+1)

                   gamma_int = gamma(i,j,k,idx)
                  
                   sumn=sumn+wgh(jj)*gamma_int
                   summ=summ+wgh(jj)*gamma_int*dgh3
                   if (lprint.and.i.eq.1.and.j.eq.1.and.imod.eq.1.and.k.gt.185.and.k.lt.190) then
                      print*, 'jj=', jj, 'gamma_int=', gamma_int, 'wgh=', wgh, 'dgh= ', dgh, 'idx=', idx
                   end if
                ENDDO

                
                if (lprint.and.i.eq.1.and.j.eq.1.and.imod.eq.1.and.k.gt.185.and.k.lt.190) then
                   print*, k, 'sum_gam=', sumn*sqrtpi_r
                endif

                

                ! Number conc. tendency (1/m3)
                dndt(i,j,k,imod)=-n(i,j,k,imod)*sqrtpi_r*sumn
                ! Mass conc. tendency (kg/m3/s)
                dmdt(i,j,k,imod)=-n(i,j,k,imod)*sqrtpi*rho_p*six_r*summ
             ENDDO
          ENDDO
       ENDDO
    ENDDO
    
  END SUBROUTINE scavenging_num

  
  SUBROUTINE init_gauss_laguerre(ngl,nu,bcoef,xgl,wgl)
    ! Subroutine to initialize Gauss-Laguerre abscissa and weights
    ! with ngl number of points.
    !
    ! Input:
    ! ------
    ! nu ... Shape parameter of size distribution
    ! bcoef ... b coefficient of terminal fall velocity power law (u=a*D^b)
    !
    IMPLICIT NONE

    INTEGER, intent(in) :: ngl
    REAL(KIND=dp), intent(in) :: nu, bcoef
    REAL(KIND=dp), dimension(ngl),intent(out) :: xgl, wgl

    ! Local variables
    REAL(KIND=dp) :: alf
    
    
    ! Initialize Gauss-Laguerre abscissa and weights
    alf=2._dp+bcoef+nu
    
    ! Calculate Gauss-Laguerre abscissa and weights
    xgl=0._dp
    wgl=0._dp
    CALL gaulag(xgl,wgl,ngl,alf)

  END SUBROUTINE init_gauss_laguerre


  SUBROUTINE init_gauss_hermite(ngh,xgh,wgh)
    ! Subroutine to initialize the Gauss-Hermite abscissa and weights
    ! with ngh number of points
    !
    ! Input:
    ! ------
    ! ngh ... Number of points
    !
    ! Output:
    ! -------
    ! xgh ... Gauss-Hermite abscissa values
    ! wgh ... Gauss-Hermite weights 
    !
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: ngh
    REAL(KIND=dp), DIMENSION(ngh), INTENT(out) :: xgh, wgh

    CALL gauher(xgh,wgh,ngh)

  END SUBROUTINE init_gauss_hermite


  FUNCTION gammln(XX)
    
    IMPLICIT NONE
    
    REAL(KIND=dp) :: GAMMLN
    REAL(KIND=dp) :: COF(6),STP,HALF,ONE,FPF,X,TMP,SER
    REAL(KIND=dp), INTENT(IN) :: xx
    INTEGER :: j

    DATA COF,STP/76.18009173E0_dp,-86.50532033E0_dp,24.01409822E0_dp, & 
         -1.231739516E0_dp,.120858003E-2_dp,-.536382E-5_dp,2.50662827465E0_dp/
    DATA HALF,ONE,FPF/0.5E0_dp,1.0E0_dp,5.5E0_dp/
    X=XX-ONE
    TMP=X+FPF
    TMP=(X+HALF)*LOG(TMP)-TMP
    SER=ONE
    DO J=1,6
       X=X+ONE
       SER=SER+COF(J)/X
    ENDDO
    GAMMLN=TMP+LOG(STP*SER)
    
  END FUNCTION gammln


  SUBROUTINE gaulag(x,w,n,alf)

    IMPLICIT NONE

    INTEGER, INTENT(in) :: n
    REAL(KIND=dp), INTENT(in) :: alf
    REAL(KIND=dp), DIMENSION(n), INTENT(out) :: w, x
    
    INTEGER, PARAMETER :: maxit=10
    REAL(KIND=dp), PARAMETER :: eps=3.E-14_dp
    INTEGER :: i, its, j
    REAL(KIND=dp) :: ai
    REAL(KIND=dp) :: p1, p2, p3, pp, z, z1
    
    
    DO i = 1, n
       IF (i.EQ.1) THEN
          z = (1._dp+alf)*(3._dp+.92_dp*alf)/(1._dp+2.4_dp*n+1.8_dp*alf)
       ELSEIF (i.EQ.2) THEN
          z = z+(15._dp+6.25_dp*alf)/(1._dp+.9_dp*alf+2.5_dp*n)
       ELSE
          ai=i-2
          z = z+((1._dp+2.55_dp*ai)/(1.9_dp*ai)+1.26_dp*ai*alf/(1._dp+3.5_dp*ai))* &
               (z-x(i-2))/(1._dp+.3_dp*alf)
       ENDIF
       DO its = 1, maxit
          p1=1.E0_dp
          p2=0.E0_dp
          DO j = 1, n
             p3=p2
             p2=p1
             p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j
          ENDDO
          pp=(n*p1-(n+alf)*p2)/z
          z1=z
          z=z1-p1/pp
          IF (ABS(z-z1) .LE. eps) GOTO 1
       ENDDO
       STOP "too many iterations in gaulag"
1      x(i)=z
       w(i)=-EXP(gammln(alf+n)-gammln(REAL(n,dp)))/(pp*n*p2)
    ENDDO
    
  END SUBROUTINE gaulag


  SUBROUTINE gauher(X,W,N)
    
    IMPLICIT NONE
    
    INTEGER, INTENT(IN) :: n
    REAL(KIND=dp), DIMENSION(n), INTENT(OUT) :: x, w
    REAL(KIND=dp), PARAMETER :: eps=3.E-14_dp
    REAL(KIND=dp), PARAMETER :: pim4=0.7511255444649425E0_dp
    INTEGER, PARAMETER :: maxit=10

    INTEGER :: i, its, j, m
    REAL(KIND=dp) :: p1, p2, p3, pp, z, z1


    m=(n+1)/2
    DO i=1,m
       IF ( i .EQ. 1 ) THEN
          z=SQRT(REAL(2*n+1,dp))-1.85575_dp*(2*n+1)**(-.16667_dp)
       ELSE IF ( i .EQ. 2 ) THEN
          z=z-1.14_dp*n**.426_dp/z
       ELSE IF ( i .EQ. 3 ) THEN
          z=1.86_dp*z-.86_dp*x(1)
       ELSE IF ( i .EQ. 4 ) THEN
          z=1.91_dp*z-.91_dp*x(2)
       ELSE
          z=2._dp*z-x(i-2)
       ENDIF
       DO its=1,maxit
          p1=pim4
          p2=0._dp
          DO j=1,n
             p3=p2
             p2=p1
             p1=z*SQRT(2._dp/j)*p2-SQRT(REAL(j-1,dp)/REAL(j,dp))*p3
          ENDDO
          pp=SQRT(2._dp*n)*p2
          z1=z
          z=z1-p1/pp
          IF (ABS(z-z1) .LE. eps) GOTO 1
       ENDDO
       STOP 'to many iterations in gauher'
1      x(i)=z
       x(n+1-i)=-z
       w(i)=2._dp/(pp*pp)
       w(n+1-i)=w(i)
    ENDDO

  END SUBROUTINE gauher

  
  SUBROUTINE locate(XX,N,X,J)

    IMPLICIT NONE
    
    INTEGER, INTENT(IN) :: N
    INTEGER, INTENT(OUT) :: J
    REAL(KIND=dp), INTENT(IN) :: X
    REAL(KIND=dp), DIMENSION(N) :: XX

    INTEGER :: JL, JU, JM
        
!A.M.>>
    !JL=0
    !JU=N+1
    JL=1
    JU=N
!A.M.<<
    DO
       IF(JU-JL.GT.1)THEN
          JM=(JU+JL)/2
          IF((XX(N).GT.XX(1)).EQV.(X.GT.XX(JM)))THEN
             JL=JM
          ELSE
             JU=JM
          ENDIF
          CYCLE
       ENDIF
       J=JL
!A.M.>> 
       ! Pick index of element that minimizes the difference 
       ! between actual and true value
       IF (ABS(X-XX(JL)) .GT. ABS(X-XX(JU))) THEN
          J=JU
       ENDIF
!A.M.<<
       EXIT
    ENDDO
      
    END SUBROUTINE locate


    FUNCTION moment_gamma(n0,lambda,nu,k)
      ! Function to calculate the k-th moment of 
      ! a gamma distribution with intercept parameter n0, 
      ! slope parameter lambda and shape parameter nu
      IMPLICIT NONE
      
      REAL(KIND=dp) :: moment_gamma
      REAL(KIND=dp), INTENT(IN) :: n0, lambda, nu, k
      REAL(KIND=dp) :: kk
      
      kk=k+nu+1._dp
      moment_gamma = n0*EXP(gammln(kk))/lambda**kk

    END FUNCTION moment_gamma
 
    FUNCTION moment_lognorm(qna,cmd,sigma,k)
      IMPLICIT NONE
      
      REAL(KIND=dp) :: moment_lognorm
      REAL(KIND=dp), INTENT(IN) :: cmd, qna, sigma, k

      moment_lognorm = qna*(cmd**k)*EXP(k*k*((LOG(sigma))**2._dp)*0.5_dp) 
      !LOG(sigma) should be an external const

    END FUNCTION moment_lognorm
    
    
    
END MODULE src_scavenging
