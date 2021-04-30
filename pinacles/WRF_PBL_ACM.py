from pinacles.wrf_physics import acm_via_cffi
import numpy as np
class ACM():

    def __init__(self, namelist, Grid, Ref, ScalarState, VelocityState, DiagnosticState, TimeSteppingController):

        self._Grid = Grid
        self._Ref = Ref
        self._ScalarSatate = ScalarState
        self._VelocityState = VelocityState
        self._DiagnosticState = DiagnosticState
        self._TimeSteppingController = TimeSteppingController

        self._acm = acm_via_cffi.ACM()

        return

    def update(self):

        print('CALLING ACM UPDATE')
        

        rho0_edge = self._Ref.p0_edge


        U3D = np.asfortranarray(self._VelocityState.get_field('u')[2:-2,2:-2,2:-2])
        V3D = np.asfortranarray(self._VelocityState.get_field('v')[2:-2,2:-2,2:-2])
        PP3D = np.zeros_like(U3D) + self._Ref.p0[np.newaxis, np.newaxis, 2:-2]
        DZ8W = np.zeros_like(U3D) + self._Grid.dx[2]
        TH3D = np.asfortranarray(self._DiagnosticState.get_field('thetav')[2:-2,2:-2,2:-2]) # TODO Check ths
        T3D =  np.asfortranarray(self._DiagnosticState.get_field('T')[2:-2,2:-2,2:-2])

        QV3D = np.asfortranarray(self._ScalarSatate.get_field('qv')[2:-2,2:-2,2:-2])
        QC3D = np.asfortranarray(self._ScalarSatate.get_field('qc')[2:-2,2:-2,2:-2])
        QI3D = np.zeros_like(QV3D)#np.asfortranarray(self._ScalarSatate.get_field('qi'))
        RR3D = np.zeros_like(U3D) + self._Ref.rho0[np.newaxis, np.newaxis, 2:-2]


        UST = np.zeros((U3D.shape[0], U3D.shape[1]), dtype=np.double, order='F')
        HFX = np.zeros_like(UST) + 100
        QFX = np.zeros_like(UST) + 100
        TSK = np.zeros_like(UST) + 310.0
        PSFC = np.zeros_like(UST) + rho0_edge[self._Grid.n_halo[0]]



        XTIME = int(self._TimeSteppingController.time/60.0)

        DTPBL = self._TimeSteppingController.dt

        EP1 = 287.1/461.5
        G = 9.81
        ROVCP = 287.1/1004.0
        RD =  287.1
        CPD = 1004.0





        PBLH = np.zeros_like(UST)
        KPBL2D = np.zeros((U3D.shape[0], U3D.shape[1]), dtype=np.int, order='F')
        EXCH_H = np.zeros_like(U3D)
        REGIME = np.zeros_like(UST)
        GZ10Z0 = np.zeros_like(UST)
        WSPD = np.zeros_like(UST)
        PSIM = np.zeros_like(UST)
        MUT = np.zeros_like(UST) + self._Ref.p0_edge[self._Grid.n_halo[0]-1] - self._Ref.p0_edge[-self._Grid.n_halo[0]]
        RMOL = np.zeros_like(UST)

        RUBLTEN = np.zeros_like(U3D)
        RVBLTEN = np.zeros_like(U3D)
        RTHBLTEN = np.zeros_like(U3D)

        RQVBLTEN = np.zeros_like(U3D)
        RQCBLTEN = np.zeros_like(U3D)
        RQIBLTEN = np.zeros_like(U3D)

        shape = U3D.shape

        ids = 0
        ims = 0
        its = 1
        
        jds = 0
        jms = 0
        jts = 1
        
        kds = 0
        kms = 0 
        kts = 1

        ide = shape[0]
        ime = shape[0]
        ite = U3D.shape[0] - 3
        
        jde = shape[1]
        jme = shape[1]
        jte = U3D.shape[1] - 3
        
        kde = shape[2]
        kme = shape[2]
        kte = U3D.shape[2] - 3

        acm = self._acm

        self._acm._lib_acm.c_acmpbl(XTIME, DTPBL, acm.as_pointer(U3D), acm.as_pointer(V3D),
                acm.as_pointer(PP3D), acm.as_pointer(DZ8W), acm.as_pointer(TH3D), acm.as_pointer(T3D), 
                acm.as_pointer(QV3D), acm.as_pointer(QC3D), acm.as_pointer(QI3D), acm.as_pointer(RR3D), 
                acm.as_pointer(UST),  acm.as_pointer(HFX), acm.as_pointer(QFX), acm.as_pointer(TSK),  
                acm.as_pointer(PSFC), EP1, G,  
                ROVCP, RD, CPD, 
                acm.as_pointer(PBLH), acm.as_pointer_int(KPBL2D), acm.as_pointer(EXCH_H), acm.as_pointer(REGIME),   
                acm.as_pointer(GZ10Z0), acm.as_pointer(WSPD), acm.as_pointer(PSIM), acm.as_pointer(MUT), 
                acm.as_pointer(RMOL),  
                acm.as_pointer(RUBLTEN), acm.as_pointer(RVBLTEN), acm.as_pointer(RTHBLTEN),         
                acm.as_pointer(RQVBLTEN), acm.as_pointer(RQCBLTEN), acm.as_pointer(RQIBLTEN),         
                ids, ide, jds, jde, kds, kde, 
                ims, ime, jms, jme, kms, kme, 
                its, ite, jts, jte, kts, kte)


        print(PBLH)

        return