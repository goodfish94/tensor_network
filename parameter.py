# input parameters
import numpy as np
from dispersion import generate_dispersion_array

class parameter:
    def __init__(self, D, ed, U, J,E0, cut = 3.0 , N_ = 200):
        """

        :param D: Max bond Dimension
        :param L: Number of site
        """


        # hamiltonian parameter
        self.d    = 2; # dim of local hilbert space
        self.U = U  # dict key : '12', '13','14','23','24','34' # density density interaction
        self.J = J  # spin spin
        self.ed = ed  # list of local energy scale
        self.E0 = E0


        self.num_of_imp = len(self.ed)
        self.model = "Anderson chain"



        # FT parameter
        self.cut = cut   # assume half band width is 1
        self.N_  = N_  # take N_ point in time domain and 2N-1 point in freq
        self.dw  = self.cut/self.N_
        self.w   = ( np.asarray( range(0, 2 * self.N_ -1 )) - (self.N_ - 1 ) ) * self.dw
        self.T   = 2.0 * np.pi / self.cut
        dt       = self.T/self.N_


        self.nw = np.asarray( range(0, 2 * self.N_ -1 )) - (self.N_ - 1 )  # range of nw
        self.nt = np.asarray( range(0, self.N_  ))  # range of nt
        self.ft = np.tensordot( np.reshape( self.nw, (len(self.nw), 1)),  ( np.reshape(self.nt, (1, len(self.nt) )) ), axes=1)
        self.ft = np.exp( 1j * np.pi * 2.0 * self.ft/ self.N_ )  # exp( 1j 2pi nw nt /N ) # Fourier transformation matrix


        # bath parameter
        self.N_bath  = 50  # 2 * N_bath -1  bath in total
        self.dw_bath = self.cut/self.N_bath
        self.w_bath  = ( np.asarray( range(0, 2 * self.N_bath -1 )) - (self.N_bath - 1 ) ) * self.dw_bath
        self.L_total = 2 * self.N_bath   # total number of site for each chain ( bath site + imp site )


        # tebd parameter
        self.Dmax = D                           # max bond dimension
        self.Dmax_imp = D   # max bond dimension for imp bond

        self.i_meas = 1                         # measure every i_meas real time step
        self.t      = dt/self.i_meas            # real time step
        if( dt < 1e-3):
            self.i_meas = 1
            self.t      = dt
        self.Nt     = self.N_ * self.i_meas     # real imte step


        self.tau    = 0.5                       # imaginary time step, init time step
        self.Ntau   = 1000                      # number of projection/imag time evolution

        self.tol_svd = 1e-10                    # smallest svd value to keep
        self.max_en_diff = 1e-5                 # max diff of gnd energy

        self.imp_list = [0]  # list of imp that calculate green's function


        # tight binding parameter
        self.N_bz = 100
        self.delta_V = -20.0
        self.mu = -1.90401066666

        k, dk, hp, hm, self.D, self.shift_band = generate_dispersion_array(self.delta_V, self.N_bz, self.mu )
        # original band are shift and rescaled: ep -> (ep - shift_band)/D. such that ep \in [-1,1]
        rho = np.ones( k.shape[0], dtype=np.float32 )

        self.rho = {}
        self.ep  = {}
        self.de  = {}

        self.ep[0] = hp.copy()
        self.ep[1] = hm.copy()
        self.ep[2] = hp.copy()
        self.ep[3] = hm.copy()

        for i in range(0,4):
            self.rho[i] = rho.copy()
            self.de[i]  = dk.copy()
            # sum over E replaced by sum over k






        # DMFT parameter
        self.delta = 1e-6 # 0+
        self.z     = self.w + 1j * self.delta
        self.mix   = 0.1

















