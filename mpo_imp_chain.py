# store density matrix as an mpo

import numpy as np
from parameter import parameter

from scipy.sparse.linalg import svds
from numpy.linalg import svd
from mpo import mpo


class mpo_imp(mpo):
# mpo
#
#                  |                |                  |                             |
#   -- lam[0] -- T[0] -- lam[1] -- T[1] -- lam[2] -- T[2] --....... --lam[L-1] -- T[L-1] -- lam[L]
#                  |                |                  |                             |
#
#

# let the initial chian geometry be (fermion order)
# imp1 - imp2 -... - chian1 - chian2 -...
# fermion sign is defined through sign op_i =  exp( i pi \sum_{j<i} c_j^\dag c_j)

    def __init__(self, para : parameter, L):

        self.L      = L  # number of site include imp
        self.d_up   = int(para.d) * np.ones(int(L), dtype=np.int)
        self.d_down   =  int(para.d) * np.ones(int(L), dtype=np.int) #


        self.T     = {}
        # dimension of T is [Dleft][d_up][d_down][Dright]

        self.D    = np.ones(self.L+1, dtype=np.int)
        #bond dimension for each "lam"
        self.log_norm = 0
        #log Normalization factor overrall

        self.center = 0

        self.id_density_matrix()


        self.tol_svd = para.tol_svd;  # smallest svd value to keep
        self.Dmax = para.Dmax # maximum bond dimension
        self.svd_full = False  # if always perform full svd
        self.current_D = 1  # current maximum bond dimension



    def update_norm_and_D(self, D, norm):
        # update parameter when perform svd for the imp outside this classes
        self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension
        self.log_norm = self.log_norm + np.log(norm)  # update normalization factor



    def d_imp_dag(self):
        """
        \sigma_1^+ \Pi (-\sigma_{bath}^z)
        :return:
        """
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )
        self.T[0] = np.tensordot( np.asarray( [[0.0, 0.0],[1.0, 0.0]], dtype = np.complex), self.T[0] , axes=1)
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )

        for i in range(1, self.L ):
            self.T[i][:, 0, :, :] = self.T[i][:,0,:,:] * ( -1.0 )
            # -sigma^z



    def d_imp(self):
        """
        \sigma_1^+ \Pi (-\sigma_{bath}^z)
        :return:
        """
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )
        self.T[0] = np.tensordot( np.asarray( [[0.0, 0.0],[1.0, 0.0]], dtype = np.complex), self.T[0] , axes=1)
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )

        for i in range(1, self.L ):
            self.T[i][:, 0, :, :] = self.T[i][:,0,:,:] * ( -1.0 )
            # -sigma^z






    def trace_with_nd(self):
        """
        take trace after multiplying n_d_imp
        return vector of dimension (D[0],)
        :return:
        """

        v = np.asarray([1], dtype=np.complex)
        for i in range(self.L-1, 0, -1):
            T = self.T[i]
            T = np.tensordot(T, v, axes=1)
            v = np.trace(T, axis1=1, axis2=2)
        T = self.T[0]
        T = np.tensordot(T, v, axes=1)
        T = np.transpose(T, (0, 2, 1) )
        T = np.tensordot(T, np.asarray([[1.0, 0.0],[0.0,0.0]]), axes = 1 )  # multuply siamg^-
        v = np.trace(T, axis1=1, axis2=2)
        return v



    def trace_with_d(self):
        """
        take trace after multiplying d
        return vector of dimension (D[0],)
        :return:
        """

        v = np.asarray([1], dtype=np.complex)
        for i in range(self.L-1, 0, -1):
            T = self.T[i]
            T = np.tensordot(T, v, axes=1)
            T[:,0,:] = -T[:,0,:]
            v = np.trace(T, axis1=1, axis2=2)
        T = self.T[0]
        T = np.tensordot(T, v, axes=1)
        T = np.transpose(T, (0, 2, 1) )
        T = np.tensordot(T, np.asarray([[0.0, 0.0],[1.0,0.0]]), axes = 1 )  # multuply siamg^-
        v = np.trace(T, axis1=1, axis2=2)
        return v

    def trace(self):
        """
        take trace
        return vector of dimension (D[0],)
        :return:
        """

        v = np.asarray([1], dtype=np.complex)
        for i in range(self.L-1, 0, -1):
            T = self.T[i]
            T = np.tensordot(T, v, axes=1)
            v = np.trace(T, axis1=1, axis2=2)

        T = self.T[0]
        T = np.tensordot(T, v, axes=1)
        v = np.trace(T, axis1=1, axis2=2)
        return v

