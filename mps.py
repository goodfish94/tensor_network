# mps class, let d_down = 1 for mpo

import numpy as np
import os
import pandas as pd

from mpo import mpo
from parameter import parameter

class mps(mpo):
# mpo
#
#                  |                |                  |                             |
#   -- lam[0] -- T[0] -- lam[1] -- T[1] -- lam[2] -- T[2] --....... --lam[L-1] -- T[L-1] -- lam[L]
#                  |                |                  |                             |
#
#


    def __init__(self, para : parameter, L, if_bath_sign = True):

        self.L = L  # number of site include imp

        self.d_up = int(para.d) * np.ones(int(L), dtype=np.int32)
        self.d_down = int(1) * np.ones(int(L), dtype=np.int32)  #down = 1

        self.T = {}
        # dimension of T is [Dleft][d_up][d_down][Dright]

        self.D = np.ones(self.L + 1, dtype=np.int)
        # bond dimension for each "lam"
        self.log_norm = 0
        # log Normalization factor overrall

        self.center = 0

        self.id_density_matrix()

        self.tol_svd = para.tol_svd;  # smallest svd value to keep
        self.Dmax = para.Dmax  # maximum bond dimension
        self.svd_full = False  # if always perform full svd
        self.current_D = 1  # current maximum bond dimension

        self.random_init(1)


        self.switch_gate = np.zeros((4,4), dtype=np.complex)
        self.switch_gate[0, 0] = -1.0
        self.switch_gate[3, 3] = 1.0
        self.switch_gate[1, 2] = 1.0
        self.switch_gate[2, 1] = 1.0
        self.switch_gate = np.reshape( self.switch_gate, (2,2,2,2)) # [i_up, j_up, i_down, j_down]


        self.switch_gate = np.resize(self.switch_gate, (2,2,2,2))  # resize to matrix

        self.if_find_gnd = False # if find and save gnd

        self.if_bath_sign = if_bath_sign # if d = sig^- Prod (-sig^z)





    def update_norm_and_D(self, D, norm):
        # update parameter when perform svd for the imp outside this classes
        self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension
        if( norm > 1e-20):
            self.log_norm = self.log_norm + np.log(norm)  # update normalization factor


    def random_init(self,D):
        """
        random initilization with bond dimension D
        :return:
        """

        self.log_norm  = 0
        self.current_D = D
        self.D = D * np.ones(self.L + 1, dtype=np.int32)
        for i in range(0, self.L):
            self.T[i] = np.random.rand( D, self.d_up[i], self.d_down[i], D )

        d_ = self.d_up[0]
        gate = np.eye( d_ * d_ )
        gate = np.reshape(gate, (d_, d_, d_, d_))
        for i in range(self.L-2, 0, -1):
            self.one_gate_projection_two_site(gate, (i,i+1), 'L') # make random mps canonical





    def trace_with_n(self,i):
        """
        take trace after multiplying n_d_imp
        <phi| n_d |phi> / <phi| phi>
        :return:
        """

        if( not i == self.center ):
            print(i,"-th site is not the center, trace with nd")

        rho_red = np.trace( np.tensordot( self.T[i], np.transpose( np.conj(self.T[i]), (3,2,1,0) ), axes=1), axis1=0, axis2=5)
        # dimension = [dout, din, din, dout]

        return rho_red[1,:,:,1]




    def hopping_energy(self, i, ep, v):
        """
        calculate  < ep c_{i+1}^dag c_{i+1} + v (c_i^dag c_{i+1} +h.c.) >

        :param i:
        :return:
        """

        if (not i == self.center):
            print(i, "-th site is not the center, trace with c_i^\dag c_{i+1}")

        Dout = self.d_up[i] * self.d_up[i + 1]
        rho_red = np.tensordot(self.T[i], self.T[i + 1], axes=1)
        rho_red = np.reshape(rho_red, (self.D[i], Dout, self.D[i + 2])) # shape = (DL, d^2, DR)
        rho_red = np.reshape( np.transpose(rho_red, (0,2,1) ), (self.D[i] * self.D[i+2], Dout) ) # shape = DLDR, d^2
        rho_red = np.transpose(np.conj(rho_red)) @ rho_red # shape = [d^2, d^2] = [Dout, Din] 

        op_ = np.zeros((Dout, Dout), dtype=np.complex)
        op_[1, 2] = v
        op_[2, 1] = v
        op_[0, 0] = ep
        op_[2, 2] = ep


        return np.trace(rho_red @ op_)



    def gnd_energy(self, chain_ham):
        """
        calculate gnd energy
        :return:
        """

        if( self.center != 0):
            raise ValueError("Chain gnd energy error start")

        gnd_en = 0.0

        ep = chain_ham.ep
        v = chain_ham.v
        ed = chain_ham.ed

        nd = self.trace_with_n(0)
        gnd_en += nd * ed
        for i in range(0, self.L - 1):
            gnd_en += self.hopping_energy(i, ep[i], v[i])
            self.switch_gate_right()  # move d  right
        for i in range(self.L - 1, 0, -1):
            self.switch_gate_left()  # move d back to the 1st

        if (self.center != 0):
            raise ValueError("Chain gnd energy error end")

        return gnd_en




    def switch_gate_right(self):
        """
        fermion switch,
        switch index i with i+1

        :return:
        """

        if( self.center == self.L - 1):
            return

        self.one_gate_projection_two_site(self.switch_gate, (self.center, self.center+1), 'R')

    def switch_gate_left(self):
        """
        fermion switch,
        switch index i with i-1

        :return:
        """

        if( self.center == 0):
            return

        self.one_gate_projection_two_site(self.switch_gate, (self.center-1, self.center), 'L')



    def save_gnd(self):
        """
        find and save gnd
        :return:
        """
        self.if_find_gnd = True
        self.T_gnd = {}
        for i in self.T:
            self.T_gnd[i] = self.T[i].copy()

        self.D_gnd = self.D.copy()
        self.log_norm_gnd = self.log_norm # log norm for gnd
        self.log_norm = 0.0 # reset log norm

    def load_gnd(self):
        """
       load saved gnd
        :return:
        """
        if( self.if_find_gnd == False ):
            raise ValueError("Not gnd to load")
        self.T = {}
        for i in self.T_gnd:
            self.T[i] = self.T_gnd[i].copy()

        self.D = self.D_gnd.copy()
        self.log_norm =0.0 #reset log norm


    def d_imp(self):
        """
        \sigma_1^+ \Pi (-\sigma_{bath}^z)
        :return:
        """
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )
        self.T[0] = np.tensordot( np.asarray( [[0.0, 0.0],[1.0, 0.0]], dtype = np.complex), self.T[0] , axes=1)
        self.T[0] = np.transpose( self.T[0], (1,0,2,3) )

        for i in range(1, self.L ):
            if( self.if_bath_sign ):
                self.T[i][:, 0, :, :] = self.T[i][:,0,:,:] * ( -1.0 )
                # -sigma^z





    def d_imp_dag(self):
        """
        \sigma_1^+ \Pi (-\sigma_{bath}^z)
        :return:
        """

        T = np.transpose( self.T[0], (1,0,2,3) )
        T = np.tensordot( np.asarray( [[0.0, 1.0],[0.0, 0.0]], dtype = np.complex), T , axes=1)
        self.T[0] = np.transpose( T, (1,0,2,3) )

        for i in range(1, self.L ):
            if (self.if_bath_sign):
                self.T[i][:, 0, :, :] = self.T[i][:,0,:,:] * ( -1.0 )
                # -sigma^z



    def string_sign(self):
        """
         multiply -sig_z for all site including imp
        :return:
        """
        for i in range(0, self.L):
            self.T[i][:, 0, :, :] = self.T[i][:, 0, :, :] * (-1.0)






    def trace_with_d(self):
        """
        d = sig^- \Pi (-sig_z)
        Take trace with    < T_gnd | d | T >
        return vector of dimension (D, D_gnd)
        :return:
        """

        v = np.reshape( np.asarray([1], dtype=np.complex), (1,1) )
        for i in range(self.L-1, -1, -1):
            T = self.T[i].copy()            #[DL, d,1, DR]
            T_gnd = self.T_gnd[i].copy()    #[DLgnd, d,1, DRgnd]
            T_gnd = np.conj(T_gnd)

            T = np.tensordot(T, v, axes=1) # shape = [DL,d,1,DRgnd]
            T = np.transpose( np.squeeze(T, axis=2), (0,2,1) ) # shape = [Dleft, Dright_gnd, d]
            if( i == 0): # multiply by sig^-
                T = np.tensordot(T,  np.asarray([[0.0,1.0],[0.0,0.0]]), axes=1) # need to take transpose of sig^- since we are actually doing T[i_phy]sig^-_{j_phy,i_phi}
            else: # multiply by -sig^z
                if( self.if_bath_sign ):
                    T[:,:,0] = - T[:,:,0]

            T_gnd = np.transpose( np.squeeze(T_gnd, axis=2), (2, 1, 0) )  # [Dright_gnd, d, Dleft_gnd]
            v = np.tensordot(T, T_gnd, axes=2 ) # shape = [Dleft, Dleft_gnd]




        return [np.reshape(v, -1), self.log_norm]


    def trace_with_d_dag(self):
        """
        d dag = sig^+ \Pi (-sig_z)
        Take trace with    < T_gnd | ddag | T >
        return vector of dimension (D, D_gnd)
        :return:
        """

        v = np.reshape(np.asarray([1], dtype=np.complex), (1, 1))
        for i in range(self.L - 1, -1, -1):
            T = self.T[i].copy()  # [DL, d,1, DR]
            T_gnd = self.T_gnd[i].copy()  # [DLgnd, d,1, DRgnd]
            T_gnd = np.conj(T_gnd)

            T = np.tensordot(T, v, axes=1)  # shape = [DL,d,1,DRgnd]
            T = np.transpose(np.squeeze(T, axis=2), (0, 2, 1))  # shape = [Dleft, Dright_gnd, d]
            if (i == 0):  # multiply by sig^-
                T = np.tensordot(T, np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                                 axes=1)  # need to take transpose of sig^+ since we are actually doing T[i_phy]sig^+_{j_phy,i_phi}
            else:  # multiply by -sig^z
                if (self.if_bath_sign):
                    T[:, :, 0] = - T[:, :, 0]

            T_gnd = np.transpose(np.squeeze(T_gnd, axis=2), (2, 1, 0))  # [Dright_gnd, d, Dleft_gnd]
            v = np.tensordot(T, T_gnd, axes=2)  # shape = [Dleft, Dleft_gnd]



        return [np.reshape(v, -1), self.log_norm]


    def trace_with_string_sign(self):
        """
        S=  sig^+ \Pi (-sig_z)
        Take trace with    < T_gnd | S | T >
        return vector of dimension (D, Dgnd)
        :return:
        """

        v = np.reshape(np.asarray([1], dtype=np.complex), (1, 1))
        for i in range(self.L - 1, -1, -1):
            T = self.T[i].copy()  # [DL, d,1, DR]
            T_gnd = self.T_gnd[i].copy()  # [DLgnd, d,1, DRgnd]
            T_gnd = np.conj(T_gnd)

            T = np.tensordot(T, v, axes=1)  # shape = [DL,d,1,DRgnd]
            T = np.transpose(np.squeeze(T, axis=2), (0, 2, 1))  # shape = [Dleft, Dright_gnd, d]
            T[:, :, 0] = - T[:, :, 0]

            T_gnd = np.transpose(np.squeeze(T_gnd, axis=2), (2, 1, 0))  # [Dright_gnd, d, Dleft_gnd]
            v = np.tensordot(T, T_gnd, axes=2)  # shape = [Dleft, Dleft_gnd]




        return [np.reshape(v,-1), self.log_norm]




    def trace_(self):
        """
        <phi_gnd | phi>, return [D,Dgnd]
        :return:
        """

        v = np.reshape(np.asarray([1], dtype=np.complex), (1, 1))
        for i in range(self.L - 1, -1, -1):
            T = self.T[i].copy()  # [DL, d,1, DR]
            T_gnd = self.T_gnd[i].copy()  # [DLgnd, d,1, DRgnd]
            T_gnd = np.conj(T_gnd)

            T = np.tensordot(T, v, axes=1)  # shape = [DL,d,1,DRgnd]
            T = np.transpose(np.squeeze(T, axis=2), (0, 2, 1))  # shape = [Dleft, Dright_gnd, d]

            T_gnd = np.transpose(np.squeeze(T_gnd, axis=2), (2, 1, 0))  # [Dright_gnd, d, Dleft_gnd]
            v = np.tensordot(T, T_gnd, axes=2)  # shape = [Dleft, Dleft_gnd]



        return [np.reshape(v, -1), self.log_norm]






    def save(self, folder):
        """
        save current T, and D and log norm
        :param folder:
        :return:
        """

        if( not os.path.isdir(folder)):
            os.mkdir(folder)

        filename = os.path.join(folder, "para.csv")
        para = {'log_norm':[self.log_norm], 'center':[self.center]}
        para = pd.DataFrame(para, columns=['log_norm','center'])
        para.to_csv(filename, index=False)

        filename = os.path.join(folder, "D.npy")
        np.save(filename, self.D)

        for i in self.T:
            filename = os.path.join(folder, "T_"+str(i)+".npy")
            np.save(filename, self.T[i])


    def load(self,folder):

        filename = os.path.join(folder, "para.csv")
        para = pd.read_csv(filename)
        self.log_norm = para.log_norm[0]
        self.center = para.center[0]


        filename = os.path.join(folder, "D.npy")
        self.D = np.load(filename)

        for i in self.T:
            filename = os.path.join(folder, "T_" + str(i) + ".npy")
            self.T[i] = np.load(filename)

