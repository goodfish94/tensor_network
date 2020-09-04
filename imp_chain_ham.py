# construct ham for single imp chain

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from numpy.linalg import svd


class single_chain_ham:
    # ham for single chain,

    def __init__(self, ep, v, ed):
        self.dtype = np.complex
        self.L = len(ep)
        # self.ep_up   = ep['up']     # \epsilon_l, 1d array shape = (,L)
        # self.ep_down = ep['down']   # \epsilon_l, 1d array shape = (,L)
        # self.v_down  = v['up']      # v_l, 1d array shape = (,L)
        # self.v_down  = v['down']    # v_l, 1d array shape = (,L)
        # self.ed_up   = ed['up']     # energy level of impurity
        # self.ed_down = ed['down']
        self.ep = np.array(ep, copy=True)    # \epsilon_l, 1d array shape = (,L)
        self.v  = np.array(v, copy=True)     # v_l, 1d array shape = (,L)
        self.ed = ed                         # energy level of impurity
        self.d = 2      # local dimension = 2
        # define useful operator



        self.sig_up   = np.array( [[0.0, 1.0],[0.0,0.0]], dtype = self.dtype )
        self.sig_down = np.array( [[0.0, 0.0],[1.0,0.0]], dtype = self.dtype )
        self.n_f      = np.array( [[1.0, 0.0],[0.0,0.0]], dtype = self.dtype )
        self.id       = np.array( [[1.0, 0.0],[0.0,1.0]], dtype = self.dtype )

        self.hop      = ( np.kron( self.sig_up, self.sig_down ) + np.kron( self.sig_down, self.sig_up ))  # hopping
        self.n_l      = np.kron( self.n_f, self.id )
        self.n_r      = np.kron( self.id, self.n_f)

        self.switch_gate = np.zeros((2, 2, 2, 2))  # [i_in][j_in][i_out][j_out] , switch two fermion
        for ni in range(0, 2):
            for nj in range(0, 2):
                if (ni == 0 and nj == 0):  # this corresponds to particle number = 1 for both site
                    self.switch_gate[ni][nj][nj][ni] = -1
                else:
                    self.switch_gate[ni][nj][nj][ni] = 1

        self.switch_gate = np.resize(self.switch_gate, (4, 4))  # resize to matrix

    def hyb_with_switch(self, k, dt, pos):
        # generate gate

        # \exp(-dt V_k (c_k^\dag d_0 +h.c.) - dt ep_k c_k^\d ag c_k
        # input : n_{d0} n_{ck}
        # ouput : n_{ck} n_{d0}
        # pos == 'L', imp on the left, 'R' imp on the right (before switch)
        # switch the impo index to the other site after switch
        # don't switch for k = L - 1
        #

        ham_ = self.v[k] * self.hop

        if( pos == 'L'):  ham_ += self.ep[k] * self.n_r
        else:             ham_ += self.ep[k] * self.n_l
        if( k == 0): # 1st  one, include ed, ed=0, include ed in the interaction
            if (pos == 'L'):  ham_ += self.ed * self.n_l
            else:             ham_ += self.ed * self.n_r

        ham_ = np.resize( ham_, ( self.d * self.d, self.d * self.d ) )
        gate = linalg.expm( - ham_ * dt)  # dimension = [i_out * j_out][i_in * j_in]


        if( k != self.L-1): # switch if not the final
            gate = self.switch_gate @ gate
            # gate = gate



        gate = np.resize( gate, ( self.d, self.d, self.d, self.d ) ) # resize back to tensor form
        # shape = [iout][jout][iin][jin]


        return gate


# ------------------------------------------------------------
#               For mps
# ------------------------------------------------------------
class int_ham:
    # ham for interaction term
    # two orb two spin

    # sum_{n<m} U_{nm} n_n n_m + J (d_1^\dag d_2 d_4^\dag d_3 + d_2^\dag d_1 d_3^\dag d_4)

    def __init__(self, para, dt):

        self.tol_svd = para.tol_svd  # smallest svd value to keep
        self.svd_full = True  # if always perform full svd


        self.dtype = np.complex
        U = para.U
        J = para.J
        ed = para.ed
        E0 = para.E0

        self.dt = dt
        self.U12 = U['12']
        self.U13 = U['13']
        self.U14 = U['14']
        self.U23 = U['23']
        self.U24 = U['24']
        self.U34 = U['34']
        self.J   = J
        self.ed = ed
        self.E0 = E0



        self.d = int(2)      # local dimension = 2
        self.current_D= 0      # Max bound dimension
        self.D     = np.ones(5, dtype=int) # D T1 D T2 D T3 D T4 D


        # define useful operator
        self.sig_up   = np.array( [[0.0, 1.0],[0.0,0.0]], dtype = self.dtype )
        self.sig_down = np.array( [[0.0, 0.0],[1.0,0.0]], dtype = self.dtype )
        self.sig_z    = np.array( [[1.0, 0.0],[0.0,-1.0]], dtype = self.dtype )
        self.n_f      = np.array( [[1.0, 0.0],[0.0,0.0]], dtype = self.dtype )
        self.id       = np.array( [[1.0, 0.0],[0.0,1.0]], dtype = self.dtype )

        n_1 = np.reshape( np.kron(np.kron(np.kron(self.n_f, self.id), self.id), self.id), (16 , 16 ) )
        n_2 = np.reshape( np.kron(np.kron(np.kron(self.id, self.n_f), self.id), self.id), (16 , 16 ) )
        n_3 = np.reshape( np.kron(np.kron(np.kron(self.id, self.id), self.n_f), self.id), (16 , 16 ) )
        n_4 = np.reshape( np.kron(np.kron(np.kron(self.id, self.id), self.id), self.n_f), (16 , 16 ) )

        c_1_d = np.reshape(np.kron(np.kron(np.kron(self.sig_up, self.id), self.id), self.id), (16, 16))
        c_2_d = np.reshape(np.kron(np.kron(np.kron(self.id, self.sig_up), self.id), self.id), (16, 16))
        c_3_d = np.reshape(np.kron(np.kron(np.kron(self.id, self.id), self.sig_up), self.id), (16, 16))
        c_4_d = np.reshape(np.kron(np.kron(np.kron(self.id, self.id), self.id), self.sig_up), (16, 16))

        c_1 = np.reshape(np.kron(np.kron(np.kron(self.sig_down, self.id), self.id), self.id), (16, 16))
        c_2 = np.reshape(np.kron(np.kron(np.kron(self.id, self.sig_down), self.id), self.id), (16, 16))
        c_3 = np.reshape(np.kron(np.kron(np.kron(self.id, self.id), self.sig_down), self.id), (16, 16))
        c_4 = np.reshape(np.kron(np.kron(np.kron(self.id, self.id), self.id), self.sig_down), (16, 16))



        Hn  = self.ed[0] * n_1  +  self.ed[1] * n_2  + self.ed[2] * n_3  + self.ed[3] * n_4
        Hn += self.U12 * n_1 @ n_2 + self.U13 * n_1 @ n_3 + self.U14 * n_1 @ n_4 + self.U23 * n_2 @ n_3 + self.U24 * n_2 @ n_4 + self.U34 * n_3 @ n_4

        flip =c_1_d @ c_2 @ c_4_d @ c_3 + c_2_d @ c_1 @ c_3_d @ c_4  # flipping part ddag d ddag d + ddag d ddag d

        self.ham = Hn + self.J * (flip) + self.E0 * np.eye( 16 ) # hamiltonian

        self.gate = linalg.expm( -self.ham  * self.dt )
        self.gate = np.resize(np.transpose(self.gate), (self.d, self.d, self.d, self.d,self.d, self.d, self.d, self.d))
        # resize back to tensor form [orb1 out][orb2 out][orb3 out][orb4 out][orb1 in][orb2 in][orb3 in][orb4 in]
        self.gate_to_mpo()

        self.gate = np.reshape(self.gate, (np.power(self.d,4), np.power(self.d,4))) # to matrix form
        self.eg, self.eg_v = np.linalg.eigh(self.ham) # get eigen value

        self.gnd_en = np.min(self.eg)


        # self.n1 = np.resize(n_1, (self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d))
        # self.n2 = np.resize(n_2, (self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d))
        # self.n3 = np.resize(n_3, (self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d))
        # self.n4 = np.resize(n_4, (self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d))

        self.n1 = n_1
        self.n2 = n_2
        self.n3 = n_3
        self.n4 = n_4


    def update_dt(self, dt):
        self.dt = dt
        self.gate =linalg.expm( -self.ham  * self.dt )
        self.gate = np.resize( self.gate,  (self.d, self.d, self.d, self.d, self.d, self.d, self.d, self.d) )
        self.gate_to_mpo()
        self.gate = np.reshape(self.gate, (np.power(self.d, 4), np.power(self.d, 4)))  # to matrix form
        self.eg, self.eg_v = np.linalg.eigh(self.ham)  # get eigen value

        self.gnd_en = self.eg[0]




    def get_gnd_energy(self):
        return self.gnd_en



    def gate_to_mpo(self):
        # get mpo formula
        # center at the imp 4

        self.T = [] # mpo tensor

        self.rho = np.reshape(self.gate, (self.d, self.d*self.d*self.d, self.d, self.d*self.d*self.d))
        self.rho = np.transpose(self.rho, (0,2, 1,3)) # [dout][din] [dout^3][din^3]
        self.rho = np.reshape(self.rho, (self.d*self.d, np.power(self.d,6)))
        [u, s, vt, D] = self.svd_()

        self.D[1] = D
        u =  np.reshape(u, (1, self.d, self.d, D) )
        self.T.append(u) # imp_1 mpo

        self.rho = np.diag(s)@vt # dimension = [D, dout^3 din^3]
        self.rho = np.reshape(self.rho, (self.D[1], self.d, self.d*self.d, self.d, self.d*self.d)) # [D, dout, dout^2, din, din^2]
        self.rho = np.transpose(self.rho, (0,1,3,2,4))# [D, dout, din, dout^2, din^2]
        self.rho = np.reshape(self.rho, (self.D[1]*self.d*self.d, np.power(self.d,4)))  #[D*dout*din, dout^2 din^2]
        [u, s, vt, D] = self.svd_()

        self.D[2] = D
        u = np.reshape(u, (self.D[1], self.d, self.d, D) )
        self.T.append(u)  # imp_2 mpo

        self.rho = np.diag(s)@vt # dimension [D, dout^2 * din^2]
        self.rho = np.reshape( self.rho, (self.D[2], self.d, self.d, self.d, self.d)) # [D, dout,dout,din,din]
        self.rho = np.transpose( self.rho, (0,1,3,2,4) ) # [D,dout, din, dout, din]
        self.rho = np.reshape(self.rho, (self.D[2] * self.d * self.d, self.d*self.d)) # [D*dout*din, dout*din]
        [u, s, vt, D] = self.svd_()

        self.D[3] = D
        u = np.reshape(u, (self.D[2], self.d, self.d, D) )
        self.T.append(u) # imp_3 mpo

        self.rho = np.diag(s) @ vt # dimension [D, dout * din]
        u = np.reshape(self.rho, (self.D[3], self.d, self.d, 1) )
        self.T.append(u) # impo_4 mpo, un-normlized






    def get_int_mpo(self, i):
        # get mpo between imp i and i+1,  i \in {0,1,2}
        return self.T[i]


    def d_dag_t_d(self, t_list):
        """
        calculate d_1(t) d_dag_1
        :param t:
        :return:
        """
        d_1_dag = np.reshape(np.kron(np.kron(np.kron(self.sig_up, self.id), self.id), self.id), (16, 16))
        d_1 = np.reshape(np.kron(np.kron(np.kron(self.sig_down, self.id), self.id), self.id), (16, 16))

        gt = []
        for t in t_list:
            self.update_dt(t)
            v_gnd = self.eg_v[:,0]

            v = d_1_dag @ v_gnd
            v = np.conj( np.transpose( self.eg_v) ) @ v
            v = np.diag( np.exp(-1j * t * self.eg) ) @ v
            v = self.eg_v @ v
            v = d_1 @ v
            v = np.sum( v_gnd * v )

            gt.append(v)

        return gt








    def svd_(self):
        """
        perform svd on mat rho
        don't normalize S
        :param mat:
        :return: [u,v,vt,D]
        """


        D_mat = min(self.rho.shape)
        D_svd = D_mat
        # keep D_svd eg values for svd
        if (D_svd == D_mat or self.svd_full):
            u, s, vt = svd(self.rho, full_matrices=False)
            # using Full svd
        else:
            u, s, vt = svds(self.rho, k=D_svd, which='LM')
            # trunctated SVD, keep largest Dmax eg vals

        # s shape = (k,)
        # u shape = (Dleft*d*d,k)
        # vt shape = (k, Dright*d*d)
        ind_ = (s > self.tol_svd)

        s = s[ind_]
        u = u[:, ind_]
        vt = vt[ind_, :]

        D = len(s)
        self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension


        return [u,s,vt,D]






