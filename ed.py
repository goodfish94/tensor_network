# ed solver for imp


import numpy as np
from scipy import linalg
from scipy.sparse.linalg import svds
from numpy.linalg import svd


class ed_solver:
        # fermion order : d_1 c_1 c_2,...,c_L, d_2, c_1,c_2,..., d_3,..., d_4,...
        def __init__(self, para, el, vl):
            self.dtype = np.complex

            self.el = el
            self.vl = vl

            self.L = len(el) # length of bath
            self.L_tot =self.L*4 + 4 # total num of site

            U = para.U
            J = para.J
            ed = para.ed
            E0 = para.E0

            self.U12 = U['12']
            self.U13 = U['13']
            self.U14 = U['14']
            self.U23 = U['23']
            self.U24 = U['24']
            self.U34 = U['34']
            self.J = J
            self.ed = ed
            self.E0 = E0

            self.sig_up = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=self.dtype)
            self.sig_down = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=self.dtype)
            self.sig_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=self.dtype)
            self.n_f = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=self.dtype)
            self.id = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=self.dtype)


        def get_c_dag(self, ind):

            if(ind == 0):
                op = self.sig_up
            else:
                op = -self.sig_z
            for i in range(1,ind):
                op = np.kron(op, -self.sig_z)
            if( ind != 0):
                op = np.kron(op, self.sig_up)
            for i in range(ind+1,self.L_tot ):
                op = np.kron(op, self.id)

            op = np.reshape(op, (np.power(2, self.L_tot), np.power(2, self.L_tot)))

            return op

        def get_c(self, ind):

            if (ind == 0):
                op = self.sig_down
            else:
                op = -self.sig_z
            for i in range(1, ind ):
                op = np.kron(op, -self.sig_z)
            if (ind != 0):
                op = np.kron(op,  self.sig_down)
            for i in range(ind + 1, self.L_tot):
                op = np.kron(op, self.id)

            op = np.reshape(op, (np.power(2, self.L_tot), np.power(2, self.L_tot)))
            return op

        def construct_ham(self):
            ham = 0.0
            for alpha in range(0,4):
                d_dag = self.get_c_dag( (self.L + 1) * alpha )
                d     = self.get_c( (self.L + 1) * alpha)
                for i in range(0,self.L):
                    c_dag = self.get_c_dag( (self.L+1) * alpha + 1 + i )
                    c     = self.get_c((self.L + 1) * alpha + 1 + i)

                    ham += self.el[i] * c_dag @ c + self.vl[i] * (c_dag @ d + d_dag @ c)


            n1 = self.get_c_dag((self.L + 1) * 0) @ self.get_c((self.L + 1) * 0)
            n2 = self.get_c_dag((self.L + 1) * 1) @ self.get_c((self.L + 1) * 1)
            n3 = self.get_c_dag((self.L + 1) * 2) @ self.get_c((self.L + 1) * 2)
            n4 = self.get_c_dag((self.L + 1) * 3) @ self.get_c((self.L + 1) * 3)
            ham += self.ed[0] * n1 + self.ed[1] * n2 + self.ed[2] * n3 + self.ed[3] * n4
            ham += self.U12 * n1 @ n2 + self.U13 * n1 @ n3 + self.U14 * n1 @ n4  + self.U23 * n2 @ n3 + self.U24 * n2 @ n4 + self.U34 * n3 @ n4


            flip  = self.get_c_dag((self.L + 1) * 0) @ self.get_c((self.L + 1) * 1) @ self.get_c_dag((self.L + 1) * 3) @ self.get_c((self.L + 1) * 2)
            flip += self.get_c_dag((self.L + 1) * 1) @ self.get_c((self.L + 1) * 0) @ self.get_c_dag((self.L + 1) * 2) @ self.get_c((self.L + 1) * 3)

            ham += flip * self.J

            self.ham = ham

        def get_eigen_value(self):
            self.eg = np.linalg.eigvals(self.ham)
            self.eg = self.eg + self.E0

        def get_gnd_energy(self):
            return np.min(self.eg)

        def free_energy(self, beta):

            fe = np.log( np.sum( np.exp( - beta * self.eg ) ) )
            fe = fe /( -beta)
            return fe


        def get_eigen_system(self):
            self.eg, self.eg_v = np.linalg.eigh(self.ham)
            self.eg = self.eg + self.E0


        def get_c_t_cdag(self, t_list, beta):
            """
            real-time Green's function of the 1st orbital
            e^{-beta H} c_1 e^{-iHt} c^dag_1
            :param t:
            :param beta:
            :return:
            """

            v_dag_cdag_1     = np.transpose( np.conj( self.eg_v ) ) @ self.get_c_dag((self.L + 1) * 0)
            c_1_v            = self.get_c((self.L + 1) * 0) @ self.eg_v
            # op for orb 1

            exp_minus_beta_H = self.eg_v @ np.diag( np.exp(-beta * self.eg) ) @ np.transpose( np.conj(self.eg_v) )
            Z = np.trace( exp_minus_beta_H)

            gt = []
            for t in t_list:
                re = ( (exp_minus_beta_H @ c_1_v) @ np.diag( np.exp( -1j * t * self.eg ) ) ) @ v_dag_cdag_1
                gt.append( np.trace( re )/Z )

            return gt

        def get_c_t_cdag_gnd(self, t_list, imp):
            """
            real-time Green's function of the 1st orbital
           <gnd| c_imp e^{-iHt} c^dag_imp |gnd>
            :param t:
            :param beta:
            :return:
            """

            gnd_index = np.argmin(self.eg)
            v_gnd = self.eg_v[:, gnd_index]

            d_1_dag = self.get_c_dag( (self.L + 1) * imp)
            d_1     = self.get_c( (self.L + 1) * imp)

            gt = []
            for t in t_list:
                v = d_1_dag @ v_gnd
                v = np.conj(np.transpose(self.eg_v)) @ v
                v = np.diag(np.exp(-1j * t * self.eg)) @ v
                v = self.eg_v @ v
                v = d_1 @ v
                v = np.sum(v_gnd * v)

                gt.append(v)

            return gt



        def get_cdag_t_c_gnd(self, t_list, imp):
            """
            real-time Green's function of the 1st orbital
           <gnd| c_imp^dag e^{-iHt} c_imp |gnd>
            :param t:
            :param beta:
            :return:
            """

            gnd_index = np.argmin(self.eg)
            v_gnd = self.eg_v[:, gnd_index]

            d_1_dag = self.get_c_dag( (self.L + 1) * imp)
            d_1     = self.get_c( (self.L + 1) * imp)

            gt = []
            for t in t_list:
                v = d_1 @ v_gnd
                v = np.conj(np.transpose(self.eg_v)) @ v
                v = np.diag(np.exp(-1j * t * self.eg)) @ v
                v = self.eg_v @ v
                v = d_1_dag @ v
                v = np.sum(v_gnd * v)

                gt.append(v)

            return gt





















