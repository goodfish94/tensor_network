#  perform exp(-tau * h) to the density matrix

from parameter import parameter
from utility import svd_utility
from mps import mps

import os
import pandas as pd
import numpy as np
import time


class TEBD_class:
    # this do tebd for single chain
    def __init__(self, para: parameter):

        self.num_of_imp = para.num_of_imp

        # self.beta_list          = []; # beta list
        # self.free_energy_list   = {}; # free energy , key is beta
        # self.norm_list          = []; # dict of norm, used to construct free energy, key is beta

        self.tol_svd = para.tol_svd  # smallest svd value to keep
        self.Dmax = para.Dmax  # maximum bond dimension
        self.D_imp_max = para.Dmax_imp  # max D of imp bond
        self.current_D = 1
        self.svd_full = False  # if always perform full svd

        self.center_imp = 0 # detnoe which chain is the center

        self.D_imp = np.ones( (self.num_of_imp) + 1, dtype=np.int)
        # Bond dimension of imp
        # D_{imp}[0] -- imp_0 -- D_{imp}[1] -- imp_1 ---... --- imp_n -- D_{imp}[n]

        self.if_print = False




    def init(self, para, chain_mpo = None):
        if( chain_mpo is None ):
            self.chain_mpo = []
            for i in range(0, 4):
                if (i % 2 == 0):
                    if_bath_sign = True
                else:
                    if_bath_sign = False
                mps_ = mps(para, para.L_total, if_bath_sign)
                self.chain_mpo.append(mps_)
        else:
            self.chain_mpo = chain_mpo



    def import_chain_ham(self, chain_ham):
        self.chain_ham = chain_ham

    def import_int_gate(self, int_gate,dt):
        self.int_gate = int_gate
        self.dt = dt



    def time_evolution(self, dt, if_calculate_gnd_en = False):
        """
        perform time evolution
        :param dt:
        :param if_calculate_gnd_en: if calculate gnd energy
        :return:
        """


        self.dt = dt
        self.int_gate.update_dt(self.dt/2.0)
        self.one_step_total(self.dt)
        if( if_calculate_gnd_en ):
            gnd_en = self.gnd_energy()
            return gnd_en
        else:
            self.move_center_from_final_to_first()
            return None




    def cal_static(self):
        """
        filling of current T
        :return:
        """
        n1 = self.trace_with_imp_mat_op( self.int_gate.n1)
        n2 = self.trace_with_imp_mat_op( self.int_gate.n2)
        n3 = self.trace_with_imp_mat_op( self.int_gate.n3)
        n4 = self.trace_with_imp_mat_op( self.int_gate.n4)

        self.n = [n1, n2, n3, n4]
        self.static = {}
        self.static['n'] = self.n



    def save_gnd(self):
        """
        find and save gnd
        :return:
        """
        for mps_ in self.chain_mpo:
            mps_.save_gnd()


        self.D_imp_gnd = self.D_imp.copy()



    def load_gnd(self):
        for mps_ in self.chain_mpo:
            mps_.load_gnd()

        self.D_imp = self.D_imp_gnd.copy()








    def one_step_chain_projection(self, imp_ind, time_step ):
        """
        loop to perform projection on one chain
        one time step

        """

        # filename = "D_"+str(mpo.Dmax) +"_L_"+str(self.L)+".txt"
        # file = open(filename,'w')
        # right sweep
        L = self.chain_ham[imp_ind].L # total length of chain, not including imp

        for i in range(0, L): # i is the bath index, L bath
            dt = time_step/2.0
            if( i == L-1): dt = time_step

            gate = self.chain_ham[imp_ind].hyb_with_switch(i, dt, 'L')  # imp at left before swap
            if( i == L-1):
                self.one_step_projection(self.chain_mpo[imp_ind], (i,i+1), gate, 'L') # make imp to be center
            else:
                self.one_step_projection(self.chain_mpo[imp_ind], (i,i+1), gate, 'R') # make imp to be center


            # perform projection for site i, switch imp and bath index (except for the last bath )
            # move center to i+1



        for i in range(L - 2, -1, -1): # i is the bath index, start from L-2 bath
            dt = time_step / 2.0
            gate = self.chain_ham[imp_ind].hyb_with_switch(i, dt, 'R')  # imp at rhs before swap
            # gate = ham.hyb_with_switch(i,dt, 'L')
            self.one_step_projection(self.chain_mpo[imp_ind], (i, i + 1), gate, 'L')  # make imp to be center


        # print(" Bond dimension = %f, time = %f min\n"%( mpo.current_D, (time_en -time_st)/60.0 ))



    def ham_mpo_on_imp(self ):
        """
        ham mpo acting on imp Tensor
        int_gate is list of tensor with shape = [D_ham_L][din][dout][D_ham_R]
        :param chain_mpo:

        :return:
        """
        if(self.center_imp != 0):
            raise RuntimeError("Center not at 0")


        for imp_ind in range(0, self.num_of_imp):
            D_imp_left, D_imp_right = self.D_imp[imp_ind], self.D_imp[imp_ind + 1]  # bound dimension of imp tensor, #D_imp_left * D_imp_right = shape_rho[0]

            gate = self.int_gate.T[imp_ind] # get the gate tensor
            self.rho = self.chain_mpo[imp_ind].get_T(0)  # get imp tensor, at 0 site
            DL, _, _, Dbath = self.rho.shape
            DhL, _, _, DhR = gate.shape
            dout = self.rho.shape[1]
            din  = self.rho.shape[2]

            self.rho = np.transpose(self.rho, (1, 2, 0, 3))  # [dout][din][DL][Dbath]
            gate = np.transpose(gate, (1, 0, 3, 2))  # [dout][DLham][DRham][din]
            self.rho = np.tensordot(gate, self.rho, axes=1)  # [dout][DLham][DRham][din][DL][Dbath]
            self.rho = np.reshape(self.rho, (dout,DhL,DhR, din, D_imp_left, D_imp_right, Dbath)) # [dout][DLham][DRham][din][DimpL][DimpR][Dbath]
            self.rho = np.transpose(self.rho, (1, 4, 0, 3, 6, 2, 5)) #[DhL][dimpL][dout][din][Dbath][DhR][DimpR]
            self.rho = np.reshape(self.rho, ( DhL*D_imp_left, dout*din*Dbath, DhR*D_imp_right))

            if( imp_ind != 0):
                self.rho = np.tensordot(vt, self.rho, axes=1) # multiply the vt from the svd of the previous imp
                new_D_imp_left = vt.shape[0] # new imp bond
            else:# first imp no need to mulitply vt
                new_D_imp_left = D_imp_left # DhL =1

            self.D_imp[imp_ind] = new_D_imp_left # update left bond

            self.rho = np.reshape(self.rho, (new_D_imp_left*dout*din*Dbath, DhR*D_imp_right))  # accumulate bonds
            [u, s, vt, norm, D] = self.svd_()
            self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension
            if( imp_ind != self.num_of_imp -1 ): # not the final one
                u = np.reshape(u,(new_D_imp_left, dout,din, Dbath,D))
                u = np.transpose(u, (0,4, 1,2,3) ) # [DimpL_new][DimpR_new][d][d][D_bath]
                u = np.reshape(u, (new_D_imp_left * D, dout,din,Dbath))
                self.chain_mpo[imp_ind].save_T(u, 0)   # save the imp Tensor and normalize it
                self.chain_mpo[imp_ind].update_norm_and_D( u.shape[0], norm) # update
                vt = np.diag(s) @ vt # vt acting on the next imp
            else: # final one
                u = u@np.diag(s)
                u = u@vt
                # using normalize tensor
                u = np.reshape(u, (new_D_imp_left, dout, din, Dbath)) # vt have dimension [D,1]
                self.chain_mpo[imp_ind].save_T(u, 0)  # save the imp Tensor and normalize it
                self.chain_mpo[imp_ind].update_norm_and_D(u.shape[0], norm)  # update

        self.center_imp = self.num_of_imp - 1 # center if the final one

    def move_imp_center_left(self):
        # normalize center one steps toward left

        imp_ind = self.center_imp

        D_imp_left, D_imp_right = self.D_imp[imp_ind], self.D_imp[imp_ind + 1]
        # bound dimension of imp tensor, #D_imp_left * D_imp_right = shape_rho[0]
        self.rho = self.chain_mpo[imp_ind].get_T(0)  # dimension = [D_imp_left * D_imp_right, d,d,Dbath]
        Dbath = self.rho.shape[3]
        dout  = self.rho.shape[1]
        din   = self.rho.shape[2]


        # self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right, dout, din, Dbath))
        self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right * dout * din * Dbath))

        [u, s, vt, norm, D] = self.svd_(  )
        self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension


        if (imp_ind != 0):  # not the first one

            vt = np.reshape(vt, (D, D_imp_right, dout, din, Dbath))
            vt = np.reshape(vt, (D * D_imp_right, dout, din, Dbath))
            self.chain_mpo[imp_ind].save_T(vt, 0)
            self.chain_mpo[imp_ind].update_norm_and_D(vt.shape[0], norm)
            u = u @ np.diag(s)

            self.rho = self.chain_mpo[imp_ind - 1].get_T(0) # get previous imp,  we need to multiply it by u

            D_imp_left, D_imp_right = self.D_imp[imp_ind-1], self.D_imp[imp_ind]
            Dbath = self.rho.shape[3]
            d = self.rho.shape[1]

            self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right, dout, din, Dbath))
            self.rho = np.transpose(self.rho, (1, 0, 2, 3, 4))
            self.rho = np.tensordot(np.transpose(u), self.rho, axes=1)  # [D][D_imp_left][d][d][Dbath]

            self.D_imp[imp_ind ] = D  # update right bond

            self.rho = np.transpose(self.rho, (1, 0, 2, 3, 4))
            self.rho = np.reshape(self.rho, (D_imp_left * D, dout, din, Dbath))

            self.chain_mpo[imp_ind - 1].save_T(self.rho, 0)
            self.chain_mpo[imp_ind].update_norm_and_D(self.rho.shape[0],1)

            self.center_imp -= 1


        else:  # 1st imp # normalzie first one
            vt = np.diag(s) @ vt
            vt = u @ vt
            vt = np.reshape(vt, (D_imp_left, D_imp_right, dout, din, Dbath))
            vt = np.reshape(vt, (D_imp_left * D_imp_right, dout, din, Dbath))
            self.chain_mpo[imp_ind].save_T(vt, 0)









    def move_center_from_final_to_first(self):
        """
        move imp center from final site to the 1st site
        :return:
        """
        if (self.center_imp != self.num_of_imp-1):
            raise RuntimeError("Center not at final")

        for imp_ind in range(self.num_of_imp-1, -1, -1 ):
            D_imp_left, D_imp_right = self.D_imp[imp_ind], self.D_imp[imp_ind + 1]
            # bound dimension of imp tensor, #D_imp_left * D_imp_right = shape_rho[0]
            self.rho = self.chain_mpo[imp_ind].get_T(0) # dimension = [D_imp_left * D_imp_right, d,d,Dbath]
            Dbath = self.rho.shape[3]
            dout  = self.rho.shape[1]
            din   = self.rho.shape[2]

            if( imp_ind == self.num_of_imp - 1):# the last impt
                self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right, dout, din, Dbath))
                self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right*dout*din*Dbath))
            else:
                self.rho = np.reshape(self.rho, (D_imp_left, D_imp_right, dout, din, Dbath))
                self.rho = np.transpose(self.rho,(1,0,2,3,4))
                self.rho = np.tensordot(np.transpose(u), self.rho, axes=1) #[D][D_imp_left][d][d][Dbath]
                self.D_imp[imp_ind+1] = D # update right bond
                self.rho = np.transpose(self.rho, (1,0,2,3,4))
                self.rho = np.reshape(self.rho, (D_imp_left, D*dout*din*Dbath))
                D_imp_right = D # new imp right

            [u, s, vt, norm, D] = self.svd_()
            self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension

            if( imp_ind != 0): # not the fist one
                vt = np.reshape(vt, (D, D_imp_right, dout, din, Dbath))
                vt = np.reshape(vt, (D*D_imp_right, dout, din, Dbath))
                self.chain_mpo[imp_ind].save_T(vt, 0)
                self.chain_mpo[imp_ind].update_norm_and_D(vt.shape[0], norm)

                u = u@np.diag(s)
            else: #1st imp
                vt = np.diag(s) @ vt
                vt = u @ vt
                vt = np.reshape(vt, (D_imp_left, D_imp_right, dout, din, Dbath))
                vt = np.reshape(vt, (D_imp_left * D_imp_right, dout, din, Dbath))
                self.chain_mpo[imp_ind].save_T(vt, 0)
                self.chain_mpo[imp_ind].update_norm_and_D(1, norm)

        self.center_imp = 0 # set 1st imp be the center






    def one_step_total(self,  dt):
        """
        # center 1st -> final
        exp(-0.5 dt H_int)
        exp(- dt H_{chain 4} )exp(- dt H_{chain 3} )exp(- dt H_{chain 2} )exp(- dt H_{chain 1} )
        exp(-0.5 dt H_int)
        :param chain_mpo:
        :param chian_ham:
        :param int_ham:
        :param dt:
        :return:
        """
        if( self.center_imp != 0):
            raise RuntimeError("imp position error ", self.center_imp)

        time_st = time.time()
        self.ham_mpo_on_imp()

        # self.print_bond_dimension(chain_mpo)


        time1 = time.time()
        for i in range(self.num_of_imp -1 ,-1, -1):
            self.one_step_chain_projection(i, dt)
            if( i != 0):
                self.move_imp_center_left()

        time2 = time.time()

        self.ham_mpo_on_imp()

        time3 = time.time()

        if( self.if_print ):
            self.print_bond_dimension()
            print("total time =%.2f, time = %.2f, %.2f, %.2f"%( (time.time()-time_st)/60.0, (time1-time_st)/60.0, (time2-time1)/60.0, (time3-time2)/60.0 ) )



    def gnd_energy(self):
        if (self.center_imp != self.num_of_imp - 1 ):
            raise RuntimeError("imp position error ", self.center_imp)

        gnd_en = 0.0
        for i in range(self.num_of_imp-1, -1, -1):
            chain_en = self.chain_mpo[i].gnd_energy(self.chain_ham[i])
            gnd_en  += chain_en
            self.move_imp_center_left()

        gnd_en += self.trace_with_imp_mat_op( self.int_gate.ham)


        return gnd_en







    def one_step_projection(self, mpo, site_index, gate, new_ceter):
        """
        do the projection
        :param mpo:
        :param site_index: (i,j) for two site, works wor j= i+1
        :param gate:
        :param new_center new_cetner = 'L', 'R', detnoe the position of new center
        :return:
        """

        (i,j) = site_index
        mpo.one_gate_projection_two_site(gate, (i,i+1), new_ceter)
        #do the projection, move the center


        return










    def svd_(self):
        """
        perform svd on mat rho
        :param mat:
        :return: [u,v,vt,norm, D]
        """

        return svd_utility(self.rho, self.D_imp_max, self.tol_svd, self.svd_full)

    def act_d_dag(self, imp_ind):
        """
        acting d_1^\dag on chain_mpo
        \sigma_1^+ \Pi (-\sigma_{bath}^z)

        :param chain_mpo:
        :return:
        """

        if (self.center_imp != 0):
            raise ValueError("Center error ddag")

        for i in range(0, imp_ind):
            self.chain_mpo[i].string_sign()
        self.chain_mpo[imp_ind].d_imp_dag()


    def act_d(self, imp_ind):
        """
       acting d_1^\dag on chain_mpo
       \sigma_1^+ \Pi (-\sigma_{bath}^z)

       :param chain_mpo:
       :return:
       """

        if (self.center_imp != 0):
            raise ValueError("Center error d")

        for i in range(0, imp_ind):
            self.chain_mpo[i].string_sign()
        self.chain_mpo[imp_ind].d_imp()


    def trace_with_d(self, imp_ind):
        """
        Trace with gnd
        :param chain_mpo:
        :return:
        """

        if (self.center_imp != 0):
            raise ValueError("Center error d")
        log_norm_sum = 0.0

        if (imp_ind == 0):
            [v,log_norm] = self.chain_mpo[0].trace_with_d()
            log_norm_sum += log_norm
            for i in range(1, self.num_of_imp):
                DL, DR, DL_gnd, DR_gnd = self.D_imp[i], self.D_imp[i + 1], self.D_imp_gnd[i], self.D_imp_gnd[i + 1]

                [u, log_norm] = self.chain_mpo[i].trace_()
                log_norm_sum += log_norm
                u = np.reshape(u, (DL, DR, DL_gnd, DR_gnd))
                u = np.transpose(u, (0, 2, 1, 3))  # DL,DL_gnd, DR, DR_gnd
                u = np.reshape(u, (DL * DL_gnd, DR * DR_gnd))
                v = v @ u


        else:
            [v, log_norm] = self.chain_mpo[0].trace_with_string_sign()
            log_norm_sum += log_norm
            for i in range(1, self.num_of_imp):
                DL, DR, DL_gnd, DR_gnd = self.D_imp[i], self.D_imp[i + 1], self.D_imp_gnd[i], self.D_imp_gnd[i + 1]

                if (i < imp_ind):
                    [u, log_norm] = self.chain_mpo[i].trace_with_string_sign()
                elif (i == imp_ind):
                    [u, log_norm] = self.chain_mpo[i].trace_with_d()
                else:
                    [u, log_norm] = self.chain_mpo[i].trace_()
                log_norm_sum += log_norm

                u = np.reshape(u, (DL, DR, DL_gnd, DR_gnd))
                u = np.transpose(u, (0, 2, 1, 3))  # DL,DL_gnd, DR, DR_gnd
                u = np.reshape(u, (DL * DL_gnd, DR * DR_gnd))
                v = v @ u

        return np.squeeze(v) * np.exp(log_norm_sum)

    def trace_with_d_dag(self, imp_ind):
        """
        Trace with gnd
        :param chain_mpo:
        :return:
        """

        if (self.center_imp != 0):
            raise ValueError("Center error d")

        log_norm_sum = 0.0
        if (imp_ind == 0):
            [v,log_norm] = self.chain_mpo[0].trace_with_d_dag()
            log_norm_sum += log_norm
            for i in range(1, self.num_of_imp):
                DL, DR, DL_gnd, DR_gnd = self.D_imp[i], self.D_imp[i + 1], self.D_imp_gnd[i], self.D_imp_gnd[i + 1]

                [u, log_norm]= self.chain_mpo[i].trace_()
                log_norm_sum += log_norm
                u = np.reshape(u, (DL, DR, DL_gnd, DR_gnd))
                u = np.transpose(u, (0, 2, 1, 3))  # DL,DL_gnd, DR, DR_gnd
                u = np.reshape(u, (DL * DL_gnd, DR * DR_gnd))
                v = v @ u


        else:
            [v, log_norm] = self.chain_mpo[0].trace_with_string_sign()
            log_norm_sum += log_norm
            for i in range(1, imp_ind - 1):
                for i in range(1, self.num_of_imp):
                    DL, DR, DL_gnd, DR_gnd = self.D_imp[i], self.D_imp[i + 1], self.D_imp_gnd[i], self.D_imp_gnd[i + 1]

                    if (i < imp_ind):
                        [u, log_norm] = self.chain_mpo[i].trace_with_string_sign()
                    elif (i == imp_ind):
                        [u, log_norm] = self.chain_mpo[i].trace_with_d_dag()
                    else:
                        [u, log_norm] = self.chain_mpo[i].trace_()
                    log_norm_sum += log_norm
                    u = np.reshape(u, (DL, DR, DL_gnd, DR_gnd))
                    u = np.transpose(u, (0, 2, 1, 3))  # DL,DL_gnd, DR, DR_gnd
                    u = np.reshape(u, (DL * DL_gnd, DR * DR_gnd))
                    v = v @ u

        return np.squeeze( v ) *np.exp(log_norm_sum)


    def trace(self):
        """
        Trace with gnd
        :param chain_mpo:
        :return:
        """

        if (self.center_imp != 0):
            raise ValueError("Center error d")


        log_norm_sum = 0.0
        [v, log_norm] = self.chain_mpo[0].trace_()
        log_norm_sum += log_norm

        for i in range(1, self.num_of_imp):
            DL, DR, DL_gnd, DR_gnd = self.D_imp[i], self.D_imp[i + 1], self.D_imp_gnd[i], self.D_imp_gnd[i + 1]

            [u,log_norm] = self.chain_mpo[i].trace_()
            log_norm_sum += log_norm
            u = np.reshape(u, (DL, DR, DL_gnd, DR_gnd))
            u = np.transpose(u, (0, 2, 1, 3))  # DL,DL_gnd, DR, DR_gnd
            u = np.reshape(u, (DL * DL_gnd, DR * DR_gnd))
            v = v @ u

        return np.squeeze( v ) *np.exo(log_norm_sum)




    def trace_with_imp_mat_op(self, ham):
        """
        trace with  matrix op
        :param chain_mpo:
        :param ham:
        :return:
        """

        u = np.ones( (1,1), dtype=np.complex)

        for i in range(0,self.num_of_imp):
            dout = self.chain_mpo[i].d_up[0]
            T=  np.tensordot( self.chain_mpo[i].T[0], np.transpose( np.conj( self.chain_mpo[i].T[0]), (3,2,1,0) ), axes=1 )
            T = np.reshape( T,  (self.D_imp[i],self.D_imp[i+1], dout, dout, self.D_imp[i], self.D_imp[i+1] ) )
            T = np.transpose( T, (0,4, 2,3,1,5) ) # [DLup, DLdn, d,d, DRup, DRdn]
            u = np.tensordot(u, T, axes=2)        # [ (dup,ddn)*n ,dup, ddn, DRup, DRdn]

        u = np.transpose( u, (0,2,4,6, 1,3,5,7, 8,9) )
        # dup1, dup2, dup3, dup4, ddn1, ddn2, ddn3, ddn4

        D_up = np.power(2,4)
        D_dn = np.power(2,4)
        u = np.reshape(u, (D_up, D_dn) ) # reduced density of state
        u = ham @ u

        return np.trace( u )






    def print_bond_dimension(self):
        print(" Bond dimension", end=" ")
        for i in range(0, self.num_of_imp):
            print("  %f  " % (self.chain_mpo[i].current_D), end="  ")
        print(self.current_D)


    def save(self, folder):
        if (not os.path.isdir(folder)):
            os.mkdir(folder)

        filename = os.path.join(folder, "para.csv")
        para = {'center_imp': [self.center_imp]}
        para = pd.DataFrame(para, columns=['center_imp'])
        para.to_csv(filename, index=False)

        filename = os.path.join(folder, "D_imp.npy")
        np.save(filename, self.D_imp)

        for i in range(0,self.num_of_imp):
            sub_fold = os.path.join(folder, "imp_"+str(i) )
            self.chain_mpo[i].save(sub_fold)

    def load(self, folder):
        filename = os.path.join(folder, "para.csv")
        para = pd.read_csv(filename)
        self.center_imp = para.center_imp[0]

        filename = os.path.join(folder, "D_imp.npy")
        self.D_imp = np.load(filename)

        for i in range(0, self.num_of_imp):
            sub_fold = os.path.join(folder, "imp_" + str(i))
            self.chain_mpo[i].load(sub_fold)
