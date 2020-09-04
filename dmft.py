import numpy as np
import pandas as pd
from numpy import linalg
from scipy.interpolate import interp1d
from scipy import integrate
# import matplotlib.pyplot as plt
import os

from imp_chain_ham import *
from mps import *
from TEBD import TEBD_class # main solver
from parameter import parameter # parameter
import time



class dmft_self_consistent:
    def __init__(self, para):
        self.para = para
        self.tebd = TEBD_class(para)

        self.data_path = os.getcwd()
        self.data_path = os.path.join(self.data_path, "data")
        if( not os.path.isdir(self.data_path)):
            os.mkdir(self.data_path)


    def initialize_(self, bath_fun = None , chain_mpo = None ):

        self.bath_fun = {}

        if( bath_fun is None ):
            cut_func = (1.0 - np.tanh(10.0 * (np.abs(self.para.w) - 1.0)) ) * 0.5
            bath_fun = -1j * np.pi * 0.5 * cut_func
            ac = - np.sum(np.imag(bath_fun)) / np.pi * self.para.dw
            bath_fun = bath_fun/ac

            for imp in self.para.imp_list:
                self.bath_fun[imp] = bath_fun.copy()
            self.get_bath_para()
        else:
            self.bath_fun = bath_fun
            self.get_bath_para()

        self.iter = 0
        self.if_print = True
        self.tebd.init(self.para, chain_mpo)

    def load_gnd(self):
        for mpo in self.tebd.chain_mpo:
            mpo.load_gnd()  # load the gnd as init
            mpo.if_find_gnd = False  # set the gnd flag false

    def save(self):
        self.tebd.save( os.path.join(self.data_path, "gnd") )
        para = {'tau': [self.para.tau], 't':[self.para.t]}
        para = pd.DataFrame(para, columns=['tau', 't'])
        para.to_csv(os.path.join(self.data_path, "para.csv"), index=False)


    def load(self, dir_name = 'gnd'):
        self.tebd.load(os.path.join( self.data_path, dir_name ) )
        filename = os.path.join(self.data_path, "para.csv")
        para = pd.read_csv(filename)
        self.para.tau = para.tau[0]
        self.para.t = para.t[0]


    def imp_solver(self, if_load_from_data = False, if_random_init = False):
        """
        if_random_init = True, then random init chain_mpo
        :return:
        """

        para = self.para


        # --------------------------------------------------------
        #                    initialization
        # --------------------------------------------------------

        el_list = self.el_list
        vl_list = self.vl_list

        # definte chain Ham
        chain_ham = []
        for i in range(0, 4):
            ham = single_chain_ham(el_list[i], vl_list[i], 0.0)
            chain_ham.append(ham)
        int_gate = int_ham(para, para.tau / 2.0)
        self.tebd.import_chain_ham(chain_ham)
        self.tebd.import_int_gate(int_gate, para.tau/2.0)

        if ( if_random_init):
            # chain mps
            print("Random init chain mps")
            self.tebd.chain_mpo = []
            for i in range(0, 4):
                if (i % 2 == 0):
                    if_bath_sign = True
                else:
                    if_bath_sign = False
                mps_ = mps(para, para.L_total, if_bath_sign)
                self.tebd.chain_mpo.append(mps_)

        if( if_load_from_data):
            self.load()

        # --------------------------------------------------------
        #     imag time evolution to find gnd
        # --------------------------------------------------------

        Ntau = para.Ntau
        max_en_diff = para.max_en_diff

        En = 999999
        i_check = 5  # check every i_check
        if( not if_load_from_data):
            for i in range(0, 5):
                tau = 2.0
                En_new = self.tebd.time_evolution(tau, if_calculate_gnd_en=True)
                if( En_new > En ):
                    En = En_new
                    break
                En = En_new

        st = time.time()
        iter_ = []
        en_ = []
        dt_ = []


        for i in range(0, Ntau):

            if (i % 20 == 0 and i != 0 and para.tau > 0.02):
                para.tau = para.tau / 2.0
            if (para.tau < 0.1):
                i_check = 2
            if (para.tau < 0.05):
                i_check = 1

            if (i % i_check == 0 or i == Ntau - 1):
                En_new = self.tebd.time_evolution(para.tau, if_calculate_gnd_en=True)
                diff = np.abs(En - En_new)
                if( En < En_new):
                    para.tau = para.tau * 0.75
                En = En_new
                if( self.if_print ):
                    print(" Imag time: ",i, " En = ", En, " En diff = ", diff, "dtau = ", para.tau, " total time = ",(time.time()-st)/60.0)
                    iter_.append(i)
                    en_.append(En)
                    dt_.append(para.tau)


                if (diff < max_en_diff):
                    if (para.tau < 0.02):
                        print("Solved")

                        ck_pt = {"iter": iter_, "En": en_, "dt":dt_}
                        df = pd.DataFrame.from_dict(ck_pt) #, columns=None)
                        df.to_csv("solved_check_point.csv")

                        break
                    else:
                        para.tau = para.tau / 2.0
            else:
                self.tebd.time_evolution(para.tau, if_calculate_gnd_en=False)

            if( i%20 == 0):
                ck_pt = {"iter":iter_, "En":en_, "dt":dt_}
                df = pd.DataFrame.from_dict(ck_pt) # , columns=None)
                df.to_csv("check_point.csv")

            if( i%30 == 0):
                self.save()

        self.save()

        self.tebd.save_gnd()

        # get static quantitiy
        self.tebd.cal_static()
        static_ = self.tebd.static

        En_gnd = En
        En_diff = diff
        if (diff > max_en_diff):
            print("Not solved")
            print("Energy diff is ", En_diff)
            print("Gnd energy is ", En_gnd)

        # --------------------------------------------------------
        #               real time evolution
        # --------------------------------------------------------

        self.g_t = {}
        for imp_ind in para.imp_list:
            self.tebd.load_gnd()
            Nt = para.Nt
            t_list = []
            g_plus = []  # d_t d_dag

            self.tebd.act_d_dag(imp_ind)

            for i in range(0, Nt):
                if (i % para.i_meas == 0):
                    gt = self.tebd.trace_with_d(imp_ind)
                    t_list.append(i * para.t)
                    g_plus.append(gt)
                    if (self.if_print):
                        print(" Real time: ", i, "  gplus ", gt," total time = ",(time.time()-st)/60.0)
                self.tebd.time_evolution(-1j * para.t, if_calculate_gnd_en=False)

            t_list = np.squeeze( np.asarray(t_list) )
            g_plus = np.squeeze( np.asarray(g_plus) ) * np.exp(1j *np.squeeze(En_gnd) * t_list)

            self.tebd.load_gnd()
            Nt = para.Nt
            t_list = []
            g_minus = []  # d_t_dag d

            self.tebd.act_d(imp_ind)
            for i in range(0, Nt):
                if (i % para.i_meas == 0):
                    gt = self.tebd.trace_with_d_dag(imp_ind)
                    t_list.append(i * para.t)
                    g_minus.append(gt)
                    if (self.if_print):
                        print(" Real time: ", i, "  gminus ", gt," total time = ",(time.time()-st)/60.0)
                self.tebd.time_evolution(1j * para.t, if_calculate_gnd_en=False)

            t_list = np.squeeze(np.asarray(t_list))
            g_minus = np.squeeze( np.asarray(g_minus) ) * np.exp(-1j *np.squeeze(En_gnd) * t_list)


            self.g_t[imp_ind] = -1j * (g_plus + g_minus)




        # save the cuurrent solution,
        self.t_list  = t_list
        self.static_ = static_
        self.gnd_en  = En_gnd
        self.En_diff =En_diff

        df = {'t':self.t_list }
        for imp_ind in self.para.imp_list:
            df['gt'+str(imp_ind)] = self.g_t[imp_ind]
        df = pd.DataFrame.from_dict(df) #, columns=None)
        df.to_csv("gt.csv")




    def gt_to_gw(self):
        df = {'w':self.para.w }
        self.g_w = {}
        for imp in self.para.imp_list:
            self.g_w[imp] = self.para.t * (self.para.ft @ self.g_t[imp]) # Fourier transf
            df['gw_' + str(imp)] = self.g_w[imp]


        df = pd.DataFrame.from_dict(df) # , columns=None)
        df.to_csv("gw.csv")



    def calculate_self_energy(self):
        self.sigma = {}
        for imp in self.g_w:
            self.sigma[imp] = self.para.z - self.bath_fun[imp] - 1.0/self.g_w[imp]

    def calculate_g_loc(self):
        """
        \int dep rho(ep)/( w - ep - Sigma[w] )
        :return:
        """
        self.g_loc = {}
        for imp in self.para.imp_list:
            de  = self.para.de[imp]
            ep  = self.para.ep[imp]
            rho = self.para.rho[imp]

            denominator = self.para.w - self.sigma[imp]
            denominator = 1.0/( np.reshape( denominator, ( len(self.para.w), 1)) - np.reshape( ep, (1,len(ep)) )  + 1j * self.para.delta )
            self.g_loc[imp] = denominator @ ( rho * de )



    def dmft_error(self):
        self.g_inv_err = {}
        self.del_g_inv = {}
        self.g_err = {}
        self.del_g = {}
        for imp in self.para.imp_list:
            self.del_g[imp] = self.g_loc[imp] - self.g_w[imp]
            self.del_g_inv[imp] = 1.0/self.g_loc[imp] - 1.0/self.g_w[imp]
            self.g_err[imp] = np.max( np.abs(self.del_g[imp]) )
            self.g_inv_err[imp] = np.max( np.abs( self.del_g_inv[imp] ) )

        if(self.iter == 0):
            return

        self.sig_err = {}
        self.del_sig = {}
        for imp in self.para.imp_list:
            self.del_sig[imp] = self.sigma[imp] - self.sigma_old[imp]
            self.sig_err[imp] = np.max( np.abs( self.del_sig[imp] ) )



    def update(self):
        mix = self.mix()
        self.sigma_old = {}
        for imp in self.para.imp_list:
            self.bath_fun[imp] = self.bath_fun[imp] - mix * self.del_g_inv[imp] # update bath function
            self.sigma_old[imp] = self.sigma[imp].copy()







    def get_bath_para(self):
        """
        g0 -> para
        :return:
        """

        self.vl_list = {}
        self.el_list = {}

        for imp in self.para.imp_list:
            rho = -np.imag( self.bath_fun[imp])/np.pi # spectrum func for bath
            if( np.min(rho) < 0.0):
                print(" spectrum function of bath, value error ", np.min(rho) )
                rho = rho + np.abs( np.min(rho) )

            vl = []
            f = interp1d(self.para.w, rho, kind='linear', bounds_error=False, fill_value=0.0)

            # interpolation of spectum funtion
            for i in range(0, 2 * self.para.N_bath -1):
                w = self.para.w_bath[i]
                vl_square = integrate.quad(f, w-self.para.dw_bath/2.0, w + self.para.dw_bath/2.0 )
                vl_square = vl_square[0]
                if( vl_square < 0.0):
                    print(" vl^2 value error ", vl_square)
                    vl_square = 0.0
                vl.append( np.sqrt(vl_square))

            el = []
            g = lambda x:f(x) * x
            for i in range(0, 2 * self.para.N_bath - 1):
                w = self.para.w_bath[i]
                el_it = integrate.quad(g, w - self.para.dw_bath / 2.0, w + self.para.dw_bath / 2.0)
                el_it = el_it[0]
                if( np.abs( vl[i] ) < 1e-10 ):
                    if( np.abs(el_it) < 1e-10 ):
                        el_it = 0.0
                    else:
                        print("el value error ", vl[i], el_it)
                el.append( el_it )


            self.vl_list[imp] = np.asarray(vl)
            self.el_list[imp] = np.asarray(el)


        self.duplicate_bath_para()


    def duplicate_bath_para(self):
        """
        if only measure one or two green's function then duplicate bath function
        :return:
        """

        if( not ( 1 in self.para.imp_list) ):
            self.vl_list[1] = self.vl_list[0].copy()
            self.el_list[1] = self.el_list[0].copy()

        if (not (2 in self.para.imp_list)):
            self.vl_list[2] = self.vl_list[0].copy()
            self.el_list[2] = self.el_list[0].copy()

        if (not (3 in self.para.imp_list)):
            self.vl_list[3] = self.vl_list[2].copy()
            self.el_list[3] = self.el_list[2].copy()



    def dmft_iteration(self, if_load_from_data = False) :
        if( self.iter != 0 and not if_load_from_data): # not the 1st iteration, load the previous gnd to be the init state
            self.load_gnd()

        self.imp_solver(if_load_from_data=if_load_from_data)
        self.gt_to_gw()
        self.calculate_self_energy()
        self.calculate_g_loc()
        self.dmft_error()
        self.update()
        self.get_bath_para()

        self.iter += 1





    def mix(self):
        return self.para.mix




































