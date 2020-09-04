import numpy as np
import pandas as pd
from numpy import linalg

from imp_chain_ham import *
from mps import *
from TEBD import TEBD # main solver
from parameter import parameter # parameter

from bath_generator import spectral


def imp_solver(para, chain_mpo  = None ):
    """
    :param para:
    :param mps_init:  init with this mps
    :return:
    """

    # --------------------------------------------------------
    #                    initialization
    # --------------------------------------------------------





    el_list = para.el_list
    vl_list = para.vl_list


    # definte chain Ham
    chain_ham = []
    for i in range(0, 4):
        ham = single_chain_ham(el_list[i], vl_list[i], 0.0)
        chain_ham.append(ham)
    int_gate = int_ham(para, para.tau / 2.0)


    if( chain_mpo is None ):
        # chain mps
        L_bath = len(el_list)
        chain_mpo = []
        for i in range(0, 4):
            if (i % 2 == 0):
                if_bath_sign = True
            else:
                if_bath_sign = False
            mps_ = mps(para, L_bath + 1, if_bath_sign)
            chain_mpo.append(mps_)
    else:
        for mpo in chain_mpo:
            mpo.load_gnd() # load the gnd as init
            mpo.if_find_gnd = False # set the gnd flag false




    # TEBD solver
    tebd_ = TEBD(para)
    tebd_.init_state(chain_mpo, chain_ham, int_gate, para.tau/2.0)

    # --------------------------------------------------------
    #                    imag time evolution to find gnd
    # --------------------------------------------------------

    Ntau = para.Ntau
    max_en_diff = para.max_en_diff

    En = 999999
    i_check = 5 # check every i_check
    for i in range(0, Ntau):
        if (i % 20 == 0 and i != 0 and para.tau > 0.02):
            para.tau = para.tau / 2.0
        if( para.tau < 0.1 ):
            i_check = 2
        if( para.tau < 0.05 ):
            i_check = 1

        if (i % i_check == 0 or i == Ntau-1):
            En_new = tebd_.time_evolution(para.tau, if_calculate_gnd_en=True)
            diff = np.abs(En - En_new)
            En = En_new
            if (diff < max_en_diff ):
                if (para.tau < 0.02):
                    print("Solved")
                    break
                else:
                    para.tau = para.tau / 2.0
        else:
            tebd_.time_evolution(para.tau, if_calculate_gnd_en=False)

    tebd_.save_gnd()

    # get static quantitiy
    tebd_.cal_static()
    static_ = tebd_.static

    En_gnd  = En
    En_diff = diff
    if( diff > max_en_diff ):
        print("Not solved")
        print("Energy diff is ", En_diff)
        print("Gnd energy is ", En_gnd)

    # --------------------------------------------------------
    #               real time evolution
    # --------------------------------------------------------

    gt = {}
    for imp_ind in para.imp_list:
        tebd_.load_gnd()
        Nt = para.Nt
        t_list = []
        g_plus = []  # d_t d_dag

        tebd_.act_d_dag(imp_ind)
        for i in range(0,Nt):
            if( i%para.i_meas == 0 ):
                gt = tebd_.trace_with_d(imp_ind)
                t_list.append(i * para.t)
                g_plus.append(gt)
            tebd_.time_evolution(-1j * para.t , if_calculate_gnd_en=False)

        t_list = np.asarray(t_list)
        g_plus = np.asarry(g_plus)  * np.exp( 1j * En_gnd * t_list )

        tebd_.load_gnd()
        Nt = para.Nt
        t_list = []
        g_minus = [] # d_t_dag d

        tebd_.act_d(imp_ind)
        for i in range(0, Nt):
            if (i % para.i_meas == 0):
                gt = tebd_.trace_with_d_dag(imp_ind)
                t_list.append(i * para.t)
                g_minus.append(gt)
            tebd_.time_evolution(1j * para.t, if_calculate_gnd_en=False)

        t_list = np.asarray(t_list)
        g_minus = np.asarry(g_minus) * np.exp(-1j * En_gnd * t_list)

        gt[imp_ind] = -1j * ( g_plus + g_minus )


    sol = {
        't':t_list,  # t list
        'gt':gt,     # gt dict
        'static':static_, # static quantity (filling)
        'mps':tebd_.chain_mpo, # mpo
        'gnd_en':En_gnd, # Gnd energy
        'en_diff':En_diff # energy diff
    }


    return sol
















