import numpy as np
import os
# import matplotlib.pyplot as plt

def ep_each_bond(k, r, t):
    """

    :param k:
    :param r:
    :param t:
    :return:   t*exp(-ikr) + conj()
    """
    im = 1j
    re = t * np.exp(- (k[0] * r[0] + k[1] * r[1]) * im)
    re += np.conj(re)
    return re


def generate_hk_plus(t_, k):
    """

    :param t_:  tbp vector
    :param k:  k vector
    :return: ep_plus

    """

    a1 = [1.0, 0.0]
    a2 = [0.5, np.sqrt(3.0) / 2.0]
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    re = 0.0

    # t1

    [m, n] = [1, 0]
    t_each = t_[0]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [0, 1]
    t_each = np.conj(t_[0])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-1, 1]
    t_each = t_[0]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    #
    # [m, n] = [-1, 0]
    # t_each = np.conj(t_[0])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [0, -1]
    # t_each = t_[0]
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [1, -1]
    # t_each = np.conj(t_[0])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)

    # t2

    [m, n] = [1, 1]
    t_each = t_[1]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-1, 2]
    t_each = np.conj(t_[1])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-2, 1]
    t_each = t_[1]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    # [m, n] = [-1, -1]
    # t_each = np.conj(t_[1])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [1, -2]
    # t_each = t_[1]
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [2,-1]
    # t_each = np.conj(t_[1])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)

    # t3
    [m, n] = [1, 2]
    t_each = t_[2]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-2, 3]
    t_each = np.conj(t_[2])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-3, 1]
    t_each = t_[2]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    # [m, n] = [-1, -2]
    # t_each = np.conj(t_[2])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [2,-3]
    # t_each = t_[2]
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [3,-1]
    # t_each = np.conj(t_[2])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)

    # t3 conj

    [m, n] = [2, 1]
    t_each = np.conj(t_[2])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-1, 3]
    t_each = (t_[2])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-3, 2]
    t_each = np.conj(t_[2])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    # [m, n] = [-2,-1]
    # t_each = (t_[2])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [1, -3]
    # t_each = np.conj(t_[2])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [3, -2]
    # t_each = (t_[2])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)

    # t4
    [m, n] = [2, 0]
    t_each = t_[3]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [0, 2]
    t_each = np.conj(t_[3])
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    [m, n] = [-2, 2]
    t_each = t_[3]
    r = a1 * m + a2 * n
    re += ep_each_bond(k, r, t_each)

    # [m, n] = [-2, 0]
    # t_each = np.conj(t_[3])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [0,-2]
    # t_each = t_[3]
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)
    #
    # [m, n] = [2, -2]
    # t_each = np.conj(t_[3])
    # r = a1 * m + a2 * n
    # re += ep_each_bond(k, r, t_each)

    re = -re

    return re


def get_t(delta_V):
    if (abs(delta_V - (-20.0)) < 0.1):
        t1 = 1.583 * np.exp(1j * np.pi * 0.169)
        t2 = -1.108
        t3 = 0.323 * np.exp(1j * np.pi * (- 0.069))
        t4 = 0.732 * np.exp(1j * np.pi * (-0.653))
    elif (abs(delta_V - (-10.0)) < 0.1):
        t1 = 1.998 * np.exp(1j * np.pi * 0.118)
        t2 = -1.330
        t3 = 0.4363 * np.exp(1j * np.pi * (- 0.035))
        t4 = 0.905 * np.exp(1j * np.pi * (-0.692))
    else:
        print(" cant't find tbp for deltaV= ", delta_V)

    t_ = np.asarray([t1, t2, t3, t4])
    return t_


def generate_dispersion_array(delta_V, N_bz,mu):
    b1 = 2.0 * np.pi * np.asarray([1.0, -1.0 / np.sqrt(3.0)])
    b2 = 2.0 * np.pi * np.asarray([0.0, 2.0 / np.sqrt(3.0)])

    t_ = get_t(delta_V)
    f  = lambda k:generate_hk_plus(t_, k) - mu

    k_list  = []
    hp_list = []
    hm_list = []
    dk_list = []
    for i in range(0, N_bz):
        for j in range(0, N_bz):
            k = float(i) / N_bz * b1 + float(j) / N_bz * b2
            k_list.append( k )
            hp_list.append( f(k) )
            hm_list.append( f(-k) )
            dk_list.append( 1.0/(N_bz/N_bz))

    k = np.asarray(k_list)
    dk = np.asarray(dk_list)
    hp = np.real( np.asarray(hp_list) )
    hm = np.real( np.asarray(hm_list) )

    max_en = np.max(hp)
    min_en = np.min(hp)
    D = (max_en - min_en)/2.0
    mid_ = (max_en + min_en)/2.0

    hp = (hp - mid_) / D
    hm = (hm - mid_) / D


    return [k, dk, hp, hm, D, mid_]


#
# delta_V = -20.0
# k, dk, hp, hm, D, mid_= generate_dispersion_array(delta_V, 100)
# plt.plot(hp)
# plt.plot(hm)
# plt.show()
