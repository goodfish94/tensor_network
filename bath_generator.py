# generate bath
import numpy as np

class spectral:
    """
    spectal function
    """

    def __init__(self):
        return

    def import_bath(self, de, ep, Lambda):
        """

        :param cut:  cut off
        :param de:   de
        :param ep:  1d array
        :param Lambda: 1d array
        """

        self.de = de
        self.ep = ep
        self.Lambda = Lambda

    def constant_dos(self, Ne):
        """
        constant dos with cutoff =1
        :param Ne:
        :return:
        """

        self.ep =-1.0 + 2.0 * np.asarray( range(0,Ne))/Ne + 1.0/ Ne
        self.Lambda = -1j * np.pi /2.0 * np.ones(Ne, dtype=np.complex)
        self.de = 2.0/Ne


    def get_bath_coeff(self, N_de):
        """
        each bath ind correspondinig to N_de interval
        :param N_de:
        :return:
        """

        vl  = []
        epl = []
        de  = []
        rho_ = np.imag( self.Lambda )
        for i in range(0, int(len(self.ep)/N_de)):
            v =  np.sqrt( - np.sum( rho_[i*N_de:(i+1) * N_de] ) * self.de /np.pi )
            vl.append(v)
            epl.append( - np.sum( rho_[i*N_de:(i+1) * N_de] * self.ep[i*N_de:(i+1) * N_de] ) * self.de/np.pi/(v * v) )
            de.append( N_de * self.de )

        return [np.asarray(vl), np.asarray(epl), np.asarray(de)]





    def get_dos_from_para(self, vl, epl,de, cut):
        N = 100
        w = -cut + 2.0 * cut/N * np.asarray( range(0,N) ) + cut/N
        gc0 = (vl*vl ) @ ( 1.0/( -w - 1j * 1e-3 + np.resize(epl, (len(epl), 1) )) )
        rc  = 2.0 * np.imag(gc0) / (2.0 * np.pi)

        return [w, rc]



