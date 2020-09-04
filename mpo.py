# store density matrix as an mpo

import os
import numpy as np
from parameter import parameter
from utility import svd_utility
import pandas as pd



class mpo:
# mpo
#
#                  |                |                  |                             |
#   -- lam[0] -- T[0] -- lam[1] -- T[1] -- lam[2] -- T[2] --....... --lam[L-1] -- T[L-1] -- lam[L]
#                  |                |                  |                             |
#
#


    def __init__(self, para : parameter):

        self.L      = para.L_total
        self.d_up   = para.d.copy()  # up layer
        self.d_down = para.d.copy()  # down layer

        self.T    = {}
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


    def id_density_matrix(self):
        """
        set the matrix to be id w/ bond dimension 1
        :return:
        """


        for i in range(0,self.L):
            self.D[i] = 1
            T_id = np.reshape(np.eye(self.d_up[i], self.d_down[i]), (1, self.d_up[i], self.d_down[i], 1))
            # d_up = d_down at this step
            self.T[i] = np.array(T_id, copy=True)

        self.D[self.L] = 1
        self.log_norm=0


    def set_norm_one(self):
        """
        set norm to be one
        :return:
        """
        self.log_norm=0

    def get_T(self,i):
        return self.T[i]

    def save_T(self, T ,i):
        self.T[i] = np.array(T, copy = True)
        self.D[i] = T.shape[0]
        self.D[i+1] = T.shape[3]
        # update tensor and bond dimension




    def svd_(self):
        """
        perform svd on mat rho
        :param mat:
        :return: [u,v,vt,norm, D]
        """
        return svd_utility(self.rho, self.Dmax, self.tol_svd, self.svd_full)


    def one_gate_projection_two_site(self, gate, site_index, center ):
        """
        update mpo

                        |                     |
                              (  Gate  )
                        |                     |


                        |                     |
            --lam1-----[T1]-------lam2-------[T2]--------lam3
                        |                     |
               D1                  D2                     D3


        :param mpo:  density operator
        :param Gate: gate acting on mpo, shape = [i_up, j_up, i_down, j_down]
        :param site_index: tuple, site that we are consider now. only (i,i+1) is valid choice
        :param center: == 'L' i is new center, 'R', i+1 is new center
        :return:
        """

        i,j = site_index # j = i+1


        Dleft   = self.D[i]
        Dright  = self.D[j+1]
        # bond dimension for the left most and right most

        dup_i   = self.d_up[i]
        ddown_i = self.d_down[i]
        dup_j   = self.d_up[j]
        ddown_j = self.d_down[j]



        self.rho     = np.tensordot( self.get_T(i), self.get_T(i+1), axes=1)
        # rho ~ lamd1 T1 lam2 T2 lamd
        # rho shape = (Dleft, d,d,d, d,Dright )







        self.rho     =np.transpose(self.rho, (1,3,0,2,4,5) )
        #[d,d,Dleft,d,d,Dright] , pick up the index

        self.rho = np.tensordot(gate, self.rho, axes=2)



        self.rho = np.transpose( self.rho, (2,0,3,1,4,5) )
        self.rho = np.reshape( self.rho, (Dleft*dup_i*ddown_i, Dright*dup_j*ddown_j))



        [u,s,vt,norm,D] = self.svd_()



        self.current_D = max(D, self.current_D)       # update the currnet maximum bond dimension
        self.log_norm = self.log_norm + np.log(norm)  # update normalization factor


        if( center == 'R'):
            self.save_T(np.reshape(u, (Dleft, dup_i, ddown_i, D)), i)
            # left canonical
            self.save_T(np.reshape(np.tensordot(np.diag(s),vt,axes=1), (D, dup_j,ddown_j, Dright)), j)
            # center

            self.center = j
            self.D[j] = D

        else:  # center == 'L'
            self.save_T(np.reshape( np.tensordot(u, np.diag(s),axes=1), (Dleft, dup_i, ddown_i, D)), i)
            # center
            self.save_T(np.reshape(vt, (D, dup_j, ddown_j, Dright)), j)
            # right canonical

            self.center = i
            self.D[j]   = D


    def one_gate_projection_three_site(self, Gate, site_index, center):
        """
        update mpo

                        |                     |                      |
                                          (  Gate  )
                        |                     |                      |


                        |                     |                      |
            --lam1-----[T1]-------lam2-------[T2]--------lam3-------[T3]--------lam4
                        |                     |                      |
               D1                  D2                     D3                     D4


        :param Gate: gate acting on mpo
        :param site_index: tuple, site that we are consider now. only (i,i+1,i+2) is valid choice
        :param center:
        =='L', new center is T1
        =='R', new center is T3
        =='M', new center is T2

        :return:
        """





        i, j, k = site_index # j= i+1, k = j+1

        if(center == 'L'):
            self.center = i
        elif(center == 'M'):
            self.center = j
        else:
            self.center = k

        Dleft   = self.D[i]
        Dright  = self.D[k+1]
        # bond dimension for the left most and right most

        dup_i   = self.d_up[i]
        ddown_i = self.d_down[i]
        dup_j = self.d_up[j]
        ddown_j = self.d_down[j]
        dup_k = self.d_up[k]
        ddown_k = self.d_down[k]


        self.rho     = np.tensordot( self.get_T(i), np.tensordot(self.get_T(j), self.get_T(k), axes=1), axes=1)
        # rho ~ T1,T2,T3
        # rho shape = (Dleft, d,d,d,d, d,d, Dright )

        self.rho     = np.transpose(self.rho, (1,3,5,0,2,4,6,7) )
        #[d*d*d,Dleft,d,d,Dright] , pick up the index

        self.rho = np.tensordot(Gate, self.rho, axes=3)
        # contract with Gate Gate [iup][jup][kup][idown][jdown][kdown]

        self.rho = np.transpose( self.rho, (3, 0, 4, 1, 5, 2, 6, 7) )
        # (Dleft, d,d,d,d,d,d, Dright)


        if( center  == 'R' or center == 'M'):
            # perform SVD from left, k or j is new center

            self.rho = np.reshape(self.rho, [Dleft*dup_i*ddown_i, dup_j*ddown_j*dup_k*ddown_k*Dright])

            [u, s, vt, norm, D] = self.svd_()

            self.current_D = max(D, self.current_D)         # update the current maximum bond dimension
            self.log_norm = self.log_norm + np.log(norm)
            self.D[j] = D


            self.save_T(np.reshape(u, (Dleft, dup_i, ddown_i, D)),i)
            # update 1st T

            Dleft = D # new D left

            self.rho = np.reshape( np.tensordot( np.diag(s), vt, axes=1), (D*dup_j*ddown_j, Dright*dup_k*ddown_k) )
            # construct new rho to update second bond. [D*d*d, d*d*Dright]

            [u, s, vt, norm, D] = self.svd_()

            self.current_D = max(D, self.current_D)
            self.log_norm = self.log_norm + np.log(norm)
            self.D[k] = D
            if( center == 'M'):
                self.save_T(np.reshape(vt, (D, dup_k, ddown_k, Dright)), k)
                self.save_T(np.reshape( np.tensordot(u, np.diag(s), axes=1), (Dleft, dup_j, ddown_j, D) ),j)
            else: # center == 'R'
                self.save_T(np.reshape(np.tensordot( np.diag(s),vt, axes=1), (D, dup_k, ddown_k, Dright)), k)
                self.save_T(np.reshape(u, (Dleft, dup_j, ddown_j, D)), j)

        else: # center == 'L'
            # perform SVD from right
            self.rho = np.reshape(self.rho, [Dleft * dup_i * ddown_i * dup_j * ddown_j, dup_k * ddown_k * Dright])

            [u, s, vt, norm, D] = self.svd_()

            self.current_D = max(D, self.current_D)  # update the current maximum bond dimension

            self.log_norm = self.log_norm + np.log(norm)  # second time reach this bond, update norm  again
            self.D[k] = D

            self.save_T(np.reshape(vt, (D, dup_k, ddown_k, Dright)),k)
            # update final T

            self.rho = np.reshape(np.tensordot(u, np.diag(s), axes=1), (Dleft * dup_i * ddown_i, D * dup_j * ddown_j))
            # construct new rho to update 1st bond.

            Dright = D
            # update left D

            [u, s, vt, norm, D] = self.svd_()
            self.current_D = max(D, self.current_D)  # update the currnet maximum bond dimension

            self.log_norm = self.log_norm + np.log(norm)
            self.D[j] = D

            self.save_T(np.reshape( np.tensordot(u, np.diag(s),axes=1), (Dleft, dup_i, dup_j, D)), i)
            self.save_T(np.reshape(vt, (D, dup_j, ddown_j, Dright)), j)




    def swap_gate(self,  i, new_center):
        """
        switch the d_up index of site i and site i+1
        require: one of the sites is the center
        :param i:
        :param new_center: =i or i+1, i or i+1 is the new center
        :return:
        """
        if( i >= self.L - 1):
            return


        Dleft   = self.D[i]
        Dright  = self.D[i+2]
        dup_i   = self.d_up[i]
        ddown_i = self.d_down[i]
        dup_j   = self.d_up[i+1]
        ddown_j = self.d_down[i+1]



        self.rho  = np.tensordot( self.get_T(i), self.get_T(i+1), axes=1)
        # [Dleft][dup_i][ddown_i][dup_j][ddown_j][Dright]
        # self.rho = self.rho[0:Dleft, 0:dup_i, 0:ddown_i, 0:dup_j, 0:ddown_j, 0:Dright]
        self.rho = np.transpose(self.rho, (0,3,2,1,4,5) )
        self.rho = np.reshape(self.rho, (Dleft*dup_j*ddown_i, Dright*dup_i*ddown_j))
        [u, s, vt, norm, D] = self.svd_()


        self.D[i+1] = D
        self.current_D = max(D, self.current_D)
        self.log_norm = self.log_norm + np.log(norm)

        self.d_up[i] = dup_j
        self.d_up[i+1] = dup_i
        # swap upper index


        if( new_center == i):
            self.save_T(np.reshape(np.tensordot(u,np.diag(s),axes=1), (Dleft, self.d_up[i], self.d_down[i], D)), i)
            self.save_T(np.reshape(vt, (D, self.d_up[i+1], self.d_down[i+1], Dright)), i+1)
            self.center = i
        else: # new center ==  i + 1
            self.save_T(np.reshape(u, (Dleft, self.d_up[i], self.d_down[i], D)), i)
            self.save_T(np.reshape(np.tensordot(np.diag(s),vt,axes=1), (D, self.d_up[i + 1], self.d_down[i + 1], Dright)), i+1)
            self.center = i + 1


    def move_center_right(self):
        """
        move center: i->i+1
        :return:
        """

        i = self.center
        if(self.center >= self.L-1):
            return

        self.rho = self.get_T(i)
        rho_shape = (self.rho).shape
        self.rho = np.reshape(self.rho, (rho_shape[0]*rho_shape[1]*rho_shape[2], rho_shape[3]) )

        [u, s, vt, norm, D] = self.svd_()

        self.D[i+1]     = D
        self.current_D  = max(D, self.current_D)
        self.log_norm   = self.log_norm + np.log(norm)

        self.save_T( np.reshape(u, (rho_shape[0], rho_shape[1],rho_shape[2],D)),i )
        # save T[i], now, T[i] is left canonical


        self.rho = np.tensordot( np.tensordot(np.diag(s),vt, axes=1), self.get_T(i+1), axes=1 )
        self.save_T( self.rho, i+1)
        # get new T[i+1] which is the new center

        self.center = i+1

    def move_center_left(self):
        """
        move center: i->i-1
        :return:
        """

        i = self.center
        if(self.center <= 0):
            return

        self.rho = self.get_T(i)
        rho_shape = (self.rho).shape
        self.rho = np.reshape(self.rho, (rho_shape[0], rho_shape[1]*rho_shape[2]*rho_shape[3]) )

        [u, s, vt, norm, D] = self.svd_()

        self.D[i]      = D
        self.current_D = max(D, self.current_D)
        self.log_norm  = self.log_norm + np.log(norm)

        self.save_T(np.reshape(vt, (D, rho_shape[1], rho_shape[2],rho_shape[3])), i)
        # save T[i], now, T[i] is left canonical

        self.rho = np.tensordot(self.get_T(i-1), np.tensordot(u,np.diag(s), axes=1), axes=1 )
        self.save_T( self.rho, i-1)
        # get new T[i+1] which is the new center

        self.center = i-1

    def move_center_to(self,dest):
        """
        move center to dest
        :param dest:
        :return:
        """


        if(dest == self.center):
            return
        i=self.center
        if( i> dest):# move center left
            for it in range(i,dest,-1):
                self.move_center_left()
        else:
            for ir in range(i,dest,1):
                self.move_center_right()




    def move_site_to_center_rhs(self,j, j_dest):
        """

        :param j:  j> self.center+1
        :return: move upper index of j to j_dest
        with j_dest < j
        """

        if( j== j_dest):
            return

        i = self.center
        self.move_center_to(j-1)
        # move center to j-1

        for it in range( j-1, j_dest-1, -1 ):
            self.swap_gate(it, it)
        # swap the gate until reach site j_dest
        # center is at j_dest now

        self.move_center_to(i)
        # move center back to i


    def move_site_to_center_lhs(self, j, j_dest):
        """

        :param j:  j<  self.center-1
        :return: move upper index of j to j_dest
        j < j_dest < self.center
        """

        if( j == j_dest):
            return

        i = self.center
        self.move_center_to(j+1)
        # move center to j+1

        for it in range(j, j_dest-1):
            self.swap_gate(it, it+1)
        # swap the gate until reach site j_dest-1
        # now the center is at j_dest

        self.move_center_to(i)
        # move center back to i

    def move_site_away_center_rhs(self, j, j_dest):
        """
        move upper index of  j to j_dest
        :param j:  i< j<j_dest
        :return:
        """

        if( j == j_dest):
            return

        i = self.center
        self.move_center_to(j)
        # move center to j

        for it in range(j, j_dest):
            self.swap_gate(it, new_center = it+1)
        # switch to j
        # now center is at j_dest

        self.move_center_to(i)
        # move center back to i

    def move_site_away_center_lhs(self, j, j_dest):
        """
        move upper index of  j to j_dest
        :param j:
        :return:
        j_dest < j < center
        """
        if( j == j_dest):
            return

        i = self.center
        self.move_center_to(j)
        # move center to j
        for it in range(j-1, j_dest-1, -1):
            self.swap_gate(it, new_center = it)
        # switch to j_dest
        # now center j_dest

        self.move_center_to(i)
        # move center back to i


    def check_d(self, d):
        for i in range(0,self.L):
            if( self.d_up[i] != d[i] or self.d_down[i] != d[i]):
                print("local dimension error , site i = ",i, " dup, ddown = ", self.d_up[i],self.d_down[i], " d = ",d[i])
                return False
        return True




    def check_left_canonical(self):
        """
        check if the site < center is left canonical
        :param center:
        :return:
        """

        center = self.center

        if_left_canonical = True

        for i in range(0,center):
            M       =  self.get_T(i)
            M       = np.reshape(M, (self.D[i]*self.d_up[i]*self.d_down[i],self.D[i+1]))
            ID      = np.matmul(np.conj(np.transpose(M)) , M )
            ID = ID / np.trace(ID)*self.D[i+1]
            diff    = ID - np.eye(self.D[i+1],self.D[i+1])
            diff    = np.sum( np.abs(diff) )

            if(diff > 1e-10):
                if_left_canonical = False

                print("site %f is not left canonical , diviation = %.30f "%(i,diff))
        return if_left_canonical

    def check_right_canonical(self):
        """
        check if the site > center is right canonical
        :param center:
        :return:
        """

        center = self.center

        if_right_canonical = True

        for i in range(center+1,self.L):
            M       = self.get_T(i)
            M       = np.reshape(M, (self.D[i],self.d_up[i]*self.d_down[i]*self.D[i+1]))
            ID      = np.matmul(M, np.conj(np.transpose(M)) )
            ID      = ID/np.trace(ID)*self.D[i]
            diff    = ID - np.eye(self.D[i],self.D[i])
            diff    = np.sum( np.abs(diff) )

            if(diff > 1e-10):
                if_right_canonical = False
                print("site %f is not right canonical, diviation = %.30f "%(i,diff))

                # raise ValueError("Not right canonical")

        return if_right_canonical













