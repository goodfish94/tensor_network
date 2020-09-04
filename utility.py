# svd subroutine
from scipy.sparse.linalg import svds
from numpy.linalg import svd
import numpy as np
import time



def svd_utility(mat, Dmax, tol_svd, svd_full):


    # ts = time.time()

    D_mat = min(mat.shape)
    D_svd = min(D_mat, Dmax)
    # keep D_svd eg values for svd
    if (D_svd == D_mat or svd_full):
        u, s, vt = svd(mat, full_matrices=False)
        # using Full svd
    else:
        u, s, vt = svds(mat, k= D_svd , which='LM')
        # trunctated SVD, keep largest Dmax eg vals

    # s shape = (k,)
    # u shape = (Dleft*d*d,k)
    # vt shape = (k, Dright*d*d)

    ind_ = (s > tol_svd)

    s = s[ind_]
    u = u[:, ind_]
    vt = vt[ind_, :]

    D = len(s)

    norm = np.linalg.norm(s)  # L2 norm
    s = s / norm

    # print("scipy svd time = ", (time.time()-ts)/60.0 )

    return [u, s, vt, norm, D]

