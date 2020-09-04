import numpy as np

a = np.reshape( np.asarray( [0,1] ), (2,1) )
b = np.reshape( np.asarray( [3,2] ), (1,2) )

sigup = np.asarray([[0,1],[0,0]])
sigdn = np.transpose(sigup)
n     = np.asarray([[1,0],[0,0]])
id_   = np.asarray([[1,0],[0,1]])

hop = np.kron( sigup, sigdn) + np.kron(sigdn, sigup)
n2  = np.kron( id_, n)
hop = np.reshape(hop, (4,4))
n2  = np.reshape(n2, (4,4))

a = np.asarray( [1,2,3,4] )
print( np.diag(a) )
