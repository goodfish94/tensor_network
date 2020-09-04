import numpy as np
# import matplotlib.pyplot as plt
import os

import pandas as pd
from numpy import linalg 
import time 
# plot setting 
# import matplotlib as mpl
# from matplotlib import rcParams
# from matplotlib import rc

# plt.style.use('default')

# rcParams[ 'axes.titlesize'] = 25
# rcParams[ 'axes.labelsize'] = 25
# rcParams[ 'lines.markersize'] = 5
# rcParams[ 'xtick.labelsize'] = 22
# rcParams[ 'ytick.labelsize'] = 22
# rcParams[ 'legend.fontsize'] = 25
# rcParams[ 'legend.frameon'] = True

# # mpl.rc('font', family='sans-serif')
# mpl.rc('text', usetex=True)

# mpl.rcParams['text.latex.preamble'] = [
#        r'\usepackage{amsmath}',
#        r'\usepackage{helvet}',    # set the normal font here
#        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#        r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
# ] 

from dmft import dmft_self_consistent
from parameter import parameter

Dmax = 300

U_hubbard = 0.5# density interaction
JH        = 0.1 # Hund's

U ={ '12':U_hubbard, '13':U_hubbard - 0.5 * JH, '14':U_hubbard, '23':U_hubbard, '24':U_hubbard - 0.5 * JH, '34':U_hubbard }
J = -0.5 * JH
ed = np.ones( 4 ) * ( -1.5 * U_hubbard ) #fix half filling
E0 = 0.0
para = parameter( D = Dmax, ed = ed, U = U , J = J ,E0 = E0, N_ = 1000 )
#--------------------------------
#      init solver
#--------------------------------

st = time.time()

solver = dmft_self_consistent(para)
solver.initialize_()

solver.dmft_iteration()

print("One iter time = ", (time.time()- st)/60.0)
