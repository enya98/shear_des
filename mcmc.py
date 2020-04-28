import numpy as np
import pickle
from comp_shear_func import comp_shear

dic_xi = pickle.load(open('/sps/lsst/users/evandena/DES_DATA/xip_xim_simu.pkl', 'rb'))

xip = dic_xi['xip']
xim = dic_xi['xim']
cov = dic_xi['cov_matrix']

# TO DO
# *ecrive une classe qui calcul le chi2
# *je pense que tu va devoir mettre tes xi_plus et tes xi_moins dans le meme format
# *je veux que tu me fasse le plot donnee /model xip/xim avec cosmologie de ton choix

