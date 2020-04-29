import numpy as np
import pickle
from comp_shear_func import comp_shear

# TO DO
# *ecrive une classe qui calcul le chi2 # DONE
# *je pense que tu va devoir mettre tes xi_plus et tes xi_moins dans le meme format # DONE
# *je veux que tu me fasse le plot donnee /model xip/xim avec cosmologie de ton choix
# * tu branches le mcmc 


def trans_vec_line(vec):
    return vec.reshape((len(vec), 1))

class comp_chi2:
    
    def __init__(self, xip, xim, theta, cov):

        self.xip = xip
        self.xim = xim
        self.xi = np.concatenate((xip, xim))
        self.weight = np.linalg.inv(cov)
        self.theta = theta

    def comp_shear_chi2(self, param, theta=None):
        
        if theta is None:
            theta = self.theta

        cs = comp_shear(Omega_m=param[0], Omega_b=param[1], AS=param[2],
                        Omega_nu_h2=param[3], H0=param[4], ns=param[5]) 
        cs.comp_photo_z()
        cs.comp_xipm(theta)

        KEY = ['0_0', '1_0', '2_0', '3_0', 
               '1_1', '2_1', '3_1',
               '2_2', '3_2',
               '3_3']

        self.xip_th = np.zeros_like(self.xip)
        self.xim_th = np.zeros_like(self.xim)

        for i in range(len(KEY)):
            #WARNING: 20 is hard coded, could be improved....
            self.xip_th[i*20:(i+1)*20] = cs.xip[KEY[i]]
            self.xim_th[i*20:(i+1)*20] = cs.xim[KEY[i]]

    def calcul_chi2(self, param):
        
        self.dof = len(self.xi) - len(param)
        self.comp_shear_chi2(param)
        self.xi_th = np.concatenate((self.xip_th, self.xim_th))
        self.residuals = self.xi - self.xi_th
        self.chi2 = self.residuals.dot(np.dot(self.weight, trans_vec_line(self.residuals)))[0]

    def return_log_L(self, param):
        self.calcul_chi2(self, param)
        return -self.chi2

if __name__ == "__main__":
    
    param = [0.3, 0.05, 2., 0.001, 70, 0.97]
    dic_xi = pickle.load(open('/sps/lsst/users/evandena/DES_DATA/xip_xim_simu.pkl', 'rb'))
    theta = dic_xi['xip']['ANG'][:20] / 60.

    import time
    
    cc = comp_chi2(dic_xi['xip']['VALUE'], 
                   dic_xi['xim']['VALUE'], 
                   theta, dic_xi['cov_matrix'])
    A = time.time()
    cc.calcul_chi2(param)
    B = time.time()
    print(B-A)






