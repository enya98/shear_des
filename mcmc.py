import numpy as np
import pickle
import pylab as plt
from comp_shear_func import comp_shear
import emcee
import corner
import parser, optparse
import os

def read_option():

    usage = "launch mcmc for shear fitting of des y1"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--rep","-r",dest="rep",help="dir output", default='')
    parser.add_option("--nwalkers","-w",dest="nwalkers", help="number of walkers", default='20')
    parser.add_option("--nsteps","-n",dest="nsteps", help="number of steps", default='2')
    parser.add_option("--seed","-s",dest="seed", help="seed of the generator", default='42')

    option,args = parser.parse_args()

    return option

def trans_vec_line(vec):
    return vec.reshape((len(vec), 1))

class comp_chi2:
    
    def __init__(self, xip, xim, theta, cov):

        self.xip = xip
        self.xim = xim
        self.xi = np.concatenate((xip, xim))
        self.cov = cov
        self.err = np.sqrt(np.diag(self.cov))
        self.weight = np.linalg.inv(cov)
        self.theta = theta

    def comp_shear_chi2(self, param, theta=None):
        
        if theta is None:
            theta = self.theta

        cs = comp_shear(Omega_m=param[0], Omega_b=param[1], AS=param[2],
                        Omega_nu_h2=param[3], H0=param[4], ns=param[5],
                        Z_SYST=[param[6], param[7], param[8], param[9]], 
                        M_SYST=[param[10], param[11], param[12], param[13]]) 
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
        print('chi2 param: ', param)
        self.dof = len(self.xi) - len(param)
        self.comp_shear_chi2(param)
        self.xi_th = np.concatenate((self.xip_th, self.xim_th))
        self.residuals = self.xi - self.xi_th
        self.chi2 = self.residuals.dot(np.dot(self.weight, trans_vec_line(self.residuals)))[0]


    def plots(self):
        # plot xiplus, ximoins

        xip_th = self.xip_th
        xim_th = self.xim_th
        theta = self.theta

        plt.figure(figsize=(12,11))
        plt.subplots_adjust(wspace=0.0, hspace=0.)

        SUBPLOTS_plus = [4, 3, 2, 1, 
                         7, 6, 5, 
                         10, 9, 
                         13]

        SUBPLOTS_minus = [21, 22, 23, 24, 
                          18, 19, 20, 
                          15, 16, 
                          12]
        
        XLIM = [2, 300]
        YLIM = [[-1,2.5], [-1,2.5], [-1,2.5], [-1,2.5],
                [-1, 3.], [-1, 3.], [-1, 3.],
                [-1, 5.], [-1, 5.],
                [-1, 6]]

        LEGEND = ['1,1', '1,2', '1,3', '1,4',
                  '2,2', '2,3', '2,4',
                  '3,3', '3,4',
                  '4,4']

        for i in range(len(SUBPLOTS_plus)):
            
            plt.subplot(6, 4, SUBPLOTS_plus[i])
            plt.plot(theta*60., theta * 60. * xip_th[i*20:(i+1)*20] * 1e4, 'b', label=LEGEND[i])
            plt.plot(XLIM, np.zeros(2), 'k')
            plt.scatter(theta*60., theta * 60. * self.xip[i*20:(i+1)*20] * 1e4, c='k')
            plt.xlim(XLIM[0], XLIM[1])
            plt.ylim(YLIM[i][0], YLIM[i][1])
            plt.xscale('log')
            plt.xticks([],[])
            if SUBPLOTS_plus[i] not in [1, 5, 9, 13]:
                plt.yticks([],[])
            else:
                plt.ylabel('$\\theta \\xi_{+}$ / $10^{-4}$', fontsize=10)

            leg = plt.legend(handlelength=0, handletextpad=0, 
                         loc=2, fancybox=True, fontsize=8)
            for item in leg.legendHandles:
                item.set_visible(False)

            ax = plt.subplot(6, 4, SUBPLOTS_minus[i])
            plt.plot(theta*60., theta * 60. * xim_th[i*20:(i+1)*20] * 1e4, 'r', label=LEGEND[i])
            plt.plot(XLIM, np.zeros(2), 'k')
            plt.scatter(theta*60., theta * 60. * self.xim[i*20:(i+1)*20] * 1e4, c='k')
            plt.xlim(XLIM[0], XLIM[1])
            plt.ylim(YLIM[i][0], YLIM[i][1])
            plt.xscale('log')

            if SUBPLOTS_minus[i] in [21, 22, 23, 24]:
                plt.xlabel('$\\theta$ (arcmin)', fontsize=12)
            else:
                plt.xticks([],[])

            plt.yticks([],[])

            leg = plt.legend(handlelength=0, handletextpad=0, 
                         loc=2, fancybox=True, fontsize=8)
            for item in leg.legendHandles:
                item.set_visible(False)

            ax2 = ax.twinx()
            ax2.set_ylim(YLIM[i][0], YLIM[i][1])

            if SUBPLOTS_minus[i] not in [12, 16, 20, 24]:
                ax2.set_yticks([],[])
            else:
                ax2.set_ylabel('$\\theta \\xi_{-}$ / $10^{-4}$', fontsize=10)

        plt.savefig('plots/xip_xim_3.png')
        

    def return_log_L(self, param):

        range_params = True

        # flat prior omega_m
        if param[0]<0.1 or param[0]>0.9:
            range_params &= False
        
        # flat prior omega_b
        if param[1]<0.03 or param[1]>0.07:
            range_params &= False

        # flat prior As*10**9
        if param[2]<0.5 or param[2]>5.:
            range_params &= False

        # flat prior Omega_nu_h**2
        if param[3]<6e-4 or param[3]>0.01:
            range_params &= False

        # flat prior H0
        if param[4]<55 or param[4]>90:
            range_params &= False

        # flat prior ns
        if param[5]<0.87 or param[5]>1.07:
            range_params &= False
            
        if range_params:
            self.calcul_chi2(param)
            return -self.chi2
        else:
            return -np.inf

    def fit_mcmc(self, starting_point=[0.3, 0.05, 2., 0.001, 70, 0.97, [0.1, -1.9, 0.9, -0.8], 
                    [1.2, 1.2, 1.2, 1.2]], nsteps = 5, nwalkers=2, seed=42):

        np.random.seed(seed)
        starting_positions = []
        
        ndim = len(starting_point)
        for j in range(ndim):
            starting_positions.append(starting_point[j] + 1e-4*np.random.randn(ndim) for i in range(nwalkers))
        
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.return_log_L)
        self.sampler.run_mcmc(starting_positions, nsteps)


if __name__ == "__main__":

    option = read_option()
    
    param = [0.3, 0.05, 2., 0.001, 70, 0.97, 0.1-5*1.6, -1.9-5*1.3, 0.9-5*1.1, -0.8-5*2.2, 
             1.2-5*2.3, 1.2-5*2.3, 1.2-5*2.3, 1.2-5*2.3]

    ##param_fail_1 = [2.97562507e-01, 4.85680140e-02, 2.00498544e+00, 
    ##                2.19841975e-04, 6.99978467e+01, 9.75383166e-01]
    ##param_fail_2 = [2.99578865e-01, 5.12096193e-02, 1.99966989e+00,
    ##                2.74596509e-04, 7.00003361e+01,  9.70221717e-01]

    dic_xi = pickle.load(open('/sps/lsst/users/evandena/DES_DATA/xip_xim_simu.pkl', 'rb'))
    theta = dic_xi['xip']['ANG'][:20] / 60.

    
    cc = comp_chi2(dic_xi['xip']['VALUE'], 
                   dic_xi['xim']['VALUE'], 
                   theta, dic_xi['cov_matrix'])
    cc.comp_shear_chi2(param)
    cc.plots()

    cc.fit_mcmc(nsteps=int(option.nsteps), 
                nwalkers=int(option.nwalkers), 
                seed=int(option.seed))

    # file_name = os.path.join(option.rep, 'sample_%s.pkl'%(option.seed))
    # file_output = open(file_name, 'wb')
    # dic_output = {'chain':cc.sampler.chain,
    #               'seed':option.seed}
    # pickle.dump(dic_output, file_output)
    # file_output.close()
