#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyccl as ccl
import copy
from astropy.io import fits
#import matplotlib.gridspec as gridspec


class comp_shear :
        
    def __init__(self, Omega_m=0.3, Omega_b=0.05, AS=2., 
                 Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, 
                 Z_SYST=[0.1, -1.9, 0.9, -0.8], 
                 M_SYST=[1.2, 1.2, 1.2, 1.2],
                 A=1., eta=1.):


        self._matter_power_spectrum = 'halofit'

        self.update_cosmology(Omega_m=Omega_m, Omega_b=Omega_b,
                              AS=AS, Omega_nu_h2=Omega_nu_h2,
                              H0=H0, ns=ns, w0=w0)

        self.load_photo_z()
        self.update_photo_z(Z_SYST)
        self.update_multiplicative_bias(M_SYST)
        self.update_intrinsic_al(A0=1., eta=2.5, z0=0.62)

        self.ell = np.arange(20, 3000)
        #self.theta = np.linspace(1./60, 2., 100)

    def update_cosmology(self, Omega_m=0.3, Omega_b=0.05, AS=2.,
                         Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1):

        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.H0 = H0
        self.AS = AS
        self.Omega_nu_h2 = Omega_nu_h2
        self.n_s = ns
        self.w0 = w0

        # c est ce que mange ccl                                                                                                                       
        self._Omega_c = self.Omega_m - self.Omega_b
        self._h = self.H0 / 100.
        self._m_nu = (self.Omega_nu_h2 / self._h**2) * 93.14
        self._sigma8 = None
        self._A_s = self.AS/10.**9

        self.cosmology = ccl.Cosmology(Omega_c=self._Omega_c, Omega_b=self.Omega_b,
                                       h=self._h, n_s=self.n_s, sigma8=self._sigma8, A_s=self._A_s,
                                       w0=self.w0, m_nu=self._m_nu,
                                       matter_power_spectrum=self._matter_power_spectrum)

    def load_photo_z(self):

        filename = '/sps/lsst/users/evandena/DES_DATA/y1_redshift_distributions_v1.fits'
        redshifts_data = fits.open(filename, memmap=True)

        self.redshift = redshifts_data[1].data['Z_MID']

        self.redshifts = []
        self.nz = []
        
        for i in range(4):
            self.redshifts.append(self.redshift)
            self.nz.append(redshifts_data[1].data['BIN%i'%(i+1)]*redshifts_data[1].header['NGAL_%i'%(i+1)])
            
        self.redshifts = np.array(self.redshifts)
        self.nz = np.array(self.nz)

    def update_photo_z(self, zsyst):
        
        self.zsyst = np.array(zsyst) * 1e-2
        for i in range(len(zsyst)):
            self.redshifts[i] = copy.deepcopy(self.redshift) - self.zsyst[i]

    def update_multiplicative_bias(self, mbias):
        self.mbias = np.array(mbias) * 1e-2
        
    def update_intrinsic_al(self, A0=1, eta=1, z0=0.62):
        self.AI = A0 * ((1+self.redshift) / (1+z0))**eta

    def comp_xipm(self, theta):
        self.wl_bin_shear = []
        self.Cl = {}
        self.xip = {}
        self.xim = {}

        for i in range(4):
            filtre = (self.redshifts[i] > 0)
            self.wl_bin_shear.append(ccl.WeakLensingTracer(self.cosmology,
                                                           dndz=(self.redshifts[i][filtre], self.nz[i][filtre]), 
                                                           has_shear=True,
                                                           ia_bias=(self.redshift, self.AI)))

        for i in range(4):
            for j in range(4):
                if i>=j:
                    # calcul de l auto-correlation pour le bin donnee
                    key = "%i_%i"%((i, j))
                    self.Cl.update({key:ccl.angular_cl(self.cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], self.ell)})
                    m_ij = (1.+self.mbias[i]) * (1.+self.mbias[j])
                    self.xip.update({key: ccl.correlation(self.cosmology, 
                                                          self.ell, 
                                                          self.Cl[key], 
                                                          theta, 
                                                          corr_type='L+', 
                                                          method='fftlog')*m_ij}) 
                    self.xim.update({key: ccl.correlation(self.cosmology, 
                                                          self.ell,
                                                          self.Cl[key], 
                                                          theta, 
                                                          corr_type='L-', 
                                                          method='fftlog')*m_ij})
        self.theta = theta

    def plots(self):
        #plot du bin en redshift

        # plt.figure()
        # C = ['r', 'b', 'g', 'y']
        
        # for i in range(4):
        #      plt.plot(self.redshift, self.nz[i], C[i], 
        #                       lw=3, label='redshift bin %i'%(i+1))

        # plt.xlim(0,2)
        # plt.xlabel('z', fontsize=14)
        # plt.ylabel('dN/dz', fontsize=14)
        # plt.legend()
        # plt.savefig('plots/test_1_photozbin.png')

        # plot xiplus, ximoins

        xip = self.xip
        xim = self.xim

        plt.figure(figsize=(12,11))
        plt.subplots_adjust(wspace=0.0, hspace=0.)

        XLIM = [2, 300]

        SUBPLOTS_plus = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13]
        KEY_plus = ['3_0', '2_0', '1_0', '0_0', '3_1', 
                    '2_1', '1_1', '3_2', '2_2', '3_3']

        YLIM = [[-1,2.5], [-1,2.5], [-1,2.5], [-1,2.5],
                [-1, 3.], [-1, 3.], [-1, 3.],
                [-1, 5.], [-1, 5.],
                [-1, 6]]

        SUBPLOTS_minus = [24, 23, 22, 21, 20, 19, 18, 16, 15, 12]
        KEY_minus = ['3_0', '2_0', '1_0', '0_0', '3_1', 
                     '2_1', '1_1', '3_2', '2_2', '3_3']
        LEGEND = ['1,4', '1,3', '1,2', '1,1', '2,4', 
                  '2,3', '2,2', '3,4', '3,3', '4,4']

        for i in range(len(SUBPLOTS_plus)):
            key = KEY_plus[i]
            plt.subplot(6, 4, SUBPLOTS_plus[i])
            plt.plot(self.theta*60., self.theta * 60. * xip[key] * 1e4, 'b', label=LEGEND[i])
            plt.plot(XLIM, np.zeros(2), 'k')
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

            key = KEY_minus[i]
            ax = plt.subplot(6, 4, SUBPLOTS_minus[i])
            plt.plot(self.theta*60., self.theta * 60. * xim[key] * 1e4, 'r', label=LEGEND[i])
            plt.plot(XLIM, np.zeros(2), 'k')
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

        #plt.savefig('plots/test_xip_xim.png')

        
if __name__ == "__main__":    

     cs = comp_shear(Omega_m=0.3, Omega_b=0.05, AS=2.,
                     Omega_nu_h2=1e-3, H0=70, ns=0.97,
                     Z_SYST=[0., -2., 1., 1.])
     theta = np.linspace(1./60, 2., 20)
     cs.comp_xipm(theta)
     cs.plots()
   
# self.matter_power_spectrum = 'halofit' # include non-linearities in the power spectrum computation if I remember
# self.Omega_b = 0.05   # baryon density
# self.Omega_c = 0.25   # dark matter density
# self.h = 0.7          # normalized Hubble constant
# self.sigma8 = 0.8     # variance of matter density perturbations at an 8 Mpc/h scale
# self.A_s = None       # power spectrum normalization
# self.n_s = 0.96       # spectral index
# self.w0 = -1.         # intercept of dark energy equation state
# self.wa = 0. 
