#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyccl as ccl
from astropy.io import fits
#import matplotlib.gridspec as gridspec


class comp_shear :
        
    def __init__(self, Omega_m=0.3, Omega_b=0.05, AS=2., 
                 Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1):

        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.H0 = H0
        self.AS = AS
        self.Omega_nu_h2 = Omega_nu_h2
        self.n_s = ns
        self.w0 = w0

        self._Omega_c = self.Omega_m - self.Omega_b
        self._h = self.H0 / 100.
        self._m_nu = self.Omega_nu_h2 * 93.
        self._sigma8 = None
        self._A_s = self.AS/10.**9

        self._matter_power_spectrum = 'halofit'

        self.ell = np.arange(20, 3000)
        self.theta = np.linspace(3./60, 200./60., 200)

    def gauss_photo_z(self, z, z0, sigma_z):
        return 3 * np.exp(-0.5 * (z-z0)**2 / sigma_z**2)

    def comp_photo_z(self):
        
        filename = '/sps/lsst/users/evandena/DES_DATA/y1_redshift_distributions_v1.fits'
        self.redshifts_data = fits.open(filename, memmap=True)
        self.redshift = self.redshifts_data[1].data['Z_MID']

        photo_z_distribution = []        
        for i in range(4):
            photo_z_distribution.append(self.redshifts_data[1].data['BIN%i'%(i+1)]*self.redshifts_data[1].header['NGAL_%i'%(i+1)])
            
        self.photo_z_distribution = photo_z_distribution
    
    
    def comp_xipm(self):
        self.wl_bin_shear = []
        self.Cl = {}
        self.xip = {}
        self.xim = {}
        cosmology = ccl.Cosmology(Omega_c=self._Omega_c, Omega_b=self.Omega_b,
                                  h=self._h, n_s=self.n_s, sigma8=self._sigma8, A_s=self._A_s,
                                  w0=self.w0, m_nu=self._m_nu,
                                  matter_power_spectrum=self._matter_power_spectrum)
        for i in range(4):
            self.wl_bin_shear.append(ccl.WeakLensingTracer(cosmology,
                                                           (self.redshift, self.photo_z_distribution[i]), has_shear=True, ia_bias=None))
            
        for i in range(4):
            for j in range(4):
                if i>=j:
                    # calcul de l auto-correlation pour le bin donnee
                    key = "%i_%i"%((i, j))
                    self.Cl.update({key:ccl.angular_cl(cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], self.ell)})
                    self.xip.update({key: ccl.correlation(cosmology, self.ell, self.Cl[key], self.theta, corr_type='L+', method='fftlog')}) 
                    self.xim.update({key: ccl.correlation(cosmology, self.ell, self.Cl[key], self.theta, corr_type='L-', method='fftlog')}) 

    def plots(self):
        #plot du bin en redshift

        plt.figure()
        C = ['r', 'b', 'g', 'y']
        
        for i in range(4):
             plt.plot(self.redshift, self.photo_z_distribution[i], C[i], 
                              lw=3, label='redshift bin %i'%(i+1))

        plt.xlim(0,2)
        plt.xlabel('z', fontsize=14)
        plt.ylabel('dN/dz', fontsize=14)
        plt.legend()
        plt.savefig('plots/test_1_photozbin.png')

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

        plt.savefig('plots/test_xip_xim.png')
        
        
  
        
if __name__ == "__main__":    

    cs = comp_shear(Omega_m=0.3, Omega_b=0.05, AS=2.,
                    Omega_nu_h2=1e-3, H0=70, ns=0.97)
    cs.comp_photo_z()
    cs.comp_xipm()
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
