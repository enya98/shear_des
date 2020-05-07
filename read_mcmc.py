import pickle
import pylab as plt
import numpy as np
import corner
import glob
import os
from select_cut_mcmc import find_cut

class read_mcmc_fit:
    
    def __init__(self, rep_output="../sps_lsst/mcmc_output/", 
                 LABEL=['$\Omega_m$', '$\Omega_b$', '$A_s \\times 10^9$', 
                        '$\Omega_{\\nu}h^2$', '$H_0$', '$n_s$']):

        self.LABEL = LABEL
        self.rep_output = rep_output

    def read_result(self, pickle_file, plot_chain=True):

        dic = pickle.load(open(pickle_file, 'rb'))
        chain = dic['chain']
        
        self.ndim = np.shape(chain)[2]
        self.nwalkers = np.shape(chain)[0]

        step_cut = find_cut(chain, step=10, atol=7)

        if plot_chain:
            for j in range(self.ndim):
                plt.figure()
                for i in range(self.nwalkers):
                    plt.plot(chain[i,:,j], 'k', alpha=0.1)
                    ylim = plt.ylim()
                    plt.plot([step_cut, step_cut], ylim, 'k', lw=3)
                    plt.ylabel(self.LABEL[j], fontsize=20)

        samples = chain[:, step_cut:, :].reshape((-1, self.ndim))

        return samples


    def read_samples(self, plot_corner=False):
        
        pkls = os.path.join(self.rep_output, '*.pkl')

        output_pkl = glob.glob(pkls)

        self.samples = []
        
        for i in range(len(output_pkl)):
            self.samples.extend(self.read_result(output_pkl[i]))
        self.samples = np.array(self.samples)
            
        # np.concatenate
            
        results = []
        errors = []
        for i in range(self.ndim):
            mcmc = np.percentile(self.samples[:,i], [16, 50, 84])
            errors.append(np.diff(mcmc))
            results.append(mcmc[1])
            
        if plot_corner :
            fig = corner.corner(self.samples, labels=self.LABEL,
                                truths=[results[0], results[1], results[2],
                                        results[3], results[4],
                                        results[5]], levels=(0.68, 0.95))
            plt.savefig('plots/mcmc_output_fixed_syste_fixed_astro.png')


if __name__ == "__main__":

    cr = read_mcmc_fit(rep_output="../sps_lsst/mcmc_output/",
                       LABEL = ['$\Omega_m$', '$\Omega_b$', '$A_s \\times 10^9$', 
                                '$\Omega_{\\nu}h^2$', '$H_0$', '$n_s$'])
    cr.read_samples(plot_corner=True)

    
