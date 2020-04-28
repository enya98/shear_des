import numpy as np
import pylab as plt
from iminuit import Minuit
import emcee
import corner


def generate_quadratique_model(x, a=2, b=1, c=-2,err_scale=0.3):
    
    y_err = np.random.normal(scale=err_scale, size=len(x))
    y = a*x*x + b*x + c
    y += y_err
    y_err = np.ones_like(y) * err_scale
    return y, y_err


class linear_fit:

    def __init__(self, x, y, y_err=None):

        self.x = x
        self.y = y
 
        if y_err is not None:
            self.y_err = y_err
        else:
            self.y_err = np.ones_like(y)

        self.w = 1./self.y_err**2

        
    def chi2_fct(self, params):

        self.residuals = self.y - params[0]*self.x**2 - params[1]*self.x - params[2] # data - model(params)
        #self.chi2 = 0
        #for i in range(len(self.residuals)):
        #    self.chi2 += (self.residuals[i] * self.resiuduals[i]) * self.w[i]
        self.chi2 = np.sum(self.residuals**2 * self.w)
        return self.chi2

    # p(f(theta)| y) ~ 1/det(C) * exp(-0.5 (y-f(theta))^T C^-1 (y-f(theta))) * prior(theta)
    # prior = 1
    # p(f(theta)| y) ~ 1/det(C) * exp(-0.5 (y-f(theta))^T C^-1 (y-f(theta)))
    # tu veux connaitre theta (a, b pour le fit lineaire) qui maximise p(f(theta)| y)
    # dp/d(theta) = 0 <-->  chi2 = (y-f(theta))^T C^-1 (y-f(theta)) <--> min chi2
    # C covariance matrix (y_err**2)

    def fit_gradient(self, starting_point=[3, -2]):
        self.m = Minuit.from_array_func(self.chi2_fct, starting_point, print_level=0)
        self.m.migrad()
        self.results = [self.m.values[key] for key in self.m.values.keys()]


    def log_likelihood(self, theta):
        chi2 = self.chi2_fct(theta)
        return - chi2    
    
    def fit_mcmc(self, starting_point=[3.1, 2.5, 0.1], step_cut=200):
        
        nsteps = 1000
        ndim, nwalkers = len(starting_point), 100
        starting_positions = [starting_point + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood)
        sampler.run_mcmc(starting_positions, nsteps)
            
        LABEL = ['a', 'b', 'c']
        for j in range(ndim):
            plt.figure()
            for i in range(nwalkers):
                plt.plot(sampler.chain[i,:,j],'k', alpha=0.1)
            plt.ylabel(LABEL[j], fontsize=20)

        self.sampler = sampler

        samples = sampler.chain[:, step_cut:, :].reshape((-1, ndim))

        self.results = []
        self.errors = []
        for i in range(ndim):
            mcmc = np.percentile(samples[:,i], [16, 50, 84])
            self.errors.append(np.diff(mcmc))
            self.results.append(mcmc[1])

        fig = corner.corner(samples, labels=["a", "b", "c"], truths=[self.results[0], 
                            self.results[1], self.results[2]], levels=(0.68, 0.95))
        plt.show()
        plt.savefig('plots/fit_mcmc.png')

if __name__ == "__main__":

    np.random.seed(42)
    x = np.random.uniform(-5, 5, 100)

    y, y_err = generate_quadratique_model(x, a=2, b=1, c=-2, err_scale=1.)


    #plt.scatter(x, y, c='b')
    #plt.errorbar(x, y, xerr=None, yerr=y_err, 
    #             linestyle='', ecolor='blue', alpha=1,
    #             marker='.', zorder=0) 

    lf = linear_fit(x, y, y_err=y_err)
    lf.fit_mcmc()

    
    #fig = corner.corner(results_mcmc)
