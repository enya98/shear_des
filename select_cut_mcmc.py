import numpy as np

def find_cut(chains, step=10, atol=8): 

    nwalk = np.shape(chains)[1]
    ndim = np.shape(chains)[2]

    ncuts = np.zeros(ndim)

    for nd in range(ndim):

        stds = np.std(chains[:,:,nd], axis=0)
        mean = np.mean(stds[-step:]) 
        std =  np.std(stds[-step:])

        nstep = int(nwalk / step) 
    
        for i in range(nstep): 
            if i == 0:
                new_mean = np.mean(stds[-(step*(i+1)):]) 
            else: 
                new_mean = np.mean(stds[-(step*(i+1)):-(step*i)]) 
            
            if abs(mean-new_mean)> atol * std: 
                break 
        ncuts[nd] = nwalk -(step*i)

    return int(np.max(ncuts))

if __name__ == "__main__":

    np.random.seed(4)
    x = np.random.uniform(-5, 5, 100)

    #y, y_err = generate_linear_model(x, a=2, b=1, err_scale=1.)

    #lf = linear_fit(x, y, y_err=y_err)
    #lf.fit_mcmc(starting_point=[2., 2.])
