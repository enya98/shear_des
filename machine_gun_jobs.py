import os
import numpy as np

n_seed = np.arange(1, 3) # [1, 2]

os.system('tar cvzf shear_des.tar.gz ../shear_des/')
rep_output = '/sps/lsst/users/evandena/mcmc_output'

for i in range(len(n_seed)):

    fichier=open('machine_gun_jobs_DESY1_%i.sh'%(i),'w')
    fichier.write('#!/bin/bash \n')
    fichier.write('\n')
    fichier.write('home=/pbs/home/e/evandena/shear_des \n')
    fichier.write('\n')
    fichier.write('cp ${home}/shear_des.tar.gz . \n')
    fichier.write('\n')
    fichier.write('tar xzvf shear_des.tar.gz \n')
    fichier.write('\n')
    fichier.write('cd shear_des/ \n')
    fichier.write('\n')

    fichier.write('python mcmc.py --nwalkers 20 --nsteps 1 --seed %i --rep %s'%((n_seed[i], rep_output)))
    
    fichier.close()

    o_log = os.path.join(rep_output, 'log', "output_o_%i.log"%(i+1))
    e_log = os.path.join(rep_output, 'log', "output_e_%i.log"%(i+1))

    os.system('qsub -P P_lsst -q long -l s_vmem=16G -l sps=1 -e %s -o %s machine_gun_jobs_DESY1_%i.sh'%((i, e_log, o_log)))
    os.system('rm machine_gun_jobs_DESY1_%i.sh*'%(i))

