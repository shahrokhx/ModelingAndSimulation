#!/usr/bin/env python
from __future__ import print_function
import cubic_ising_lattice, ising_hamiltonian
import numpy as np
import os
import matplotlib.pyplot as plt

# plotting function
def Plot_fn(file_name):
        
        trace = np.loadtxt( file_name )
        fig, ax = plt.subplots(
            1,2, gridspec_kw = {'width_ratios':[3, 1]})
        ax[0].set_xlabel("index", fontsize=14)
        ax[0].set_ylabel("data" , fontsize=14)
        ax[1].set_xlabel("freq.", fontsize=14)
        ax[1].get_yaxis().tick_right()

        # plot trace
        ax[0].plot(trace,c='black')

        # plot histogram
        wgt,bins,patches = ax[1].hist(trace, bins=30, normed=True
            , fc='gray', alpha=0.5, orientation='horizontal')
        # moving averge to obtain bin centers
        bins = np.array( [(bins[i-1]+bins[i])/2. for i in range(1,len(bins))] )
        def _gauss1d(x,mu,sig):
            norm  = 1./np.sqrt(2.*sig*sig*np.pi)
            gauss = np.exp(-(x-mu)*(x-mu)/(2*sig*sig)) 
            return norm*gauss

        Mymean = np.mean(trace)
        Mystd  = np.std(trace)
        ax[1].plot(_gauss1d(bins,Mymean,Mystd),bins,lw=2,c="black")
        ax[1].set_xticks([0,0.5,1])

        # overlay statistics
        for myax in ax:
            myax.axhline( Mymean, c='b', lw=2, label="mean = %1.4f" % Mymean )
            myax.axhline( Mymean+Mystd, ls="--", c="gray", lw=2, label="std = %1.2f" % Mystd )
            myax.axhline( Mymean-Mystd, ls="--", c="gray", lw=2 )

        ax[0].legend(loc='best')

        plt.show()

        auto_time = corr(trace,Mymean,Mystd)
        stderr    = error(trace,Mystd,auto_time)
        return auto_time

def corr(trace,mymean,mystd):
    num = len(trace)
    n   = num
    R   = [0]* n
    for k in range(1,n):
            temp_sum1 = 0
            for t in range(0,n-k):
                temp_sum1 += (trace[t]-mymean)*(trace[t+k]-mymean)
            
            R[k]=(1./ ((n-k)*mystd**2)) * temp_sum1
            if R[k]<0:
                break
            
    
    eff_R = R[1:k]
    auto_time = 1+2*sum(eff_R)

    # calculate auto correlation time
    return auto_time
def error(trace,mystd,Auto_time):
    N_eff        = len(trace)/Auto_time
    stderr_wrong = mystd/np.sqrt(len(trace))   #FIXME : This is currently unused 
    stderr       = mystd/np.sqrt(N_eff)
    
    # calculate standard error
    return stderr


def mc_loop(
    spin_mag       ,
    isingJ         ,
    temperature    ,
    spins_per_side ,
    nsweep         ,
    ndump          ,
    seed           ,
    traj_file      ,
    scalar_file    ,
    restart        = False,
    ):
    """ perform Monte Carlo simulation of the Ising model """

    beta = 1./temperature
    np.random.seed(seed)

    vol_frac_array =np.zeros((nsweep,1))
    
    # initialize the Ising model
    # ---------------------------
    lat = cubic_ising_lattice.CubicIsingLattice(spins_per_side,spin_mag=spin_mag)
    ham = ising_hamiltonian.IsingHamiltonian(isingJ,lat)
    if restart: # check traj_file
        assert os.path.exists(traj_file), "restart file %s not found" % traj_file
        lat.load_frame(traj_file)

    # setup outputs
    # ---------------------------
    if not restart:
        # destroy!
        with open(scalar_file,'w') as fhandle:
            fhandle.write('# energy    M^2/spin\n')

    print_fmt   = "{isweep:4d}  {energy:10.4f}  {Mperspin2:9.6f}"
    print(" isweep   energy   M2 ")

    # initialize observables
    # ---------------------------
    tot_energy  = ham.compute_energy()
    
    tot_energy_ini=tot_energy ;  #FIXME: for future usage 

    # MC loop
    # ---------------------------
    num_accept  = 0

    Ni = 10.0 # FIXME: hard-coded values -- number of initial recrystallized sites
    Nd = 3.0  # FIXME: hard-coded values -- number of adding new nucleations in each step
    nbr_type='Extended'

    # Initial nucleation
    for ini_c in range(int(Ni)):
        ispin=np.random.randint(lat.size())     
        lat.flip_spin(ispin)            

    total_crys = Ni  # total # of crystallized sites
    vol_frac = total_crys/lat.size()

    for isweep in range(nsweep):

        num_accept=0
        switch_list=np.zeros(lat.size())
        # report and dump configuration
        if isweep%ndump==0:
            print(print_fmt.format(**{'isweep':isweep,'energy':vol_frac
                ,'Mperspin2':0}))
            if isweep==0 and restart:
                continue # do not duplicate last configuration

            lat.append_frame(traj_file)

        # save scalar data
        with open(scalar_file,'a') as fhandle:
            fhandle.write('{energy:10.4f}  {Mperspin2:9.6f}\n'.format(**{'energy':vol_frac,'Mperspin2':0}))

        for ispin in range(lat.size()):
            
            # nucleation 
            if Nd/lat.size() > np.random.rand():  
                if lat.spin(ispin)==1:              
                    switch_list[ispin]=33 
                         
            
            # growth
            spin_sum=0

            if  nbr_type=='Normal':             # normal 4 neighbors
                if lat.spin(ispin)==1: 
                    for jj in lat.nb_list()[ispin]:
                        spin_sum = spin_sum + lat.spin(jj)  # sum of neighbors spin
                    if  spin_sum < 4:           # atleast one is already recrystallized    
                        switch_list[ispin]=33   # make it to the list            
                        
                              

        # end for ispin
        for ispin in range(lat.size()):
            if switch_list[ispin]==33 and lat.spin(ispin)==1:
               lat.flip_spin(ispin)   # recrystallize the current ispin (make it -1)
               num_accept = num_accept + 1  
        
        total_crys=total_crys+num_accept
        # Saving for plotting
        vol_frac = total_crys/lat.size()
        vol_frac_array[isweep] =np.log(1/(1-vol_frac))


    
    # initial output
    np.savetxt('Energy.txt',vol_frac_array,newline='\n')  
    auto_time_energy = Plot_fn('Energy.txt')
