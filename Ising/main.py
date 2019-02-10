import os
from mc_ising_single import mc_loop

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of the recrystallization process')
    parser.add_argument('-b','--beta',type=float,default=0.25,help='inverse temperature, reasonable range 0.25-1.0')
    parser.add_argument('-n','--nsweep',type=int,default=100,help='number of Monte Carlo sweeps')
    parser.add_argument('-sps','--spins_per_side',type=int,default=20,help='number of spins on each dimension of the cubic Ising lattice')
    parser.add_argument('-s','--seed',type=int,default=1,help='random number generator seed')
    parser.add_argument('-r','--restart',action='store_true',help='restart simulation from trajectory file, default to traj.dat')
    parser.add_argument('-j','--isingJ',type=float,default=1.0,help='Ising model\'s J parameter')
    parser.add_argument('-mag','--spin_mag',type=float,default=1.0,help='magnitude of the spins')
    args = parser.parse_args()

    # hard-coded inputs (Temporary)
    # ==============================
    ndump       = 25
    empty_files = True 
    traj_file   = 'traj.dat'
    scalar_file = 'scalar.dat'



    # parse inputs
    # ==============================
    spins_per_side = args.spins_per_side
    beta           = args.beta
    nsweep         = args.nsweep
    restart        = args.restart
    seed           = args.seed
    isingJ         = args.isingJ
    spin_mag       = args.spin_mag

    # check input & output locations
    # ==============================
    if restart: # restart from trajfile
        # check trajectory file
        if not os.path.exists(traj_file):
            print( "WARNING: no trajectory file found. Starting from scratch" )
            restart = False


    if not restart: # from scratch
        if not empty_files: # stop if any output is about to be overwritten
            for output in [traj_file,scalar_file]:
                if os.path.exists(output):
                    raise RuntimeError("%s exists!" % output)

        # destroy!
        if os.path.exists(traj_file):
            open(traj_file,'w').close()


    # Call MC
    # ==============================
    mc_loop(
        spins_per_side = spins_per_side,
        temperature    = 1./beta,
        nsweep         = nsweep,
        ndump          = ndump,
        restart        = restart,
        seed           = seed,
        traj_file      = traj_file,
        scalar_file    = scalar_file,
        isingJ         = isingJ,
        spin_mag       = spin_mag
    )
