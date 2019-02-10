import numpy as np


class CubicIsingLattice:
    '''
    Usage:
        lat = CubicIsingLattice(3) will initialize a 3x3 lattice of spins
        lat.size()                       the total number of spins on the lattice
        lat.spins()                      all spin states 
        lat.spin_mag()                   magnitude of the spins (assumed to be the same) 
        lat.spin(ispin)                  spin ispin state
        lat.nside()                      number of spins per dimension 
        lat.shape()                      shape of latice (default: [lat.nside(),lat.nside()]) 
        lat.nb_list()                    neighbour list 
        lat.flip_spin(ispin)             to change spin state of ispin
    '''

    def __init__(self,spins_per_dim,ndim=2,spin_mag=1.0):

        # save input quantities
        self._spins_per_dim = spins_per_dim # number of spins along each dimension of the cubic lattice
        self._ndim          = ndim          # number of dimensions
        self._spin_mag      = spin_mag      # magnitude of a single spin (like mass)
        self._nb_per_dim    = 2             # 2 neighbours per dimension on a cubic lattice

        # save calculated quantities
        self._shape = [spins_per_dim]*ndim # default: (spins_per_dim,spins_per_dim)
        self._nspin = spins_per_dim**ndim  # total number of spins

        # initialize all spins to be pointing up
        self._spins = np.ones(self._shape,dtype=int) # initialize cubic lattice of spins

        # allocate and initialize neighbor list to establish the topology of the lattice
        self._nb_list = np.zeros([self._nspin,self._nb_per_dim*self._ndim],dtype=int)
        self._nb_list_extended = np.zeros([self._nspin,self._nb_per_dim*self._ndim*2],dtype=int)
        self._nb_list_extended_double = np.zeros([self._nspin,self._nb_per_dim*self._ndim*2+16],dtype=int)
        for ispin in range(self._nspin): # calculate and save the neighbours of each spin
            self._nb_list[ispin,:] = self.neighbours(ispin) 
            self._nb_list_extended[ispin,:] = self.neighbours_extended(ispin)
            self._nb_list_extended_double[ispin,:] = self.neighbours_extended_double(ispin)   

    def __str__(self):
        return str(self._spins)

    def ndim(self):
        return self._ndim

    def size(self):
        return self._nspin

    def nside(self):
        return self._spins_per_dim

    def spin_mag(self):
        return self._spin_mag

    def shape(self):
        return self._shape[:] # return a copy to deter external modification

    def spins(self,copy=True):
        if copy:
            return self._spins.copy()
        else:
            return self._spins


    def spin(self,ispin):
        spin_idx = self.multi_idx(ispin)
        return self._spins[spin_idx]


    def nb_list(self,copy=True):
        if copy:
            return self._nb_list.copy()
        else:
            return self._nb_list


    def nb_list_extended(self,copy=True):
        if copy:
            return self._nb_list_extended.copy()
        else:
            return self._nb_list_extended


    def nb_list_extended_double(self,copy=True):
        if copy:
            return self._nb_list_extended_double.copy()
        else:
            return self._nb_list_extended_double


    def linear_idx(self,tuple_idx):
        # locate the linear index of a spin on the lattice: this method takes a 
        # multi-dimensional index and returns a single integer that labels the selected spin

        # guards
        out_lower_bound = np.where(tuple_idx<0)
        out_upper_bound = np.where(tuple_idx>=self.nside())

        ispin = np.ravel_multi_index(tuple_idx,self.shape())
        return ispin


    def multi_idx(self,ispin):
        # locate the multi-dimensional index of a spin on the lattice: this method takes a index and returns a multi-dimentional index
        if ispin >= self.size() or ispin<0: # guard against misuse
            raise IndexError("linear spin index %d is out of bounds."%ispin)

        return np.unravel_index(ispin,self.shape())


    def flip_spin(self,ispin):
        # flip spin ispin and return the change in total magnetization 
        dmag = self.mag_change(ispin)    # change of total magnetization after flip
        spin_idx = self.multi_idx(ispin) # find the spin to flip
        self._spins[spin_idx] *= -1      # flip the spin
        return dmag

    
    def append_frame(self,filename):
        fhandle = open(filename,'a')
        fhandle.write("%d\n"%self.size())
        for irow in range(self.nside()):
            row_text = " ".join(["%2d"%spin for spin in self._spins[irow]])
            fhandle.write(row_text+"\n")
        fhandle.close()

    def load_frame(self,filename,ref_frame=-1,max_nframe=10**6):
        fhandle = open(filename,'r+')
	from mmap import mmap
        mm = mmap(fhandle.fileno(),0)

        # first line should specify the number of spins
        first_line  = mm.readline()
        nspin = int(first_line)
        mm.seek(0) # rewind file

        # find starting positions of each frame in trajectory file
        frame_starts = []
        for iframe in range(max_nframe):
            # stop if end of file is reached    
            if mm.tell() >= mm.size():
                break

            # locate starting line (should be "%d\n" % nspin) 
            idx = mm.find(first_line)
            if idx == -1:
                break

            frame_starts.append(idx)
            mm.seek(idx)
            myline = mm.readline()


        # go to desired frame
        mm.seek(frame_starts[ref_frame])
        mm.readline() # skip first line with nspin

        # read spin states
        for irow in range(self.nside()):
            self._spins[irow] = np.array(mm.readline().split(),dtype=int)

    def visualize(self,ax):
        # draw the lattice on a matplotlib axes object ax 
        if self._ndim != 2:
            raise NotImplementedError("visualization only implemented for 2D lattice")

        # draw spins
        ax.pcolormesh(self.spins().T,vmin=-self._spin_mag,vmax=self._spin_mag)

        # set ticks to spin centers
        ax.set_xticks(np.arange(self.nside())+0.5)
        ax.set_yticks(np.arange(self.nside())+0.5)
        # rename ticks
        ax.set_xticklabels(np.arange(self.nside()))
        ax.set_yticklabels(np.arange(self.nside()))

    def neighbours(self,ispin):
        # return a list of indices pointing to the neighbours of spin ispin
        spin_idx    = self.multi_idx(ispin)
        neighb_list = np.zeros(self._nb_per_dim*self.ndim(),dtype=int)
    
        # LEFT
        x_minus =spin_idx[0]-1
        if  x_minus<0:
            x_minus=self._shape[0]-1 
        elif x_minus>self._shape[0]-1:
            x_minus=0

        # RIGHT
        x_plus  =spin_idx[0]+1
        if  x_plus<0:
            x_plus=self._shape[0]-1
        elif x_plus>self._shape[0]-1:
            x_plus=0

        # BOTTOM
        y_minus =spin_idx[1]-1
        if  y_minus<0:
            y_minus=self._shape[0]-1
        elif y_minus>self._shape[0]-1:
            y_minus=0

        # TOP
        y_plus  =spin_idx[1]+1 
        if  y_plus<0:
            y_plus=self._shape[0]-1
        elif y_plus>self._shape[0]-1:
            y_plus=0

        neighb_list[0]=self.linear_idx((x_minus,spin_idx[1]))
        neighb_list[1]=self.linear_idx((x_plus,spin_idx[1]))
        neighb_list[2]=self.linear_idx((spin_idx[0],y_minus))
        neighb_list[3]=self.linear_idx((spin_idx[0],y_plus))

        return neighb_list

    
    def neighbours_extended(self,ispin):
        # return a list of indices pointing to the neighbours of spin ispin
        spin_idx    = self.multi_idx(ispin)
        neighb_list = np.zeros(self._nb_per_dim*self.ndim()*2,dtype=int)
   
        # LEFT
        x_minus =spin_idx[0]-1
        if  x_minus<0:
            x_minus=self._shape[0]-1 
        elif x_minus>self._shape[0]-1:
            x_minus=0

        # RIGHT
        x_plus  =spin_idx[0]+1
        if  x_plus<0:
            x_plus=self._shape[0]-1
        elif x_plus>self._shape[0]-1:
            x_plus=0

        # BOTTOM
        y_minus =spin_idx[1]-1
        if  y_minus<0:
            y_minus=self._shape[0]-1
        elif y_minus>self._shape[0]-1:
            y_minus=0

        # TOP
        y_plus  =spin_idx[1]+1 
        if  y_plus<0:
            y_plus=self._shape[0]-1
        elif y_plus>self._shape[0]-1:
            y_plus=0

        neighb_list[0]=self.linear_idx((x_minus,spin_idx[1]))
        neighb_list[1]=self.linear_idx((x_plus,spin_idx[1]))
        neighb_list[2]=self.linear_idx((spin_idx[0],y_minus))
        neighb_list[3]=self.linear_idx((spin_idx[0],y_plus))
        neighb_list[4]=self.linear_idx((x_minus,y_minus))
        neighb_list[5]=self.linear_idx((x_minus,y_plus))
        neighb_list[6]=self.linear_idx((x_plus,y_plus))
        neighb_list[7]=self.linear_idx((x_plus,y_minus))

        return neighb_list

    
    def neighbours_extended_double(self,ispin):
        # return a list of indices pointing to the neighbours of spin ispin
        spin_idx    = self.multi_idx(ispin)
        neighb_list = np.zeros(24,dtype=int)

        # LEFT
        x_minus =spin_idx[0]-1
        if  x_minus<0:
            x_minus=self._shape[0]-1 
        elif x_minus>self._shape[0]-1:
            x_minus=0

        x_minus2=x_minus-1
        if  x_minus2<0:
            x_minus2=self._shape[0]-1 
        elif x_minus2>self._shape[0]-1:
            x_minus2=0

        # RIGHT
        x_plus  =spin_idx[0]+1
        if  x_plus<0:
            x_plus=self._shape[0]-1
        elif x_plus>self._shape[0]-1:
            x_plus=0

        x_plus2=x_plus+1
        if  x_plus2<0:
            x_plus2=self._shape[0]-1
        elif x_plus2>self._shape[0]-1:
            x_plus2=0

        # BOTTOM
        y_minus =spin_idx[1]-1
        if  y_minus<0:
            y_minus=self._shape[0]-1
        elif y_minus>self._shape[0]-1:
            y_minus=0

        y_minus2=y_minus-1
        if  y_minus2<0:
            y_minus2=self._shape[0]-1
        elif y_minus2>self._shape[0]-1:
            y_minus2=0

        # TOP
        y_plus  =spin_idx[1]+1 
        if  y_plus<0:
            y_plus=self._shape[0]-1
        elif y_plus>self._shape[0]-1:
            y_plus=0

        y_plus2=y_plus+1
        if  y_plus2<0:
            y_plus2=self._shape[0]-1
        elif y_plus2>self._shape[0]-1:
            y_plus2=0

        neighb_list[0]=self.linear_idx((x_minus,spin_idx[1]))
        neighb_list[1]=self.linear_idx((x_plus,spin_idx[1]))
        neighb_list[2]=self.linear_idx((spin_idx[0],y_minus))
        neighb_list[3]=self.linear_idx((spin_idx[0],y_plus))
        neighb_list[4]=self.linear_idx((x_minus,y_minus))
        neighb_list[5]=self.linear_idx((x_minus,y_plus))
        neighb_list[6]=self.linear_idx((x_plus,y_plus))
        neighb_list[7]=self.linear_idx((x_plus,y_minus))

        neighb_list[8]=self.linear_idx((x_minus2,spin_idx[1]))
        neighb_list[9]=self.linear_idx((x_minus2,y_minus))
        neighb_list[10]=self.linear_idx((x_minus2,y_minus2))
        neighb_list[11]=self.linear_idx((x_minus2,y_plus))
        neighb_list[12]=self.linear_idx((x_minus2,y_plus2))

        neighb_list[13]=self.linear_idx((x_plus2,spin_idx[1]))
        neighb_list[14]=self.linear_idx((x_plus2,y_minus))
        neighb_list[15]=self.linear_idx((x_plus2,y_minus2))
        neighb_list[16]=self.linear_idx((x_plus2,y_plus))
        neighb_list[17]=self.linear_idx((x_plus2,y_plus2))

        neighb_list[18]=self.linear_idx((spin_idx[0],y_minus2))
        neighb_list[19]=self.linear_idx((x_minus,y_minus2))
        neighb_list[20]=self.linear_idx((x_plus,y_minus2))


        neighb_list[21]=self.linear_idx((spin_idx[0],y_plus2))
        neighb_list[22]=self.linear_idx((x_minus,y_plus2))
        neighb_list[23]=self.linear_idx((x_plus,y_plus2))

        return neighb_list
      
    def magnetization(self):
        # total magnetization of the lattice
        tot_mag = 0.0
       
        tot_mag=np.sum(self.spins()*self.spin_mag())

        return tot_mag

    def mag_change(self,ispin):
        # change in magnetization if ispin is flipped 
        dmag     = 0.0
        
        dmag     =2*self.spin(ispin)*self.spin_mag()
        return dmag


if __name__ == '__main__':

    # test flip_spin()
    lat = CubicIsingLattice(4)
    assert np.isclose(lat.spins()[1,1],1)
    lat.flip_spin(5) # row major -> second row, second column
    assert np.isclose(lat.spins()[1,1],-1)

    # for each spin, visualize its neighbours
    lat = CubicIsingLattice(4)
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(4,4)
    for ispin in range(lat.size()):

        # flip'em
        neighbours = lat.nb_list()[ispin]
        for nb in neighbours:
            lat.flip_spin(nb)

        # plot'em
        spin_idx = lat.multi_idx(ispin)
        ix,iy = spin_idx
        lat.visualize(ax[ix][iy])

        # flip'em back
        for nb in neighbours:
            lat.flip_spin(nb)

    plt.show()

