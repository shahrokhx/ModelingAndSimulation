import numpy as np
from copy import deepcopy

class IsingHamiltonian:

    def __init__(self,J,lat):
        self._isingJ   = J      # OK to use one-letter variable once in constructor
        self._lat_ref = lat     # keep a reference to an IsingLattice

    def isingJ(self):
        return self._isingJ

    def lattice(self,copy=True):
        if copy:
            return deepcopy(self._lat_ref)
        else:
            return self._lat_ref

    def compute_energy(self):
        
        """ calculate the total energy of the tracked ising lattice
        note: the state of the spins (up/down) is stored in lattice.spins(), and the magnitude of the spins are stored separately in lattice.spin_mag() """
        tot_energy = 0.0
        H          = 0.0
        for ii in range(self._lat_ref.size()):  # iath spin
            for kk in range(self._lat_ref._nb_per_dim*self._lat_ref._ndim): 
                jj=self._lat_ref.nb_list()[ii][kk]  # jth spin
                sisj=self._lat_ref.spin(ii)* self._lat_ref.spin(jj)*(self._lat_ref.spin_mag()**2)    ###si * sj
                H   = H + sisj
        tot_energy = - self._isingJ*H/2.0       
   
        return tot_energy

    def compute_spin_energy(self,ispin):
        """ calculate the energy associated with one spin
        note: the state of the spins (up/down) is stored in lattice.spins(), and the magnitude of the spins are stored separately in lattice.spin_mag() """
        energy = 0.0
        for kk in range(self._lat_ref._nb_per_dim*self._lat_ref._ndim):
            jj=self._lat_ref.nb_list()[ispin][kk]  # jth spin
            sisj=self._lat_ref.spin(ispin)* self._lat_ref.spin(jj)*(self._lat_ref.spin_mag()**2)    ###si * sj
            energy = energy -  self._isingJ*sisj

        return energy


if __name__ == '__main__':
    from cubic_ising_lattice import CubicIsingLattice

    # 16 spins -> 32 n.n. bonds -> each bond holds -0.25 energy -> -8 total
    lat = CubicIsingLattice(4,spin_mag=0.5)
    ham = IsingHamiltonian(1.0,lat)
    assert np.isclose(-8,ham.compute_energy())

    # turn 4 bonds to 4 anti-bonds -> cost -0.25*4*2 = -2.0 energy
    lat.flip_spin(0)
    assert np.isclose(-6,ham.compute_energy())

