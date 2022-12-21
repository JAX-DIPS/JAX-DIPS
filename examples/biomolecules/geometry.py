from jax import random, numpy as jnp, jit
import jax
from src.jaxmd_modules.util import f32


from examples.biomolecules.load_pqr import base
from examples.biomolecules.units import *

import numpy as np
import pdb
import os
currDir = os.path.dirname(os.path.realpath(__file__))


def get_initial_level_set_fn(file_name = 'keytruda.pqr'):
    
    address = os.path.join(currDir, 'pqr_input_mols')
    bs = base(address, file_name)
    num_atoms = len(bs.atoms['x'])
    print(f'\n number of atoms = {num_atoms} \n ')
    
    atom_locations = np.stack([np.array(bs.atoms['x']), 
                                np.array(bs.atoms['y']), 
                                np.array(bs.atoms['z'])
                                ], axis=-1) * Angstroms_to_nm_coeff                  # was in Angstroms, converted to nm   
    
    sigma_i = np.array(bs.atoms['R']) * Angstroms_to_nm_coeff                       # was Angstroms, converted to nm
    sigma_s = 0.65 * Angstroms_to_nm_coeff                                          # was Angstroms, converted to nm
    atom_sigmas = sigma_i + sigma_s                                                 # is in nm
    # atom_epsilon = np.full_like(atom_sigmas, 0.039 * kcal_to_kJ_coeff )             # was kcal/mol, converted to kJ/mol >> LJ epsilon, not needed for now.
    
    atom_charges = np.array(bs.atoms['q'])                                          # partial charges, in units of electron charge e
    atom_xyz_rad = jnp.concatenate((atom_locations, atom_sigmas[..., jnp.newaxis]), axis=1)
    
    @jit
    def unperturbed_phi_fn(r):
        """
        Level-set function for the interface
        """
        x = r[0]
        y = r[1]
        z = r[2]

        def initialize(carry, xyzs):
            phi_, = carry
            xc, yc, zc, sigma = xyzs
            phi_  = jnp.min( jnp.array([ phi_, jnp.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2) - sigma ] ) )
            phi_= jnp.nan_to_num(phi_)
            return (phi_,), None

        phi_ = 1e9
        (phi_,), _ = jax.lax.scan(initialize, (phi_,), atom_xyz_rad)

        return phi_
    
    return unperturbed_phi_fn