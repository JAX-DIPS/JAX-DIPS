from jax import vmap, numpy as jnp
from src import interpolate, geometric_integrations_per_point
from examples.biomolecules_GZ17.units import *
import pdb
import numpy as onp
"""
    * Based on equation 9 in `Eﬃcient calculation of fully resolved electrostatics around large biomolecules`
    
    * Should be based on equation 19/20 in Micu et al 1997: 
        'Numerical Considerations in the Computation of the Electrostatic Free Energy of Interaction within the Poisson-Boltzmann Theory'
"""

def get_free_energy(gstate, psi_hat, atom_xyz_rad_chg):
    
    #---- First term in the equation 
    xyz, rad_chg = jnp.split(atom_xyz_rad_chg, [3], axis=1)
    sigma, chg = jnp.split(rad_chg, [1], axis=1)
    
    psi_hat_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(psi_hat, gstate)                                
                                      
    sfe = jnp.sum( psi_hat_interp_fn(xyz) * jnp.squeeze(chg)  ) * eC_per_Angs_to_kcal_per_mol_eC   # convert to kcal/mol/eC
    return sfe
    
    