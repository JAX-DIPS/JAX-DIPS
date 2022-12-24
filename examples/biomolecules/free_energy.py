from jax import vmap, numpy as jnp
from src import interpolate, geometric_integrations_per_point
from examples.biomolecules.units import *
import pdb

"""
    * Based on equation 9 in `Eﬃcient calculation of fully resolved electrostatics around large biomolecules`
    
    * Should be based on equation 19/20 in Micu et al 1997: 
        'Numerical Considerations in the Computation of the Electrostatic Free Energy of Interaction within the Poisson-Boltzmann Theory'
"""

def get_free_energy(gstate, phi, u, uhat, atom_xyz_rad_chg):
    
    #---- First term in the equation 
    xyz, rad_chg = jnp.split(atom_xyz_rad_chg, [3], axis=1)
    sigma, chg = jnp.split(rad_chg, [1], axis=1)
    uhat_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(uhat, gstate)
    KbTper2z = K_B * T / 2.0 / z_solvent                                                                  
    sfe_component_1 = jnp.sum( uhat_interp_fn(xyz) * jnp.squeeze(chg) * KbTper2z ) * N_avogadro / (kcal_in_kJ * 1000)   # from Joules to kcal/mol
    
    
    #---- Second term in the equation 
    KbTnl3 = K_B * T * n_tilde * l_tilde**3                                                                   
    integrand = KbTnl3 * ( u * jnp.sinh(u) - 2 * (jnp.cosh(u) - 1.0) )
    
    phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(-phi, gstate)
    integral_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(integrand, gstate)
    get_vertices_of_cell_intersection_with_interface_at_point, is_cell_crossed_by_interface = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(phi_interp_fn)
    _, integrate_in_negative_domain_at_point = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_point, 
                                                                                                                 is_cell_crossed_by_interface, 
                                                                                                                 integral_interp_fn)
    sfe_component_2 = jnp.sum(vmap(integrate_in_negative_domain_at_point, (0, None, None, None))(gstate.R, gstate.dx, gstate.dy, gstate.dz))
    sfe_component_2 *= N_avogadro / (kcal_in_kJ * 1000)                                                                 # convert from Joules to kcal/mol
    
    return sfe_component_1 + sfe_component_2
    
    