from jax import vmap, numpy as jnp
from src import interpolate, geometric_integrations_per_point
from examples.biomolecules_GZ17.units import *
import pdb
from examples.biomolecules_GZ17.coefficients import get_rho_fn
"""
    * Based on equation 9 in `Eﬃcient calculation of fully resolved electrostatics around large biomolecules`
    
    * Should be based on equation 19/20 in Micu et al 1997: 
        'Numerical Considerations in the Computation of the Electrostatic Free Energy of Interaction within the Poisson-Boltzmann Theory'
"""

def get_free_energy(gstate, phi, psi_hat, atom_xyz_rad_chg, epsilon_grad_psi_sq, psi_solution, epsilon_grad_psi_star_sq, epsilon_grad_psi_hat_sq):
    
    #---- First term in the equation 
    xyz, rad_chg = jnp.split(atom_xyz_rad_chg, [3], axis=1)
    sigma, chg = jnp.split(rad_chg, [1], axis=1)
    
    psi_hat_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(psi_hat, gstate)                                
                                      
    sfe = jnp.sum( psi_hat_interp_fn(xyz) * jnp.squeeze(chg)  ) * e2_per_Angs_to_kcal_per_mol   # convert to kcal/mol/eC
    
    
    
    
    # # method 2:
    # epsilon_E_sq = 0.5 * epsilon_grad_psi_sq - 0.5 * epsilon_grad_psi_star_sq
    # E_correction = epsilon_E_sq.sum() * (gstate.dx*gstate.dy*gstate.dz) * e2_per_Angs_to_kcal_per_mol * 1e-10
    # sfe_2 = sfe - E_correction
    
    # method 3
    # rho_fn = get_rho_fn(atom_xyz_rad_chg)
    # chg_density = vmap(rho_fn)(gstate.R)
    # integrand = chg_density * psi_solution - epsilon_E_sq
    # integral_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(integrand, gstate)
    
    # phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(phi, gstate)
    # get_vertices_of_cell_intersection_with_interface_at_point, is_cell_crossed_by_interface = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(phi_interp_fn)
    # _, integrate_in_negative_domain_at_point = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_point, is_cell_crossed_by_interface, integral_interp_fn)
    
    # phi_p_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(-phi, gstate)
    # get_vertices_of_cell_intersection_with_interface_at_point, is_cell_crossed_by_interface = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(phi_p_interp_fn)
    # _, integrate_in_positive_domain_at_point = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(get_vertices_of_cell_intersection_with_interface_at_point, is_cell_crossed_by_interface, integral_interp_fn)
    
    
    # partial_integrals_negative = vmap(integrate_in_negative_domain_at_point, (0, None, None, None))(gstate.R, gstate.dx, gstate.dy, gstate.dz)
    # partial_integrals_positive = vmap(integrate_in_positive_domain_at_point, (0, None, None, None))(gstate.R, gstate.dx, gstate.dy, gstate.dz)
    # sfe_3 = ( jnp.sum(partial_integrals_negative) + jnp.sum(partial_integrals_positive) ) * eC_per_Angs_to_kcal_per_mol_eC
    
    
    
    return sfe
    
    