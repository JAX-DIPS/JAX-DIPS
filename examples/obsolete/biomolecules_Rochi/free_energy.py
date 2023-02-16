from jax import vmap, numpy as jnp
from jax_dips import interpolate, geometric_integrations_per_point
from examples.obsolete.biomolecules_Rochi.units import *
import pdb
import numpy as onp

"""
    * Based on equation 9 in `Eﬃcient calculation of fully resolved electrostatics around large biomolecules`

    * Should be based on equation 19/20 in Micu et al 1997:
        'Numerical Considerations in the Computation of the Electrostatic Free Energy of Interaction within the Poisson-Boltzmann Theory'
"""


def get_free_energy(gstate, phi, psi, psi_hat, atom_xyz_rad_chg):
    # ---- First term in the equation
    KbTper2z = K_B * T / 2.0 / z_solvent

    xyz, rad_chg = jnp.split(atom_xyz_rad_chg, [3], axis=1)
    sigma, chg = jnp.split(rad_chg, [1], axis=1)
    psi_hat_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(psi_hat, gstate)
    sfe_component_1 = (
        jnp.sum(psi_hat_interp_fn(xyz) * jnp.squeeze(chg) * KbTper2z) * N_avogadro / (kcal_in_kJ * 1000)
    )  # from Joules to kcal/mol

    sfe_3 = sfe_component_1 * 2.0

    # ---- Second term in the equation
    KbTnl3 = K_B * T * n_tilde * l_tilde**3

    diag = 0.0  # jnp.sqrt(gstate.dx**2 + gstate.dy**2 + gstate.dz**2)
    mask_p = 0.5 * (jnp.sign(phi - diag) + 1)
    psi_clipped = mask_p * psi_hat
    integrand = psi_clipped * KbTnl3 * onp.sinh(psi_clipped) - 2 * KbTnl3 * (onp.cosh(psi_clipped) - 1.0)

    phi_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(-phi, gstate)
    integral_interp_fn = interpolate.nonoscillatory_quadratic_interpolation(integrand, gstate)
    (
        get_vertices_of_cell_intersection_with_interface_at_point,
        is_cell_crossed_by_interface,
    ) = geometric_integrations_per_point.get_vertices_of_cell_intersection_with_interface(phi_interp_fn)
    (
        _,
        integrate_in_negative_domain_at_point,
    ) = geometric_integrations_per_point.integrate_over_gamma_and_omega_m(
        get_vertices_of_cell_intersection_with_interface_at_point,
        is_cell_crossed_by_interface,
        integral_interp_fn,
    )
    partial_integrals = vmap(integrate_in_negative_domain_at_point, (0, None, None, None))(
        gstate.R, gstate.dx, gstate.dy, gstate.dz
    )
    sfe_component_2 = (
        jnp.sum(partial_integrals * mask_p) * N_avogadro / (kcal_in_kJ * 1000)
    )  # convert from Joules to kcal/mol

    return sfe_component_1, sfe_component_2, sfe_3
