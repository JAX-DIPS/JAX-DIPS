# Simulation Parameters
LINEAR_PB                = True                                     # if True solves linear Poisson-Boltzmann, otherwise solves nonlinear PB.


# Physical constants in CGS units
T                        = 298.0                                    # Kelvin
ionic_strength           = 1.0 / 8.486902807                        # molar density of solvent mol/liter
eps_m_r                  = 1.0                                      # relative electric permittivity of molecule
eps_p_r                  = 80.0                                     # relative electric permittivity of water


kappa_sq_p               = 8.486902807 * ionic_strength             # Angstrom^-2
kappa_sq_m               = 0.0
C                        = 1.0                                      # C = e_C/(K_B T) term in the rhs of eqn 2 in Geng & Zhao's two-component formulation

# Unit conversions
e2_per_Angs_to_kcal_per_mol = 332.06364 * (T / 298.0)


