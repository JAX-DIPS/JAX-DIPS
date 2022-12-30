# Simulation Parameters
LINEAR_PB                = True                                     # if True solves linear Poisson-Boltzmann, otherwise solves nonlinear PB.


# Physical constants in SI units
C                        = 1.0                                      # C = e_C/(K_B T) term in the rhs of eqn 2 in Geng & Zhao's two-component formulation

T                        = 298.0                                    # Kelvin
ionic_strength           = 1e-3                                     # molar density of solvent mol/liter

eps_m_r                  = 1.0                                      # relative electric permittivity of molecule
eps_p_r                  = 78.54                                    # relative electric permittivity of water
kappa_sq_p               = 8.430325455 * ionic_strength / eps_p_r
kappa_sq_m               = 0.0



# Unit conversions
eC_per_Angs_to_kcal_per_mol_eC = 332.0716


