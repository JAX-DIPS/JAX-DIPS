# Simulation Parameters
LINEAR_PB                = True                                     # if True solves linear Poisson-Boltzmann, otherwise solves nonlinear PB.

# For scaling length
l_tilde                  = 1.0e-8                                  # l tilde in meters

# Physical constants in SI units
N_avogadro               = 6.022e23                                 # number per 1 mole
K_B                      = 1.380649e-23                             # m^2 kh s^-2 K^-1
e_tilde                  = 1.60217663e-19                           # Coloumbs
epsilon_0                = 8.8541878e-12                            # in F/m = 5.7276574e-4 (mol.e^2/kJ/nm)
eps_m_r                  = 1.0                                      # relative electric permittivity of molecule
eps_s_r                  = 78.54                                    # relative electric permittivity of water
eps_m                    = eps_m_r * epsilon_0
eps_s                    = eps_s_r * epsilon_0


# Unit conversions
Angstrom_in_m            = 1e-10                                    # 1 Ang = 1e-10 m
Angstrom_in_nm           = 0.1                                      # conversion factor from Ang to nm
kcal_in_kJ               = 4.184                                    # conversion factor from kcal to kJ
liter_in_nm_cubed        = 1e24
liter_in_m_cubed         = 1e-3
nm_in_m                  = 1e-9


# Solvent parameters
import numpy as np

T                        = 298.15                                                         # Kelvin
molar_density            = 1e-6                                                           # molar density of solvent mole/liter
n_tilde                  = molar_density * (N_avogadro / liter_in_m_cubed)                # number density = number/m^3. 
z_solvent                = 1.0                                                            # overall charge of solvent in electron units
solvent_chg_tilde        = z_solvent * e_tilde    
lambda_tilde = np.sqrt( epsilon_0 * K_B * T / (2*z_solvent**2 * e_tilde**2 * n_tilde) )   # should be 1.08575 * Angstrom_in_m for an ionic strength of 0.1 molar
kappa_p_sq               = (l_tilde / lambda_tilde)**2                                    # Square of nondimensionalized coeff in nonlinear PB term in \Omega^+
kappa_m_sq               = 0.0                                                            # Square of nondimensionalized coeff in nonlinear PB term in \Omega^-


# # For nonpolar terms (excluded for now)
# water_density            = 33.4567                                  # number / nm^3
# rho_0                    = water_density / nm_in_m**3               # number/m^3
# bulk_density_coefficient = 2                                        # rho_0 / y
# Y                        = 27.893                                   # kJ/mol/nm^2
# pressure_coeff           = 0.2                                      # p / y



