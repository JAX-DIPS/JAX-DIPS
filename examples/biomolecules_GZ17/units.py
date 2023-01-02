# Simulation Parameters
LINEAR_PB                = True                                     # if True solves linear Poisson-Boltzmann, otherwise solves nonlinear PB.


# Physical constants in CGS units
T                        = 298.0                                    # Kelvin
ionic_strength           = 9.48955                                     # molar density of solvent mol/liter = mol/cm^3
eps_m_r                  = 1.0                                      # relative electric permittivity of molecule
eps_p_r                  = 80.0                                     # relative electric permittivity of water



## CGS system.
PI = 3.141592653589793
e_C = 4.8032424e-10     # in esu. Charge unit is esu in cgs system
Kb = 1.3806620e-16      # erg/K
erg_per_kcal = 4.184e10  # 1 kcal = 4.184e10 erg
N_A = 6.0220450e23      # Avogadro number
epsilon_0 = 1.0         # in CGS epsilon_0 is 1
e2_per_Angs_to_kcal_per_mol = e_C**2 * N_A * 1e8 / erg_per_kcal  # 1 esu^2/A = 332.06364 kcal/mol

# 1 erg = 1e-8 esu^2/Angstrom -> esu/erg = 1e8 Angstrom/esu

KbT_in_kcal_per_mol = N_A * Kb * T / erg_per_kcal      # in kcal/mol; KbT  = 0.592183 * (T/298.0)  in kcal/mol



kappa_sq_in_angs2 = 1e-16 * ( 8 * PI * e_C**2 * N_A / (1000 * Kb * T) ) * ionic_strength / eps_p_r   # this is in units of Angstrom^-2
kappa_bar_sq_p               = kappa_sq_in_angs2 * eps_p_r                                               # equivalent to 8.486902807 * ionic_strength in Angstrom^-2
kappa_bar_sq_m               = 0.0


# The point charges come into PBE with coefficient:
rhs_esu2_per_erg = 4*PI*e_C**2/(Kb*T)       # in esu^2/erg; useful in sum{ 4*PI*e_C**2/(Kb*T) z_i * delta(x-x_i)}
rhs_per_Angs2 = rhs_esu2_per_erg * 1e8      # is 7046.5288 Angs^-2