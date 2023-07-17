# https://github.com/bzlu-Group/INN/blob/main/INN_E4/main.py

kbT = 0.592783
Is = 1e-5
PI = 3.141592653589793

# Physical constants in dimensionless units
alpha_1 = 2.0  # relative electric permittivity of molecule
alpha_2 = 80.0  # relative electric permittivity of water

beta_1 = 0.0
kappa = (8.4869 * Is / alpha_2) ** 0.5  # this is dimensionless
beta_2 = alpha_2 * kappa**2

omega = 1.0  # originally was 7.0465e3, now solution is per omega. Absolute errors in solution should be multiplied by 7.0465e3 at the end.
