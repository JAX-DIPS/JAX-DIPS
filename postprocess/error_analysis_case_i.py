import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


xgrid = np.linspace(8,128, 1000)
ygrid = 6 / xgrid

# case I
nx = np.array([8, 16, 32, 64, 128])
rmse_regress = np.array([3.7e-2, 7.1e-3, 5.9e-3, 4.1e-3, 2.64e-3])
linf_regress = np.array([3.25e-1, 1.1e-1, 8.36e-2, 6.44e-2, 3.53e-2])


fig, ax = plt.subplots(figsize=(8,8))
ax.plot(nx, rmse_regress, color='b', linewidth=2, label='RMSE, regress $\partial_n$')
ax.plot(nx, linf_regress, color='r', linewidth=2, linestyle='-.', label=r'$\rm L^\infty$, regress $\partial_n$')
ax.scatter(nx, rmse_regress, color='k', marker='D')
ax.scatter(nx, linf_regress, color='r')
ax.plot(xgrid, ygrid, color='k', linestyle='--', label=r'$\mathcal{O}(N_x^{-1})$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-3,1e0])
ax.set_xlabel(r'$\rm N_x$', fontsize=25)
ax.set_ylabel('error', fontsize=25)
plt.xticks(nx, ['8', '16', '32', '64', '128'])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.legend(fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig('../manuscript/DRM/figures/case_I_regression.png')
plt.close()


# case II
nx = np.array([8, 16, 32, 64, 128])
rmse_regress = np.array([1.36e-1, 7.98e-2, 4.36e-2, 2.43e-2, 2.64e-3])
linf_regress = np.array([1.27, 8.23e-1, 3.85e-1, 2.28e-1, 3.53e-2])


fig, ax = plt.subplots(figsize=(8,8))
ax.plot(nx, rmse_regress, color='b', linewidth=2, label='RMSE, regress $\partial_n$')
ax.plot(nx, linf_regress, color='r', linewidth=2, linestyle='-.', label=r'$\rm L^\infty$, regress $\partial_n$')
ax.scatter(nx, rmse_regress, color='k', marker='D')
ax.scatter(nx, linf_regress, color='r')
ax.plot(xgrid, ygrid, color='k', linestyle='--', label=r'$\mathcal{O}(N_x^{-1})$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-3,3e0])
ax.set_xlabel(r'$\rm N_x$', fontsize=25)
ax.set_ylabel('error', fontsize=25)
plt.xticks(nx, ['8', '16', '32', '64', '128'])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.legend(fontsize=15, frameon=False)
plt.tight_layout()
plt.savefig('../manuscript/DRM/figures/case_II_regression.png')
plt.close()