import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb


# case I
nx = np.array([8, 16, 32, 64, 128])
rmse_regress = np.array([3.7e-2, 7.1e-3, 5.9e-3, 4.1e-3, 2.64e-3])
linf_regress = np.array([3.25e-1, 1.1e-1, 8.36e-2, 6.44e-2, 3.53e-2])


fig, ax = plt.subplots(figsize=(8,8))
ax.plot(nx, rmse_regress, color='b', linewidth=2, label='RMSE, regress $\partial_n$')
ax.plot(nx, linf_regress, color='r', linewidth=2, linestyle='-.', label=r'$\rm L^\infty$, regress $\partial_n$')
ax.scatter(nx, rmse_regress, color='k', marker='D')
ax.scatter(nx, linf_regress, color='r')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([1e-3,1e0])
ax.set_xlabel(r'$\rm N_x$', fontsize=25)
ax.set_ylabel('error', fontsize=25)

plt.xticks(nx, ['8', '16', '32', '64', '128'])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig('../manuscript/figures/case_I_regression.png')