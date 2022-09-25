import numpy as np
import pdb
import matplotlib.pyplot as plt

gpu_count = np.array([1,2, 4, 8])

training_times_64 = np.array([18.16, 13.13, 8.099, 7.67]) / 20.0
total_times_64 = np.array([198.7, 164.63, 157.7, 167.33]) / 20.0

training_times_128 = np.array([139.19, 117.86, 72.58, 66.80]) / 20.0
total_times_128 = np.array([324.94, 274.31, 229.84, 224.82 ]) / 20.0

training_times_256 = np.array([1105.73, 947.2, 565.22, 535.97]) / 20.0
total_times_256 = np.array([1349, 1148.1, 747.79, 719.05]) / 20.0

training_times_512 = np.array([8769.0, 7579.50, 4534.5, 4097.5]) / 20.0
total_times_512 = np.array([9749.4, 8375.8, 5082.6, 4510.14]) / 20.0


compile_time_64  = total_times_64  - training_times_64
compile_time_128 = total_times_128 - training_times_128
compile_time_256 = total_times_256 - training_times_256
compile_time_512 = total_times_512 - training_times_512


fig, ax = plt.subplots(figsize=(8,8))

ax.plot(gpu_count, training_times_64 , color='b', linewidth=1, label=r'$\rm 64^3$')
ax.plot(gpu_count, training_times_128, color='r', linewidth=1, label=r'$\rm 128^3$')
ax.plot(gpu_count, training_times_256, color='g', linewidth=1, label=r'$\rm 256^3$')
ax.plot(gpu_count, training_times_512, color='k', linewidth=1, label=r'$\rm 512^3$')
ax.scatter(gpu_count, training_times_64, color='b')
ax.scatter(gpu_count, training_times_128, color='r')
ax.scatter(gpu_count, training_times_256, color='g')
ax.scatter(gpu_count, training_times_512, color='k')

ax.plot(gpu_count, compile_time_64 , color='b', linewidth=1, linestyle='-.', label=r'$\rm 64^3$')
ax.plot(gpu_count, compile_time_128, color='r', linewidth=1, linestyle='-.', label=r'$\rm 128^3$')
ax.plot(gpu_count, compile_time_256, color='g', linewidth=1, linestyle='-.', label=r'$\rm 256^3$')
ax.plot(gpu_count, compile_time_512, color='k', linewidth=1, linestyle='-.', label=r'$\rm 512^3$')
ax.scatter(gpu_count, compile_time_64, color='b')
ax.scatter(gpu_count, compile_time_128, color='r')
ax.scatter(gpu_count, compile_time_256, color='g')
ax.scatter(gpu_count, compile_time_512, color='k')

# ax.plot(gpu_count, total_times_64 , color='b', linewidth=1 , linestyle='-.', label=r'$\rm 64^3,\ w\ compile$')
# ax.plot(gpu_count, total_times_128, color='r', linewidth=1 , linestyle='-.', label=r'$\rm 128^3,\ w\ compile$')
# ax.plot(gpu_count, total_times_256, color='g', linewidth=1, linestyle='-.', label=r'$\rm 256^3,\ w\ compile$')
# ax.plot(gpu_count, total_times_512, color='k', linewidth=1, linestyle='-.', label=r'$\rm 512^3,\ w\ compile$')
# ax.scatter(gpu_count, total_times_64, color='b')
# ax.scatter(gpu_count, total_times_128, color='r')
# ax.scatter(gpu_count, total_times_256, color='g')
# ax.scatter(gpu_count, total_times_512, color='k')


# ax.plot(xgrid, ygrid, color='k', linestyle='--', label=r'$\mathcal{O}(N_x^{-1})$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([0.10,1e3])
ax.set_xlabel('# A100 GPUs', fontsize=20)
ax.set_ylabel('time per epoch [s]', fontsize=20)
ax.set_xticklabels(['1', '2', '', '4', '', '', '',  '8'])
ax.set_xticks([1,2, 3, 4,5, 6, 7, 8])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
ax.legend(fontsize=15, frameon=False, ncol=2)
plt.tight_layout()
plt.savefig('./scaling.png')
plt.close()

pdb.set_trace()