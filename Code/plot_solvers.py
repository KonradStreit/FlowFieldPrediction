"""
Created on Fri May  1 17:19:15 2020

@author: konrad
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

resolutions = np.array((50**2, 100**2, 150**2, 200**2, 300**2, 400**2, 500**2,
                        600**2, 700**2, 800**2, 1000**2))
Dense_totals = np.array((0.211, 6.356, 79.737, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, np.nan))
Sparse_totals = np.array((0.003, 0.033, 0.080, 0.236, 0.478, 0.976, 1.782, 3.7, 4.813, 7.061, 14.890))
Sparse_solver = np.array((0.001, 0.03, 0.076, 0.227, 0.463, 0.949, 1.738, 3.603, 4.698, 6.894, 14.697))
Banded_totals = np.array((0.002, 0.042, 0.123, 0.318, 1.24, 2.528, 5.507, np.nan, np.nan, np.nan, np.nan))
Banded_solver = np.array((0.001, 0.035, 0.097, 0.261, 1.054, 2.084, 4.669, np.nan, np.nan, np.nan, np.nan))

plt.figure()
plt.title('Total Times')
plt.plot(resolutions, Sparse_totals, 'x-', label='Sparse')
plt.plot(resolutions, Banded_totals, 'x-', label='Banded')
plt.ylabel('Time [s]')
plt.xlabel('Number of points')
plt.legend()

plt.figure()
plt.title('Solver Times')
plt.plot(resolutions, Sparse_solver, 'x--', label='Sparse')
plt.plot(resolutions, Banded_solver, 'x--', label='Banded')
plt.ylabel('Time [s]')
plt.xlabel('Number of points')
plt.legend()

# %%
plt.figure()
plt.title('Times')
plt.plot(resolutions, Sparse_totals, 'rx-', label='Sparse Total')
plt.plot(resolutions, Sparse_solver, 'ro--', label='Sparse Solver')
plt.plot(resolutions, Banded_totals, 'gx-', label='Banded Total')
plt.plot(resolutions, Banded_solver, 'go--', label='Banded Solver')
plt.ylabel('Time [s]')
plt.xlabel('Number of points')
plt.legend()

# %%
f, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(resolutions, Sparse_totals, 'rx-', label='Sparse Total')
ax1.plot(resolutions, Banded_totals, 'gx-', label='Banded Total')
ax2.plot(resolutions, (resolutions**2))


# %%
f, ax1 = plt.subplots()
# ax2 = ax1.twinx()
ax1.loglog(resolutions, Dense_totals, 'rx-', label='Dense')
ax1.loglog(resolutions, Sparse_totals, 'rx-', label='Sparse')
ax1.loglog(resolutions, Banded_totals, 'gx-', label='Banded')
ax1.loglog(resolutions, (resolutions**2)*1e-10, 'k--', label='$N²$')
ax1.loglog(resolutions, (resolutions**3)*1e-15, 'b--', label='$N³$')
ax1.loglog(resolutions, (resolutions*np.log(resolutions))*5e-7, 'y--', label='$N\logN$')

plt.legend()
plt.ylabel('Time')
plt.xlabel('Resolution')
plt.grid(which='minor')