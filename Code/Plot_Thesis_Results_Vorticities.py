#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:43:54 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ot
import time
from scipy.interpolate import griddata
from skimage.measure import block_reduce
from scipy.spatial.distance import cdist

import VortexLine as VL
import Physical_new as PC

# %%
AoA = (0, 10, 20)
weights = (.5, .5)
step = 1

# %% Read Simulation Data
x_full, y_full, u_full, v_full,\
    vort_full, u_std, v_std, Cont, Mom = PC.Read_Data(AoA, step=step)

x, y, u, v, vort = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=step)

# %% Read OT Results
x_OT = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_x.csv"
               .format(AoA[0], AoA[1], AoA[2], weights[0], weights[1]), delimiter=",")

y_OT = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_y.csv"
               .format(AoA[0], AoA[1], AoA[2], weights[0], weights[1]), delimiter=",")

vort_OT_pos = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_pos.csv"
               .format(AoA[0], AoA[1], AoA[2], weights[0], weights[1]), delimiter=",")

vort_OT_neg = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_neg.csv"
               .format(AoA[0], AoA[1], AoA[2], weights[0], weights[1]), delimiter=",")

sums = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_sums.csv"
               .format(AoA[0], AoA[1], AoA[2], weights[0], weights[1]), delimiter=",")

vort_OT = vort_OT_pos*np.sum(weights*sums[0])\
        - vort_OT_neg*np.sum(weights*sums[1])
        
# %% Calculate Linear Interpolation

vort_lin = (vort[0]*weights[0] + vort[2]*weights[1])

# %% PLOTS
vortlim = .1
nlevels = 100
c_m = cm.RdBu_r
xlim = -200
ylim = 200

plt.figure()
plt.contourf(x, y, vort[1], cmap=c_m, extend='both',
             levels=np.linspace(-vortlim, vortlim, nlevels))
plt.colorbar(ticks=np.linspace(-vortlim, vortlim, 7))

plt.axis('equal')
plt.ylim(-ylim, ylim)
plt.xlim(xlim)
plt.xticks(())
plt.yticks(())


plt.figure()
plt.contourf(x, y, vort_OT, cmap=c_m, extend='both',
             levels=np.linspace(-vortlim, vortlim, nlevels))
plt.colorbar(ticks=np.linspace(-vortlim, vortlim, 7))

plt.axis('equal')
plt.ylim(-ylim, ylim)
plt.xlim(xlim)
plt.xticks(())
plt.yticks(())

plt.figure()
plt.contourf(x, y, vort_lin, cmap=c_m, extend='both',
             levels=np.linspace(-vortlim, vortlim, nlevels))
plt.colorbar(ticks=np.linspace(-vortlim, vortlim, 7))

plt.axis('equal')
plt.ylim(-ylim, ylim)
plt.xlim(xlim)
plt.xticks(())
plt.yticks(())
