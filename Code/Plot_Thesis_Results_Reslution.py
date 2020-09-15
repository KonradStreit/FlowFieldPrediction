#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:39:39 2020

@author: konrad
"""


import numpy as np
import time
import matplotlib.pyplot as plt
from decimal import Decimal
from skimage.measure import block_reduce
import ot
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

import Physical_new as PC
import VortexLine as VL

# %%
def exvelo_base(xt, yt, ut, vt):
    u_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 ut.flatten(), np.vstack((xt, yt)).transpose())
    v_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 vt.flatten(), np.vstack((xt, yt)).transpose())
    return u_out, v_out


# %% Read files & Statistics
AoA = np.linspace(0, 20, 3, dtype=int)  # [0, 5, 10]
reg = 1e-6
steps = (1, 2, 3, 4, 5)
weights=(.5, .5)
vort_thr = .1
plot = True

# %%
start = time.time()
print('Reading Files')
x_full, y_full, u_full, v_full, vort_full,\
    u_std, v_std, Cont, Mom_full = PC.Read_Data(AoA)
print('Finished reading files after {:02f}s'.format(time.time()-start))

# %% Preparing Arrays

o2_Mom_full = np.zeros((len(AoA), ))
o2_Mom_OT = np.zeros((len(steps), ))


time_OT = np.zeros((len(steps), ))

x_arc, y_arc = PC.Gen_Arc_full_res(AoA[1])
Arc = VL.VortexLine(x_arc, y_arc)

# %% Calculate initial Momentum Error
mask = np.logical_and(abs(x_full)<500, abs(y_full)<500)
l = int(np.sqrt(np.sum(mask)))
x_sq = np.ones((l, ))
y_sq = np.ones((l, ))
for i in range(len(AoA)):
    u_sq = u_full[i][mask].reshape((l, l))
    v_sq = v_full[i][mask].reshape((l, l))
    vort_sq = vort_full[i][mask].reshape((l, l))
    
    Mom_sq = PC.Momentum(vort_sq, u_sq, v_sq, x_sq, y_sq)
    
    o2_Mom_full[i] = np.linalg.norm(Mom_full[i], ord=2)

# %% Loop over steps
for j, step in enumerate(steps):
    
    print("Generating Square Fields")
    x, y, u, v, vort, Mom = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=step, Mom=Mom_full)
    print("Prepping pos neg")
    start_Prep = time.time()
    vort_pos = np.zeros((2, vort.shape[1], vort.shape[2]))
    vort_neg = np.zeros((2, vort.shape[1], vort.shape[2]))
    
    vort_pos[0] = vort[0]
    vort_neg[0] = -vort[0]
    vort_pos[1] = vort[-1]
    vort_neg[1] = -vort[-1]
    
    vort_pos[vort_pos < 0] = 0
    vort_neg[vort_neg < 0] = 0
    
    sum_pos = np.zeros((2,))
    sum_neg = np.zeros((2,))
    for i in range(2):
        sum_pos[i] = np.sum(vort_pos[i])
        vort_pos[i] = vort_pos[i] / sum_pos[i]
        sum_neg[i] = np.sum(vort_neg[i])
        vort_neg[i] = vort_neg[i] / sum_neg[i]
        
    vort_pos_1 = np.array(vort[1])
    vort_neg_1 = np.array(-vort[1])
    
    vort_pos_1[vort_pos_1 < 0] = 0
    vort_neg_1[vort_neg_1 < 0] = 0
    
    sum_pos_1 = np.sum(vort_pos_1)
    sum_neg_1 = np.sum(vort_neg_1)
    
    vort_pos_1 /= sum_pos_1
    vort_neg_1 /= sum_neg_1
    
    
    dx = np.gradient(x[0, :])
    dy = np.gradient(y[:, 0])
    
    print("Prepping Done after {:.2f}s".format(time.time()-start_Prep))
    print("Starting B-S for square Sims")
    
    print('Starting OT')
    start_OT = time.time()
    vort_pos_OT = ot.bregman.convolutional_barycenter2d(vort_pos, reg, weights)
    vort_neg_OT = ot.bregman.convolutional_barycenter2d(vort_neg, reg, weights)
    vort_OT = vort_pos_OT*np.sum(weights*sum_pos)\
        - vort_neg_OT*np.sum(weights*sum_neg)
    
    time_OT[i] = time.time() - start_OT
    print('OT finished after {:.0f}mins'.format(time_OT[i]/60))
    
    # %% Velocity from Vorticity
    
    print("Starting OT u_omega")
    start_uom = time.time()
    mask_vort = abs(vort_OT) > vort_thr*np.mean(abs(vort_OT))
    u_OT_vort, v_OT_vort = PC.u_omega(x, y, x[mask_vort], y[mask_vort],
                                      vort_OT[mask_vort], h=step)
    print('Finished after {:.0}mins'.format((time.time()-start_uom)/60))

    # %% Vortex Line
    print('Creating & Solving Vortex Line')
    start_VL = time.time()
    
    exvelo_OT = lambda xl, yl: exvelo_base(xl, yl, u_OT_vort+1, v_OT_vort)
    
    gamma_OT = Arc.solve_gamma(exvelo_OT)
    
    u_OT_vl, v_OT_vl = Arc.velocity_ext(gamma_OT, x, y)
    
    
    u_OT_tot = u_OT_vort - u_OT_vl + 1
    v_OT_tot = v_OT_vort - v_OT_vl
    
    print('VL OT finished after {:.0}mins'.format((time.time()-start_VL)/60))
    
    o2_Mom_OT[i] = np.linalg.norm(PC.Momentum(vort_OT, u_OT_tot, v_OT_tot,
                                           dx, dy), ord=2)

    

# %% PLOTS

plt.figure()
plt.scatter(steps, o2_Mom_OT/o2_Mom_full[1], marker='x',
            label='{:2.0f}°'.format(AoA[1]))

plt.ylabel('Error relative to Simulation')
plt.xlabel('Coarsening factor')
plt.xticks(ticks=steps)
plt.legend()

    

