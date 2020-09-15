#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:14:00 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
import time
from scipy.interpolate import griddata
from skimage.measure import block_reduce
from scipy.spatial.distance import cdist

import VortexLine as VL
import PhysicalCalculations as PC

# %% exvelo_base
def exvelo_base(xt, yt, ut, vt):
    u_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 ut.flatten(), np.vstack((xt, yt)).transpose())
    v_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 vt.flatten(), np.vstack((xt, yt)).transpose())
    return u_out, v_out

# %% Setup

AoA = (0, 5, 10)

n_Thr = 61

weights = (.5, .5)

step = 1
reg = 1e-5
order = 2

vort_thr = np.linspace(.001, 1, n_Thr)

# %% Read Data
x_full, y_full, u_full, v_full,\
    vort_full, u_std, v_std, Cont, Mom_full = PC.Read_Data(AoA)


# %%
x, y, u, v, vort, Mom = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=step, Mom=Mom_full)


dx = np.gradient(x[0, :])
dy = np.gradient(y[:, 0])

Mom_sq = PC.Momentum(vort[1], u[1], v[1], dx, dy)
o2_Mom_sq = np.linalg.norm(Mom_sq, ord=2)
# %% Divide Pos & Neg


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

# %% Empty arrays
Mom_OT = np.zeros((n_Thr, ))
Mom_OT_vort = np.zeros_like(Mom_OT)
vel_OT = np.zeros_like(Mom_OT)

vel_lin = np.zeros_like(Mom_OT)
Mom_lin = np.zeros_like(Mom_OT)
time_OT = np.zeros_like(Mom_OT)
time_lin = np.zeros_like(Mom_OT)
n_OT = np.zeros_like(Mom_OT)
n_lin = np.zeros_like(Mom_OT)


u_OT_tot = np.zeros((n_Thr, u.shape[1], u.shape[2]))
v_OT_tot = np.zeros_like(u_OT_tot)
u_OT_vort = np.zeros_like(u_OT_tot)
v_OT_vort = np.zeros_like(u_OT_tot)
u_lin_tot = np.zeros_like(u_OT_tot)
v_lin_tot = np.zeros_like(u_OT_tot)

mask_vort_OT = np.zeros_like(u_OT_tot, dtype=bool)
mask_vort_lin = np.zeros_like(u_OT_tot, dtype=bool)

# %% OT

print('Starting OT')
start_OT = time.time()
vort_pos_OT = ot.bregman.convolutional_barycenter2d(vort_pos, reg, weights)
vort_neg_OT = ot.bregman.convolutional_barycenter2d(vort_neg, reg, weights)
vort_OT = vort_pos_OT*np.sum(weights*sum_pos)\
    - vort_neg_OT*np.sum(weights*sum_neg)

print('OT finished after {:.0f}mins'.format((time.time()-start_OT)/60))

vort_OT_norm = np.linalg.norm(abs(vort_OT-vort[1]), ord=order)


# %% Linear
print('Calculating Linear Interpolation')
start_lin = time.time()
vort_lin = vort[0]*weights[0] + vort[2]*weights[1]

print('Lin finished after {:.0f}s'.format((time.time()-start_lin)))
vort_lin_norm = np.linalg.norm(abs(vort_lin-vort[1]), ord=order)


# %% Initialize Arc
x_arc, y_arc = PC.Gen_Arc_full_res(AoA[1])
dist = np.zeros_like(x)
for i in range(len(x)):
    dist[i] = np.min(cdist(np.vstack((x_arc, y_arc)).transpose(),
                           np.vstack((x[i], y[i])).transpose()), axis=0)

Arc = VL.VortexLine(x_arc, y_arc)
    
# %% Velocity from Vorticity
for i, Thr in enumerate(vort_thr):
    
    print("Starting OT u_omega")
    start_uom = time.time()
    mask_vort_OT[i] = abs(vort_OT) > Thr*np.max(abs(vort_OT))
    n_OT[i] = np.sum(mask_vort_OT[i])
    u_OT_vort[i], v_OT_vort[i] = PC.u_omega(x, y, x[mask_vort_OT[i]], y[mask_vort_OT[i]],
                                      vort_OT[mask_vort_OT[i]], h=step)
    time_OT[i] = time.time() - start_uom
    print('Finished after {:.0}mins'.format((time.time()-start_uom)/60))
    Mom_OT_vort[i] = np.linalg.norm(PC.Momentum(vort_OT, u_OT_vort[i]+1, v_OT_vort[i],
                                           dx, dy), ord=order)
    
    
    
    # %% Vortex Line
    print('Creating & Solving Vortex Line')
    start_VL = time.time()
    
    exvelo_OT = lambda xl, yl: exvelo_base(xl, yl, u_OT_vort[i]+1, v_OT_vort[i])
    
    gamma_OT = Arc.solve_gamma(exvelo_OT)
    
    u_OT_vl, v_OT_vl = Arc.velocity(gamma_OT, x, y)
    
    
    u_OT_tot[i] = u_OT_vort[i] - u_OT_vl + 1
    v_OT_tot[i] = v_OT_vort[i] - v_OT_vl
    
    print('VL OT finished after {:.0}mins'.format((time.time()-start_VL)/60))
    
    # Mom_OT[i] = np.linalg.norm(PC.Momentum(vort_OT, u_OT_tot[i], v_OT_tot[i],
    #                                        dx, dy), ord=order)
    
    Mom_OT[i] = np.linalg.norm(PC.Momentum(vort_OT, u_OT_tot[i], v_OT_tot[i],
                                           dx, dy), ord=order)
    
    vel_sum = np.sqrt(((u_OT_tot[i]-1))**2 + v_OT_tot[i]**2)
    vel_OT[i] = np.linalg.norm(abs(vel_sum - np.sqrt((u[1]-1)**2 + v[1]**2)))
    
# %% Lin    

    print("Starting Lin u_omega")
    start_uom = time.time()
    mask_vort_lin[i] = abs(vort_lin) > Thr*np.mean(abs(vort_lin))
    n_lin[i] = np.sum(mask_vort_lin[i])
    u_lin_vort, v_lin_vort = PC.u_omega(x, y, x[mask_vort_lin[i]], y[mask_vort_lin[i]],
                                        vort_lin[mask_vort_lin[i]], h=step)
    time_lin[i] = time.time() - start_uom
    print('Finished after {:.0}mins'.format((time.time()-start_uom)/60))
    

    exvelo_lin = lambda xl, yl: exvelo_base(xl, yl, u_lin_vort+1, v_lin_vort)
    gamma_lin = Arc.solve_gamma(exvelo_lin)
    u_lin_vl, v_lin_vl = Arc.velocity(gamma_lin, x, y)
    
    u_lin_tot[i] = u_lin_vort - u_lin_vl + 1
    v_lin_tot[i] = v_lin_vort - v_lin_vl
    
    Mom_lin[i] = np.linalg.norm(PC.Momentum(vort_lin, u_lin_tot[i], v_lin_tot[i],
                                            dx, dy), ord=order)
    
    vel_sum = np.sqrt((u_lin_tot[i])**2 + v_lin_tot[i]**2)
    vel_lin[i] = np.linalg.norm(abs(vel_sum - np.sqrt(u[1]**2 + v[1]**2)))
    

# %% PLOTS
# %% Velocity Error plot
start = 1
f, ax = plt.subplots()

ax.plot(vort_thr[start:], vel_OT[start:]/np.min(vel_OT[start:]), 'b')

ax0 = ax.twinx()
ax0.plot(vort_thr[start:], n_OT[start:], 'r')

ax.tick_params(axis='y', labelcolor='b')
ax0.tick_params(axis='y', labelcolor='r')

ax.set_ylabel('Velocity Error', color='b')
ax.set_xlabel('Threshold')
ax0.set_ylabel('No. Points', color='r')
ax.ticklabel_format(useOffset=False)

# %% Momentum Error plot
start = 1
f, ax = plt.subplots()

ax.plot(vort_thr[start:], Mom_OT[start:]/np.min(Mom_OT[start:]), 'b')

ax0 = ax.twinx()
ax0.plot(vort_thr[start:], n_OT[start:], 'r')

ax.tick_params(axis='y', labelcolor='b')
ax0.tick_params(axis='y', labelcolor='r')

ax.set_ylabel('Momentum Error', color='b')
ax.set_xlabel('Threshold')
ax0.set_ylabel('No. Points', color='r')
ax.ticklabel_format(useOffset=False)

