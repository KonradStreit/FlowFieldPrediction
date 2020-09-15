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
from matplotlib import cm



import VortexLine as VL
import Physical_new as PC

# %%
def exvelo_base(xt, yt, ut, vt):
    u_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 ut.flatten(), np.vstack((xt, yt)).transpose())
    v_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 vt.flatten(), np.vstack((xt, yt)).transpose())
    return u_out, v_out

# %% Setup

AoA = (0, 5, 10)


weights = (.5, .5)

step = 2
reg = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
order = 2

vort_thr = .3

# %% Read Data
x_full, y_full, u_full, v_full,\
    vort_full, u_std, v_std, Cont, Mom = PC.Read_Data(AoA, step=step)


# %%
x, y, u, v, vort = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=2)


dx = np.gradient(x[0, :])
dy = np.gradient(y[:, 0])

Mom_sq = np.linalg.norm(PC.Momentum(vort[1], u[1], v[1], dx, dy), ord=order)
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


Mom_OT = np.zeros((len(reg), ))
time_OT = np.zeros_like(Mom_OT)
vort_OT_norm = np.zeros_like(Mom_OT)

u_OT_tot = np.zeros((len(reg), u.shape[1], u.shape[2]))
v_OT_tot = np.zeros_like(u_OT_tot)
vort_OT = np.zeros_like(u_OT_tot)

# %% OT
x_arc, y_arc = PC.Gen_Arc_full_res(AoA[1])
Arc = VL.VortexLine(x_arc, y_arc)
    
for i, r in enumerate(reg):
    print('Starting OT')
    start_OT = time.time()
    vort_pos_OT = ot.bregman.convolutional_barycenter2d(vort_pos, r, weights)
    vort_neg_OT = ot.bregman.convolutional_barycenter2d(vort_neg, r, weights)
    vort_OT[i] = vort_pos_OT*np.sum(weights*sum_pos)\
        - vort_neg_OT*np.sum(weights*sum_neg)
    
    time_OT[i] = time.time() - start_OT
    print('OT finished after {:.0f}mins'.format((time.time()-start_OT)/60))
    
    vort_OT_norm[i] = np.linalg.norm(abs(vort_OT[i]-vort[1]), ord=order)


    # %% Velocity from Vorticity
    
    print("Starting OT u_omega")
    start_uom = time.time()
    mask_vort = abs(vort_OT[i]) > vort_thr*np.mean(abs(vort_OT[i]))
    u_OT_vort, v_OT_vort = PC.u_omega(x, y, x[mask_vort], y[mask_vort],
                                      vort_OT[i][mask_vort], h=step)
    print('Finished after {:.0}mins'.format((time.time()-start_uom)/60))

    
    # %% Vortex Line
    print('Creating & Solving Vortex Line')
    start_VL = time.time()
    
    exvelo_OT = lambda xl, yl: exvelo_base(xl, yl, u_OT_vort+1, v_OT_vort)
    
    gamma_OT = Arc.solve_gamma(exvelo_OT)
    
    u_OT_vl, v_OT_vl = Arc.velocity_ext(gamma_OT, x, y)
    
    
    u_OT_tot[i] = u_OT_vort - u_OT_vl + 1
    v_OT_tot[i] = v_OT_vort - v_OT_vl
    
    print('VL OT finished after {:.0}mins'.format((time.time()-start_VL)/60))
    
    Mom_OT[i] = np.linalg.norm(PC.Momentum(vort_OT[i], u_OT_tot[i], v_OT_tot[i],
                                           dx, dy), ord=order)


# %% PLOTS
# %% Error & Time Plots
f, ax = plt.subplots()
# f.suptitle("Regularization Investigation")
ax.plot(reg, vort_OT_norm/np.min(vort_OT_norm), 'b')
ax.tick_params(axis='y', labelcolor='b')
ax.set_ylabel('Vorticity Error', color='b')
ax.set_xlabel('Regularization')
ax.set_xscale('log')
# ax.set_yscale('log')

ax2 = ax.twinx()
ax2.plot(reg, time_OT/np.min(time_OT), 'r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylabel('Calculation Time', color='r')
ax2.set_yscale('log')


# %% Velocity Plots
skip = 15
c_m = cm.RdBu
for i, r in enumerate(reg):
    plt.figure()
    # plt.title('Reg = {:0}'.format(r))
    plt.contourf(x, y, np.sqrt(u_OT_tot[i]**2 + v_OT_tot[i]**2), cmap=c_m)
    plt.colorbar()
    plt.quiver(x[::skip, ::skip], y[::skip, ::skip],
               u_OT_tot[i][::skip, ::skip], v_OT_tot[i][::skip, ::skip])


# %% Vorticity Plots
for i, r in enumerate(reg):
    lim = np.minimum(abs(np.max(vort_OT[i])), abs(np.min(vort_OT[i])))
    lev = np.linspace(-lim, lim)
    plt.figure()
    # plt.title('Reg = {:0}'.format(r))
    plt.contourf(x, y, vort_OT[i], cmap=c_m, extend='both', levels=lev)
    plt.colorbar(format='%.2e')
        
    
