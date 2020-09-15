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
import PhysicalCalculations as PC

# %% Exvelo base
def exvelo_base(xt, yt, ut, vt):
    u_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 ut.flatten(), np.vstack((xt, yt)).transpose())
    v_out = griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                 vt.flatten(), np.vstack((xt, yt)).transpose())
    return u_out, v_out


# %%Setup 
AoA = (0, 10, 20)
n_weights = 31

temp = np.linspace(0., 1, n_weights)
weights = np.vstack((temp, 1-temp)).transpose()

step = 1
order = 2
vort_thr = .3

# %% Read Simulation Data
x_full, y_full, u_full, v_full,\
    vort_full, u_std, v_std, Cont, Mom = PC.Read_Data(AoA, step=step)

x, y, u, v, vort = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=step)

Mom_OT = np.zeros((n_weights, ))
Mom_lin = np.zeros_like(Mom_OT)
vort_OT_norm = np.zeros_like(Mom_OT)
vort_lin_norm = np.zeros_like(Mom_OT)

dx = np.gradient(x[0, :])
dy = np.gradient(y[:, 0])
Mom_sq = PC.Momentum(vort[1], u[1], v[1], dx, dy)

# %% Read OT Results
for i, w in enumerate(weights):
    x_OT = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_x.csv"
                   .format(AoA[0], AoA[1], AoA[2], w[0], w[1]), delimiter=",")
    
    y_OT = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_y.csv"
                   .format(AoA[0], AoA[1], AoA[2], w[0], w[1]), delimiter=",")
    
    vort_OT_pos = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_pos.csv"
                   .format(AoA[0], AoA[1], AoA[2], w[0], w[1]), delimiter=",")
    
    vort_OT_neg = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_neg.csv"
                   .format(AoA[0], AoA[1], AoA[2], w[0], w[1]), delimiter=",")
    
    sums = np.genfromtxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_sums.csv"
                   .format(AoA[0], AoA[1], AoA[2], w[0], w[1]), delimiter=",")
    
    vort_OT = vort_OT_pos*np.sum(w*sums[0])\
            - vort_OT_neg*np.sum(w*sums[1])
            
    vort_OT_norm[i] = np.linalg.norm(abs(vort_OT-vort[1]), ord=order)
    
    # %% Calcualte Velocities
    mask_vort = abs(vort_OT) > vort_thr*np.max(abs(vort_OT))
    u_OT_vort, v_OT_vort = PC.u_omega(x, y, x[mask_vort], y[mask_vort],
                                      vort_OT[mask_vort], h=step)
    
    print('Creating & Solving Vortex Line')
    start_VL = time.time()
    x_arc, y_arc = PC.Gen_Arc_full_res(AoA[1])
    
    Arc = VL.VortexLine(x_arc, y_arc)
    
    exvelo_OT = lambda xl, yl: exvelo_base(xl, yl, u_OT_vort+1, v_OT_vort)
    
    gamma_OT = Arc.solve_gamma(exvelo_OT)
    
    u_OT_vl, v_OT_vl = Arc.velocity_ext(gamma_OT, x, y)
    
    
    u_OT_tot = u_OT_vort - u_OT_vl + 1
    v_OT_tot = v_OT_vort - v_OT_vl
    
    Mom_OT[i] = np.linalg.norm(PC.Momentum(vort_OT, u_OT_tot, v_OT_tot,
                                           dx, dy), ord=order)
    
    # %% Calculate Linear Interpolation
    
    vort_lin = (vort[0]*w[0] + vort[2]*w[1])
    vort_lin_norm[i] = np.linalg.norm(abs(vort_lin-vort[1]), ord=order)

    # %% Calculate Velocities
    
    mask_vort = abs(vort_lin) > vort_thr*np.mean(abs(vort_lin))
    u_lin_vort, v_lin_vort = PC.u_omega(x, y, x[mask_vort], y[mask_vort],
                                        vort_lin[mask_vort], h=step)
    
    exvelo_lin = lambda xl, yl: exvelo_base(xl, yl, u_lin_vort+1, v_lin_vort)
    gamma_lin = Arc.solve_gamma(exvelo_lin)
    u_lin_vl, v_lin_vl = Arc.velocity_ext(gamma_lin, x, y)
    
    u_lin_tot = u_lin_vort - u_lin_vl + 1
    v_lin_tot = v_lin_vort - v_lin_vl
    
    Mom_lin[i] = np.linalg.norm(PC.Momentum(vort_lin, u_lin_tot, v_lin_tot,
                                            dx, dy), ord=order)
# %% PLOTS
# %% Vorticity & Momentum
f, ax = plt.subplots(2, 1, sharex=True)
# f.suptitle("{:.0f}° - {:.0f}° - {:.0f}°, Momentum and Vorticity Error".format(AoA[0], AoA[1], AoA[2]))

ax[0].plot(weights[:][:, 0], Mom_OT/np.linalg.norm(Mom_sq, ord=order),
           'b', label='OT')
ax[0].plot(weights[:][:, 0], Mom_lin/np.linalg.norm(Mom_sq, ord=order),
           'r', label='Linear')
ax[0].scatter(weights[Mom_OT.argmin(), 0],
              np.min(Mom_OT)/np.linalg.norm(Mom_sq, ord=order),
              marker='x', color='b')
ax[0].scatter(weights[Mom_lin.argmin(), 0],
              np.min(Mom_lin)/np.linalg.norm(Mom_sq, ord=order),
           marker='x', color='r')
ax[0].legend()

ax[1].plot(weights[:][:, 0], vort_OT_norm/np.linalg.norm(vort[1], ord=order),
           'b', label='Vorticity OT')
ax[1].plot(weights[:][:, 0], vort_lin_norm/np.linalg.norm(vort[1], ord=order),
           'r', label='Vorticity Linear')
ax[1].scatter(weights[vort_OT_norm.argmin(), 0],
              np.min(vort_OT_norm)/np.linalg.norm(vort[1], ord=order),
              marker='x', color='b')
ax[1].scatter(weights[vort_lin_norm.argmin(), 0],
              np.min(vort_lin_norm)/np.linalg.norm(vort[1], ord=order),
              marker='x', color='r')

ax[1].set_xlabel("Weight {:.0f}° Sim".format(AoA[0]))
ax[1].set_ylabel("Vorticity Error")
ax[0].set_ylabel("Momentum Error")


# %% Momentum
f, ax = plt.subplots(1, 1, sharex=True)
# f.suptitle("{:.0f}° - {:.0f}° - {:.0f}°, Momentum and Vorticity Error".format(AoA[0], AoA[1], AoA[2]))

ax.plot(weights[:][:, 0], Mom_OT/np.linalg.norm(Mom_sq, ord=order),
           'b', label='OT')
ax.plot(weights[:][:, 0], Mom_lin/np.linalg.norm(Mom_sq, ord=order),
           'r', label='Linear')
ax.scatter(weights[Mom_OT.argmin(), 0],
              np.min(Mom_OT)/np.linalg.norm(Mom_sq, ord=order),
              marker='x', color='b')
ax.scatter(weights[Mom_lin.argmin(), 0],
              np.min(Mom_lin)/np.linalg.norm(Mom_sq, ord=order),
           marker='x', color='r')
ax.legend()

ax.set_xlabel("Weight {:.0f}° Sim".format(AoA[0]))
ax.set_ylabel("Momentum Error")
