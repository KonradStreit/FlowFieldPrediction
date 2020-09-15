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
# %% Vorticity
c_m = cm.RdBu_r
ylim = .1
levs = np.linspace(-ylim, ylim, 51)
skip = 40

plt.figure()
plt.contourf(x, y, vort[1], cmap=c_m, extend='both', levels=levs)
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.axis('equal')

plt.figure()
plt.contourf(x, y, np.sqrt(u[1]**2 + v[1]**2), levels=np.linspace(0, 1.4), extend='max')
plt.colorbar()
plt.quiver(x[::skip, ::skip], y[::skip, ::skip],
           u[1][::skip, ::skip], v[1][::skip, ::skip])
plt.xticks(())
plt.yticks(())
plt.axis('equal')


# %% Arc Vortexline
skip = 20
yVL, xVL  = np.mgrid[-200:200:1, -200:200:1]
xVL = xVL.astype(float)
yVL = yVL.astype(float)
u_uni = np.ones_like(xVL, dtype=float)
v_uni = np.zeros_like(xVL, dtype=float)

x_arc, y_arc = PC.Gen_Arc(10)

Arc = VL.VortexLine(x_arc, y_arc)

exvelo = lambda x, y: (np.ones_like(x, dtype=float), np.zeros_like(y, dtype=float))

gamma = Arc.solve_gamma(exvelo)
u_indu, v_indu = Arc.velocity(gamma, xVL, yVL)

u_VL = 1 - u_indu
v_VL = -v_indu


plt.figure()
cont_VL = plt.contourf(xVL, yVL, np.sqrt(u_VL**2 + v_VL**2),
                       levels=np.linspace(0, 1.8, 19), extend='max')
plt.colorbar()
plt.quiver(xVL[::skip, ::skip], yVL[::skip, ::skip],
           u_VL[::skip, ::skip], v_VL[::skip, ::skip])
plt.plot(x_arc, y_arc)
plt.xticks(())
plt.yticks(())
plt.axis('equal')


plt.figure()
plt.contourf(xVL, yVL, np.sqrt(u_uni**2 + v_uni**2),
             levels=cont_VL.levels, extend='max')
plt.colorbar()
plt.quiver(xVL[::skip, ::skip], yVL[::skip, ::skip],
           u_uni[::skip, ::skip], v_uni[::skip, ::skip])
plt.plot(x_arc, y_arc)
plt.xticks(())
plt.yticks(())
plt.axis('equal')
