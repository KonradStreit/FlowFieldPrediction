#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:07:34 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

import PhysicalCalculations as PC
import VortexLine as VL

# %%

AoA = (0, 5, 10)
weights = (.5, .5)
Thr = .3

x_full, y_full, u_full, v_full, vort_full, u_std, v_std, Cont_full, Mom_full = \
PC.Read_Data(AoA)

x, y, u, v, vort = PC.make_square(x_full, y_full, u_full, v_full, vort_full, 1000, step=1)

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


# %%
mask_OT = abs(vort_OT) > Thr * np.max(abs(vort_OT))
u_OT, v_OT = PC.u_omega(x_OT, y_OT, x_OT[mask_OT],
                        y_OT[mask_OT], vort_OT[mask_OT], h=1)

u_OT += 1
Mom_OT = PC.Momentum(vort_OT, u_OT, v_OT,
                     np.gradient(x_OT[0, :]), np.gradient(y_OT[:, 0]))


# %% 
x_arc, y_arc = PC.Gen_Arc(AoA[1])
Arc = VL.VortexLine(x_arc, y_arc)

exvelo = lambda xt, yt: (griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                               u_OT.flatten(),
                               np.vstack((xt.flatten(), yt.flatten())).transpose()),
                         griddata(np.vstack((x.flatten(), y.flatten())).transpose(),
                               v_OT.flatten(),
                               np.vstack((xt.flatten(), yt.flatten())).transpose()))

gamma = Arc.solve_gamma(exvelo)
unorm, vnorm = exvelo(Arc.xc, Arc.yc)

u_gam, v_gam = Arc.velocity(gamma, Arc.xc, Arc.yc)
u_indu, v_indu = Arc.velocity_ext(gamma, x, y)

upw_uni = -unorm * Arc.sy + vnorm * Arc.sx
upw_gamma = -(unorm - u_gam) * Arc.sy + (vnorm - v_gam) * Arc.sx

# %% Motivation
skip = 50
plt.figure()
plt.contourf(x_full, y_full, np.sqrt(u_full[2]**2, + v_full[2]**2))
plt.quiver(x_full[::skip, ::skip], y_full[::skip, ::skip],
           u_full[2][::skip, ::skip], v_full[2][::skip, ::skip])
plt.xticks(())
plt.yticks(())


# %% Method Flowchart
c_m = cm.RdBu_r
ylim = .1
levs = np.linspace(-ylim, ylim, 51)

plt.figure()
plt.contourf(x, y, vort[0], cmap=c_m, extend='both', levels=levs)
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.axis('equal')

plt.figure()
plt.contourf(x, y, vort[2], cmap=c_m, extend='both', levels=levs)
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.axis('equal')

plt.figure()
plt.contourf(x_OT, y_OT, vort_OT, cmap=c_m, extend='both', levels=levs)
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.axis('equal')

plt.figure()
plt.contourf(x_OT, y_OT, Mom_OT, extend='both', levels=np.linspace(0, 1e-4, 51))
plt.colorbar()
plt.xticks(())
plt.yticks(())
plt.axis('equal')


# %% Interpolation Method

plt.figure()
plt.contourf(x, y, np.ones_like(x))
plt.colorbar()
plt.quiver(x[::skip, ::skip], y[::skip, ::skip],
           np.ones_like(x)[::skip, ::skip], np.zeros_like(x)[::skip, ::skip])
plt.xticks(())
plt.yticks(())


plt.figure()
plt.contourf(x_OT, y_OT, np.sqrt(u_OT**2 + v_OT**2))
plt.colorbar()
plt.quiver(x_OT[::skip, ::skip], y_OT[::skip, ::skip], u_OT[::skip, ::skip],
           v_OT[::skip, ::skip])
plt.xticks(())
plt.yticks(())
plt.axis('equal')


# %%
limits = 60
skip = 10
plt.figure()
plt.contourf(x_OT, y_OT, np.sqrt(u_OT**2 + v_OT**2))
plt.colorbar()
plt.quiver(x_OT[::skip, ::skip], y_OT[::skip, ::skip], u_OT[::skip, ::skip],
           v_OT[::skip, ::skip], scale=20)
plt.xticks(())
plt.yticks(())
plt.axis('equal')
plt.xlim(-limits, limits)
plt.ylim(-limits, limits)
plt.plot(x_arc, y_arc)

plt.figure()
plt.contourf(x_OT, y_OT, np.sqrt((u_OT-u_indu)**2 + (v_OT-v_indu)**2))
plt.colorbar()
plt.quiver(x_OT[::skip, ::skip], y_OT[::skip, ::skip], (u_OT - u_indu)[::skip, ::skip],
           (v_OT - v_indu)[::skip, ::skip], scale=10)
plt.xticks(())
plt.yticks(())
plt.axis('equal')
plt.xlim(-limits, limits)
plt.ylim(-limits, limits)
plt.plot(x_arc, y_arc)

skip = 50
plt.figure()
plt.contourf(x_OT, y_OT, np.sqrt((u_OT-u_indu)**2 + (v_OT-v_indu)**2))
plt.colorbar()
plt.quiver(x_OT[::skip, ::skip], y_OT[::skip, ::skip], (u_OT - u_indu)[::skip, ::skip],
           (v_OT - v_indu)[::skip, ::skip])
plt.xticks(())
plt.yticks(())
plt.axis('equal')
# plt.xlim(-limits, limits)
# plt.ylim(-limits, limits)
plt.plot(x_arc, y_arc)


# %% BIOT - SAVART

vort = np.zeros_like(x)
mask = np.logical_and(abs(x)<1, abs(y)<1)
vort[mask] = 10

u_BS, v_BS = PC.u_omega(x, y, x[mask], y[mask], vort[mask], h=1)
u_BS[mask] = 0
v_BS[mask] = 0

# %% Plots
skip = 20
lim = 200
plt.figure()
plt.contourf(x, y, np.sqrt(u_BS**2 + v_BS**2), extend='max',
             levels=np.linspace(0, 1, 101), cmap=cm.Reds)
plt.quiver(x[::skip, ::skip], y[::skip, ::skip], 
           u_BS[::skip, ::skip], v_BS[::skip, ::skip])
plt.ylim(-lim, lim)
plt.xlim(-lim, lim)
plt.xticks(())
plt.yticks(())


# %% Flat Plate Vortex Line
yVL, xVL = np.mgrid[-50:50, -50:50]
xVL = xVL.astype(float)
yVL = yVL.astype(float)

alpha = np.radians(20)
exvelo2 = lambda xt, yt: (np.ones_like(xt, dtype=float),
                         np.zeros_like(xt, dtype=float))

x_plate = np.arange(-5, 5, .2) * np.cos(alpha)+.5
y_plate = np.arange(5, -5, -.2) * np.sin(alpha)
plate = VL.VortexLine(x_plate, y_plate)

gamma = plate.solve_gamma(exvelo2)
u_indu, v_indu = plate.velocity_ext(gamma, xVL, yVL)

u_n, v_n = exvelo2(xVL, yVL)

u_tot = u_n - u_indu
v_tot = v_n - v_indu

skip = 6
plt.figure()
plt.contourf(xVL, yVL, np.ones_like(xVL))
plt.quiver(xVL[::skip, ::skip], yVL[::skip, ::skip],
           np.ones_like(xVL)[::skip, ::skip],
           np.zeros_like(xVL)[::skip, ::skip])
plt.plot(x_plate, y_plate, 'k')

# skip = 1
plt.figure()
plt.contourf(xVL, yVL, np.sqrt(u_tot**2 + v_tot**2),
             extend='max', levels=np.linspace(0, 2, 21))
plt.colorbar()
plt.quiver(xVL[::skip, ::skip], yVL[::skip, ::skip],
           u_tot[::skip, ::skip], v_tot[::skip, ::skip])
plt.plot(x_plate, y_plate, 'k')

