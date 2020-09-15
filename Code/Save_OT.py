#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 16:06:10 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
import time
from scipy.interpolate import griddata
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

AoA = (0, 5, 10)  # np.linspace(0, 10, 3, dtype=int)

n_weights = 11

temp = np.linspace(0, 1, n_weights)
weights = np.vstack((temp, 1-temp)).transpose()

step = 1
reg = 1e-8
order = 2

# %% Read Data
x_full, y_full, u_full, v_full,\
    vort_full, u_std, v_std, Cont, Mom = PC.Read_Data(AoA, step=step)

x, y, u, v, vort = PC.make_square(x_full, y_full, u_full, v_full, vort_full,
                                  1000, step=1)

# %% Divide Pos & Neg

dx = np.gradient(x[0, :])
dy = np.gradient(y[:, 0])

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

# %% OT

for i, w in enumerate(weights):
    print('Starting OT')
    start_OT = time.time()
    vort_pos_OT = ot.bregman.convolutional_barycenter2d(vort_pos, reg, w)
    vort_neg_OT = ot.bregman.convolutional_barycenter2d(vort_neg, reg, w)
    vort_OT = vort_pos_OT*np.sum(w*sum_pos)\
        - vort_neg_OT*np.sum(w*sum_neg)
    
    print('OT finished after {:.0}mins'.format((time.time()-start_OT)/60))

    # %% Save
    
    np.savetxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_x.csv"
               .format(AoA[0], AoA[1], AoA[2], w[0], w[1]),
               x, delimiter=",")
    
    np.savetxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_y.csv"
               .format(AoA[0], AoA[1], AoA[2], w[0], w[1]),
               y, delimiter=",")

    np.savetxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_pos.csv"
               .format(AoA[0], AoA[1], AoA[2], w[0], w[1]),
               vort_pos_OT, delimiter=",")
    
    np.savetxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_neg.csv"
               .format(AoA[0], AoA[1], AoA[2], w[0], w[1]),
               vort_neg_OT, delimiter=",")
    
    np.savetxt("../Data/OT_Results/{:.0f}_{:.0f}_{:.0f}_{:.2f}_{:.2f}_sums.csv"
               .format(AoA[0], AoA[1], AoA[2], w[0], w[1]),
               [sum_pos, sum_neg], delimiter=",")
    

