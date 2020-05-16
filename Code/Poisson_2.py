# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:10 2020

@author: kvstr
"""

import numpy as np
# import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp
# from scipy import linalg
import Physical_Calculations as PC
import time

import scipy.sparse as sparse
from scipy.sparse import linalg


# Choose Flow to be modelled, 'Shear', 'TGV', 'TGV_compare' or 'Body'
FlowType = 'TGV'

# Grid size
nx = 100
ny = nx

Re = 1
kappa = 1
t = 0
# %% Shear Flow
if FlowType == 'Shear':
    vort = np.ones((nx, ny))*-1
    # vort[2, 4] = -5
    u_top = np.ones((nx, 1)) * (ny-1)
    u_bot = np.ones((nx, 1)) * 0
    v_left = np.ones((ny, 1)) * 0
    v_right = np.ones((ny, 1)) * 0
    u, v = PC.solve_Poisson(vort, u_top, u_bot, v_left, v_right)
    
    err_mom = PC.Momentum(vort, u, v)
    err_cont = PC.Continuity(u, v)
    
    err_mom_sum = np.sqrt(np.sum(err_mom**2))
    err_cont_sum = np.sqrt(np.sum(err_cont**2))
    print('Momentum Error: %.2f' %err_mom_sum)
    print('Continuity Error: %.2f' %err_cont_sum)
    
    # %%
    cm = 'Greys'
    
    plt.figure()
    plt.title('u')
    plt.imshow(u, cmap=cm)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.colorbar()
    
    plt.figure()
    plt.title('v')
    plt.imshow(v, cmap=cm)
    plt.colorbar()
    plt.ylabel('y')
    plt.xlabel('x')
    
    plt.figure()
    plt.title('Momentum Error')
    plt.imshow(err_mom, cmap=cm)
    plt.colorbar()
    plt.ylabel('y')
    plt.xlabel('x')

        
# %% Taylor Green Vortex
# TODO change nested loop to enumerate, check time
if FlowType == 'TGV':
    Ft = np.exp(-2*kappa**2*t/Re)
    x, h = np.linspace(0, 2*np.pi, nx, retstep=True)
    y = np.linspace(0, 2*np.pi, ny)
    # x = x[1:-1]
    # y = y[1:-1]
    
    u_ana = np.zeros((nx, ny))
    v_ana = np.zeros_like(u_ana)
    for i in range(ny):
        for j in range(nx):
            u_ana[i, j] = np.cos(x[j]) * np.sin(y[-(i+1)]) * Ft
            v_ana[i, j] = -np.sin(x[j]) * np.cos(y[-(i+1)]) * Ft

    vort = -2* kappa * np.cos(kappa*x) * np.cos(kappa*y[:, np.newaxis])\
        * Ft
    u_top = u_ana[0, :]
    u_bot = u_ana[-1, :]
    v_left = v_ana[:, 0]
    v_right = v_ana[:, -1]
    u, v, Psi = PC.solve_Poisson_periodic(vort, u_top, u_bot, v_left, v_right,
                                          h=h, periodic=True)


# %%

    # b_ana =  A.dot(u_ana.reshape(nx**2))
    # g_ana = b_ana - vort.reshape((nx**2))*h**2
    # # vort_ana = PC.Gradient(v_ana, 0, h) - PC.Gradient(u_ana, 1, h)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    im_ga = ax1.imshow(u_ana.reshape((nx, ny)))
    f.colorbar(im_ga, ax=ax1)
    ax1.set_title('Analytic')
    im_g = ax2.imshow(u.reshape((nx, ny)))
    f.colorbar(im_g, ax=ax2)
    ax2.set_title('Discrete')
    
    plt.figure()
    plt.imshow((u[1:-1, :]-u_ana[1:-1, :])**2)
    plt.colorbar()
    
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot(u[0, :])
    # ax2.plot(u_ana[0, :])

    # # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # # im_va = ax1.imshow(vort_ana)
    # # ax1.set_title('Analytic')
    # # f.colorbar(im_va, ax=ax1)
    # # im_v = ax2.imshow(vort)
    # # f.colorbar(im_v, ax=ax2)
    # # ax2.set_title('Discrete')
    
    # # plt.figure()
    # # plt.imshow(vort-vort_ana)
    # # plt.colorbar()
    
    
    # # %%
    # cm = 'Greys'

    # plt.figure()
    # plt.imshow(u, cmap=cm)
    # plt.title('u')
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(u_ana, cmap=cm)
    # plt.title('u Analytic')
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(u-u_ana, cmap=cm)
    # plt.title('u-error')
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(v-v_ana, cmap=cm)
    # plt.title('v-error')
    # plt.colorbar()
    
    # #  Boundary Condition? Error on boundary zero
    # #  Poisson check matrix multiplication with correct matrix
    # # CHeck Continuity & Momentum
    # # Turn the source term near the body
    # # Implement distance function for circular arc
    # # Apply on-off switch - turn off source trm very close and then smoothly ramp between
    # linpack
    # LU-decomposition when in same geometry?
    0

# %% Taylor Green, compare Solver speeds

if FlowType == 'TGV_compare':
    Ft = np.exp(-2*kappa**2*t/Re)
    x, h = np.linspace(0, 2*np.pi, nx, retstep=True)
    y = np.linspace(0, 2*np.pi, ny)
    # x = x[1:-1]
    # y = y[1:-1]
    
    u_ana = np.zeros((nx, ny))
    v_ana = np.zeros_like(u_ana)
    for i in range(ny):
        for j in range(nx):
            u_ana[i, j] = np.cos(x[j]) * np.sin(y[-(i+1)]) * Ft
            v_ana[i, j] = -np.sin(x[j]) * np.cos(y[-(i+1)]) * Ft

    vort = -2* kappa * np.cos(kappa*x) * np.cos(kappa*y[:, np.newaxis])\
        * Ft
    u_top = u_ana[0, :]
    u_bot = u_ana[-1, :]
    v_left = v_ana[:, 0]
    v_right = v_ana[:, -1]
    start = time.time()
    u, v, A = PC.solve_Poisson(vort, u_top, u_bot, v_left, v_right, h=h)
    t_dense = time.time()-start
    print('Time Dense: %.3f' %t_dense)
    print('Times Sparse:')
    start = time.time()
    us, vs, As = PC.solve_Poisson_sparse(vort, u_top, u_bot, v_left, v_right,
                                         h=h, timeit=True)
    t_sparse = time.time() - start
    print('Total Time sparse: %.3f' %t_sparse)
    print('Times Banded:')
    start = time.time()
    ub, vb = PC.solve_Poisson_banded(vort, u_top, u_bot, v_left, v_right, h=h)
    t_banded = time.time()-start
    print('Total Time banded: %.3f' %t_banded) 


# %% Immersed Plate at 10<y<20
if FlowType == 'Body':
    radius = 5
    
    vort = np.ones((nx, ny))*0
    centre_2D = np.array((vort.shape[0] // 2, vort.shape[1] // 2))
    for i in range(radius):
        for j in range(radius):
            if np.sqrt(i**2 + j**2) < radius:
                vort[centre_2D[0]+i, centre_2D[1]+j] = 1
                vort[centre_2D[0]+i, centre_2D[1]-j] = -1
                vort[centre_2D[0]-i, centre_2D[1]-j] = 1
                vort[centre_2D[0]-i, centre_2D[1]+j] = -1
                
    u_top = np.ones((nx, 1)) * (nx-1)
    u_bot = np.ones((nx, 1)) * (nx-1)
    v_left = np.ones((ny, 1)) * 0
    v_right = np.ones((ny, 1)) * 0
    u, v, A = PC.solve_Poisson_sparse(vort, u_top, u_bot, v_left, v_right, h=1)
    
    cm = 'Greys'
    
    plt.imshow(u, cmap=cm)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(v, cmap=cm)
    plt.colorbar()