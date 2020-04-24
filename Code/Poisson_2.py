# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:10 2020

@author: kvstr
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp
import Physical_Calculations as PC

# Choose Flow to be modelled, 'Shear' or 'TGV'
FlowType = 'TGV'

# Grid size
nx = 20
ny = 20

Re = 1
kappa = 1
t = .1
# %% Shear Flow
if FlowType == 'Shear':
    vort = np.ones((nx, ny))*-1
    # vort[2, 4] = -5
    u_top = np.ones((nx, 1)) * 19
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
if FlowType == 'TGV':
    Ft = np.exp(-2*kappa**2*t/Re)
    x, h = np.linspace(0, 2*np.pi, nx, retstep=True)
    y = np.linspace(0, 2*np.pi, ny)
    vort = -2* kappa * np.cos(kappa*x) * np.cos(kappa*y[:, np.newaxis])\
        * Ft
    u_top = np.cos(x) * np.sin(y[-1]) * Ft
    u_bot = np.cos(x) * np.sin(y[0]) * Ft
    v_left = -np.sin(x[0]) * np.cos(y) * Ft
    v_right = -np.sin(x[-1]) * np.cos(y) * Ft
    u, v = PC.solve_Poisson(vort, u_top, u_bot, v_left, v_right, h=h)
    
    u_ana = np.zeros_like(u)
    v_ana = np.zeros_like(v)
    for i in range(ny):
        for j in range(nx):
            u_ana[i, j] = np.cos(x[j]) * np.sin(y[-(i+1)]) * Ft
            v_ana[i, j] = -np.sin(x[j]) * np.cos(y[-(i+1)]) * Ft
    plt.figure()
    plt.imshow(u)
    plt.title('u')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(u_ana)
    plt.title('Analytic')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(u-u_ana)
    plt.colorbar()
