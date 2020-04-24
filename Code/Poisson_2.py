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
FlowType = 'Shear'

# Grid size
nx = 20
ny = 20

Re = 1
kappa = 2
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
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)
    vort = 2* kappa * np.cos(kappa*x) * np.cos(kappa*y[:, np.newaxis]) * np.exp(-2*kappa**2*t/Re)
    u, v = PC.solve_Poisson(vort, u_top, u_bot, v_left, v_right)
    
    plt.imshow(vort)
