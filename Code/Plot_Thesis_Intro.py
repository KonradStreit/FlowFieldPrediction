#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:31:16 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt
import ot

import Physical_new as PC


# %%

n = 100
n_img = 5
Circ1 = np.zeros((n, n))
Circ2 = np.zeros_like(Circ1)

for i in range(n):
    for j in range(n):
        if ((i-50)**2 + (j-20)**2) < n/5:
            Circ1[i, j] = 1
        
        if ((i-50)**2 + (j-80)**2) < n/5:
            Circ2[i, j] = 1

A = []
A.append(Circ1)
A.append(Circ2)
A =  np.array(A)
# %%

reg = .005

f, ax = plt.subplots(1, 6, figsize=(6, 1))
pl = ax[0].contourf(Circ1, levels=np.linspace(0, 1, 21))
ax[0].axis('equal')
ax[5].contourf(Circ2, levels=pl.levels)
ax[5].axis('equal')

ax[1].contourf(ot.bregman.convolutional_barycenter2d(A, reg, (.8, .2)), levels=pl.levels)
ax[1].axis('equal')
ax[2].contourf(ot.bregman.convolutional_barycenter2d(A, reg, (.6, .4)), levels=pl.levels)
ax[2].axis('equal')
ax[3].contourf(ot.bregman.convolutional_barycenter2d(A, reg, (.4, .6)), levels=pl.levels)
ax[3].axis('equal')
ax[4].contourf(ot.bregman.convolutional_barycenter2d(A, reg, (.2, .8)), levels=pl.levels)
ax[4].axis('equal')

plt.setp(ax, xticks=(), yticks=())

# %%
f, ax = plt.subplots(1, 6, figsize=(6, 1))
pl = ax[0].contourf(Circ1, levels=np.linspace(0, 1, 21))
ax[0].axis('equal')
ax[5].contourf(Circ2, levels=pl.levels)
ax[5].axis('equal')

ax[1].contourf(Circ1*.8 + Circ2*.2, levels=pl.levels)
ax[1].axis('equal')
ax[2].contourf(Circ1*.6 + Circ2*.4, levels=pl.levels)
ax[2].axis('equal')
ax[3].contourf(Circ1*.4 + Circ2*.6, levels=pl.levels)
ax[3].axis('equal')
ax[4].contourf(Circ1*.2 + Circ2*.8, levels=pl.levels)
ax[4].axis('equal')

plt.setp(ax, xticks=(), yticks=())

