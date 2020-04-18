# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:10 2020

@author: kvstr
"""

import numpy as np
import pylab as pl
import ot
# import matplotlib.pyplot as plt



# %%
##############################################################################
# Data preparation

n1 = 200
n2 = 202  # len(temp)
x = np.transpose(np.linspace(0, 1, np.max([n1, n2])))

A = []
f1 = np.zeros([n1, n2])
f2 = np.zeros([n1, n2])
f3 = np.zeros([n1, n2])
f4 = np.zeros([n1, n2])
r = 0.5
for i in range(n1):
    for j in range(n2):
        if (x[i]-r)**2+(x[j]-r)**2 <= r**2:
            f1[i, j] = 1
        f2[i, j] = i
        if ((x[i] <= x[j]+0.3 and x[i] >= x[j]-0.3)
            or (x[i] >= 1-(x[j]+0.3) and x[i] <= 1-(x[j]-0.3))):
            f4[i, j] = 1
        if (x[i] >= 0.3 and x[i] <= 0.7) or (x[j] >= 0.3 and x[j] <= 0.7):
            f3[i, j] = 1
norm = np.max([np.sum(f1), np.sum(f2), np.sum(f3), np.sum(f4)])
f1 = f1 / np.sum(f1)
f2 = f2 / np.sum(f2)
f3 = f3 / np.sum(f3)
f4 = f4 / np.sum(f4)
A.append(f1)
A.append(f2)
A.append(f3)
A.append(f4)
A = np.array(A)
if not A.shape[1] == A.shape[2]:
        padding = np.abs(A.shape[2] - A.shape[1])
        pad1 = int(np.floor(padding/2))
        pad2 = int(padding - pad1)
        if A.shape[1] < A.shape[2]:
            A = np.pad(A, ((0, 0),(pad1, pad2), (0, 0)))
        elif A.shape[2] < A.shape[2]:
            A = np.pad(A, ((0, 0), (0,0), (pad1, pad2)))

nb_images = 5

# those are the four corners coordinates that will be interpolated by bilinear
# interpolation
v1 = np.array((1, 0, 0, 0))
v2 = np.array((0, 1, 0, 0))
v3 = np.array((0, 0, 1, 0))
v4 = np.array((0, 0, 0, 1))


##############################################################################
#%% Barycenter computation and visualization
# ----------------------------------------
#

pl.figure(figsize=(10, 10))
pl.title('Convolutional Wasserstein Barycenters in POT')
cm = 'Greys'
# regularization parameter
reg = 0.008
for i in range(nb_images):
    for j in range(nb_images):
        pl.subplot(nb_images, nb_images, i * nb_images + j + 1)
        tx = float(i) / (nb_images - 1)
        ty = float(j) / (nb_images - 1)

        # weights are constructed by bilinear interpolation
        tmp1 = (1 - tx) * v1 + tx * v2
        tmp2 = (1 - tx) * v3 + tx * v4
        weights = (1 - ty) * tmp1 + ty * tmp2

        # Square inverse distance
        # tmp1 = (1 - tx)**2 * v1 + tx**2 * v2
        # tmp2 = (1 - tx)**2 * v3 + tx**2 * v4
        # weights = (1 - ty) * tmp1 + ty * tmp2
        
        if i == 0 and j == 0:
            pl.imshow(f1, cmap=cm)
            pl.axis('off')
        elif i == (nb_images - 1) and j == 0:
            pl.imshow(f2, cmap=cm)
            pl.axis('off')
        elif i == 0 and j == (nb_images - 1):
            pl.imshow(f3, cmap=cm)
            pl.axis('off')
        elif i == (nb_images - 1) and j == (nb_images - 1):
            pl.imshow(f4, cmap=cm)
            pl.axis('off')
        else:
            # call to barycenter computation
            pl.imshow(ot.bregman.convolutional_barycenter2d(A, reg, weights),
                        cmap=cm)

            # My own one
            # pl.imshow(convo_barycenter2d(A, reg, weights)[:n1, :n2],
            #           cmap=cm)

            # Linear Interpolation
            # pl.imshow(A[0]*weights[0] + A[1]*weights[1]
            #           + A[2]*weights[2] + A[3]*weights[3], cmap=cm)
            pl.axis('off')
pl.show()
