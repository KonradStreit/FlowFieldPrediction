# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:10 2020

@author: kvstr
"""

import numpy as np
import pylab as pl
import ot
import matplotlib.pyplot as plt
import scipy as sp
import Physical_Calculations as PC

# %%
# vort = np.random.rand(5, 5)
vort = np.ones((10, 10))*-1
# u_top, u_bot, v_left, v_right
u_top = 9
u_bot = 0
v_left = 0
v_right = 0
Psi = PC.solve_Poisson(vort, u_top, u_bot, v_left, v_right)
u = PC.Gradient(Psi, 1)
v = -PC.Gradient(Psi, 0)
# print(Psi)
print(u)
print(v)
plt.imshow(u)
plt.colorbar()
plt.figure()
plt.imshow(v)
