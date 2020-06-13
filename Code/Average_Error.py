#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:33:15 2020

@author: konrad
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import Physical_Calculations as PC

import matplotlib.pyplot as plt


def read_vtr(file):
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(file)
    reader.Update()
    data = reader.GetOutput()
    pointData = data.GetPointData()
    
    sh = data.GetDimensions()[::-1]
    ndims = len(sh)
    
    # Get Vector Field
    v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims, ))
    Velocity = []
    for d in range(ndims):
        a = v[..., d]
        Velocity.append(a)
    # Get Sccalar field
    Pressure = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1, )).squeeze()
    
    u = Velocity[0].squeeze()
    v = Velocity[1].squeeze()
    # Obtain Grid
    x = np.array(data.GetXCoordinates())
    y = np.array(data.GetYCoordinates())
    
    return u, v, Pressure, x, y

# %% Read files & Statistics
AoAs = [10]
start = 50
n_files = 100-start
for alpha in AoAs:

    u0, v0, Press0, xlin, ylin = read_vtr("../Data/arc_{:03d}_Re_150/dat0x0x0/fluid.{:01d}.vtr".format(alpha, start))
    u_files = np.empty((n_files, u0.shape[0], u0.shape[1]))
    u_files[0] = u0
    v_files = np.empty((n_files, v0.shape[0], v0.shape[1]))
    v_files[0] = v0
    Press_files = np.empty((n_files, Press0.shape[0], Press0.shape[1]))
    Press_files[0] = Press0
    for i in range(1, n_files):
        file = "../Data/arc_{:03d}_Re_150/dat0x0x0/fluid.{:01d}.vtr".format(alpha, i+start)
        u_files[i], v_files[i], Press_files[i], xlin, ylin = read_vtr(file)
    x, y = np.meshgrid(xlin, ylin)
    
    # %% Statistics
    u = np.mean(u_files, axis=0)
    v = np.mean(v_files, axis=0)
    Press = np.mean(Press_files, axis=0)
    u_std = np.std(u_files, axis=0)
    v_std = np.std(v_files, axis=0)
    
    print('Mean std u:              %.8f' % np.mean(u_std))
    print('Mean std v:              %.8f' % np.mean(v_std))
    print('Max std u:               %.8f' % np.max(u_std))
    print('Max std v:               %.8f' % np.max(v_std))
    
    dx, dy = PC.CellSizes(x[0, :], y[:, 0])
    vort = PC.Vorticity(u, v, dx, dy)
    
    Cont_err = PC.Continuity(u, v, x, y)
    Mom_err = PC.Momentum(vort, u, v, dx, dy)
    
    print('Max Continuity Error:    %.8f' % np.max(Cont_err))
    print('Max Momentum Error:      %.8f' % np.max(Mom_err))
    
    
    # %% PLOTTING
    plt.figure()
    plt.contourf(x, y, abs(Cont_err))
    plt.colorbar()
    plt.axis('equal')
    plt.title('Continuity Error')
    plt.scatter(x[np.unravel_index(abs(Mom_err).argmax(), Mom_err.shape)],
            y[np.unravel_index(abs(Mom_err).argmax(), Mom_err.shape)], marker='x', c='k')

    
    plt.figure()
    plt.contourf(x, y, abs(Mom_err))
    plt.colorbar()
    plt.axis('equal')
    plt.title('Momentum Error')
    plt.scatter(x[np.unravel_index(abs(Mom_err).argmax(), Mom_err.shape)],
                y[np.unravel_index(abs(Mom_err).argmax(), Mom_err.shape)], marker='x', c='k')
    