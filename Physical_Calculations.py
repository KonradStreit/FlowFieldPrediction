# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:23:05 2020

@author: kvstr
"""
import numpy as np


def Gradient(field, direction, d=1):  # TODO use known boundary conditions?
    size = field.shape
    gradient = np.zeros(size)
    if direction == 0:
        for i in range(1, size[0]-1):
            for j in range(0, size[1]):
                gradient[i, j] = (field[i+1, j] - field[i-1, j]) / (2*d)
                gradient[0, j] = (field[1, j] - field[0, j]) / (d)
                gradient[-1, j] = (field[-1, j] - field[-2, j]) / (d)
    elif direction == 1:
        for i in range(0, size[0]):
            for j in range(1, size[1]-1):
                gradient[i, j] = (field[i, j-1] - field[i, j+1]) / (2*d)
            gradient[i, 0] = (field[i, 0] - field[i, 1]) / d
            gradient[i, -1] = (field[i, -2] - field[i, -1]) / d
    else:
        print('Invalid Direction')
    return gradient


def Continuity(gradient1, gradient2):
    if not np.shape(gradient1) == np.shape(gradient2):
        print('Fields have different sizes')
        return None
    else:
        n = len(gradient1)
        error = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                error[i, j] = gradient1[i, j] + gradient2[i, j]
    return error


def Momentum(vort, u, v):
    nu = 1.05e-6
    if not (np.shape(vort) == np.shape(u) and np.shape(vort) == np.shape(v)):
        print('Shape mismatch')
        return np.infty
    else:
        s = np.shape(vort)
        vortx1 = Gradient(vort, 0, 1)
        vortx2 = Gradient(vort, 1, 1)
        vortxx1 = Gradient(vortx1, 0, 1)
        vortxx2 = Gradient(vortx2, 1, 1)
        eps = np.zeros_like(vort)
        for i in range(s[0]):
            for j in range(s[1]):
                eps[i, j] = u[i, j] * vortx1[i, j] + v[i, j]*vortx2[i, j] \
                    - nu*(vortxx1[i, j]+vortxx2[i, j])
    # error = np.sum(vort)
    return eps


def solve_Poisson(vort, u_top, u_bot, v_left, v_right, h=1):
    """
    
    

    Parameters
    ----------
    vort : Vorticity Field \n
    u_top : u at top Boundary \n
    u_bot : u at bottom Boundary \n
    v_left : v at left Boundary \n
    v_right : v at right boundary \n
    h : scalar, optional
        Grid spacing. The default is 1.

    Returns
    -------
    Psi : Field of Stream function values

    """
    size = vort.shape
    # Building Matrix A for A*Psi = b
    main_B = np.ones((size[0], 1)) * 4  # Main Diagonal entries
    off_B = np.ones((size[0]-1, 1)) * -1  # Off Diagonal entries
    #  Assemble B
    B = np.diagflat(main_B, 0)
    B += np.diagflat(off_B, 1)
    B += np.diagflat(off_B, -1)
    # Overwrite at Boundaries
    B[0, 1] = -2
    B[-1, -2] = -2
    I = np.identity(size[0])
    fill = np.zeros((size[0], size[0]*(size[0]-2)))
    # Generate rows of full Matrix A
    row1 = np.hstack((B, -2*I, fill))
    row2 = np.hstack((-1*I, B, -1*I, fill[:, size[0]:]))
    rown_1 = np.hstack((fill[:, size[0]:], -1*I, B, -1*I))
    rown = np.hstack((fill, -2*I, B))
    if size[0] > 4:
        row = 0
        for i in range(2, size[0]-2):
            del row
            row = np.hstack((np.zeros((size[0], size[0]*(i-1))), -1*I, B, -1*I,
                             np.zeros((size[0], size[0]*(size[0]-(i+2))))))
            if i == 2:
                rows = row
            else:
                rows = np.vstack((rows, row))
        # Assemble A
        A = np.vstack((row1, row2, rows, rown_1, rown))
    elif size[0] == 4:
        A = np.vstack((row1, row2, rown_1, rown))
    else:
        print('Not implemented for M < 4')
    # Build RHS
    # g: BC's
    g = np.zeros((size[0]**2))
    for i in range(size[0]):
        # Top Boundary, normal derivative of stream function = u_inf
        g[i*size[0]] += u_top * 2
        # Bottom Boundary, as above but sign inverted
        g[i*size[0]+size[0]-1] -= u_bot * 2
        # Left Boundary
        g[i] += v_left*2
        # Right Boundary
        g[-(i+1)] -= v_right*2
    # print(g)
    # Sum Vorticities and BCs
    b = vort.reshape(size[0]*size[1], order='F')*h**2 + g*h
    # plt.imshow(A)
    # Solve for Psi
    Psi = np.linalg.solve(A, b)
    # Reshape Psi to original shape of vorticity field
    # print(Psi)
    Psi = Psi.reshape(size)
    return Psi