# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:23:05 2020

@author: kvstr
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import solve_banded
import time


def Gradient(field, direction, d=1):
    """
    Calculates the first derivative on a discrete grid using central
    difference inside and one-sided difference at the boundaries

    Parameters
    ----------
    field : NxM array
        function values at grid points, with (N, 0) being closest to the
        origin and (0, M) being the point furthest away
    direction : scalar, 0 or 1
        Direction in which to take the derivative, with 0 being in y and
        1 in x
    d : scalar value, optional
        Grid spacing in the direction of the derivative. The default is 1.

    Returns
    -------
    gradient : NxM array
        Field of the derivative values on each grid point.

    """
    

    size = field.shape
    gradient = np.zeros(size)
    if direction == 0:
        for i in range(0, size[0]):
            for j in range(1, size[1]-1):
                gradient[i, j] = (field[i, j+1] - field[i, j-1]) / (2*d)
                gradient[i, 0] = (field[i, 1] - field[i, 0]) / (d)
                gradient[i, -1] = (field[i, -1] - field[i, -2]) / (d)
    elif direction == 1:
        for i in range(1, size[0]-1):
            for j in range(0, size[1]):
                gradient[i, j] = (field[i-1, j] - field[i+1, j]) / (2*d)
                gradient[0, j] = (field[0, j] - field[1, j]) / d
                gradient[-1, j] = (field[-2, j] - field[-1, j]) / d
    else:
        print('Invalid Direction')
    return gradient


def Continuity(u, v):
    """
    Calculation of the continuity error in a 2D flow field

    Parameters
    ----------
    dudx : NxM array
        Derivative of u in x on each grid point.
    dvdy : NxM array
        Derivative of v in y on each grid point.

    Returns
    -------
    error : NxM array
        Continuity error on each grid point

    """
    if not u.shape == v.shape:
        print('Fields have different sizes')
        return None
    else:
        dudx = Gradient(u, 0)
        dvdy = Gradient(v, 1)
        n = len(dudx)
        error = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                error[i, j] = dudx[i, j] + dvdy[i, j]
    return error


def Momentum(vort, u, v):
    """
    Calculation of the momentum error in a 2D flow field

    Parameters
    ----------
    vort : NxM array
        Vorticity value on each grid point.
    u : NxM array
        u-velocity value on each grid point.
    v : NxM array
        v-velocity value on each grid point.

    Returns
    -------
    error: NxM array
        Momentum error on each grid point.

    """
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
        error = np.zeros_like(vort)
        for i in range(s[0]):
            for j in range(s[1]):
                error[i, j] = u[i, j] * vortx1[i, j] + v[i, j]*vortx2[i, j] \
                    - nu*(vortxx1[i, j]+vortxx2[i, j])
    return error


def solve_Poisson(vort, u_top, u_bot, v_left, v_right, h=1):
    size = vort.shape
    # Building Matrix A for A*Psi = b
    start = time.time()
    main_A = np.ones((size[0]**2, 1)) * 4  # Main Diagonal entries
    off_Au = np.ones(((size[0]**2-1), 1)) * -1  # Off Diagonal entries - upper
    off_Al = np.ones(((size[0]**2-1), 1)) * -1  # Off Diagonal entries - lower
    for i in range(size[0]):    
        off_Au[size[0]*i] *= 2
        off_Al[-(size[0]*i+1)] *= 2
        if i > 0:
            off_Au[size[0]*i-1] = 0
            off_Al[-(size[0]*i)] = 0
    off_AI = np.ones((size[0]*(size[0]-1))) * -1  # Off Diag - Identity
    #  Assemble B
    A = np.diagflat(main_A, 0)
    A += np.diagflat(off_Au, 1)
    A += np.diagflat(off_Al, -1)
    A += np.diagflat(off_AI, size[0])
    A += np.diagflat(off_AI, -size[0])
    for i in range(size[0]):
        A[i, size[1]+i] *= 2
        A[-(i+1), -(size[1]+1+i)] *= 2
    end = time.time()
    print('Build Matrix: %.3f' %(end-start))
    # Build RHS
    # g: BC's
    start = time.time()
    g = np.zeros(size)
    for i in range(size[0]):
        # Left Boundary
        g[i, 0] += v_left[i]*2
        # Right Boundary
        g[i, -1] -= v_right[i]*2
    for j in range(size[1]):
        # Top Boundary, normal derivative of stream function = u_inf
        g[0, j] += u_top[j] * 2
        # Bottom Boundary, as above but sign inverted
        g[-1, j] -= u_bot[j] * 2

    b = vort.reshape(size[0]*size[1], order='F')*h**2\
        + g.reshape(size[0]*size[1], order='F')*h
    end = time.time()
    print('Build Source: %.3f' %(end-start))
    # plt.imshow(A)
    # Solve for Psi
    start = time.time()
    Psi = np.linalg.solve(A, b)
    end = time.time()
    print('Dense Solver: %.3f' %(end-start))
    # Reshape Psi to original shape of vorticity field
    # print(Psi)
    Psi = Psi.reshape(size, order='F')
    grad = np.gradient(Psi, h)
    u = -grad[0]
    v = -grad[1]
    return u, v, A


def solve_Poisson_sparse(vort, u_top, u_bot, v_left, v_right, h=1, timeit=False):
    size = vort.shape
    if timeit:
        start = time.time()
    # Building Matrix A for A*Psi = b
    main_A = np.ones((size[0]**2, )) * 4  # Main Diagonal entries
    off_Au = np.ones(((size[0]**2-1), )) * -1  # Off Diagonal entries - upper
    off_Al = np.ones(((size[0]**2-1), )) * -1  # Off Diagonal entries - lower
    off_AIu = np.ones((size[0]*(size[0]-1))) * -1  # Off Diag - Identity - upper
    off_AIl = np.ones((size[0]*(size[0]-1))) * -1  # Off Diag - Identity - lower
    for i in range(size[0]):    
        off_Au[size[0]*i] *= 2
        off_Al[-(size[0]*i+1)] *= 2
        off_AIu[i] *= 2
        off_AIl[-(i+1)] *= 2
        if i > 0:
            off_Au[size[0]*i-1] = 0
            off_Al[-(size[0]*i)] = 0
    #  Assemble A
    A = sparse.diags([main_A, off_Au, off_Al, off_AIu, off_AIl],
                     [0, 1, -1, size[0], -size[0]], format="csr")
    if timeit:
        end = time.time()
        print('Build sparse Matrix: %.3f' %(end-start))
    # Build RHS
    # g: BC's
    if timeit:
        start = time.time()
    g = np.zeros(size)
    for i in range(size[0]):
        # Left Boundary
        g[i, 0] += v_left[i]*2
        # Right Boundary
        g[i, -1] -= v_right[i]*2
    for j in range(size[1]):
        # Top Boundary, normal derivative of stream function = u_inf
        g[0, j] += u_top[j] * 2
        # Bottom Boundary, as above but sign inverted
        g[-1, j] -= u_bot[j] * 2

    b = vort.reshape(size[0]*size[1], order='F')*h**2\
        + g.reshape(size[0]*size[1], order='F')*h
    if timeit:
        end = time.time()
        print('Build Source: %.3f' %(end-start))
    # plt.imshow(A)
    # Solve for Psi
    if timeit:
        start = time.time()
    Psi = sparse.linalg.spsolve(A, b)
    if timeit:
        end = time.time()
        print('Sparse Solver: %.3f' %(end-start))
    # Reshape Psi to original shape of vorticity field
    # print(Psi)
    Psi = Psi.reshape(size, order='F')
    grad = np.gradient(Psi, h)
    u = -grad[0]
    v = -grad[1]
    return u, v, Psi

def solve_Poisson_banded(vort, u_top, u_bot, v_left, v_right, h=1):
    if not vort.shape[0] == vort.shape[1]:
        print('Non-square')
        return False
    N = vort.shape[0]
    start = time.time()
    locs = np.linspace(0, N-1, N, dtype=int)
    # Building Matrix A for A*Psi = b
    LU = np.ones((2*N+1, N**2))*-1
    LU[1:N-1, :] *= 0
    LU[N+2:-1, :] *= 0
    LU[N, :] *= -4
    LU[0, 0:N] *= 0
    LU[0, N:2*N] *= 2
    LU[-1, -(N):] *= 0
    LU[-1, -(2*N):-(N)] *= 2
    LU[N-1, 0] = 0
    LU[N+1, -1] = 0
    LU[N-1, locs*N+1] *= 2
    LU[N-1, locs*N] = 0
    LU[N+1, locs*N+N-2] *= 2
    LU[N+1, locs*N+N-1] = 0
    end = time.time()
    print('Build Band Matrix: %.3f' %(end-start))
    # print(end)
    # Build RHS
    # g: BC's
    start = time.time()
    g = np.zeros((N, N))
    for i in range(N):
        # Left Boundary
        g[i, 0] += v_left[i]*2
        # Right Boundary
        g[i, -1] -= v_right[i]*2
    for j in range(N):
        # Top Boundary, normal derivative of stream function = u_inf
        g[0, j] += u_top[j] * 2
        # Bottom Boundary, as above but sign inverted
        g[-1, j] -= u_bot[j] * 2

    b = vort.reshape(N**2, order='F')*h**2\
        + g.reshape(N**2, order='F')*h
    end = time.time()
    print('Build source vector: %.3f' %(end-start))
    # plt.imshow(A)
    # Solve for Psi
    start = time.time()
    Psi = solve_banded((N, N), LU, b)
    end = time.time()
    print('Banded Solver: %.3f' %(end-start))
    # Reshape Psi to original shape of vorticity field
    # print(Psi)
    Psi = Psi.reshape((N, N), order='F')
    grad = np.gradient(Psi, h)
    u = -grad[0]
    v = -grad[1]
    return u, v

