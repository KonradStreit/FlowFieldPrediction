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
from numba import njit
from numba import prange


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


# @njit(parallel=True)
def Continuity(u, v, x, y):
    """
    Calculation of the continuity error in a 2D flow field

    Parameters
    ----------
    u:  MxN Array
        u-velocity matrix
    v:  MxN Array
        v-velocity matrix
    x:  Nx1 vector
        x-coordinates of points
    y:  Mx1 vector
        y-coordinates of points

    Returns
    -------
    error : NxM array
        Continuity error on each grid point

    """
    if not u.shape == v.shape:
        print('Fields have different sizes')
        return None
    else:
        error = abs(np.divide((u[:-1, 1:] - u[:-1, :-1]),\
                    np.gradient(x, axis=1)[:-1, :-1])\
          +np.divide(v[1:, :-1] - v[:-1, :-1],\
                    np.gradient(y, axis=0)[:-1, :-1]))
    error = np.pad(error, ((0, 1),), constant_values=0)
    return error


# @njit  # (parallel=True)
def Momentum(vort, u, v, dx, dy):
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
    x:  Mx1 array
        half cell sizes in x direction
    y:  Nx1 array
        half cell sizes in y direction

    Returns
    -------
    error: NxM array
        Momentum error on each grid point.

    """
    nu = 1.05e-6
    if not (np.shape(vort) == np.shape(u) and np.shape(vort) == np.shape(v)):
        print('Shape mismatch')
        return None
    else:
       # Vorticity Gradient x
       vortx = np.zeros_like(vort)
       vortxx = np.zeros_like(vortx)
       vortx[:, -1] = np.divide(vort[:, -1]-vort[:, -2], dx[-1]+dx[-2])
       for i in range(vort.shape[1]-1):
           vortx[:, i] = (np.divide(vort[:, i+1]*dx[i] - vort[:, i]*dx[i+1],
                                    dx[i]+dx[i+1])
                          -np.divide(vort[:, i]*dx[i-1] - vort[:, i-1]*dx[i],
                                     dx[i]+dx[i-1])) / 2*dx[i]
       vortxx[:, -1] = np.divide(vortx[:, -1]-vortx[:, -2], dx[-1]+dx[-2])
       for i in range(vortx.shape[1]-1):
            vortxx[:, i] = (np.divide(vortx[:, i+1]*dx[i] - vortx[:, i]*dx[i+1],
                                      dx[i]+dx[i+1])
                           -np.divide(vortx[:, i]*dx[i-1] - vortx[:, i-1]*dx[i],
                                      dx[i]+dx[i-1])) / 2*dx[i]
        
       # Vorticity Gradient y
       vorty = np.zeros_like(vort)
       vortyy = np.zeros_like(vortx)
       vorty[-1, :] = np.divide(vort[-1, :]-vort[-2, :], dy[-1]+dy[-2])
       for i in range(vort.shape[0]-1):
           vorty[i, :] = (np.divide(vort[i+1, :]*dy[i] - vort[i, :]*dy[i+1],
                                    dy[i]+dy[i+1])
                          -np.divide(vort[i, :]*dy[i-1] - vort[i-1, :]*dy[i],
                                     dy[i]+dy[i-1])) / 2*dy[i]
       vortyy[-1, :] = np.divide(vorty[-1, :]-vorty[-2, :], dy[-1]+dy[-2])
       for i in range(vorty.shape[0]-1):
           vortyy[i, :] = (np.divide(vorty[i+1, :]*dy[i] - vorty[i, :]*dy[i+1],
                                    dy[i]+dy[i+1])
                          -np.divide(vorty[i, :]*dy[i-1] - vorty[i-1, :]*dy[i],
                                     dy[i]+dy[i-1])) / 2*dy[i]

       t1 = np.multiply(u, vortx)
       t2 = np.multiply(v, vorty)
       t3 = nu * (vortxx+vortyy)
       error = abs(np.subtract(t1+t2, t3))
    return error

def CellSizes(x, y):
    """
    Calculates the distance from cell centre to cell face in either direction

    Parameters
    ----------
    x : Mx1 Array
        x-Coordinates of cell centers.
    y : Nx1 Array
        y-Coordinates of cell centers.

    Returns
    -------
    dx : Mx1 Array
        x-distance cell centre-face.
    dy : Nx1
        y-distance cell centre-face.

    """
    # Calcuating Cell sizes x-direction
    first = np.where(np.gradient(x) == 1)[0][0]
    last = np.where(np.gradient(x) == 1)[0][-1]
    dx = np.ones_like(x)*.5
    for i in np.linspace(first-1, 0, first, dtype=int):
        dx[i] = x[i+1] - x[i] - dx[i+1]
    for i in range(last, x.shape[0]):
        dx[i] = x[i] - x[i-1] - dx[i-1]
    
    # Calculating cell sizes in y-direction
    first = np.where(np.gradient(y) == 1)[0][0]
    last = np.where(np.gradient(y) == 1)[0][-1]
    dy = np.ones_like(y)*.5
    for i in np.linspace(first-1, 0, first, dtype=int):
        dy[i] = y[i+1] - y[i] - dy[i+1]
    
    for i in range(last, y.shape[0]):
        dy[i] = y[i] - y[i-1] -dy[i-1]
    return dx, dy

def Vorticity(u, v, dx, dy):
    """
    Calculates the Vorticity from velocity Components and Cell sizes

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    dy : TYPE
        DESCRIPTION.

    Returns
    -------
    vort : TYPE
        DESCRIPTION.

    """

    # Gradient v-velocity
    dvdx = np.zeros_like(v)
    for i in range(1, v.shape[1]-1):
        vpl = np.divide(v[:, i]*dx[i+1] + v[:, i+1]*dx[i], dx[i]+dx[i+1])
        vmi = np.divide(v[:, i]*dx[i-1] + v[:, i-1]*dx[i], dx[i-1]+dx[i])
        dvdx[:, i] = np.divide(vpl - vmi, 2*dx[i])
    # Gradient u-velocity
    dudy = np.zeros_like(u)
    for i in range(1, u.shape[0]-1):
        upl = np.divide(u[i, :]*dy[i+1] + u[i+1, :]*dy[i], dy[i]+dy[i+1])
        umi = np.divide(u[i, :]*dy[i-1] + u[i-1, :]*dy[i], dy[i]+dy[i-1])
        dudy[i, :] = np.divide(upl-umi, 2*dy[i])
        
    vort = dvdx - dudy
    return vort

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


def solve_Poisson_periodic(vort, u_top, u_bot, v_left, v_right, h=1,
                           periodic=False, timeit=False):
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
    
    if not periodic:
        grad = np.gradient(Psi, h)
        u = -grad[0]
        v = -grad[1]
        return u, v, Psi
    else:
        temp = np.zeros((size[0]+2, size[0]+2))
        temp[1:-1, 1:-1] = Psi
        for i, right in enumerate(Psi[:, -2]):
            temp[i+1, 0] = right
            temp[i+1, -1] = Psi[i, 1]
        for i, bot in enumerate(temp[-3, :]):
            temp[0, i] = bot
            temp[-1, i] = temp[2, i]
        grad = np.gradient(temp, h)
        u = -grad[0][1:-1, 1:-1]
        v = -grad[1][1:-1, 1:-1]
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

