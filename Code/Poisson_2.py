# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:19:10 2020

@author: kvstr
"""

import numpy as np
import numpy.linalg
# import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp
# from scipy import linalg
import Physical_Calculations as PC
import time
import VortexPanel as vp

import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.spatial.distance import euclidean as spdist

# %%
def wipe_rows_csr(matrix, rows):
    assert isinstance(matrix, sparse.csr_matrix)
    
    
    for i in rows:
        matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] = 0.0
    
    d = matrix.diagonal()
    d[rows] = 1.0
    matrix.setdiag(d)
    
    return


# %%


# Choose Flow to be modelled, 'Shear', 'TGV', 'TGV_compare' or 'Body'
FlowType = 'Body'

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
    u, v, Psi = PC.solve_Poisson_periodic(vort, u_top, u_bot, v_left, v_right)
    
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
    plt.title('Error (max(u)=%.2f)' % (np.max(u)))
    
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


# %% Immersed Arc
if FlowType == 'Body':
    
    vort = np.ones((nx, ny))*0
    x = np.linspace(0, nx-1, nx)
    
    angles = [0, -45]
    h = 4/nx
    
    # Arc radius
    radius = 1  # vort.shape[0] //3 * h
    # Arc end angles
    alpha = np.array([np.deg2rad(angles[0]), np.deg2rad(angles[1])])
    # Arc centre
    centre_2D = np.array((vort.shape[0] //8 * 5 * h, vort.shape[1] // 2 * h))
    # Arc thickness
    th = .05
    
    # %% Vortex Panel
    
    arc = vp.make_arc(32, angles[0], angles[1], th, ycen=-.5)
    arc.solve_gamma()
    arc.plot_flow()
    plt.grid()
    
    # %%    
    # Calculates Arc endpoints
    end = np.array([[centre_2D[0] - np.cos(alpha[0])*radius,
            centre_2D[1]-np.sin(alpha[0])*radius],
           [centre_2D[0] - np.cos(alpha[1])*radius,
            centre_2D[1]-np.sin(alpha[1])*radius]])
    
    dist = np.zeros((nx, nx))
    r = np.zeros_like(dist)
    gamma = np.zeros_like(dist)
    for i in range(nx):
        for j in range(nx):
            if any((all([i*h, j*h] == end[0, :]), all([i*h, j*h] == end[1, :]))):
                dist[i, j] = -th/2
                gamma[i, j] = np.arctan(((centre_2D[1]-j*h)/(centre_2D[0]-i*h)))
            elif i*h >= centre_2D[0]:
                dist[i, j] = 1
                gamma[i, j] = (2*np.pi
                               + np.arctan(((centre_2D[1]-j*h)
                                            / (centre_2D[0]-i*h))))
            else:
                gamma[i, j] = np.arctan(((centre_2D[1]-j*h)/(centre_2D[0]-i*h)))
                r[i, j] = np.sqrt((centre_2D[1]-j*h)**2+(centre_2D[0]-i*h)**2)
                # print(centre_2D[0]-i)
                # print((j-centre_2D[1]))
                # print(np.rad2deg(gamma))
                if gamma[i, j] >= alpha[1]:
                    # print('Greater')
                    if gamma[i, j] <= alpha[0]:
                        dist[i, j] = np.abs(radius-r[i, j])-th/2
                    else:
                        # print('Too big')
                        dist[i, j] = spdist([i*h, j*h], end[0, :])
                        # np.sqrt((i-end[1, 0])**2
                        #                      + (j-end[1, 1])**2) * h
                elif gamma[i, j] < alpha[1]:
                    dist[i, j] = spdist([i*h, j*h], end[1, :])
                     # dist[i, j] = np.sqrt((end[0, 0]-i)**2
                     #                         + (end[0, 1]-j)**2) * h

    # plt.figure()
    # plt.imshow(dist)
    # plt.colorbar()

    u_top = np.ones((nx, 1)) * 1  # * (nx-1)
    u_bot = np.ones((nx, 1)) * 1  # (nx-1)
    v_left = np.ones((ny, 1)) * 0
    v_right = np.ones((ny, 1)) * 0
    # h = 1
    
    top_boundary = np.where(np.logical_and
                            (np.logical_and((dist < h),(dist > 0)),
                             r > radius))
    top_boundary_1D = top_boundary[1]*nx + top_boundary[0]
    
    bot_boundary = np.where(np.logical_and
                            (np.logical_and((dist < h),(dist > 0)),
                             r < radius))
    bot_boundary_1D = bot_boundary[1]*nx + bot_boundary[0]
    
    body_points_2D = np.vstack(np.where(dist < 0)).transpose()
    body_points_1D = body_points_2D[:, 1]*nx + body_points_2D[:, 0]
    
    gamma_1D = gamma.reshape((nx**2, 1), order='F')
    
    # u, v, A = PC.solve_Poisson_sparse(vort, u_top, u_bot, v_left, v_right, h=1)
# %%
# def solve_Poisson_periodic(vort, u_top, u_bot, v_left, v_right, h=1,
#                            periodic=False, timeit=False):
    timeit = False
    periodic = False
    body = True
    
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

    vort_1D = vort.reshape(size[0]*size[1], order='F')
    b = vort_1D*h**2\
        + g.reshape(size[0]*size[1], order='F')*h
    if timeit:
        end = time.time()
        print('Build Source: %.3f' %(end-start))
        
        
    if body:
        wipe_rows_csr(A, body_points_1D)
        wipe_rows_csr(A, top_boundary_1D)
        wipe_rows_csr(A, bot_boundary_1D)
        # Stream function on body = 0
        for i, ind in enumerate(body_points_1D):
            b[ind] = 0
        # Stream function one step in = -.5*vort at body
        # Using gamma to determine which direction to take
        for i, ind in enumerate(top_boundary_1D):
            if np.sin(gamma_1D[ind]) > 0:
                # print(b[ind])
                # vort_1D[ind+1] = 100
                b[ind] = (-0.5 * vort_1D[ind + 1] * h**2) #* np.cos(gamma_1D[ind])
                          # -0.5 * vort_1D[ind + nx] * h**2 * np.sin(gamma_1D[ind]))
                # print(b[ind])
            else:
                b[ind] = (-0.5 * vort_1D[ind + 1] * h**2) #* np.cos(gamma_1D[ind])
                          # -0.5 * vort_1D[ind - nx] * h**2 * np.sin(gamma_1D[ind]))
        for i, ind in enumerate(bot_boundary_1D):
            # vort_1D[ind-1] = -100
            if np.sin(gamma_1D[ind]) < 0:
                b[ind] = (-0.5 * vort_1D[ind - 1] * h**2) #* np.cos(gamma_1D[ind])
                          # -0.5 * vort_1D[ind + nx] * h**2 * np.sin(gamma_1D[ind]))
            else:
                b[ind] = (-0.5 * vort_1D[ind - 1] * h**2) #* np.cos(gamma_1D[ind])
                          # -0.5 * vort_1D[ind - nx] * h**2 * np.sin(gamma_1D[ind]))
            # TODO  angles
        
            
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
        grad = np.gradient(Psi, h, h)
        u = -grad[0]
        v = -grad[1]
        # return u, v, Psi
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
        # return u, v, Psi



    

# %%
    # cm = 'Greys'
    
    # plt.figure()
    # plt.imshow(u)
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(v)
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(np.sqrt(u**2+v**2))
    # plt.colorbar()
    
    x = np.linspace(-2, 2, nx)
    
    plt.figure(figsize=(9,7))                # set size
    plt.xlabel('x', fontsize=14)             # label x
    plt.ylabel('y', fontsize=14, rotation=0) # label y

    # plot contours
    m = np.sqrt(u**2+v**2)
    velocity = plt.contourf(x, np.flip(x), m, vmin=0, vmax=None)
    cbar = plt.colorbar(velocity)
    cbar.set_label('Velocity magnitude', fontsize=14);

    # plot vector field

    plt.quiver(x[::4], np.flip(x[::4]),
                  u[::4,::4], v[::4,::4])
    plt.grid()
    
    # %%
    X, Y = np.meshgrid(x, np.flip(x))
    magvp = np.sqrt(arc.velocity(X, Y)[0]**2 + arc.velocity(X, Y)[1]**2)
    
    # plt.figure()
    # plt.imshow(m)
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(magvp)
    # plt.colorbar()
    