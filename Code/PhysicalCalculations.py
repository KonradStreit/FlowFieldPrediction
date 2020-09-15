# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:23:05 2020

@author: kvstr
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import solve_banded
from scipy.interpolate import griddata
import time
from numba import njit
from numba import prange
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from skimage.measure import block_reduce

# %% Continuity

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


# %% Momentum
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
    dx : Mx1 Array
        x-distance cell centre-face.
    dy : Nx1
        y-distance cell centre-face.

    Returns
    -------
    error: NxM array
        Momentum error on each grid point.

    """
    nu = 1.05e-6
    if not (np.shape(vort) == np.shape(u) and np.shape(vort) == np.shape(v)):
        print('Momentum: Shape mismatch')
        return None
    else:
       # Vorticity Gradient x
       vortx = np.zeros_like(vort)
       vortxx = np.zeros_like(vortx)
       vortx[:, -1] = np.divide(vort[:, -1]-vort[:, -2], dx[-1]+dx[-2])
       vortx[:, 0] = np.divide(vort[:, 1]-vort[:, 0], dx[1]+dx[0])
       for i in range(1, vort.shape[1]-1):
           vortx[:, i] = (np.divide(vort[:, i+1]*dx[i] - vort[:, i]*dx[i+1],
                                    dx[i]+dx[i+1])
                          -np.divide(vort[:, i]*dx[i-1] - vort[:, i-1]*dx[i],
                                     dx[i]+dx[i-1])) / 2*dx[i]
       vortxx[:, -1] = np.divide(vortx[:, -1]-vortx[:, -2], dx[-1]+dx[-2])
       vortxx[:, 0] = np.divide(vortx[:, 1]-vortx[:, 0], dx[0]+dx[1])
       for i in range(1, vortx.shape[1]-1):
            vortxx[:, i] = (np.divide(vortx[:, i+1]*dx[i] - vortx[:, i]*dx[i+1],
                                      dx[i]+dx[i+1])
                           -np.divide(vortx[:, i]*dx[i-1] - vortx[:, i-1]*dx[i],
                                      dx[i]+dx[i-1])) / 2*dx[i]
        
       # Vorticity Gradient y
       vorty = np.zeros_like(vort)
       vortyy = np.zeros_like(vortx)
       vorty[-1, :] = np.divide(vort[-1, :]-vort[-2, :], dy[-1]+dy[-2])
       vorty[0, :] = np.divide(vort[1, :]-vort[0, :], dy[0]+dy[1])
       for i in range(1, vort.shape[0]-1):
           vorty[i, :] = (np.divide(vort[i+1, :]*dy[i] - vort[i, :]*dy[i+1],
                                    dy[i]+dy[i+1])
                          -np.divide(vort[i, :]*dy[i-1] - vort[i-1, :]*dy[i],
                                     dy[i]+dy[i-1])) / 2*dy[i]
       vortyy[-1, :] = np.divide(vorty[-1, :]-vorty[-2, :], dy[-1]+dy[-2])
       vortyy[0, :] = np.divide(vorty[1, :]-vorty[0, :], dy[0]+dy[1])
       for i in range(1, vorty.shape[0]-1):
           vortyy[i, :] = (np.divide(vorty[i+1, :]*dy[i] - vorty[i, :]*dy[i+1],
                                    dy[i]+dy[i+1])
                          -np.divide(vorty[i, :]*dy[i-1] - vorty[i-1, :]*dy[i],
                                     dy[i]+dy[i-1])) / 2*dy[i]

       t1 = np.multiply(u, vortx)
       t2 = np.multiply(v, vorty)
       t3 = nu * (vortxx+vortyy)
       error = abs(np.subtract(t1+t2, t3))
    return error


# %% CellSizes
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



# %% Vorticity
def Vorticity(u, v, dx, dy):
    """
    Calculates the Vorticity from velocity Components and Cell sizes

    Parameters
    ----------
    u : NxM Array
        u-velocity at each grid point.
    v : NxM Array
        v-velocity at each grid point.
    dx : Mx1 Array
        Half cell sizes in x-direction.
    dy : Nx1 Array
        Half cell sizes in y-direction.

    Returns
    -------
    vort : NxM Array
        Vorticity at each grid point.

    """

    # Gradient v-velocity
    dvdx = np.zeros_like(v)
    dvdx[:, 0] = np.divide(v[:, 1] - v[:, 0], dx[0]+dx[1])
    dvdx[:, -1] = np.divide(v[:, -1]-v[:, -2], dx[-1]+dx[-2])
    for i in range(1, v.shape[1]-1):
        vpl = np.divide(v[:, i]*dx[i+1] + v[:, i+1]*dx[i], dx[i]+dx[i+1])
        vmi = np.divide(v[:, i]*dx[i-1] + v[:, i-1]*dx[i], dx[i-1]+dx[i])
        dvdx[:, i] = np.divide(vpl - vmi, 2*dx[i])
    # Gradient u-velocity
    dudy = np.zeros_like(u)
    dudy[0, :] = np.divide(u[1, :] - u[0, :], dy[0]+dy[1])
    dudy[-1, :] = np.divide(u[-1, :] - u[-2, :], dy[0]+dy[1])
    for i in range(1, u.shape[0]-1):
        upl = np.divide(u[i, :]*dy[i+1] + u[i+1, :]*dy[i], dy[i]+dy[i+1])
        umi = np.divide(u[i, :]*dy[i-1] + u[i-1, :]*dy[i], dy[i]+dy[i-1])
        dudy[i, :] = np.divide(upl-umi, 2*dy[i])
        
    vort = dvdx - dudy
    return vort



# %% Pressure
def Pressure(x, y, u, v, x_arc, y_arc, step=1, rho=1, nu=128/150):
    dudx = np.gradient(u, step, axis=1)
    dudy = np.gradient(u, step, axis=0)
    
    dpdx = -(rho*(u*dudx + v * dudy)
             + nu*(np.gradient(dudx, step, axis=1))
                   + np.gradient(dudy, step, axis=0))
    
    dvdx = np.gradient(v, step, axis=1)
    dvdy = np.gradient(v, step, axis=0)
    
    dpdy = -(rho*(u*dvdx + v * dvdy)
             + nu*(np.gradient(dvdx, step, axis=1)
                   + np.gradient(dvdy, step, axis=0)))
    
    p = np.empty_like(x)
    p[:, 0] = step * dpdx[:, 0]
    i = 1
    while (x[0, i] < np.min(x_arc)) and i < len(x):
        p[:, i] = p[:, i-1] + step * dpdx[:, i]
        i += 1
    
    for k in range(i, len(x)):
        p[0, k] = p[0, k-1] + step * dpdx[0, k]
        p[-1, k] = p[-1, k-1] + step * dpdx[-1, k]
    
    k = 1
    while (y[k, 0] < np.min(y_arc)) and k < len(x):
        p[k, :] = p[k-1, :] + step * dpdy[k, :]
        k += 1
        
    l = len(x)-2
    while (y[l, 0] > np.max(y_arc)) and l > 0:
        p[l, :] = p[l+1, :] - step * dpdy[l, :]
        l -= 1
    
    while x[0, i] <= np.max(x_arc):
        yl = np.interp(x[0, i], x_arc, y_arc)
        ind = abs(yl - y[:, 0]).argmin()
        for m in range(k, ind):
            p[m, i] = p[m-1, i] + step * dpdy[k, i]
        for m in range(l-ind):
            p[l-m, i] = p[l-m+1, i] - step * dpdy[l-m, i]
        
        i += 1
    
    yl = np.interp(x[0, i], x_arc, y_arc)
    ind = abs(yl - y[:, 0]).argmin()
    for m in range(k, ind):
        p[m, i] = p[m-1, i] + step * dpdy[k, i]
    for m in range(l-ind):
        p[l-m, i] = p[l-m+1, i] - step * dpdy[l-m, i]
    
    i += 1
    for m in range(k, l+1):
        p[m, i:] = p[m-1, i:] + step * dpdy[m, i:]
    return p


# %% Forces
def Forces(x, y, u, v, p, xb, yb, Sb, dr, Angles, chord, nu=128/150):
    """
    

    Parameters
    ----------
    x : meshgrid x
    y : TYPE
        meshgrid y.
    u : TYPE
        x-velocity field.
    v : TYPE
        y-velocity field.
    p : TYPE
        pressure field.
    xb : TYPE
        Body x-coordinates.
    yb : TYPE
        Body y-coordinates.
    dr : TYPE
        Body half thickness.
    Angles : TYPE
        Body panel angles.
    chord : TYPE
        Body Chord length.
    nu : TYPE, optional
        Viscosity. The default is 128/150.

    Returns
    -------
    Lift : TYPE
        Lift COefficient.
    Drag : TYPE
        Drag Coefficient.

    """
    
    ptop = np.zeros_like(xb)
    pbot = np.zeros_like(ptop)
    top = np.zeros((len(xb), 2))
    bot = np.zeros_like(top)
    i = 0
    for xt, yt in zip(xb, yb):
        
        top[i, 0] = xt - np.sin(Angles[i]) * dr
        top[i, 1] = yt + np.cos(Angles[i]) * dr
        
        bot[i, 0] = xt + np.sin(Angles[i]) * dr
        bot[i, 1] = yt - np.cos(Angles[i]) * dr

        i += 1
    
    maskx = np.logical_and(x > np.min(xb)-5, x<np.max(xb)+5)
    masky = np.logical_and(y > np.min(yb)-5, y<np.max(yb)+5)
    mask = np.logical_and(maskx, masky)
    
    ptop = griddata(np.vstack((x[mask], y[mask])).transpose(),
                    p[mask], top)
    pbot = griddata(np.vstack((x[mask], y[mask])).transpose(),
                    p[mask], bot)
    
    veltop = griddata(np.vstack((x[mask], y[mask])).transpose(),
                    np.sqrt(u**2 + v**2)[mask], top)
    velbot = griddata(np.vstack((x[mask], y[mask])).transpose(),
                    np.sqrt(u**2 + v**2)[mask], bot)

    Lift = 2 * (np.sum(np.cos(Angles)*(pbot - ptop)* Sb)
                + nu * np.sum(np.sin(Angles)*(veltop+velbot)* Sb)) / chord
    Drag = 2 * (-np.sum(np.sin(Angles)*(pbot - ptop)* Sb)
                + nu * np.sum(np.cos(Angles)*(veltop+velbot)* Sb))/ chord
    
    return Lift, Drag

# %% Read VTR
def read_vtr(file):
    """
    

    Parameters
    ----------
    file : PATH
        Path to file to be read.

    Returns
    -------
    u : NxM Array
        u-velocity at each grid point.
    v : NxM Array
        v-velocity at each grid point..
    Pressure : NxM Array
        Pressure at each grid point..
    x : Mx1 Array
        x-coordinates of gridpoints.
    y : Nx1 Array
        y-coordinates of gridpoints.

    """
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


# %% Read Data
def Read_Data(AoAs, start=50, timesteps=100, step=1, verbose=False,
              getPressure=False):
    """
    

    Parameters
    ----------
    AoAs : tuple with N scalar entries
        Angles of attack in degrees for which to read data.
    start : scalar, optional
        First tmestep to use. The default is 50.
    timesteps : TYPE, optional
        Total number of timesteps, will use start-timesteps. The default is 100.

    Returns
    -------
    x : MxO Array
        x-coordinates of grid.
    y : MxO Array
        y-coordinates of grid.
    u : NxMxO Array
        u-velocity at each AoA and grid point.
    v : NxMxO Array
        v-velocity at each AoA and grid point.
    vort : NxMxO Array
        vorticity at each AoA and grid point.
    u_std : NxMxO Array
        u standard deviation at each AoA and grid point.
    v_std : NxMxO Array
        v standard deviation at each AoA and grid point.
    Cont : NxMxO Array
       Continuity error at each AoA and grid point.
    Mom : NxMxO Array
        Momentum error at each AoA and grid point.

    """
    n_files = timesteps-start
    j = 0
    for alpha in AoAs:
        print('alpha = {:03d}deg'.format(alpha))
        
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
        x_stretch, y_stretch = np.meshgrid(xlin, ylin)
        
        # Statistics
        if j == 0:
            u_std = np.zeros((len(AoAs), u0.shape[0], u0.shape[1]))
            v_std = np.zeros((len(AoAs), u0.shape[0], u0.shape[1]))
        u_stretch = np.mean(u_files, axis=0)
        v_stretch = np.mean(v_files, axis=0)
        Press = np.mean(Press_files, axis=0)
        u_std[j] = np.std(u_files, axis=0)
        v_std[j] = np.std(v_files, axis=0)
        
        if verbose:
            print('Mean std u:              %.8f' % np.mean(u_std[j]))
            print('Mean std v:              %.8f' % np.mean(v_std[j]))
            print('Max std u:               %.8f' % np.max(u_std[j]))
            print('Max std v:               %.8f' % np.max(v_std[j]))
        
        dx, dy = CellSizes(xlin, ylin)
        vort_stretch = Vorticity(u_stretch, v_stretch, dx, dy)
        
        # Interpolate to regular grid
        if j == 0:
            xmin = np.ceil(np.min(xlin))
            xmax = np.floor(np.max(xlin))
            ymin = np.ceil(np.min(ylin))
            ymax = np.floor(np.max(ylin))
            x, y = (np.mgrid[xmin+1:xmax:step, ymin:ymax:step] - .5)
            x = x.transpose().astype(float)
            y = y.transpose().astype(float)
            u = np.zeros((len(AoAs), x.shape[0], x.shape[1]))
            v = np.zeros_like(u)
            m = np.zeros_like(u)
            vort = np.zeros_like(u)
            Cont = np.zeros_like(u)
            Mom = np.zeros_like(u)
            
            if getPressure:
                p = np.zeros_like(u)
            
        u[j] = griddata(np.array([x_stretch.flatten(), y_stretch.flatten()]).transpose(),
                         u_stretch.flatten(), (x, y))
        v[j] = griddata(np.array([x_stretch.flatten(), y_stretch.flatten()]).transpose(),
                         v_stretch.flatten(), (x, y))
        vort[j] = griddata(np.array([x_stretch.flatten(), y_stretch.flatten()]).transpose(),
                         vort_stretch.flatten(), (x, y))
        if getPressure:
            p[j] = griddata(np.array([x_stretch.flatten(), y_stretch.flatten()]).transpose(),
                            Press.flatten(), (x, y))
        dx, dy = np.ones_like(x[0, :])*.5, np.ones_like(y[:, 0])*.5
        # vort[j] = Vorticity(u[j], v[j], dx, dy)
        
        Cont[j] = Continuity(u[j], v[j], x, y)
        Mom[j] = Momentum(vort[j], u[j], v[j], dx, dy)
    
        print('Max Continuity Error:    %.8f' % np.max(Cont[j]))
        print('Max Momentum Error:      %.8f' % np.max(Mom[j]))
        
        j += 1


    if getPressure:
        return x, y, u, v, vort, p, u_std, v_std, Cont, Mom
    else:        
        return x, y, u, v, vort, u_std, v_std, Cont, Mom



# %% make_square
def make_square(x, y, u, v, vort, square, step, p=None, Mom=None, xOffset=0):
    
    # size0 = int(square/step)
    
    mask = np.logical_and(abs(x-xOffset)<(square/2), abs(y)<(square/2))

    size0 = int(np.sqrt(np.sum(mask)))
    
    # TODO change order
    vort_square = vort[:, mask].reshape((len(u), size0, size0))
    
    x_square = x[mask].reshape((size0, size0))
    
    y_square = y[mask].reshape((size0, size0))

    u_square = u[:, mask].reshape((vort.shape[0], size0, size0))

    v_square = v[:, mask].reshape((vort.shape[0], size0, size0))
    
    if p is not None:
        p_square = p[:, mask].reshape((vort.shape[0], size0, size0))
    
    if step is not 1:
        vort_square = block_reduce(vort_square, block_size=(1, step, step), func=np.mean)
        x_square = block_reduce(x_square, block_size=(step, step), func=np.mean)
        y_square = block_reduce(y_square, block_size=(step, step), func=np.mean)
        u_square = block_reduce(u_square, block_size=(1, step, step), func=np.mean)
        v_square = block_reduce(v_square, block_size=(1, step, step), func=np.mean)
        if p is not None:
            p_square = block_reduce(p_square, block_size=(1, step, step), func=np.mean)

    
    if Mom is not None:
        Mom_square = Mom[:, mask].reshape((vort.shape[0], size0, size0))
        Mom_square = block_reduce(Mom_square, block_size=(1, step, step),
                                  func=np.mean)
        return x_square, y_square, u_square, v_square, vort_square, Mom_square
    if p is not None:
        return x_square, y_square, u_square, v_square, vort_square, p_square
    else:
        return x_square, y_square, u_square, v_square, vort_square 

    
# %% u_omega
@njit(parallel=True)
def u_omega(x, y, xi, yi, omega, h):
    """
    
    Parameters
    ----------
    x : Vector
        x-location of points to be evaluated.
    y : Vector
        y-location of points to be evaluated.
    xi : Vector
        x-location of point with non-negligible vorticity.
    yi : Vector
        y-location of point with non-negligible vorticity.
    omega : Vector
        Vorticity as above specified points.
    h : Scalar
        Step size.

    Returns
    -------
    u : Vector
        induced velocity in x-direction at evaluated points.
    v : Vector
        induced velocity in x-direction at evaluated points

    """
    # u, v = np.zeros_like(x), np.zeros_like(x)
    # for xp, yp, op in zip(xi, yi, omega):
    #     rx, ry = x-xp, y-yp
    #     a = op/(rx**2 + ry**2 + 0.5*h**2) # +.5h**2 to avoid division by zero
    #     u += -a*ry
    #     v += a*rx
    # u = u*h**2 / (2*np.pi)
    # v = v*h**2 / (2*np.pi)
    # return u, v
    u, v = np.zeros_like(x), np.zeros_like(x)
    for i in prange(len(xi)):
    # for xp, yp, op in zip(xi, yi, omega):
        xp = xi[i]
        yp = yi[i]
        op = omega[i]
        rx, ry = x-xp, y-yp
        a = op/(rx**2 + ry**2 + 0.5*h**2) # +.5h**2 to avoid division by zero
        u += -a*ry
        v += a*rx
    u = u*h**2 / (2*np.pi)
    v = v*h**2 / (2*np.pi)
    if np.max(u) == 0:
        print('u is 0')
    return u, v


def u_omega_nojit(x, y, xi, yi, omega, h):
    """
    
    Parameters
    ----------
    x : Vector
        x-location of points to be evaluated.
    y : Vector
        y-location of points to be evaluated.
    xi : Vector
        x-location of point with non-negligible vorticity.
    yi : Vector
        y-location of point with non-negligible vorticity.
    omega : Vector
        Vorticity as above specified points.
    h : Scalar
        Step size.

    Returns
    -------
    u : Vector
        induced velocity in x-direction at evaluated points.
    v : Vector
        induced velocity in x-direction at evaluated points

    """
    # u, v = np.zeros_like(x), np.zeros_like(x)
    # for xp, yp, op in zip(xi, yi, omega):
    #     rx, ry = x-xp, y-yp
    #     a = op/(rx**2 + ry**2 + 0.5*h**2) # +.5h**2 to avoid division by zero
    #     u += -a*ry
    #     v += a*rx
    # u = u*h**2 / (2*np.pi)
    # v = v*h**2 / (2*np.pi)
    # return u, v
    u, v = np.zeros_like(x), np.zeros_like(x)
    for i in prange(len(xi)):
    # for xp, yp, op in zip(xi, yi, omega):
        xp = xi[i]
        yp = yi[i]
        op = omega[i]
        rx, ry = x-xp, y-yp
        a = op/(rx**2 + ry**2 + 0.5*h**2) # +.5h**2 to avoid division by zero
        u += -a*ry
        v += a*rx
    u = u*h**2 / (2*np.pi)
    v = v*h**2 / (2*np.pi)
    if np.max(u) == 0:
        print('u is 0')
    return u, v

# %%
# @njit(parallel=True)
def u_indu(x, y, xi, yi, omega, h, cell=1):
    """
    
    Parameters
    ----------
    x : Vector
        x-location of points to be evaluated.
    y : Vector
        y-location of points to be evaluated.
    xi : Vector
        x-location of point with non-negligible vorticity.
    yi : Vector
        y-location of point with non-negligible vorticity.
    omega : Vector
        Vorticity as above specified points.
    h : Scalar
        Step size.

    Returns
    -------
    u : Vector
        induced velocity in x-direction at evaluated points.
    v : Vector
        induced velocity in x-direction at evaluated points

    """
    u, v = np.zeros_like(x), np.zeros_like(x)
    for i in prange(len(xi)):
        xp = xi[i]
        yp = yi[i]
        op = omega[i]
        rx, ry = x-xp, y-yp
        r = np.sqrt(rx**2 + ry**2) / cell
        a = op/(rx**2 + ry**2 + 0.5*h**2) # +.5h**2 to avoid division by zero
        a[r<1] = (op * r[r<1]) / cell**2
        u += -a*ry
        v += a*rx
    u = u*h**2 / (2*np.pi)
    v = v*h**2 / (2*np.pi)
    return u, v

# %% Gen_Arc

def Gen_Arc(alpha):
    alpharad = np.deg2rad(-alpha)

    dr = 5
    camber = 0.12
    chord= 128.
    
        
    R = chord / (2 * (4.*camber/(4*camber**2+1)))
    s = 8*R*camber**2 / (4*camber**2+1)
    
    center = [0, -R+s]
    
    theta = np.linspace(0, 2*np.pi, 200)
    x1 = np.cos(theta)*R + center[0]
    y1 = np.sin(theta)*R + center[1]
    
    x1 = x1[np.where(y1 >= 0)]
    y1 = y1[np.where(y1 >= 0)]
    
    x_rot = np.cos(alpharad) * x1 - np.sin(alpharad) * y1
    y_rot = np.sin(alpharad) * x1 + np.cos(alpharad) * y1
    
    return np.flip(x_rot), np.flip(y_rot)

def Gen_Arc_full_res(alpha):
    return Gen_Arc(alpha)


# %% Plot Vel Contour & Quiver

def plot_flow(x, y, u, v, skip=20):
    plt.figure()
    plt.contourf(x, y, np.sqrt(u**2 + v**2))
    plt.colorbar()
    plt.quiver(x[::skip, ::skip], y[::skip, ::skip],
               u[::skip, ::skip], v[::skip, ::skip])

    