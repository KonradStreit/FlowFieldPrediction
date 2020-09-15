#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:34:11 2020

@author: konrad
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Vortex Line Code from Weymouth
def influence(x, y): # unit panel influence functions
    lr = 0.5*np.log(((x-1)**2+y**2)/(x**2+y**2))
    lr[np.isinf(lr)] = 0
    dt = np.arctan2(y, x-1)-np.arctan2(y, x)
    factors = np.array([-y*lr+(1-x)*dt, (1-x)*lr+y*dt-1, y*lr+x*dt, x*lr-y*dt+1])/(2*np.pi)
    # factors[:, np.where((x**2+y**2) == 0)] = 0
    # factors[0, abs(x)<=1] = (x**2 + y**2)[abs(x)<=1]
    # factors[np.isinf(factors)] = 0
    return factors



class VortexLine(object):
    def __init__(self, x, y): # This geometry stuff only needs to be done once
        self.x,self.y = x,y
        self.xc = 0.5*(x[1:]+x[:-1])
        self.yc = 0.5*(y[1:]+y[:-1])
        self.sx = x[1:]-x[:-1]
        self.sy = y[1:]-y[:-1]
        self.S = np.sqrt(self.sx**2+self.sy**2)
        self.sx /= self.S; self.sy /= self.S
        self.N = len(self.S)
        self.d = np.append(0.,np.cumsum(self.S))

    def velocity(self,gamma,x,y):
        u, v = np.zeros_like(x),np.zeros_like(x)
        for i in range(self.N):
            a,b,c,d = self.f(i,x,y)
            
            u += gamma[i]*a+gamma[i+1]*c
            v += gamma[i]*b+gamma[i+1]*d
        return u,v
    
    def velocity_ext(self,gamma,x,y):
        u, v = np.zeros_like(x),np.zeros_like(x)
        for i in range(self.N):
            a,b,c,d = self.f(i,x,y)
            
            r = np.sqrt((x-self.x[i])**2 + (y-self.y[i])**2) / self.S[i]
            a[r<1] = (gamma[i]*r[r<1]) / (self.S[i]**2*2*np.pi)
            b[r<1] = (gamma[i]*r[r<1]) / (self.S[i]**2*2*np.pi)
            
            r = np.sqrt((x-self.x[i+1])**2 + (y-self.y[i+1])**2) / self.S[i]
            c[r<1] = (gamma[i+1]*r[r<1]) / (self.S[i]**2*2*np.pi)
            d[r<1] = (gamma[i+1]*r[r<1]) / (self.S[i]**2*2*np.pi)
            
            u += gamma[i]*a+gamma[i+1]*c
            v += gamma[i]*b+gamma[i+1]*d
        return u,v

    def solve_gamma(self,exvelo):
        # Fill matrix by adding up panel influences
        A = np.zeros((self.N,self.N))
        for i in range(self.N):
            a,b,c,d = self.f(i,self.xc,self.yc)
            
            A[:,i] += -a*self.sy+b*self.sx
            if i+1<self.N: # Kutta condition, gamma[N]=0
                A[:,i+1] += -c*self.sy+d*self.sx

        # Determine upwash by evaluating the external velocity function
        u,v = exvelo(self.xc,self.yc)
        w = -u*self.sy+v*self.sx
        return np.append(np.linalg.solve(A, w),0.) # return


    def f(self,i,x,y): # Influence function of panel i
        # Normalize x,y
        xt,yt = (x-self.x[i])/self.S[i], (y-self.y[i])/self.S[i]
        # Rotate x,y and then rotate a,b,c,d back
        sx, sy = self.sx[i], self.sy[i]
        a,b,c,d = influence(xt*sx+yt*sy, yt*sx-xt*sy)
        return a*sx-b*sy,b*sx+a*sy,c*sx-d*sy,d*sx+c*sy