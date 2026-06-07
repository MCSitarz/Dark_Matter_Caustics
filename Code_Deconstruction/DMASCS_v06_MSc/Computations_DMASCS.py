#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:20:41 2024
@author: michaelsitarz
"""
import numpy as np
# import numba as nb

# @nb.jit
def _interpolation_call(x, shape, data):    
    X1 = np.floor(x) % shape
    X1 = X1.astype(np.int64)    
    X2 = np.ceil(x) % shape
    X2 = X2.astype(np.int64)
    xm = x % 1.0
    xn = 1.0 - xm
    
    f1 = data[X1[:,0], X1[:,1]]
    f2 = data[X2[:,0], X1[:,1]]
    f3 = data[X1[:,0], X2[:,1]]    
    f4 = data[X2[:,0], X2[:,1]]

    Value = f1 * xn[:,0] * xn[:,1] + \
            f2 * xm[:,0] * xn[:,1] + \
            f3 * xn[:,0] * xm[:,1] + \
            f4 * xm[:,0] * xm[:,1]    
    return Value  

class Interp2D:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
      
    def __call__(self, x):
        return _interpolation_call(x, np.array(self.shape), np.array(self.data))  


class Computations:
    def __init__(self, UP):
        self.UP = UP
    
    '''Fast 2D gradient function.'''
    def gradient_dual(self, F, i):
        return F - np.roll(F, 1, axis=i)

    def fix_Boundries(self, x):
        for i in range(len(np.ravel(x[0]))):
            if np.ravel(x[0])[i] > self.UP.L:
                np.ravel(x[0])[i] = np.ravel(x[0])[i] - self.UP.L
            if np.ravel(x[1])[i] > self.UP.L:
                np.ravel(x[1])[i] = np.ravel(x[1])[i] - self.UP.L
            if np.ravel(x[0])[i] < 0:
                np.ravel(x[0])[i] = np.ravel(x[0])[i] + self.UP.L
            if np.ravel(x[1])[i] < 0:
                np.ravel(x[1])[i] = np.ravel(x[1])[i] + self.UP.L
        return x

    '''Function to split particle grid into a force grid for CIC based calculations.'''
    def subdiv_unitcell_gen(self, n):
        subdiv = np.array( \
            [[0.0, 0.0],
              [0.0, 0.5],
              [0.5, 0.0],
              [0.5, 0.5]])
        if (n == 0):
            return np.array([0,0])
        else:
            P = subdiv[:,np.newaxis] + \
                0.5 * self.subdiv_unitcell_gen(n-1)[np.newaxis,:]
            return P.reshape((int(P.size/2), 2))
        
