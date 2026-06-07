#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:52:40 2024
@author: michaelsitarz
"""
import os
import sys
import scipy as sp
import numpy as np
import seaborn as sns
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.stats.kde import gaussian_kde

class Plotting_Functions:
    def __init__(self, UP, B, C):
        self.UP = UP
        self.B = B
        self.C = C
        
    def Initial_Field(self, Matrix, COLOR, TITLE, SAVE):
        fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        g = sns.heatmap(Matrix, cmap=COLOR, ax=ax2, cbar_kws={"shrink": 0.6})
        g.tick_params(left=False, bottom=False)
        ax2.invert_yaxis()
        ax1.contour(Matrix, cmap=COLOR)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
                
    def Heatmap_Plotting(self, Matrix, COLOR, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        ax = sns.heatmap(Matrix, cmap=COLOR)
        ax.invert_yaxis()
        plt.savefig(SAVE + '/'+ TITLE + '_Heatmap.png')
        plt.close()
        plt.clf()
        
    def Contour_Plotting(self, Matrix, COLOR, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        ax = plt.contour(Matrix, cmap=COLOR)
        plt.savefig(SAVE + '/'+ TITLE + '_Contour.png')
        plt.close()
        plt.clf()
        
    def Contour_Plots(self, a, X, Y, Z, LEVELS, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.tricontour(X, Y, Z, cmap='nipy_spectral', levels=LEVELS)
        plt.colorbar()
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def Power_Spectrum(self, Matrix, TITLE, SAVE, Matrix_Label):
        kbins = np.arange(0.5, self.UP.Np//2+1, 1.) #/ (2*np.pi / self.UP.Np)
        kvals = 0.5 * (kbins[1:] + kbins[:-1]) #/ (2*np.pi / self.UP.Np)
        fourier_image = sp.fftpack.fftn(np.ravel(Matrix))
        fourier_amplitudes = np.abs(fourier_image)**2
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins, _, _ = stats.binned_statistic(self.B.k_norm.flatten(), fourier_amplitudes, statistic = "mean", bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)        
        plt.loglog(kvals, Abins, label="$Log(k_{Cutoff}) = $" + str(round(np.log10(self.UP.k_cutoff),3)))
        plt.ylabel("log(P(k))")
        plt.xlabel("$log(k/h^3 \ Mpc^{-3})$")
        plt.tight_layout()  
        # plt.legend()
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()  
    
    def D2_Scatter(self, x, y, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(x, y, s=1, c='k')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf() 
        
    def density_Plots(self, x, y, TITLE, SAVE):
        fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        counts, xbins, ybins, p1 = ax1.hist2d(x,y,(self.UP.Np * self.UP.qpf,self.UP.Np * self.UP.qpf), cmap=plt.cm.gist_ncar, vmin=0, vmax=35)
    
        k = gaussian_kde(np.vstack([x, y]), bw_method=0.1)
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        p2 = ax2.contour(xi, yi, zi.reshape(xi.shape) * len(x), alpha=0.5, cmap = plt.cm.jet, vmin=0, vmax=35)
        
        ax1.set_xlim(0, self.UP.L)
        ax1.set_ylim(0, self.UP.L)
        ax2.set_xlim(0, self.UP.L)
        ax2.set_ylim(0, self.UP.L)
    
        plt.colorbar(p1, ax=ax1, fraction=0.046, pad=0.04)
        plt.colorbar(p2, ax=ax2, fraction=0.046, pad=0.04)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def D2_Scatter_with_Color(self, x, y, z, TITLE, SAVE, PALETTE, VMIN, VMAX):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(x, y, s=1, c=z, cmap=PALETTE, vmin=VMIN, vmax=VMAX)
        plt.colorbar()
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()   

    def Singularity_Value_Tag(self, a, X, Y, Z, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        S = np.argwhere(Z > 1)
        plt.scatter(X, Y, s=1, c='white')
        plt.scatter(X[S], Y[S], s=1, c='red')
        plt.xlim(0,self.UP.L)
        plt.ylim(0,self.UP.L)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()  
        
    def Singularity_Value_Tag_Dbl(self, a, X, Y, R, S, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(X, Y, s=1, c='white')
        plt.scatter(X[np.argwhere(R > 1)], Y[np.argwhere(R > 1)], s=1, c='red')
        plt.scatter(X[np.argwhere(S > 1)], Y[np.argwhere(S > 1)], s=1, c='blue')
        plt.xlim(0,self.UP.L)
        plt.ylim(0,self.UP.L)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def LXD(self, a, X, Y, Z, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(X, Y, s=1, c=Z, cmap='Spectral', vmax=1.0)
        plt.colorbar()
        plt.xlim(0,self.UP.L)
        plt.ylim(0,self.UP.L)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()   
        
    def LXD_Sing(self, a, X, Y, Z, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(X[Z > 1], Y[Z > 1], s=1, c=Z[Z > 1], cmap='Spectral', vmax=1.0)
        plt.colorbar()
        plt.xlim(0,self.UP.L)
        plt.ylim(0,self.UP.L)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf() 
    
    def Single_Contour_Eigenvalue(self, X, Y, Z, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.tricontour(X, Y, Z, cmap='nipy_spectral')
        plt.colorbar()
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
    
    def DT_Double_Eigenvectors(self, a, X, Y, U, V, S, T, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        n = 2
        plt.quiver(X[::n], Y[::n], (U/2)[::n], (V/2)[::n], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        plt.quiver(X[::n], Y[::n], (S/2)[::n], (T/2)[::n], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red',  headlength=0, headaxislength=0, pivot='middle')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def DT_Double_Eigenvectors_Zoom(self, a, X, Y, U, V, S, T, XLIM, YLIM, TITLE, SAVE):
        if a == 0.0:
            self.Index = np.where((X <= XLIM[1]) & (X >= XLIM[0]) & (Y <= YLIM[1]) & (Y >= YLIM[0]))
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U/2, V/2, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        plt.quiver(X, Y, S/2, T/2, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red',  headlength=0, headaxislength=0, pivot='middle')
        plt.xlim(min(X[self.Index]),max(X[self.Index]))
        plt.ylim(min(Y[self.Index]),max(Y[self.Index]))
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def DT_EigenVectors(self, a, Q, R, X, Y, U, V, TITLE, SAVE, SAVE2, SAVE3):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, np.arctan2(V, U), cmap=plt.cm.turbo, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', headlength=0, headaxislength=0, pivot='middle')
        plt.colorbar()
        plt.savefig(SAVE2 + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, np.hypot(V, U), cmap=plt.cm.turbo, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', headlength=0, headaxislength=0, pivot='middle')
        plt.colorbar()
        plt.savefig(SAVE3 + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def DT_EigenVectors_Zoom(self, a, Q, R, X, Y, U, V, XLIM, YLIM, TITLE, SAVE, SAVE2, SAVE3):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(YLIM[0], YLIM[1])
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, np.arctan2(V, U), cmap=plt.cm.turbo, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', headlength=0, headaxislength=0, pivot='middle')
        plt.colorbar()
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(YLIM[0], YLIM[1])
        plt.savefig(SAVE2 + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.quiver(X, Y, U, V, np.hypot(V, U), cmap=plt.cm.turbo, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', headlength=0, headaxislength=0, pivot='middle')
        plt.colorbar()
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(YLIM[0], YLIM[1])
        plt.savefig(SAVE3 + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
     
    def Double_Evectors_Contours(self, a, Qx, Qy, Xx, Xy, X, Y, U, V, S, T, LEVELS, TITLE, SAVE):
        k = gaussian_kde(np.vstack([X, Y]), bw_method=0.1)
        xi, yi = np.mgrid[X.min():X.max():X.size**0.5*1j,Y.min():Y.max():Y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        ax4.set_aspect('equal')
        
        ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', alpha=0.3)
        p1 = ax1.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)

        ax2.quiver(X, Y, S, T, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', alpha=0.3)
        p2 = ax2.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        

        ax3.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', alpha=0.3)
        ax3.quiver(X, Y, S, T, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls=':', fc='none', ec='red', alpha=0.3)
        p3 = ax3.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        
        p4 = ax4.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        
        plt.colorbar(p1, ax=ax1)
        plt.colorbar(p2, ax=ax2)
        plt.colorbar(p3, ax=ax3)
        plt.colorbar(p4, ax=ax4)
        ax1.set_title('$\\vec{v_1}$')
        ax2.set_title('$\\vec{v_2}$')
        ax3.set_title('$\\vec{v}_{ij}')
        ax4.set_title('$\\rho_{init}$')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
    
    def Double_Evectors_Contours_Zoom(self, a, Qx, Qy, Xx, Xy, X, Y, U, V, S, T, XLIM, YLIM, LEVELS, TITLE, SAVE):
        k = gaussian_kde(np.vstack([X, Y]), bw_method=0.1)
        xi, yi = np.mgrid[X.min():X.max():X.size**0.5*1j,Y.min():Y.max():Y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        ax4.set_aspect('equal')
        
        ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', alpha=0.3)
        p1 = ax1.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)

        ax2.quiver(X, Y, S, T, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', alpha=0.3)
        p2 = ax2.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        

        ax3.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', alpha=0.3)
        ax3.quiver(X, Y, S, T, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls=':', fc='none', ec='red', alpha=0.3)
        p3 = ax3.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        
        p4 = ax4.contour(xi, yi, zi.reshape(xi.shape), cmap = plt.cm.viridis)
        
        plt.colorbar(p1, ax=ax1)
        plt.colorbar(p2, ax=ax2)
        plt.colorbar(p3, ax=ax3)
        plt.colorbar(p4, ax=ax4)

        ax1.set_title('$\\vec{v_1}$')
        ax2.set_title('$\\vec{v_2}$')
        ax3.set_title('$\\vec{v}_{ij}')
        ax4.set_title('$\\rho_{init}$')
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(YLIM[0], YLIM[1])
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def Vector_Direction(self, a, X, Y, S, T, U, V, TITLE, SAVE):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        ax4.set_aspect('equal')
        
        if a == 0.0:
            self.Curr_x_Direction_Sign_v1 = np.sign(S - X)
            self.Curr_x_Direction_Sign_v2 = np.sign(U - X)
            self.Curr_y_Direction_Sign_v1 = np.sign(T - Y)
            self.Curr_y_Direction_Sign_v2 = np.sign(V - Y)
        if a != 0.0:
            self.Prev_x_Direction_Sign_v1 = self.Curr_x_Direction_Sign_v1
            self.Prev_x_Direction_Sign_v2 = self.Curr_x_Direction_Sign_v2
            self.Prev_y_Direction_Sign_v1 = self.Curr_y_Direction_Sign_v1
            self.Prev_y_Direction_Sign_v2 = self.Curr_y_Direction_Sign_v2
            
            self.Curr_x_Direction_Sign_v1 = np.sign(S - X)
            self.Curr_x_Direction_Sign_v2 = np.sign(U - X)
            self.Curr_y_Direction_Sign_v1 = np.sign(T - Y)
            self.Curr_y_Direction_Sign_v2 = np.sign(V - Y)
            
        ax1.quiver(X, Y, S, T, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='black', headlength=0, headaxislength=0, pivot='middle', alpha=0.5)
        if a != 0.0:
            ax1.quiver(X[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], Y[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], U[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], V[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', headlength=0, headaxislength=0, pivot='middle')
            ax1.quiver(X[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], Y[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], U[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], V[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        
        ax2.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='black', headlength=0, headaxislength=0, pivot='middle', alpha=0.5)
        if a != 0.0:
            ax2.quiver(X[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], Y[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], U[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], V[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', headlength=0, headaxislength=0, pivot='middle')
            ax2.quiver(X[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], Y[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], U[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], V[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        
        if a != 0.0:
            ax3.quiver(X[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], Y[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], U[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], V[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', headlength=0, headaxislength=0, pivot='middle')
            ax3.quiver(X[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], Y[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], U[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], V[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        
        if a != 0.0:
            ax4.quiver(X[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], Y[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], U[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], V[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='red', headlength=0, headaxislength=0, pivot='middle')
            ax4.quiver(X[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], Y[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], U[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], V[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], angles='xy', scale_units='xy', scale=1, units='xy', linewidth=1, ls='-', fc='none', ec='blue', headlength=0, headaxislength=0, pivot='middle')
        
        
        ax1.set_title('$\\vec{v_1}$')
        ax2.set_title('$\\vec{v_2}$')
        ax3.set_title('$\\vec{v_1}$')
        ax4.set_title('$\\vec{v_2}$')
        fig.suptitle(TITLE)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def Altered_Vector_Direction(self, a, Qx, Qy, Xx, Xy, S, T, U, V, TITLE, SAVE):
        if a == 0.0:
            self.LAnchorE1X = Qx + S
            self.LAnchorE1Y = Qy + T
            self.LAnchorE2X = Qx + U
            self.LAnchorE2Y = Qy + V
            
            self.Curr_x_Direction_Sign_v1 = np.sign(self.LAnchorE1X - Xx)
            self.Curr_y_Direction_Sign_v1 = np.sign(self.LAnchorE1Y - Xy)
            self.Curr_x_Direction_Sign_v2 = np.sign(self.LAnchorE2X - Xx)
            self.Curr_y_Direction_Sign_v2 = np.sign(self.LAnchorE2Y - Xy)
        
        if a != 0.0:
            self.Prev_x_Direction_Sign_v1 = self.Curr_x_Direction_Sign_v1
            self.Prev_y_Direction_Sign_v1 = self.Curr_y_Direction_Sign_v1
            self.Prev_x_Direction_Sign_v2 = self.Curr_x_Direction_Sign_v2
            self.Prev_y_Direction_Sign_v2 = self.Curr_y_Direction_Sign_v2
            
            self.Curr_x_Direction_Sign_v1 = np.sign(self.LAnchorE1X - Xx)
            self.Curr_y_Direction_Sign_v1 = np.sign(self.LAnchorE1Y - Xy)
            self.Curr_x_Direction_Sign_v2 = np.sign(self.LAnchorE2X - Xx)
            self.Curr_y_Direction_Sign_v2 = np.sign(self.LAnchorE2Y - Xy)
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), sharex = True, sharey = True)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        ax4.set_aspect('equal')
        
        ax1.scatter(Qx, Qy, c='black', s=1)
        if a != 0.0:
            ax1.scatter(Qx[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], Qy[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], c='red', s=1)
            ax1.scatter(Qx[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], Qy[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], c='blue', s=1)
        
        ax2.scatter(Qx, Qy, c='black', s=1)
        if a != 0.0:
            ax2.scatter(Qx[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], Qy[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], c='red', s=1)
            ax2.scatter(Qx[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], Qy[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], c='blue', s=1)
        
        ax3.scatter(Xx, Xy, c='black', s=1)
        if a != 0.0:
            ax3.scatter(Xx[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], Xy[self.Prev_x_Direction_Sign_v1 != self.Curr_x_Direction_Sign_v1], c='red', s=1)
            ax3.scatter(Xx[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], Xy[self.Prev_y_Direction_Sign_v1 != self.Curr_y_Direction_Sign_v1], c='blue', s=1)
        
        ax4.scatter(Xx, Xy, c='black', s=1)
        if a != 0.0:
            ax4.scatter(Xx[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], Xy[self.Prev_x_Direction_Sign_v2 != self.Curr_x_Direction_Sign_v2], c='red', s=1)
            ax4.scatter(Xx[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], Xy[self.Prev_y_Direction_Sign_v2 != self.Curr_y_Direction_Sign_v2], c='blue', s=1)
        
        ax1.set_title('$\\vec{v_1}$')
        ax2.set_title('$\\vec{v_2}$')
        ax3.set_title('$\\vec{v_1}$')
        ax4.set_title('$\\vec{v_2}$')
        fig.suptitle(TITLE + ' Red:x-flip, Blue:y-flip')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def Jacobian_Tagging(self, X, Y, Z, TITLE, SAVE):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(X, Y, s=1, c='black')
        plt.scatter(X[Z <= 0], Y[Z <= 0], s=1, c='red')
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()  
        
    def Lagrange_Jacobian_Category_Plotting_Total_Switch(self, Particles, Volumes, Counts, SAVE_0, TITLE):
        SAVE = SAVE_0 + '/Lagrange_Grid_Total'
        if not os.path.exists(SAVE):
            os.mkdir(SAVE)
        
        Size = 2
        Qx = Particles[:,0]
        Qy = Particles[:,1]

        fig = plt.figure()
        plt.gca().set_aspect('equal')
        colors = ['red', 'green', 'blue', 'cyan', 'orange', 
                  'yellow', 'purple', 'pink', 'brown', 'grey', 
                  'indigo', 'olive', 'salmon', 'darkorange', 'lightcoral']
        cmap = mpl.colors.ListedColormap(colors[0:len(np.unique(Counts))])
        for c in range(len(np.unique(Counts))):
            Color_Correspondence_Index = np.where(np.isin(Counts,c))
            for Volume in Color_Correspondence_Index:
                plt.scatter(Qx[Volume], Qy[Volume], color = cmap(c), s=Size)

        norm = mpl.colors.Normalize(vmin=0, vmax=len(np.unique(Counts)))
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        labels = np.arange(0, len(colors) - 1, 1)
        loc = labels + 0.5
        cb = plt.colorbar(mappable)
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        # plt.show()
        plt.close()
        plt.clf()
        
        SAVE = SAVE_0 + '/Lagrange_Grid_Layered'
        if not os.path.exists(SAVE):
            os.mkdir(SAVE)
        
        for c in range(len(np.unique(Counts))):
            Color_Correspondence_Index = np.where(np.isin(Counts,c))
            for Volume in Color_Correspondence_Index:
                plt.scatter(Qx[Volume], Qy[Volume], color = cmap(c), s=Size)
            title = TITLE + " Count Number " + str(c)
            norm = mpl.colors.Normalize(vmin=0, vmax=len(np.unique(Counts)))
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array([])
            labels = np.arange(0, len(colors) - 1, 1)
            loc = labels + 0.5
            cb = plt.colorbar(mappable)
            cb.set_ticks(loc)
            cb.set_ticklabels(labels)
            plt.gca().set_aspect('equal')
            plt.xlim(0, self.UP.L)
            plt.ylim(0, self.UP.L)
            plt.savefig(SAVE + '/' + title + '.png')
            plt.close()
            plt.clf()
        
    def Euler_Jacobian_Category_Plotting_Total_Switch(self, Particles, Volumes, Counts, SAVE_0, TITLE):
        SAVE = SAVE_0 + '/Euler_Grid_Total'
        if not os.path.exists(SAVE):
            os.mkdir(SAVE)
        
        Size = 2
        Xx = Particles[:,2]
        Xy = Particles[:,3]
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        colors = ['red', 'green', 'blue', 'cyan', 'orange', 
                  'yellow', 'purple', 'pink', 'brown', 'grey', 
                  'indigo', 'olive', 'salmon', 'darkorange', 'lightcoral']
        cmap = mpl.colors.ListedColormap(colors[0:len(np.unique(Counts))])
        for c in range(len(np.unique(Counts))):
            Color_Correspondence_Index = np.where(np.isin(Counts,c))
            for Volume in Color_Correspondence_Index:
                plt.scatter(Xx[Volume], Xy[Volume], color = cmap(c), s=Size)
        norm = mpl.colors.Normalize(vmin=0, vmax=len(np.unique(Counts)))
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        labels = np.arange(0, len(colors) - 1, 1)
        loc = labels + 0.5
        cb = plt.colorbar(mappable)
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        SAVE = SAVE_0 + '/Euler_Grid_Layered'
        if not os.path.exists(SAVE):
            os.mkdir(SAVE)
        
        for c in range(len(np.unique(Counts))):
            Color_Correspondence_Index = np.where(np.isin(Counts,c))
            for Volume in Color_Correspondence_Index:
                plt.scatter(Xx[Volume], Xy[Volume], color = cmap(c), s=Size)
            title = TITLE + " Count Number " + str(c)
            plt.gca().set_aspect('equal')
            plt.xlim(0, self.UP.L)
            plt.ylim(0, self.UP.L)
            norm = mpl.colors.Normalize(vmin=0, vmax=len(np.unique(Counts)))
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array([])
            labels = np.arange(0, len(colors) - 1, 1)
            loc = labels + 0.5
            cb = plt.colorbar(mappable)
            cb.set_ticks(loc)
            cb.set_ticklabels(labels)
            plt.savefig(SAVE + '/'+ title + '.png')
            plt.close()
            plt.clf()
    
    def Minkowski_Curve_Structures(self, Connected_Curves, Piecewise_Curves, a, TITLE, SAVE):        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        spacer_1 = []
        if len(Connected_Curves) > 0:
            for i in range(len(Connected_Curves)):
                Index = np.array(Connected_Curves[i]) * (self.UP.L / self.UP.Np)
                plt.plot(Index[:,0], Index[:,1])
                plt.xlim(0, self.UP.L)
                plt.ylim(0, self.UP.L)
        if len(Piecewise_Curves) > 0:
            for i in range(len(Piecewise_Curves)):
                Index = np.array(Piecewise_Curves[i]) * (self.UP.L / self.UP.Np)
                plt.plot(Index[:,0], Index[:,1], '-.')
                plt.xlim(0, self.UP.L)
                plt.ylim(0, self.UP.L)
        plt.xlim(0, self.UP.L)
        plt.ylim(0, self.UP.L)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        if len(Connected_Curves) > 0:
            for i in range(len(Connected_Curves)):
                Index = np.array(Connected_Curves[i]) * (self.UP.L / self.UP.Np)
                plt.plot(Index[:,0], Index[:,1])
        if len(Piecewise_Curves) > 0:
            for i in range(len(Piecewise_Curves)):
                Index = np.array(Piecewise_Curves[i]) * (self.UP.L / self.UP.Np)
                plt.plot(Index[:,0], Index[:,1], '-.')
        plt.xlim(35, 60)
        plt.ylim(40, 60)
        plt.savefig(SAVE + '/'+ TITLE + '_Zoom' + '.png')
        plt.close()
        plt.clf()

    def Geometric_Caustics(self, a, Particles, causticVector, causticParticle, TITLE, SAVE):
        SAVE01 = SAVE + '/Caustic_Vector_Lagrangian_Line_Plots'
        if not os.path.exists(SAVE01):
            os.mkdir(SAVE01)  
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for V in causticVector:
            plt.plot([Particles[V[0]][0], Particles[V[1]][0]], 
                     [Particles[V[0]][1], Particles[V[1]][1]], 
                     'k-', linewidth = .25)  
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.savefig(SAVE01 + '/'+ TITLE  + '.png')
        plt.close()
        plt.clf()
        
        SAVE02 = SAVE + '/Caustic_Particle_Lagrangian_Dot_Plots'
        if not os.path.exists(SAVE02):
            os.mkdir(SAVE02)  
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for C in causticParticle:
            plt.scatter(Particles[C][0], Particles[C][1], c='k', s=1)  
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.savefig(SAVE02 + '/'+ TITLE + '_Zoom' + '.png')
        plt.close()
        plt.clf()
        
        SAVE03 = SAVE + '/Caustic_Vector_Eulerian_Line_Plots'
        if not os.path.exists(SAVE03):
            os.mkdir(SAVE03)  
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for V in causticVector:
            plt.plot([Particles[V[0]][2], Particles[V[1]][2]], 
                     [Particles[V[0]][3], Particles[V[1]][3]], 
                     'k-', linewidth = .25)  
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.savefig(SAVE03 + '/'+ TITLE + '_Zoom' + '.png')
        plt.close()
        plt.clf()
        
        SAVE04 = SAVE + '/Caustic_Particle_Eulerian_Dot_Plots'
        if not os.path.exists(SAVE04):
            os.mkdir(SAVE04)  
        
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for C in causticParticle:
            plt.scatter(Particles[C][2], Particles[C][3], c='k', s=1)  
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.savefig(SAVE04 + '/'+ TITLE + '_Zoom' + '.png')
        plt.close()
        plt.clf()

    def Tessellations(self, Triangles, Particles, TITLE, SAVE):
       fig = plt.figure()
       plt.gca().set_aspect('equal')
       i = 0
       for tess in Triangles:
           tess_x = []
           tess_y = []
           for idx in tess:
               tess_x.append(Particles[idx][2])
               tess_y.append(Particles[idx][3])
           tess_x.append(Particles[tess[0]][2])
           tess_y.append(Particles[tess[0]][3])
           for i in range(len(tess_x) - 1):
               x1 = tess_x[i]
               x2 = tess_x[i + 1]
               y1 = tess_y[i]
               y2 = tess_y[i + 1]
               line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
               if line_length <= self.UP.L/2:
                   plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25) 
       plt.savefig(SAVE + '/'+ TITLE + '.png')
       plt.close()
       plt.clf()
            
    def Negative_Area_Lagrangian_Tessellation(self, Particles, TITLE, SAVE, Current_Sign):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for t in range(len(self.B.Triangles)):
            Fill_Counter = 0
            tess_x = []
            tess_y = []
            for idx in self.B.Triangles[t]:
                tess_x.append(Particles[idx][0])
                tess_y.append(Particles[idx][1])
            tess_x.append(Particles[self.B.Triangles[t][0]][0])
            tess_y.append(Particles[self.B.Triangles[t][0]][1])
            for i in range(len(tess_x) - 1):
                x1 = tess_x[i]
                x2 = tess_x[i + 1]
                y1 = tess_y[i]
                y2 = tess_y[i + 1]
                line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                if line_length <= self.UP.L/2:
                    Fill_Counter += 1
                    # plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25, color = 'k') 
                    if Fill_Counter == 3 and Current_Sign[t] > 0:
                        plt.fill(tess_x, tess_y, c='k')   
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()  
        
    def Negative_Area_Euelerian_Tessellation(self, Particles, TITLE, SAVE, Current_Sign):
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        for t in range(len(self.B.Triangles)):
            Fill_Counter = 0
            tess_x = []
            tess_y = []
            for idx in self.B.Triangles[t]:
                tess_x.append(Particles[idx][2])
                tess_y.append(Particles[idx][3])
            tess_x.append(Particles[self.B.Triangles[t][0]][2])
            tess_y.append(Particles[self.B.Triangles[t][0]][3])
            for i in range(len(tess_x) - 1):
                x1 = tess_x[i]
                x2 = tess_x[i + 1]
                y1 = tess_y[i]
                y2 = tess_y[i + 1]
                line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                if line_length <= self.UP.L/2:
                    Fill_Counter += 1
                    # plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25, color = 'k') 
                    if Fill_Counter == 3 and Current_Sign[t] > 0:
                        plt.fill(tess_x, tess_y, c='k')   
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def Lagrange_Flipflops(self, a, Particles, FFM1_Total_Flip_Flops, TITLE_1, TITLE_2, SAVE):                
        colored_flops = []
        for u in np.unique(FFM1_Total_Flip_Flops):
            unique_flops = []
            
            for f in range(len(FFM1_Total_Flip_Flops)):
                if FFM1_Total_Flip_Flops[f] == u:
                    unique_flops.append(self.B.Triangles[f])
            
            colored_flops.append(unique_flops)
            
            fig = plt.figure()
            plt.gca().set_aspect('equal')
            i = 0
            plt.title(str(u) + TITLE_1)
            for tess in unique_flops:
                tess_x = []
                tess_y = []
                for idx in tess:
                    tess_x.append(Particles[idx][0])
                    tess_y.append(Particles[idx][1])
                tess_x.append(Particles[tess[0]][0])
                tess_y.append(Particles[tess[0]][1])
                for i in range(len(tess_x) - 1):
                    x1 = tess_x[i]
                    x2 = tess_x[i + 1]
                    y1 = tess_y[i]
                    y2 = tess_y[i + 1]
                    line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                    if line_length <= self.UP.L/2:
                        plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25) 
            plt.savefig(SAVE + '/' + str(u) + TITLE_1 + '.png')
            plt.close()
            plt.clf()
  
        cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan', 'orange', 'yellow'])
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.title(TITLE_2)
        for c in range(len(colored_flops)):
            i = 0
            Volume_Flop = []
            for tess in colored_flops[c]:
                Volume_Flop.append(tess)
                tess_x = []
                tess_y = []
                for idx in tess:
                    tess_x.append(Particles[idx][0])
                    tess_y.append(Particles[idx][1])
                tess_x.append(Particles[tess[0]][0])
                tess_y.append(Particles[tess[0]][1])
                for i in range(len(tess_x) - 1):
                    x1 = tess_x[i]
                    x2 = tess_x[i + 1]
                    y1 = tess_y[i]
                    y2 = tess_y[i + 1]
                    line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                    if line_length <= self.UP.L/2:
                        plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25, color = cmap(c)) 
        norm = mpl.colors.Normalize(vmin=0, vmax=len(colored_flops))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(0, len(colored_flops), len(colored_flops) + 1))
        plt.savefig(SAVE + '/' + TITLE_2 + '.png')
        plt.close()
        plt.clf()
        
    def Euler_Flipflops(self, a, Particles, FFM1_Total_Flip_Flops, TITLE_1, TITLE_2, SAVE):        
        colored_flops = []
        for u in np.unique(FFM1_Total_Flip_Flops):
            unique_flops = []
            
            for f in range(len(FFM1_Total_Flip_Flops)):
                if FFM1_Total_Flip_Flops[f] == u:
                    unique_flops.append(self.B.Triangles[f])
            
            colored_flops.append(unique_flops)
            
            fig = plt.figure()
            plt.gca().set_aspect('equal')
            i = 0
            plt.title(str(u) + TITLE_1)
            for tess in unique_flops:
                tess_x = []
                tess_y = []
                for idx in tess:
                    tess_x.append(Particles[idx][2])
                    tess_y.append(Particles[idx][3])
                tess_x.append(Particles[tess[0]][2])
                tess_y.append(Particles[tess[0]][3])
                for i in range(len(tess_x) - 1):
                    x1 = tess_x[i]
                    x2 = tess_x[i + 1]
                    y1 = tess_y[i]
                    y2 = tess_y[i + 1]
                    line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                    if line_length <= self.UP.L/2:
                        plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25) 
            plt.savefig(SAVE + '/' + str(u) + TITLE_1 + '.png')
            plt.close()
            plt.clf()
            
        cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan', 'orange', 'yellow'])
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.title(TITLE_2)
        for c in range(len(colored_flops)):
            i = 0
            Volume_Flop = []
            for tess in colored_flops[c]:
                Volume_Flop.append(tess)
                tess_x = []
                tess_y = []
                for idx in tess:
                    tess_x.append(Particles[idx][2])
                    tess_y.append(Particles[idx][3])
                tess_x.append(Particles[tess[0]][2])
                tess_y.append(Particles[tess[0]][3])
                for i in range(len(tess_x) - 1):
                    x1 = tess_x[i]
                    x2 = tess_x[i + 1]
                    y1 = tess_y[i]
                    y2 = tess_y[i + 1]
                    line_length = np.sqrt(np.power((x2 - x1),2) + np.power((y2 - y1),2))
                    if line_length <= self.UP.L/2:
                        plt.plot([x1, x2],[y1, y2], 'k-', linewidth = .25, color = cmap(c)) 
        norm = mpl.colors.Normalize(vmin=0, vmax=len(colored_flops))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(0, len(colored_flops), len(colored_flops) + 1))
        plt.savefig(SAVE + '/' + TITLE_2 + '.png')
        plt.close()
        plt.clf()
        
    def thetaCollapse_Lagrangian_Line(self, Particles, P1, P3, theta, TITLE, SAVE):
        P1, P3, theta = np.array(P1).astype(int), np.array(P3).astype(int), np.array(theta).astype(int)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        norm = mpl.colors.Normalize(vmin=0, vmax=180)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(len(P1)):
            plt.plot([Particles[P1[i]][0], Particles[P3[i]][0]],
                      [Particles[P1[i]][1], Particles[P3[i]][1]],
                      c=cmap.to_rgba(theta[i]))
        plt.colorbar(cmap)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
        
    def thetaCollapse_Eulerian_Line(self, Particles, P1, P3, theta, TITLE, SAVE):
        P1, P3, theta = np.array(P1).astype(int), np.array(P3).astype(int), np.array(theta).astype(int)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        norm = mpl.colors.Normalize(vmin=0, vmax=180)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        for i in range(len(P1)):
            plt.plot([Particles[P1[i]][2], Particles[P3[i]][2]],
                      [Particles[P1[i]][3], Particles[P3[i]][3]],
                      c=cmap.to_rgba(theta[i]))
        plt.colorbar(cmap)
        plt.savefig(SAVE + '/'+ TITLE + '.png')
        plt.close()
        plt.clf()
    
    def thetaCollapse_Lagrangian_Dot(self, Particles, P2, theta, TITLE, SAVE):
        P2, theta = np.array(P2).astype(int), np.array(theta)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(Particles[P2][:,0], Particles[P2][:,1], s=1, c=theta, cmap='coolwarm', vmin=0, vmax=180)
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.colorbar()
        plt.savefig(SAVE + '/'+ TITLE  + '.png')
        plt.close()
        plt.clf()
        
    def thetaCollapse_Eulerian_Dot(self, Particles, P2, theta, TITLE, SAVE):
        P2, theta = np.array(P2).astype(int), np.array(theta)
        fig = plt.figure()
        plt.gca().set_aspect('equal')
        plt.scatter(Particles[P2][:,0], Particles[P2][:,1], s=1, c=theta, cmap='coolwarm', vmin=0, vmax=180)
        plt.xlim(10, 90)
        plt.ylim(10, 90)
        plt.colorbar()
        plt.savefig(SAVE + '/'+ TITLE  + '.png')
        plt.close()
        plt.clf()
        