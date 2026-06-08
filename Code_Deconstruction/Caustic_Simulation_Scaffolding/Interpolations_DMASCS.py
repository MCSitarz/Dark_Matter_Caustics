#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:42:49 2023
@author: michaelsitarz
"""
import numpy as np
import scipy as sp
from scipy.fftpack import fft

class Interpolation:
    def __init__(self, UP, C, JBM, B):
        self.UP = UP
        self.C = C
        self.JBM = JBM
        self.B = B

        '''Importing Parameters'''
        self.L = self.UP.L
        self.Np = self.UP.Np
        self.q = self.JBM.q
        
        '''Calulation of Multi-Use Values'''
        self.Qx_Int = self.q[0] + 0.5
        self.Qy_Int = self.q[1] + 0.5        
        self.Mass_Resolution = self.L / self.Np
        self.Bottom_Left = [self.q[0], self.q[1]]
        self.Bottom_Right = [np.roll(self.q[0], -1, axis=0), np.roll(self.q[1], -1, axis=0)]
        self.Top_Left = [np.roll(self.q[0], -1, axis=1), np.roll(self.q[1], -1, axis=1)]
        self.Top_Right = [np.roll(self.q[0], -1, axis=0), np.roll(self.q[1], -1, axis=1)]
#============================================================================== 
    def make_FFT_Interpolation(self, F):
        '''Take the displacment and FFT the array.'''
        F_Tilde = sp.fftpack.fft2(F)
        F_Tilde_Interp = np.zeros_like((F_Tilde))
        idx = np.indices((len(F_Tilde), len(F_Tilde[0])))
        '''Compute the interpolated displacment in FFT space given the new lagrangian coordinate. 
           This is then used to compute the interpoalted eulerian coordinates.'''
        K = np.where(idx > len(F_Tilde)/2, idx - len(F_Tilde), idx) #* (2*np.pi / len(F_Tilde))
        for k in range(len(F_Tilde)):
            for l in range(len(F_Tilde[k])):
                delta_term = ( 0.5 * K[0][k][l] * (len(F_Tilde))**-1 ) + ( 0.5 * K[1][k][l] * (len(F_Tilde[k]))**-1 )
                phase_angle = 2 * np.pi * 1j
                F_Tilde_Interp[k][l] = F_Tilde[k][l] * np.exp(phase_angle * delta_term)
        F_Interp = sp.fftpack.ifft2(F_Tilde_Interp)
        return np.array(F_Interp).real
#==============================================================================    
    def make_Linear_Interpolation(self, Fii, Fij, Fji, Fjj):
        '''Simple linear interpoaltion function.'''
        Fii = np.reshape(Fii, (self.Np, self.Np))
        Fij = np.reshape(Fij, (self.Np, self.Np))
        Fji = np.reshape(Fji, (self.Np, self.Np))
        Fjj = np.reshape(Fjj, (self.Np, self.Np))
        Fii_Interp = []
        Fij_Interp = []
        Fji_Interp = []
        Fjj_Interp = []
        for r in range(len(self.q[0])):
            for s in range(len(self.q[0][r])):
               '''Divide the values by 4???'''
               Fii_Interp.append(Fii[int(self.Bottom_Right[0][r][s])][int(self.Bottom_Right[1][r][s])] + \
                             Fii[int(self.Bottom_Left[0][r][s])][int(self.Bottom_Left[1][r][s])] + \
                             Fii[int(self.Top_Left[0][r][s])][int(self.Top_Left[1][r][s])] + \
                             Fii[int(self.Top_Right[0][r][s])][int(self.Top_Right[1][r][s])])
               Fij_Interp.append(Fij[int(self.Bottom_Right[0][r][s])][int(self.Bottom_Right[1][r][s])] + \
                             Fij[int(self.Bottom_Left[0][r][s])][int(self.Bottom_Left[1][r][s])] + \
                             Fij[int(self.Top_Left[0][r][s])][int(self.Top_Left[1][r][s])] + \
                             Fij[int(self.Top_Right[0][r][s])][int(self.Top_Right[1][r][s])])
               Fji_Interp.append(Fji[int(self.Bottom_Right[0][r][s])][int(self.Bottom_Right[1][r][s])] + \
                             Fji[int(self.Bottom_Left[0][r][s])][int(self.Bottom_Left[1][r][s])] + \
                             Fji[int(self.Top_Left[0][r][s])][int(self.Top_Left[1][r][s])] + \
                             Fji[int(self.Top_Right[0][r][s])][int(self.Top_Right[1][r][s])])
               Fjj_Interp.append(Fjj[int(self.Bottom_Right[0][r][s])][int(self.Bottom_Right[1][r][s])] + \
                             Fjj[int(self.Bottom_Left[0][r][s])][int(self.Bottom_Left[1][r][s])] + \
                             Fjj[int(self.Top_Left[0][r][s])][int(self.Top_Left[1][r][s])] + \
                             Fjj[int(self.Top_Right[0][r][s])][int(self.Top_Right[1][r][s])])
        
        return np.ravel(Fii_Interp), np.ravel(Fij_Interp), np.ravel(Fji_Interp), np.ravel(Fjj_Interp)
#==============================================================================
    def make_Particles(self, a, X, S, SJ, SD):     
        '''Find the interpoalted particle positions and jacobians.'''       
        Q_Total_i = [np.ravel(self.q[0])]
        Q_Total_j = [np.ravel(self.q[1])]
        X_Total_i = [np.ravel(X[0])]
        X_Total_j = [np.ravel(X[1])]
        S_Total_i = [np.ravel(S[0])]
        S_Total_j = [np.ravel(S[1])]
        
        Jac_Total_ii = [np.ravel(SJ[:,0,0])]
        Jac_Total_ij = [np.ravel(SJ[:,0,1])]
        Jac_Total_ji = [np.ravel(SJ[:,1,0])]
        Jac_Total_jj = [np.ravel(SJ[:,1,1])]
        Det_Total = [np.ravel(SD)]
        
        '''Find the interpoalted displacment vector.'''
        Si_Int = self.make_FFT_Interpolation(S[0])
        Sj_Int = self.make_FFT_Interpolation(S[1])
        X = np.array([self.Qx_Int, self.Qy_Int]) * self.Mass_Resolution + np.array([Si_Int, Sj_Int])
        Q_Total_i.append(np.ravel(self.Qx_Int))
        Q_Total_j.append(np.ravel(self.Qy_Int))
        X_Total_i.append(np.ravel(X[0]))
        X_Total_j.append(np.ravel(X[1]))
        S_Total_i.append(np.ravel(Si_Int))
        S_Total_j.append(np.ravel(Sj_Int))
        

        '''Find the interpoalted jacobian elements.'''        
        Jac_ii_Int, Jac_ij_Int, Jac_ji_Int, Jac_jj_Int = \
            self.make_Linear_Interpolation(Jac_Total_ii, Jac_Total_ij, Jac_Total_ji, Jac_Total_jj)
                     
        Jac_Total_ii.append(np.ravel(Jac_ii_Int))
        Jac_Total_ij.append(np.ravel(Jac_ij_Int))
        Jac_Total_ji.append(np.ravel(Jac_ji_Int))
        Jac_Total_jj.append(np.ravel(Jac_jj_Int))
        
        
        '''Find the interpolated spcific volumes.'''
        Interpolated_Jacobians = np.array([[[np.ravel(Jac_ii_Int)[i], np.ravel(Jac_ij_Int)[i]],[np.ravel(Jac_ji_Int)[i], np.ravel(Jac_jj_Int)[i]]] for i in range(len(np.ravel(self.q[0])))])
        self.Interpolated_Volumes = self.JBM.Specific_Volume(Interpolated_Jacobians)
        Det_Total.append(self.Interpolated_Volumes)
                
        Q_Total_i = np.concatenate(Q_Total_i)
        Q_Total_j = np.concatenate(Q_Total_j)
        X_Total_i = np.concatenate(X_Total_i)
        X_Total_j = np.concatenate(X_Total_j)
        S_Total_i = np.concatenate(S_Total_i)
        S_Total_j = np.concatenate(S_Total_j)
        
        Jac_Total_ii = np.concatenate(Jac_Total_ii)
        Jac_Total_ij = np.concatenate(Jac_Total_ij)
        Jac_Total_ji = np.concatenate(Jac_Total_ji)
        Jac_Total_jj = np.concatenate(Jac_Total_jj)
        Det_Total = np.concatenate(Det_Total)
        
        lambda_1_Int = np.ravel(self.make_FFT_Interpolation(self.B.L1.reshape((self.UP.Np, self.UP.Np))))
        lambda_2_Int = np.ravel(self.make_FFT_Interpolation(self.B.L2.reshape((self.UP.Np, self.UP.Np))))
        L1 = np.concatenate([self.B.L1, lambda_1_Int]) 
        L2 = np.concatenate([self.B.L2, lambda_2_Int]) 
        lambda1 = L1
        lambda2 = L2
        
        L1 = L1 * self.UP.a0 * np.power((a)/self.UP.a0, 2/3)
        L2 = L2 * self.UP.a0 * np.power((a)/self.UP.a0, 2/3)
        
        v10_int = np.ravel(self.make_FFT_Interpolation(self.B.V1[:,0].reshape((self.UP.Np, self.UP.Np))))
        v11_int = np.ravel(self.make_FFT_Interpolation(self.B.V1[:,1].reshape((self.UP.Np, self.UP.Np))))        
        V10 = np.concatenate((self.B.V1[:,0], v10_int))
        V11 = np.concatenate((self.B.V1[:,1], v11_int))

        v20_int = np.ravel(self.make_FFT_Interpolation(self.B.V2[:,0].reshape((self.UP.Np, self.UP.Np))))
        v21_int = np.ravel(self.make_FFT_Interpolation(self.B.V2[:,1].reshape((self.UP.Np, self.UP.Np))))        
        V20 = np.concatenate((self.B.V2[:,0], v20_int))
        V21 = np.concatenate((self.B.V2[:,1], v21_int))
        
        HD_Int = np.ravel(self.make_FFT_Interpolation(self.B.I2.reshape((self.UP.Np, self.UP.Np))))
        Hess_Det = np.concatenate([self.B.I2, HD_Int]) 
        
        HT_Int = np.ravel(self.make_FFT_Interpolation(self.B.I1.reshape((self.UP.Np, self.UP.Np))))
        Hess_Trace = np.concatenate([self.B.I1, HT_Int]) 
        
        '''Create the particle data structure.'''
        Particles = []
        for i in range(len(X_Total_i)):
            Particles.append([Q_Total_i[i], Q_Total_j[i], X_Total_i[i], X_Total_j[i], S_Total_i[i], S_Total_j[i], Jac_Total_ii[i], Jac_Total_ij[i], Jac_Total_ji[i], Jac_Total_jj[i], Det_Total[i], L1[i], L2[i], V10[i], V11[i], V20[i], V21[i], Hess_Det[i], Hess_Trace[i]])
        Particles = np.array(Particles)  
        Particles = Particles[np.lexsort((Particles[:,0], Particles[:,1]))]

        for part in range(len(Particles)):
            Particles[part][0] = Particles[part][0] * self.Mass_Resolution
            Particles[part][1] = Particles[part][1] * self.Mass_Resolution
        
        '''Enforce periodic boundary conditions and sort structure by x then y lagrangian coordinate.'''
        for part in range(len(Particles)):
            if Particles[part][2] <= 0:
                Particles[part][2] = Particles[part][2] + self.L
            if Particles[part][2] >= self.L:
                Particles[part][2] = Particles[part][2] - self.L
            if Particles[part][3] <= 0:
                Particles[part][3] = Particles[part][3] + self.L
            if Particles[part][3] >= self.L:
                Particles[part][3] = Particles[part][3] - self.L
        Particles = np.array(Particles)   
        Particles = Particles[np.lexsort((Particles[:,0], Particles[:,1]))]                
    
        return Particles, self.Interpolated_Volumes, lambda1, lambda2
#==============================================================================
