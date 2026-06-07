#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:21:57 2024
@author: michaelsitarz
"""
import random
import numpy as np
import scipy as sp
np.random.seed(10292022)
from scipy.fftpack import fft

class Box:
    def __init__(self, UP, C):
        self.UP = UP
        self.C = C
        
        '''Initialize and set up grids for simualtion.'''
        self.Mass_Resolution = self.UP.L / self.UP.Np
        self.Force_Resolution = self.UP.L / self.UP.Nf
        
        self.Force_Indices = np.indices((self.UP.Nf, self.UP.Nf)).astype(float)
        self.Mass_Indices_Grid = self.Force_Indices * self.Force_Resolution
        
        self.Mass_Indices = np.indices((self.UP.Np, self.UP.Np)).astype(float)
        self.Mass_Indices_Grid = self.Mass_Indices * self.Mass_Resolution
                
        subdiv_unitcell = [self.C.subdiv_unitcell_gen(n) for n in range(self.UP.subdiv)]
        
        self.XX = (self.Mass_Indices.transpose([1,2,0]).reshape([self.UP.Np**2,2])[:,np.newaxis,:] + \
                   subdiv_unitcell[self.UP.qpf]).reshape([self.UP.Np**2 * 2**(2*self.UP.qpf), 2 ])

        '''Compute initial wave numebers for each resolution.'''
        '''k-values, divide by resolution to get physical scales'''
        self.Km = self.make_K(self.UP.Np)
        self.Km_Squared = (self.Km**2).sum(axis=0)
        self.Km_Squared[0, 0] = 1
        
        self.Kf = self.make_K(self.UP.Nf)
        self.Kf_Squared = (self.Kf**2).sum(axis=0)
        self.Kf_Squared[0, 0] = 1
        
        ''' Creating wavenumber (k) frequency bins for the Matter Transfer Function'''
        kfreq = np.fft.fftfreq(self.UP.Np) * self.UP.Np
        self.kfreq2D = np.meshgrid(kfreq, kfreq)
        self.k_norm = np.sqrt(self.kfreq2D[0]**2 + self.kfreq2D[1]**2)
        
        '''Initilize Lagrangian coordinates.'''
        self.q = self.Mass_Indices
        self.dqi = self.q[0][1,0] - self.q[0][0,0]
        self.dqj = self.q[1][0,1] - self.q[1][0,0]
                
        '''Compute initial density fluctuations and density.'''
        self.delta_Initial, self.rho_Initial, self.Window_Function = self.make_Initial_Density_Field()

        '''Compute initial potential.'''
        self.phi_Initial = self.make_Initial_Velocity_Potential(self.rho_Initial, self.Km_Squared, self.Mass_Resolution)

        '''Compute initial displacment.'''
        self.v =  self.make_Initial_Displacement(self.rho_Initial, self.Km, self.Mass_Resolution, self.Km_Squared)
        
        '''Compute Deformation Tensor of the particles in the box.'''
        self.L1, self.L2, self.I1, self.I2, self.V1, self.V2 = self.Distortion_Tensor_Analysis()
        
        '''Making Tessellations for Geometric and Steiner Methods'''
        self.Triangles = self.make_Triangular_Tessellation()

    
    '''Calcualte wave numbers'''
    def make_K(self, N):
        index = np.indices((N, N))        
        return np.where(index > N/2, index - N, index) * (2*np.pi / N)

    
    '''Calculate smoothing scale filter.'''
    def make_Window_Function(self): 
        '''Sharp K Cutoff'''
        return lambda K: np.where(np.sqrt(np.square(K[0]) + np.square(K[1])) <= self.UP.k_cutoff, 1, 0)

    
    '''Define and calculate the matter transfer function.'''
    def make_Matter_Transfer(self):
        if self.UP.Cosmology == 'EdS':
            '''Power Law'''
            return lambda k: np.where(k > 0, (k/(self.UP.L / self.UP.Np))**self.UP.n, 0)

    
    '''Calculate the primordial density fluctuations and smoothed density field.'''
    def make_Initial_Density_Field(self):
        MT = self.make_Matter_Transfer()
        SF = self.make_Window_Function()
        initial_perturbations = np.loadtxt(self.UP.Output_Folder + '/Delta_Initial_Field.txt')
        Window_Function = SF(self.make_K(self.UP.Np))
        Tilde_Delta_Field = sp.fftpack.fftn(initial_perturbations) * np.sqrt(MT(self.k_norm)) * Window_Function
        Initial_Density_Field = sp.fftpack.ifftn(Tilde_Delta_Field).real
        Initial_Density_Field /= Initial_Density_Field.std()
        Initial_Density_Field *= 10
        return initial_perturbations, Initial_Density_Field, Window_Function

   
    '''Calcualte initial velocity potential.'''
    def make_Initial_Velocity_Potential(self, dens, Km_Squared, Mass_Resolution):
        dens_Tilde = sp.fftpack.fftn(dens)
        phi_Initial = sp.fftpack.ifftn(dens_Tilde / Km_Squared * Mass_Resolution**2).real             
        return phi_Initial

    
    '''Find the initial displacment of the particles by the fields.'''
    def make_Initial_Displacement(self, dens, Km, Mass_Resolution, km_squared):
        '''This differentiation is correct within the edges of order -1.5 (known computational error)'''
        '''We now confirm and assume the differentiation is correct.'''
        dens_Tilde = sp.fftpack.fftn(dens)
        pot_Tilde = dens_Tilde / km_squared * Mass_Resolution**2
        vx = sp.fftpack.ifftn(pot_Tilde * -1j * np.sin(Km[0])).real  / Mass_Resolution
        vy = sp.fftpack.ifftn(pot_Tilde * -1j * np.sin(Km[1])).real  / Mass_Resolution
        v = np.array([vx,vy])
        return v

    
    def Distortion_Tensor_Analysis(self):
        '''d_ij = d^2Phi/dqidqj'''
        '''This was confirmed symmetric on order of e-13'''
        dP_dqx = np.gradient(self.phi_Initial, self.dqi, axis=0)
        dP_dqy = np.gradient(self.phi_Initial, self.dqj, axis=1)
        
        '''Added Normalization due to DT_LA_Test results.'''
        dP2_dxdx = np.gradient(dP_dqx, self.dqi, axis=0) / self.Mass_Resolution
        dP2_dxdy = np.gradient(dP_dqx, self.dqj, axis=1) / self.Mass_Resolution
        dP2_dydx = np.gradient(dP_dqy, self.dqi, axis=0) / self.Mass_Resolution
        dP2_dydy = np.gradient(dP_dqy, self.dqj, axis=1) / self.Mass_Resolution
        
        '''These list comprehension appears correct off of small index testing.'''
        DT = np.array([[[np.ravel(dP2_dxdx)[i], np.ravel(dP2_dxdy)[i]],[np.ravel(dP2_dydx)[i], np.ravel(dP2_dydy)[i]]] for i in range(len(np.ravel(self.q[0])))])
        I1 = np.array([np.ravel(dP2_dxdx)[i] + np.ravel(dP2_dydy)[i] for i in range(len(np.ravel(self.q[0])))])
        I2 = np.array([(np.ravel(dP2_dxdx)[i] * np.ravel(dP2_dydy)[i]) - (np.ravel(dP2_dxdy)[i] * np.ravel(dP2_dydx)[i]) for i in range(len(np.ravel(self.q[0])))])
         
        '''These were checked to be correctly oriented magnitudinally.'''
        '''Lambdas checked against characteristic eqution using Invariants.'''
        Eigenvalues, Eigenvectors = np.linalg.eigh(DT)
        L2 = Eigenvalues[:,0]
        L1 = Eigenvalues[:,1]
        V2 = Eigenvectors[:,0]
        V1 = Eigenvectors[:,1]

        return L1, L2, I1, I2, V1, V2
           
   
    def make_Triangular_Tessellation(self):
        '''Compute the Bounds of the Tessallation Triangles'''   
        Triangles = []
        for i in range(self.UP.Np):
            for j in range(self.UP.Np):
                '''Body of Grid'''
                if i < (self.UP.Np - 1) and j < (self.UP.Np - 1):
                    BL = j + (i * (2*self.UP.Np))
                    BR = BL + 1
                    TL = BL + (2*self.UP.Np)
                    TR = TL + 1
                    C = BL + self.UP.Np
                    Triangles.append([BL, C, BR])
                    Triangles.append([TL, C, BL])
                    Triangles.append([TR, C, TL])
                    Triangles.append([BR, C, TR])
                
                '''Top and Bottom Connection (Not Corner)'''
                if i == (self.UP.Np - 1) and j < (self.UP.Np - 1):
                    BL = j + (i * (2*self.UP.Np))
                    BR = BL + 1
                    C = BL + self.UP.Np
                    TL = j
                    TR = j + 1
                    Triangles.append([BL, C, BR])
                    Triangles.append([TL, C, BL])
                    Triangles.append([TR, C, TL])
                    Triangles.append([BR, C, TR])
                    
                '''Left and Right Connection (Not Corner)'''
                if j == (self.UP.Np - 1) and i < (self.UP.Np - 1):
                    BL = j + (i * (2*self.UP.Np))
                    BR = BL - (self.UP.Np - 1)
                    C = BL + self.UP.Np
                    TL = BL + (2 * self.UP.Np)
                    TR = TL - (self.UP.Np - 1)
                    Triangles.append([BL, C, BR])
                    Triangles.append([TL, C, BL])
                    Triangles.append([TR, C, TL])
                    Triangles.append([BR, C, TR])
        
        '''Corner'''
        TR = 0
        TL = self.UP.Np - 1
        C = ((2*self.UP.Np) * self.UP.Np) - 1
        BL = C - self.UP.Np
        BR = BL - (self.UP.Np - 1)
        Triangles.append([BL, C, BR])
        Triangles.append([TL, C, BL])
        Triangles.append([TR, C, TL])
        Triangles.append([BR, C, TR])
        return np.array(Triangles)