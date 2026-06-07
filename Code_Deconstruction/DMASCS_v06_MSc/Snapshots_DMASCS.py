#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:06:33 2024
@author: michaelsitarz
"""
import os
import sys
import numpy as np
import scipy as sp
from scipy.fftpack import fft
from Computations_DMASCS import Interp2D

import matplotlib as mpl
import matplotlib.pyplot as plt

class Snapshots:
    def __init__(self, UP, B, C, PCH, IH, JBM, GCBM, TOP):
        self.B = B
        self.UP = UP
        self.C = C
        self.PCH = PCH
        self.IH = IH
        self.JBM = JBM
        self.GCBM = GCBM
        self.TOP = TOP
        
        self.zaFree = np.zeros_like(self.UP.snap_array)
        self.zaFilaments = np.zeros_like(self.UP.snap_array)
        self.zaHalos = np.zeros_like(self.UP.snap_array)
        self.jacFree = np.zeros_like(self.UP.snap_array)
        self.jacFilaments = np.zeros_like(self.UP.snap_array)
        self.jacHalos = np.zeros_like(self.UP.snap_array)
        self.jacExtra = np.zeros_like(self.UP.snap_array)
        self.tessFree = np.zeros_like(self.UP.snap_array)
        self.tessFilaments = np.zeros_like(self.UP.snap_array)
        self.tessHalos = np.zeros_like(self.UP.snap_array)
        self.tessExtra = np.zeros_like(self.UP.snap_array)
        
        # self.jacFreeVol = np.zeros_like(self.UP.snap_array)
        # self.jacFilamentsVol = np.zeros_like(self.UP.snap_array)
        # self.jacHalosVol = np.zeros_like(self.UP.snap_array)
        # self.jacExtraVol = np.zeros_like(self.UP.snap_array)
        # self.tessFreeVol = np.zeros_like(self.UP.snap_array)
        # self.tessFilamentsVol = np.zeros_like(self.UP.snap_array)
        # self.tessHalosVol = np.zeros_like(self.UP.snap_array)
        # self.tessExtraVol = np.zeros_like(self.UP.snap_array)
        
        '''Importing Parameters'''
        self.duration = self.UP.duration
        self.a0 = self.UP.a0
        self.da = self.UP.da
        self.L = self.UP.L
        self.H0 = self.UP.H0
        self.Omega_Lambda = self.UP.Omega_Lambda
        self.Omega_m = self.UP.Omega_m
        self.Omega_K = self.UP.Omega_K 
        
        '''Initial Displacment'''
        self.q = self.B.q
        self.v = self.B.v
        self.psi = np.zeros_like(self.v)
        a = self.a0
        
        '''Particle Momenta -> p = a²ẋ'''
        '''p = v adot * a0^2'''
        self.p = self.v * ( self.H0 * a * np.sqrt(self.Omega_Lambda \
                + self.UP.Omega_m * a**-3 + self.Omega_K * a**-2))  * self.a0**2
        
        '''Accelleration -> acc = -a∇φ = aṗ'''
        self.acc = np.zeros_like(self.v)
        
        '''Simulation using the kick-drift method.'''
        for i in range(self.duration):
            if a == self.a0:
                a = self.first_Snap(self.a0, self.da, i)
                '''Displacement -> psi = x - q'''
                self.psi = self.v * self.a0 
                continue
            
            if a != 0.0 and a != self.a0:
                a = self.next_Snap(a, self.da, i)
        
        # SAVE_SEED = self.UP.Output_Folder + '/Participation_Counts_Arrays'
        # if not os.path.exists(SAVE_SEED):
        #     os.mkdir(SAVE_SEED)
            
        
        # np.savetxt(SAVE_SEED + '/' + 'zaFree.txt', self.zaFree)
        # np.savetxt(SAVE_SEED + '/' + 'zaFilaments.txt', self.zaFilaments)
        # np.savetxt(SAVE_SEED + '/' + 'zaHalos.txt', self.zaHalos)
        # np.savetxt(SAVE_SEED + '/' + 'jacFree.txt', self.jacFree)
        # np.savetxt(SAVE_SEED + '/' + 'jacFilaments.txt', self.jacFilaments)
        # np.savetxt(SAVE_SEED + '/' + 'jacHalos.txt', self.jacHalos)
        # np.savetxt(SAVE_SEED + '/' + 'jacExtra.txt', self.jacExtra)
        # np.savetxt(SAVE_SEED + '/' + 'tessFree.txt', self.tessFree)
        # np.savetxt(SAVE_SEED + '/' + 'tessFilaments.txt', self.tessFilaments)
        # np.savetxt(SAVE_SEED + '/' + 'tessHalos.txt', self.tessHalos)
        # np.savetxt(SAVE_SEED + '/' + 'tessExtra.txt', self.tessExtra)
        
        # np.savetxt(SAVE_SEED + '/' + 'jacFreeVol.txt', self.jacFreeVol)
        # np.savetxt(SAVE_SEED + '/' + 'jacFilamentsVol.txt', self.jacFilamentsVol)
        # np.savetxt(SAVE_SEED + '/' + 'jacHalosVol.txt', self.jacHalosVol)
        # np.savetxt(SAVE_SEED + '/' + 'jacExtraVol.txt', self.jacExtraVol)
       
#==============================================================================
    def Snapshot_Procedures(self, a, b):
        x = self.q * self.B.Mass_Resolution + self.psi
        x = self.C.fix_Boundries(x)
        
        Simulated_Jacobians = self.JBM.Jacobians(self.psi)
        Simulated_Determinates = self.JBM.Specific_Volume(Simulated_Jacobians)
        if a == 0:
            self.Jx_0 = Simulated_Determinates[0]
        Simulated_Determinates /= self.Jx_0
         
        Particles, Interpolated_Determinates, TL1, TL2 = self.IH.make_Particles(a, x, self.psi, Simulated_Jacobians, Simulated_Determinates)
        Sorted_Volumes = self.JBM.Total_Struct_Determinate_Sorting(Particles[:,10], a)  
        curves, flipCategories = self.JBM.Caustic_Connectivity(Simulated_Determinates, Interpolated_Determinates, self.JBM.Counts, Particles, Sorted_Volumes)
        triangleMorph, causticVector, causticParticle, totalFlips, currentSign, previousSign, uniqueFlips, collapseTheta, eulerianVolumes = self.GCBM.Tessellation_Studies(Particles, a)  
 
       
        '''Data and Plotting Group Calls'''
        '''Control Model'''
        self.PCH.Control_Model(a, Particles)
        
        '''Zel'dovich Model'''
        self.PCH.Zeldovich_Approximation_Model(a, Particles, TL1, TL2, x, self.psi)
        
        '''Jacobian Model'''
        self.PCH.Jacobian_Model(a, Particles, Sorted_Volumes, self.JBM.Counts)  
         
        '''Topology Model'''
        # if len(curves) > 0:
            # self.PCH.Caustic_Shell_Modeling(a, Particles, curves, flipCategories)
   
        '''Geometric Model'''
        # self.PCH.Geometric_Model(a, Particles, causticVector, causticParticle, totalFlips, currentSign, triangleMorph)

        '''Angle of Collapse Model'''
        if len(collapseTheta) > 0:
            self.PCH.Collapse_Angle_Model(a, Particles, collapseTheta)
        
            # '''[Tess Indx, P2 Index, P1 Index, P3 Index, Theta, P1xtC, P1ytC, P2xtC, P2ytC, P3xtC, P3ytC, a]'''  
            # collapseTheta = np.array(collapseTheta)
            # ind = np.lexsort((collapseTheta[:,0],collapseTheta[:,11]))
            # collapseTheta = collapseTheta[ind]
            # # 1 13 14, 2 15 16
            # filCollapseVec = []
            # halCollapseVec  = []
            # haloAngles = []
            # for i in range(len(collapseTheta)):
            #     if i == 0:
            #         filCollapseVec.append([Particles[int(collapseTheta[i][1])][13], Particles[int(collapseTheta[i][1])][14], collapseTheta[i - 1][4]])
            #     if i > 0:
            #         if collapseTheta[i][0] == collapseTheta[i - 1][0]:
            #             P2i = int(collapseTheta[i][1])
            #             P2i1 = int(collapseTheta[i - 1][1])
            #             if Particles[P2i][0] > 10 and Particles[P2i][0] < 90:
            #                 if Particles[P2i][1] > 10 and Particles[P2i][1] < 90:
            #                     if Particles[P2i1][0] > 10 and Particles[P2i1][0] < 90:
            #                         if Particles[P2i1][1] > 10 and Particles[P2i1][1] < 90:                            
            #                             haloAngles.append([collapseTheta[i - 1][4], collapseTheta[i][4]])
            #                             filCollapseVec.append([Particles[P2i1][13], Particles[P2i1][14], collapseTheta[i - 1][4]])
            #                             halCollapseVec.append([Particles[P2i][15], Particles[P2i][16], collapseTheta[i][4]])
            #         else:
            #             filCollapseVec.append([Particles[int(collapseTheta[i][1])][13], Particles[int(collapseTheta[i][1])][14], collapseTheta[i - 1][4]])
            
            # fig = plt.figure()
            # plt.gca().set_aspect('equal')
            # plt.scatter(np.array(haloAngles)[:,0], np.array(haloAngles)[:,1], c=abs(np.array(haloAngles)[:,0] - np.array(haloAngles)[:,1]), s=1)
            # plt.xlabel('$\\theta_{Fil}$')
            # plt.ylabel('$\\theta_{Halo}$')
            # clb = plt.colorbar()
            # clb.set_label('$|\\theta_{Fil} - \\theta_{Halo}|$')
            # plt.show()
            # plt.close()
            # plt.clf()
            
            # fig = plt.figure()
            # plt.gca().set_aspect('equal')
            # plt.scatter(np.array(filCollapseVec)[:,0], np.array(filCollapseVec)[:,1], c=np.array(filCollapseVec)[:,2], s=3, marker='P', label='Filaments')
            # plt.xlabel('$\\vec{v_x}$')
            # plt.ylabel('$\\vec{v_y}$')
            # plt.legend()
            # clb = plt.colorbar()
            # clb.set_label('$\\theta_{Collapse}$')
            # plt.show()
            # plt.close()
            # plt.clf()
            
            # fig = plt.figure()
            # plt.gca().set_aspect('equal')
            # plt.scatter(np.array(halCollapseVec)[:,0], np.array(halCollapseVec)[:,1], c=np.array(halCollapseVec)[:,2], s=3, marker='o', label='Halos')
            # plt.xlabel('$\\vec{v_x}$')
            # plt.ylabel('$\\vec{v_y}$')
            # plt.legend()
            # clb = plt.colorbar()
            # clb.set_label('$\\theta_{Collapse}$')
            # plt.show()
            # plt.close()
            # plt.clf()
            
            # fig = plt.figure()
            # plt.gca().set_aspect('equal')
            # plt.scatter(np.array(filCollapseVec)[:,0], np.array(filCollapseVec)[:,1], c=np.array(filCollapseVec)[:,2], s=2, marker='P', label='Filaments')
            # plt.scatter(np.array(halCollapseVec)[:,0], np.array(halCollapseVec)[:,1], c=np.array(halCollapseVec)[:,2], s=2, marker='o', label='Halos')
            # plt.xlabel('$\\vec{v_x}$')
            # plt.ylabel('$\\vec{v_y}$')
            # plt.legend()
            # clb = plt.colorbar()
            # clb.set_label('$\\theta_{Collapse}$')
            # plt.show()
            # plt.close()
            # plt.clf()

                # sys.exit()
    
        # SAVE_SEED = self.UP.Directory + '/Participation_Counts_Arrays'
        # if not os.path.exists(SAVE_SEED):
        #     os.mkdir(SAVE_SEED)
            
        # '''Hard Coded to Work on Data Model Comparison'''
        # self.zaFree[b] = (len(Particles) - len(np.where(Particles[:,11] > 1)[0]) - len(np.where(Particles[:,11] > 1)[0]))
        # self.zaFilaments[b] = len(np.where(Particles[:,11] > 1)[0])
        # self.zaHalos[b] = len(np.where(Particles[:,12] > 1)[0])

        # for c in range(len(np.unique(self.JBM.Counts))):
        #     Count_Correspondence_Index = np.where(np.isin(self.JBM.Counts, c))
        #     if c == 0:
        #         self.jacFree[b] = len(Count_Correspondence_Index[0])
        #         # self.jacFreeVol[b] = abs(sum(Particles[Count_Correspondence_Index[0]][:,10]))
        #     if c == 1:
        #         self.jacFilaments[b] = len(Count_Correspondence_Index[0])
        #         # self.jacFilamentsVol[b] = abs(sum(Particles[Count_Correspondence_Index[0]][:,10]))
        #     if c == 2:
        #         self.jacHalos[b] = len(Count_Correspondence_Index[0])
        #         # self.jacHalosVol[b] = abs(sum(Particles[Count_Correspondence_Index[0]][:,10]))
        #     if c > 2:
        #         self.jacExtra[b] = len(Count_Correspondence_Index[0])
        #         # self.jacExtraVol[b] = abs(sum(Particles[Count_Correspondence_Index[0]][:,10]))
            
        # for c in range(len(np.unique(totalFlips))):
        #     Count_Correspondence_Index = np.where(np.isin(totalFlips, c))
        #     if c == 0:
        #         self.tessFree[b] = len(Count_Correspondence_Index[0])
        #     if c == 1:
        #         self.tessFilaments[b] = len(Count_Correspondence_Index[0])
        #     if c == 2:
        #         self.tessHalos[b] = len(Count_Correspondence_Index[0])
        #     if c > 2:
        #         self.tessExtra[b] = len(Count_Correspondence_Index[0])
        
#==============================================================================
    def Deploy(self, a):
        '''Deploy the particles on the grid.'''
        '''Find directional components of potential.'''
        psi_x = Interp2D(self.psi[0])
        psi_y = Interp2D(self.psi[1])
        '''Deploy particles and move based on potential.'''
        xx = self.B.XX[:,0] * self.B.Mass_Resolution + psi_x(self.B.XX)
        xy = self.B.XX[:,1] * self.B.Mass_Resolution + psi_y(self.B.XX)
        '''Compute the density of the simulated particles on the force grid.'''
        self.density = self.find_Density(np.c_[xx,xy], self.B.Force_Resolution, self.UP.Nf, self.UP.qpf, self.B.Mass_Resolution) - 1
#==============================================================================    
    '''Find the simualted particle density on the force grid.'''
    def find_Density(self, Particles, Force_Resolution, Nf, qpf, Mass_Resolution):
        idc = ((Particles / Force_Resolution) % Nf).astype(int)
        idx = np.sort(idc[:,0] * Nf + idc[:,1])
        cumdens = np.searchsorted(np.r_[idx,Nf**2], np.arange(Nf**2))
        idens = (np.r_[cumdens[1:],idx.size]-cumdens).reshape((Nf,Nf))
        final_Product = idens.astype(float) / (2**(2*qpf)) * (Mass_Resolution / Force_Resolution)**2
        return final_Product        
#==============================================================================       
    def Gravitate(self):
        '''Find the accelleration of the particles due to gravity.'''
        potential_Tilde = -sp.fftpack.fftn(self.density) / self.B.Kf_Squared * self.B.Force_Resolution**2 * self.UP.GN
        potential_Tilde[0,0] = 0
        self.Phi = sp.fftpack.ifftn(potential_Tilde).real
        
        self.e_acc_x = - self.C.gradient_dual(self.Phi, 0) / self.B.Force_Resolution
        self.e_acc_y = - self.C.gradient_dual(self.Phi, 1) / self.B.Force_Resolution
        acc_x = Interp2D(self.e_acc_x)
        acc_y = Interp2D(self.e_acc_y)
        
        x = ((self.B.Mass_Indices_Grid + self.psi) % self.UP.L).transpose([1,2,0]).reshape((self.UP.Np**2, 2)) / self.B.Force_Resolution
        
        '''Compute and return new particle accellerations.'''
        self.acc[0,:,:] = acc_x(x - [0, 0.5]).reshape((self.UP.Np, self.UP.Np))
        self.acc[1,:,:] = acc_y(x - [0.5, 0]).reshape((self.UP.Np, self.UP.Np))
#==============================================================================        
    def Kick(self, a, da):
        '''Kick the particles and return the new momentas.'''
        adot = self.H0 * a * np.sqrt(self.Omega_Lambda \
                + self.Omega_m * a**-3 \
                + self.Omega_K * a**-2) 
        '''Calculate the new particle momentas.'''
        self.p += da / (adot * a) * self.acc          
        # print(self.p.shape)
#==============================================================================        
    def Drift(self, a, da):
        '''Compute the particle drift due to acceleration and return the new particle position.'''
        adot = self.H0 * a * np.sqrt(self.Omega_Lambda \
                + self.Omega_m * a**-3 \
                + self.Omega_K * a**-2) 
        '''Compute new Displacment'''
        self.psi += da / (adot * np.square(a)) * self.p
#==============================================================================                    
    def first_Snap(self, a, da, b):
        print()
        print('========================')
        print('Snap:', str(round(0.0, 2)))
        
        '''Deploy the particles on the initial grid (a = 0.0) for the calculations to the intial snap shot.'''
        self.Deploy(0.0)
        self.Snapshot_Procedures(0.0, b)
        
        '''Accellerate the particles to the initial snap shot due to potentials based on the initial grid (a = 0.0).'''
        self.Gravitate()
        return self.next_Snap(a, da, b)
#==============================================================================
    def next_Snap(self, a, da, b):
        print()
        print('========================')
        print('Snap:', str(round(a, 2)))
        
        '''Kick the particles to update momenta.'''
        self.Kick(a, da/2)
        '''Allow the particles to drift to update position.'''
        self.Drift(a, da)
        '''Redeploy the particles according to the new potentials.'''
        self.Deploy(a)
        '''Accelerate the particles.'''
        self.Gravitate()
        '''Kick the particles to update momenta.'''
        self.Kick(a + da, da/2)
        
        self.Snapshot_Procedures(a, b)
        return a + da
#==============================================================================   
