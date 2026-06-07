#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:11:02 2024
@author: michaelsitarz
"""
import os
import sys
import numpy as np


class Plotting_Call_Hub:
    def __init__(self, UP, B, C, PF):
        self.UP = UP
        self.B = B
        self.C = C
        self.PF = PF
    
    
    def Initial_Modeling(self):
        '''Initial Fields - Only Need to plot once per Seed_#'''
        self.UP.initialDirectories()

            
    def Control_Model(self, a, Particles):
        if a == 0.0:
            self.UP.controlDirectories()
            
            self.PF.Initial_Field(self.B.delta_Initial,'RdYlBu_r', '$\delta_{Init}$', self.UP.OFI)
            self.PF.Initial_Field(self.B.rho_Initial,'RdYlBu_r', '$\\rho_{Init}$', self.UP.OFI)
            self.PF.Initial_Field(self.B.phi_Initial, 'RdYlBu_r','$\\phi_{Init}$', self.UP.OFI)
            self.PF.Initial_Field(self.B.v[0], 'RdYlBu_r','$v_{Init, y}$', self.UP.OFI)
            self.PF.Initial_Field(self.B.v[1], 'RdYlBu_r','$v_{Init, x}$', self.UP.OFI)
            
            self.PF.Heatmap_Plotting(self.B.delta_Initial, 'RdYlBu_r', '$\delta_{Init}$', self.UP.OFI)
            self.PF.Heatmap_Plotting(self.B.rho_Initial, 'RdYlBu_r', '$\\rho_{Init}$', self.UP.OFI)
            self.PF.Heatmap_Plotting(self.B.phi_Initial, 'RdYlBu_r','$\\phi_{Init}$', self.UP.OFI)
            self.PF.Heatmap_Plotting(self.B.v[0], 'RdYlBu_r', '$v_{Init, y}$', self.UP.OFI)
            self.PF.Heatmap_Plotting(self.B.v[1], 'RdYlBu_r', '$v_{Init, x}$', self.UP.OFI)
            
            self.PF.Contour_Plotting(self.B.delta_Initial, 'RdYlBu_r', '$\delta_{Init}$', self.UP.OFI)
            self.PF.Contour_Plotting(self.B.rho_Initial, 'RdYlBu_r', '$\\rho_{Init}$', self.UP.OFI)
            self.PF.Contour_Plotting(self.B.phi_Initial, 'RdYlBu_r', '$\\phi_{Init}$', self.UP.OFI)
            self.PF.Contour_Plotting(self.B.v[0], 'RdYlBu_r', '$v_{Init, y}$', self.UP.OFI)
            self.PF.Contour_Plotting(self.B.v[1], 'RdYlBu_r', '$v_{Init, x}$', self.UP.OFI)
            
            self.PF.Power_Spectrum(self.B.Window_Function, '$W_k(|vec{k}|)$', self.UP.OFI, '$W_k(k)$')
            self.PF.Power_Spectrum(self.B.delta_Initial, '$\delta^2(k)$', self.UP.OFI, '$\delta_{Init}(k)$')
            self.PF.Power_Spectrum(self.B.rho_Initial, '$\\rho(k)$', self.UP.OFI, '$\\rho(k)$')
                
        '''Eulerian Coordinates'''
        if a == 0.0:
            self.PF.D2_Scatter(Particles[:,0], Particles[:,1], 'a = 0.0, Lagrangian Coordiantes', self.UP.OFX) 
            self.PF.D2_Scatter(Particles[:,2], Particles[:,3], 'a = 0.0, Eulerian Coordiantes', self.UP.OFX)    
        if a == self.UP.a0:
            self.PF.D2_Scatter(Particles[:,2], Particles[:,3], '$a_0$ = ' + str(round(a, 3)) + ', $\\vec{x}$', self.UP.OFX)  
        if a != 0.0 and a != self.UP.a0:
            self.PF.D2_Scatter(Particles[:,2], Particles[:,3], 'a = ' + str(round(a, 3)) + ', $\\vec{x}$', self.UP.OFX)  
        
        '''Particle Density on Grid'''
        if a != 0.0:
            self.PF.density_Plots(Particles[:,2], Particles[:,3],'a = ' + str(round(a, 3)), self.UP.OFR)     
      
            
    def Zeldovich_Approximation_Model(self, a, Particles, TL1, TL2, x, psi):   
        if a == 0.0:
            self.UP.zeldovichDirectories()
        
        '''[Q_Total_i[i], 
           Q_Total_j[i],     X_Total_i[i],    X_Total_j[i],    S_Total_i[i],   S_Total_j[i],   Jac_Total_ii[i], 
           Jac_Total_ij[i], Jac_Total_ji[i], Jac_Total_jj[i], Det_Total[i],   Lambda1[i],     Lambda2[i], 
           EVec10[i],      EVec11[i],       EVec20[2],       EVec21[2],      Hessian_Det[i], Hessian_Trace[i]]'''

        '''Initial Eigenvalues'''
        if a == self.UP.a0:
            self.PF.Single_Contour_Eigenvalue(Particles[:,0], Particles[:,1], TL1, '$\lambda_1^0$ Contour Distribution', self.UP.OFLI)
            self.PF.Single_Contour_Eigenvalue(Particles[:,0], Particles[:,1], TL2, '$\lambda_2$^0 Contour Distribution', self.UP.OFLI)
            self.PF.D2_Scatter_with_Color(Particles[:,0], Particles[:,1], TL1, '$\lambda_1^0$ Distribution', self.UP.OFLI, 'RdYlBu_r', None, None)
            self.PF.D2_Scatter_with_Color(Particles[:,0], Particles[:,1], TL2, '$\lambda_2^0$ Distribution', self.UP.OFLI, 'RdYlBu_r', None, None)  
            
        '''Scatter with Eigenvalue Color'''
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,11],'a = ' + str(round(a, 3)) + ', $\lambda_1$', self.UP.OFL1, 'gist_rainbow', None, None)
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,12],'a = ' + str(round(a, 3)) + ', $\lambda_2$', self.UP.OFL2, 'PuOr', None, None)
       
        '''Hessian Determinate and Trace'''
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,17],'a = ' + str(round(a, 3)) + ', $|H_{ij}|$', self.UP.HD, 'turbo', -100, 150)
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,18],'a = ' + str(round(a, 3)) + ', $Tr(H_{ij})$', self.UP.HT, 'turbo', -20, 20)
    
        '''Linear Density Fluctuations'''
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], (self.UP.a0 * np.power(a/self.UP.a0, 2/3)) * (Particles[:,11] + Particles[:,12]),'a = ' + str(round(a, 3)) + ', $a\Sigma\lambda_i$', self.UP.LDF, 'turbo', None, None)
    
        '''Singularity Isolation Value Tagging'''
        self.PF.Singularity_Value_Tag(a, Particles[:,0], Particles[:,1], Particles[:,11],'a = ' + str(round(a, 3)) + ', Filament Values', self.UP.LSingVal1)
        self.PF.Singularity_Value_Tag(a, Particles[:,0], Particles[:,1], Particles[:,12],'a = ' + str(round(a, 3)) + ', Halo Values', self.UP.LSingVal2)
        self.PF.Singularity_Value_Tag(a, Particles[:,2], Particles[:,3], Particles[:,11],'a = ' + str(round(a, 3)) + ', Filament Values', self.UP.SingVal1)
        self.PF.Singularity_Value_Tag(a, Particles[:,2], Particles[:,3], Particles[:,12],'a = ' + str(round(a, 3)) + ', Halo Values', self.UP.SingVal2)
        self.PF.Singularity_Value_Tag_Dbl(a, Particles[:,0], Particles[:,1], Particles[:,11], Particles[:,12],'a = ' + str(round(a, 3)) + ', Structure Values', self.UP.LSingVal12)
        self.PF.Singularity_Value_Tag_Dbl(a, Particles[:,2], Particles[:,3], Particles[:,11], Particles[:,12],'a = ' + str(round(a, 3)) + ', Structure Values', self.UP.SingVal12)
   
        '''Values of Lambda multiplied by Growth Factor'''
        self.PF.LXD(a, Particles[:,0], Particles[:,1], Particles[:,11],'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LD1)
        self.PF.LXD_Sing(a, Particles[:,0], Particles[:,1], Particles[:,11],'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LDS1)
        self.PF.LXD(a, Particles[:,0], Particles[:,1], Particles[:,12],'a = ' + str(round(a, 3)) + ', $D(a)\lambda_2$ Value', self.UP.LD2)
        self.PF.LXD_Sing(a, Particles[:,0], Particles[:,1], Particles[:,12],'a = ' + str(round(a, 3)) + ', $D(a)\lambda_2$ Value', self.UP.LDS2)
        
        '''Contour Plots of L1D, and L2D'''
        if a != 0.0:
            self.PF.Contour_Plots(a, Particles[:,0], Particles[:,1], Particles[:,11], [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LXD1ContL)
            self.PF.Contour_Plots(a, Particles[:,0], Particles[:,1], Particles[:,12], [-3.3, -3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2], 'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LXD2ContL)
            self.PF.Contour_Plots(a, Particles[:,2], Particles[:,3], Particles[:,11], [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LXD1ContE)
            self.PF.Contour_Plots(a, Particles[:,2], Particles[:,3], Particles[:,12], [-3.3, -3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2], 'a = ' + str(round(a, 3)) + ', $D(a)\lambda_1$ Value', self.UP.LXD2ContE)
        
        '''Eigenvectors of Distortion Tensor - WIP (Using magnitude or angle as vector color).'''
        self.PF.DT_EigenVectors(a, np.ravel(self.B.q[0]), np.ravel(self.B.q[1]), np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), 'a = ' + str(round(a, 3)) + ' $\\vec{v_1}$', self.UP.EEVEC1, self.UP.EEVEC1_Ang, self.UP.EEVEC1_Mag)
        self.PF.DT_EigenVectors_Zoom(a, np.ravel(self.B.q[0]), np.ravel(self.B.q[1]), np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), [50,60], [50,60], 'a = ' + str(round(a, 3)) + ' $\\vec{v_1}$', self.UP.EEVEC1Z, self.UP.EEVEC1Z_Ang, self.UP.EEVEC1Z_Mag)
        self.PF.DT_EigenVectors(a, np.ravel(self.B.q[0]), np.ravel(self.B.q[1]), np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]),'a = ' + str(round(a, 3)) + ' $\\vec{v_2}$', self.UP.EEVEC2, self.UP.EEVEC2_Ang, self.UP.EEVEC2_Mag)
        self.PF.DT_EigenVectors_Zoom(a, np.ravel(self.B.q[0]), np.ravel(self.B.q[1]), np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), [50,60], [50,60], 'a = ' + str(round(a, 3)) + ' $\\vec{v_2}$', self.UP.EEVEC2Z, self.UP.EEVEC2Z_Ang, self.UP.EEVEC2Z_Mag)
        
        '''Dual Eigenvectors'''
        self.PF.DT_Double_Eigenvectors(a, np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), 'a = ' + str(round(a, 3)) + ' $\\vec{v_i}$', self.UP.DTDEV)
        self.PF.DT_Double_Eigenvectors_Zoom(a, np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), [50,80], [60,90], 'a = ' + str(round(a, 3)) + ' $\\vec{v_i}$', self.UP.DTDEVZ)
        
        '''Density Contours with Eigenvectors'''
        if a != 0.0:
            self.PF.Double_Evectors_Contours(a, Particles[:,0], Particles[:,1], Particles[:,2], Particles[:,3], np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 'a = ' + str(round(a, 3)) + ' $\\vec{v_i}$ + $\\rho_N$', self.UP.DTDEVEV1)
            self.PF.Double_Evectors_Contours_Zoom(a, Particles[:,0], Particles[:,1], Particles[:,2], Particles[:,3], np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), [40,60], [40,60], [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], 'a = ' + str(round(a, 3)) + ' $\\vec{v_i}$ + $\\rho_N$', self.UP.DTDEVEV1Z)

        '''Eigenvector Directions'''
        self.PF.Vector_Direction(a, np.ravel(x[0]), np.ravel(x[1]), np.ravel(self.B.V1[:,0]), np.ravel(self.B.V1[:,1]), np.ravel(self.B.V2[:,0]), np.ravel(self.B.V2[:,1]), 'a = ' + str(round(a, 3)), self.UP.EVFD)
        self.PF.Altered_Vector_Direction(a, Particles[:,0], Particles[:,1], Particles[:,2], Particles[:,3], Particles[:,13] * TL1, Particles[:,14] * TL1, Particles[:,15] * TL2, Particles[:,16] * TL2, 'a = ' + str(round(a, 3)), self.UP.AEVD)
    
    
    def Jacobian_Model(self,a, Particles, Jacobian_Volume_Signs, Jacobian_Volume_Counts):
        if a == 0.0:
            self.UP.jacobianDirectories()
            
        '''Jacobian Determinate'''
        self.PF.D2_Scatter_with_Color(Particles[:,0], Particles[:,1], Particles[:,10],'a = ' + str(round(a, 3)) + ', $|J_{qx}|$', self.UP.LOFJ, 'turbo', 0, 120)
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,10],'a = ' + str(round(a, 3)) + ', $|J_{qx}|$', self.UP.EOFJ, 'turbo', 0, 120)
        
        '''Zeldovich Density - Inverse of Jacobian Determinate'''
        self.PF.D2_Scatter_with_Color(Particles[:,0], Particles[:,1], Particles[:,10]**-1,'a = ' + str(round(a, 3)) + ', $|J_{qx}|^{-1}$', self.UP.LZAD, 'turbo', 0, 20)
        self.PF.D2_Scatter_with_Color(Particles[:,2], Particles[:,3], Particles[:,10]**-1,'a = ' + str(round(a, 3)) + ', $|J_{qx}|^{-1}$', self.UP.EZAD, 'turbo', 0, 20)
       
        '''Jacobian (Specific Volume) Tagging'''
        self.PF.Jacobian_Tagging(Particles[:,0], Particles[:,1], Particles[:,10],'a = ' + str(round(a, 3)) + ', Specific Volume Tagging', self.UP.LSpVT)          
        self.PF.Jacobian_Tagging(Particles[:,2], Particles[:,3], Particles[:,10],'a = ' + str(round(a, 3)) + ', Specific Volume Tagging', self.UP.SpVT)          
        
        '''Determinate Categories'''            
        self.PF.Lagrange_Jacobian_Category_Plotting_Total_Switch(Particles, Jacobian_Volume_Signs, Jacobian_Volume_Counts, self.UP.LJC, 'Total Switches Lagrange Volume Category Plotting for a =' + str(round(a, 3)))
        self.PF.Euler_Jacobian_Category_Plotting_Total_Switch(Particles, Jacobian_Volume_Signs, Jacobian_Volume_Counts, self.UP.EJC, 'Total Switches Euler Volume Category Plotting for a =' + str(round(a, 3)))


    def Caustic_Shell_Modeling(self, a, Particles,  curves, flipCategories):
        SAVE_TOP = self.UP.Output_Folder + '/Topological_Curves_Repository'
        if not os.path.exists(SAVE_TOP):
            os.mkdir(SAVE_TOP)
        
        SAVE_CC = SAVE_TOP + '/Connected_Curves'
        if not os.path.exists(SAVE_CC):
            os.mkdir(SAVE_CC)
            
        SAVE_CC_a = SAVE_CC + '/Snapshot_' + str(round(a,3))
        if not os.path.exists(SAVE_CC_a):
            os.mkdir(SAVE_CC_a)
        
        import matplotlib.pyplot as plt

        '''Structure Category Seperation'''
        for i in range(len(flipCategories)):
            print(flipCategories[i])           
        
            fig = plt.figure()
            plt.gca().set_aspect('equal')
            spacer_1 = []
            for j in range(len(curves[i])):
                Index = np.array(curves[i][j]) * (self.UP.L / self.UP.Np)
                np.savetxt(SAVE_CC_a + '/' + str(i) + '_' + str(j) + '.txt', Index)
                plt.plot(Index[:,0], Index[:,1])
                plt.xlim(0, self.UP.L)
                plt.ylim(0, self.UP.L)
                plt.savefig(SAVE_CC_a + '/' + str(i) + '_' + str(j) + '.png')
                plt.close()
                plt.clf()
        
       
        
    def Geometric_Model(self, a, Particles, causticVector, causticParticle, totalFlips, currentSign, triangleMorph):
        if a == 0.0:
            self.UP.geometricDirectories()

        '''Total Tessellation Grid'''
        # self.PF.Tessellations(self.B.Triangles, Particles, 'a = ' + str(round(a, 3)) + ' Tessellation Grid', self.UP.TG)   
        
        '''Geometric Method Caustics'''
        self.PF.Geometric_Caustics(a, Particles, causticVector, causticParticle, 'a = ' + str(round(a, 3)) + ' Geometrical Volume Caustics', self.UP.GVCM)
            
        '''Colored Flops (Non-Structural: Cummulative)'''
        self.PF.Lagrange_Flipflops(str(round(a, 3)), Particles, totalFlips, ' Lagrangian Flop(s) in a = ' + str(round(a, 3)), 'Lagrangian All Colored Flops for a = ' + str(round(a, 3)), self.UP.LCF)
        # self.PF.Euler_Flipflops(str(round(a, 3)), Particles, totalFlips, ' Eulerian Flop(s) in a = ' + str(round(a, 3)), 'Eulerian All Colored Flops for a = ' + str(round(a, 3)), self.UP.ECF)

        '''Tessellations with Negative Area'''
        # self.PF.Negative_Area_Lagrangian_Tessellation(Particles, 'a = ' + str(round(a, 3)) + ' Geometrical Volume Caustics', self.UP.NAL, currentSign)
        # self.PF.Negative_Area_Euelerian_Tessellation(Particles, 'a = ' + str(round(a, 3)) + ' Geometrical Volume Caustics', self.UP.NAE, currentSign)
        
    def Collapse_Angle_Model(self, a, Particles, collapseTheta):
        self.UP.angleDirectories()
            
        '''[Tess Indx, P2 Index, P1 Index, P3 Index, Theta, P1xtC, P1ytC, P2xtC, P2ytC, P3xtC, P3ytC, a]'''    
        
        '''Geometric Caustics'''
        self.PF.thetaCollapse_Lagrangian_Line(Particles, collapseTheta[:,2], collapseTheta[:,3], collapseTheta[:,4], 'a = ' + str(round(a, 3)) + ' Angle of Caustic Collapse', self.UP.CAoGCL)
        self.PF.thetaCollapse_Eulerian_Line(Particles, collapseTheta[:,2], collapseTheta[:,3], collapseTheta[:,4], 'a = ' + str(round(a, 3)) + ' Angle of Caustic Collapse', self.UP.CAoGCE)

        '''Caustic Collapser'''
        self.PF.thetaCollapse_Lagrangian_Dot(Particles, collapseTheta[:,1], collapseTheta[:,4], 'a = ' + str(round(a, 3)) + ' Angle of Caustic Collapse', self.UP.DPAoCL)
        self.PF.thetaCollapse_Eulerian_Dot(Particles, collapseTheta[:,1], collapseTheta[:,4], 'a = ' + str(round(a, 3)) + ' Angle of Caustic Collapse', self.UP.DPAoCE)
        
        
        '''
        print('Performing Minkowski Functional Analysis')
        if len(V2_Zero_Point_Connected_Curves) != 0:
            self.MFBM.define_Jacobian_Connection_Minkowski_Functional(V2_Zero_Point_Connected_Curves, V2_Zero_Point_Piecewise_Curves, a, 'Zero_Point_Connectivity')
        for i in range(len(Transition_Categories)):
            print()
            print(Transition_Categories[i])
            print(len(V2_Morph_CC[i]), len(V2_Morph_PC[i]))
            print('Loop Progress: Category', i + 1, '/', len(Transition_Categories))
            print()
            self.MFBM.define_Jacobian_Connection_Minkowski_Functional(V2_Morph_CC[i], V2_Morph_PC[i], a, 'Jacobian_Flip_Transitions_' + str(Transition_Categories[i][0]) + '_to_' + str(Transition_Categories[i][1]))
            if len(V2_Morph_CC[i]) != 0:
                self.MFBM.define_Jacobian_Connection_Minkowski_Functional(V2_Morph_CC[i], V2_Morph_PC[i], a, 'Jacobian_Flip_Transitions_' + str(Transition_Categories[i][0]) + '_to_' + str(Transition_Categories[i][1]))
            self.save_Snapshot(V2_Morph_CC[i], a, 'Jacobian_Flip_'+ str(Transition_Categories[i][0]) + '_to_' + str(Transition_Categories[i][1]) + '_V2_Connected_Curves')
            self.save_Snapshot(V2_Morph_PC[i], a, 'Jacobian_Flip_'+ str(Transition_Categories[i][0]) + '_to_' + str(Transition_Categories[i][1]) + '_V2_Piecewise_Curves')
        print('Total Lines Sorted and Saved', time.time() - start)
        '''

