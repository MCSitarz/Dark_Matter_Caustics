#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:19:39 2024
@author: michaelsitarz
"""
import os
import numpy as np

class Universal_Parameters:
    def __init__(self): 
        print('========================================================================')
        '''General notes and comments to be written into the records file.'''
        
        self.Random_Seed = 'EllenoraMeeting'  
        self.Directory = '/Users/michaelsitarz/Documents/Stuff_Local/GitHub/Zeldovich/Dark_Matter_Caustics/'
        self.output_parent_directory = 'DMASCSvariant_' + self.Random_Seed
        print('Random Seed Loaded:' + self.Random_Seed)
        print()
        
        '''Initial statistical parameters for Gaussian random fields.'''
        self.sigma = 1.0
        self.mu = 0.0
        
        '''Box side length, L'''
        self.L = 100.0 #Mpc NEED UNIT CHECK
        
        '''Resolution Options'''
        self.Np = 2**9
        self.qpf = 2
        self.Nf = self.Np * self.qpf
        self.subdiv = 4
        self.k_cutoff = 5 * (2*np.pi / self.Np)
        print('Number of Simulated Particles:', self.Np)
        print()


        '''Time Domain Options'''      
        self.a0 = 0.01 # NEED UNIT CHECK
        self.da = 0.01
        self.duration = 10
        self.snap_array = np.arange(self.a0, self.a0 + (self.da * (self.duration)), step = self.da)
        self.snap_array = np.insert(self.snap_array, 0, 0)
        print('Snap Array to be used:', self.snap_array)
        print()
        print('Growth Factor on a:', np.power(self.snap_array, 2/3))
        print()
        
        self.Cosmology = 'EdS' #Is this Einstein - de Sitter?
        print('Cosmology to be used:', self.Cosmology)
        
        if self.Cosmology == 'EdS':
            self.n = 0.0 # spectral index
            self.H0 = 71.0 # Hubble uncertainty parameter
            self.Omega_m = 1.0 # matter density
            self.Omega_Lambda = 0.0 # dark energy density
            self.Omega_K = 1 - self.Omega_m - self.Omega_Lambda # Curvature under 1 - self.Omega_m - self.Omega_Lambda
            self.GN = 3.2 * self.Omega_m * self.H0**2 # m3 /(kg s), Newtonian Gravity
            self.rho_crit = (3 * self.H0**2) / (8 * np.pi * self.GN)  # g/cm^3, using the formula (3H0^2)/(8piGN)
            
        '''Main Output Folder'''
        self.Output_Folder = self.Directory + self.output_parent_directory
        if not os.path.exists(self.Output_Folder):
            os.mkdir(self.Output_Folder)   
            
        print('========================================================================')
        print()
        
    # '''Hella Plot Folders

    def controlDirectories(self):
        self.Control = self.Output_Folder + '/Control_Model_Figures'
        if not os.path.exists(self.Control):
            os.mkdir(self.Control)  
            
        self.OFI = self.Output_Folder + '/Initial_Fields'
        if not os.path.exists(self.OFI):
            os.mkdir(self.OFI)
        
        self.OFX = self.Control + '/Eulerian_Coordinates_Dot_Plots'
        if not os.path.exists(self.OFX):
            os.mkdir(self.OFX)  
            
        self.OFR = self.Control + '/Particle_Density'
        if not os.path.exists(self.OFR):
            os.mkdir(self.OFR)  
        
    def zeldovichDirectories(self):
        self.Zeldovich = self.Output_Folder + '/Zeldovich_Model_Figures'
        if not os.path.exists(self.Zeldovich):
            os.mkdir(self.Zeldovich)
            
        self.OFLI = self.Zeldovich + '/Lambda_Initial_Plots'
        if not os.path.exists(self.OFLI):
            os.mkdir(self.OFLI)  
        
        self.OFL1 = self.Zeldovich + '/Lambda_1_Eulerian_Dot_Plots'
        if not os.path.exists(self.OFL1):
            os.mkdir(self.OFL1)  
            
        self.OFL2 = self.Zeldovich + '/Lambda_2_Eulerian_Dot_Plots'
        if not os.path.exists(self.OFL2):
            os.mkdir(self.OFL2)  
            
        self.HD = self.Zeldovich + '/Hessian_Determinate_Eulerian_Dot_Plots'
        if not os.path.exists(self.HD):
            os.mkdir(self.HD)
            
        self.HT = self.Zeldovich + '/Hessian_Trace_Eulerian_Dot_Plots'
        if not os.path.exists(self.HT):
            os.mkdir(self.HT)
            
        self.LDF = self.Zeldovich + '/Linear_Density_Fluctuations_Eulerian_Dot_Plots'
        if not os.path.exists(self.LDF):
            os.mkdir(self.LDF)
            
        self.LSingVal1 = self.Zeldovich + '/Lambda_1_Singularity_Isolation_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LSingVal1):
            os.mkdir(self.LSingVal1)  
            
        self.SingVal1 = self.Zeldovich + '/Lambda_1_Singularity_Isolation_Eulerian_Dot_Plots'
        if not os.path.exists(self.SingVal1):
            os.mkdir(self.SingVal1)  
            
        self.LSingVal2 = self.Zeldovich + '/Lambda_2_Singularity_Isolation_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LSingVal2):
            os.mkdir(self.LSingVal2)
            
        self.SingVal2 = self.Zeldovich + '/Lambda_2_Singularity_Isolation_Eulerian_Dot_Plots'
        if not os.path.exists(self.SingVal2):
            os.mkdir(self.SingVal2)
            
        self.SingVal12 = self.Zeldovich + '/Lambda_1_2_Singularity_Isolation_Eulerian_Dot_Plots'
        if not os.path.exists(self.SingVal12):
            os.mkdir(self.SingVal12)
            
        self.LSingVal12 = self.Zeldovich + '/Lambda_1_2_Singularity_Isolation_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LSingVal12):
            os.mkdir(self.LSingVal12)
            
        self.LD1 = self.Zeldovich + '/Lambda_1_Da_Values_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LD1):
            os.mkdir(self.LD1)
            
        self.LDS1 = self.Zeldovich + '/Lambda_1_Da_Values_Singularity_Only_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LDS1):
            os.mkdir(self.LDS1)
            
        self.LD2 = self.Zeldovich + '/Lambda_2_Da_Values_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LD2):
            os.mkdir(self.LD2)
            
        self.LDS2 = self.Zeldovich + '/Lambda_2_Da_Values_Singularity_Only_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LDS2):
            os.mkdir(self.LDS2)
            
        self.LXD1ContL = self.Zeldovich + '/Lambda_1_Da_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LXD1ContL):
            os.mkdir(self.LXD1ContL)
            
        self.LXD2ContL = self.Zeldovich + '/Lambda_2_Da_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LXD2ContL):
            os.mkdir(self.LXD2ContL)
            
        self.LXD1ContE = self.Zeldovich + '/Lambda_1_Da_Eulerian_Dot_Plots'
        if not os.path.exists(self.LXD1ContE):
            os.mkdir(self.LXD1ContE)
            
        self.LXD2ContE = self.Zeldovich + '/Lambda_2_Da_Eulerian_Dot_Plots'
        if not os.path.exists(self.LXD2ContE):
            os.mkdir(self.LXD2ContE)

        self.EEVEC1 = self.Zeldovich + '/v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1):
            os.mkdir(self.EEVEC1)  
            
        self.EEVEC2 = self.Zeldovich + '/v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2):
            os.mkdir(self.EEVEC2)
            
        self.EEVEC1Z = self.Zeldovich + '/Zoom_v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1Z):
            os.mkdir(self.EEVEC1Z)  
            
        self.EEVEC2Z = self.Zeldovich + '/Zoom_v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2Z):
            os.mkdir(self.EEVEC2Z)
            
        self.EEVEC1_Ang = self.Zeldovich + '/Angle_v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1_Ang):
            os.mkdir(self.EEVEC1_Ang)  
            
        self.EEVEC2_Ang = self.Zeldovich + '/Angle_v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2_Ang):
            os.mkdir(self.EEVEC2_Ang)
            
        self.EEVEC1Z_Ang = self.Zeldovich + '/Zoom_Angle_v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1Z_Ang):
            os.mkdir(self.EEVEC1Z_Ang)  
            
        self.EEVEC2Z_Ang = self.Zeldovich + '/Zoom_Angle_v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2Z_Ang):
            os.mkdir(self.EEVEC2Z_Ang)
            
        self.EEVEC1_Mag = self.Zeldovich + '/Magnitude_v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1_Mag):
            os.mkdir(self.EEVEC1_Mag)  
            
        self.EEVEC2_Mag = self.Zeldovich + '/Magnitude_v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2_Mag):
            os.mkdir(self.EEVEC2_Mag)
            
        self.EEVEC1Z_Mag = self.Zeldovich + '/Zoom_Magnitude_v_1_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC1Z_Mag):
            os.mkdir(self.EEVEC1Z_Mag)  
            
        self.EEVEC2Z_Mag = self.Zeldovich + '/Zoom_Magnitude_v_2_Eulerian_Quiver_Plots'
        if not os.path.exists(self.EEVEC2Z_Mag):
            os.mkdir(self.EEVEC2Z_Mag)

        self.DTDEV = self.Zeldovich + '/v_1_2_Quiver_Plots'
        if not os.path.exists(self.DTDEV):
            os.mkdir(self.DTDEV)
            
        self.DTDEVZ = self.Zeldovich + '/Zoom_v_1_2_Quiver_Plots'
        if not os.path.exists(self.DTDEVZ):
            os.mkdir(self.DTDEVZ)
            
        self.DTDEVEV1 = self.Zeldovich + '/v_1_2_Quiver_Plots_Density_Contours'
        if not os.path.exists(self.DTDEVEV1):
            os.mkdir(self.DTDEVEV1)
            
        self.DTDEVEV1Z = self.Zeldovich + '/Zoom_v_1_2_Quiver_Plots_Density_Contours'
        if not os.path.exists(self.DTDEVEV1Z):
            os.mkdir(self.DTDEVEV1Z)
            
        self.EVFD = self.Zeldovich + '/v_1_2_Direction_Flip_Flop_Flagging_Quiver_Plots'
        if not os.path.exists(self.EVFD):
            os.mkdir(self.EVFD)
            
        self.AEVD = self.Zeldovich + '/Alternate_v_1_2_Direction_Flip_Flop_Flagging_Quiver_Plots'
        if not os.path.exists(self.AEVD):
            os.mkdir(self.AEVD)
        
    def topologyDirectories(self):
        self.Isocurve_Out = self.Output_Folder + '/Particle_Volume_Isocurves'
        if not os.path.exists(self.Isocurve_Out):
            os.mkdir(self.Isocurve_Out)  
            
        self.ZpCMS = self.Isocurve_Out + '/Isocurve_Structures'
        if not os.path.exists(self.ZpCMS):
            os.mkdir(self.ZpCMS)
            
        self.ZpCMST = self.Isocurve_Out + '/Isocurve_Structures_Categories'
        if not os.path.exists(self.ZpCMST):
            os.mkdir(self.ZpCMST) 
       
    def jacobianDirectories(self):
        self.Jacobian = self.Output_Folder + '/Jacobian_Model'
        if not os.path.exists(self.Jacobian):
            os.mkdir(self.Jacobian)  
        
        self.LOFJ = self.Jacobian + '/Jacobian_Determinate_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LOFJ):
            os.mkdir(self.LOFJ) 
            
        self.EOFJ = self.Jacobian + '/Jacobian_Determinate_Eulerian_Dot_Plots'
        if not os.path.exists(self.EOFJ):
            os.mkdir(self.EOFJ) 
            
        self.LJC = self.Jacobian + '/Jacobian_Determinate_Categories_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LJC):
            os.mkdir(self.LJC) 
            
        self.EJC = self.Jacobian + '/Jacobian_Determinate_Categories_Eulergian_Dot_Plots'
        if not os.path.exists(self.EJC):
            os.mkdir(self.EJC) 
            
        self.LSpVT = self.Jacobian + '/Fluid_Element_Specific_Volume_Tagging_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LSpVT):
            os.mkdir(self.LSpVT)
            
        self.SpVT = self.Jacobian + '/Fluid_Element_Specific_Volume_Tagging_Eulerian_Dot_Plots'
        if not os.path.exists(self.SpVT):
            os.mkdir(self.SpVT)
            
        self.LZAD = self.Jacobian + '/Zeldovich_Density_Lagrangian_Dot_Plots'
        if not os.path.exists(self.LZAD):
            os.mkdir(self.LZAD)
            
        self.EZAD = self.Jacobian + '/Zeldovich_Density_Eulerian_Dot_Plots'
        if not os.path.exists(self.EZAD):
            os.mkdir(self.EZAD)
        
    def geometricDirectories(self):
        self.Geometric = self.Output_Folder + '/Geometric_Caustic_Model_Figures'
        if not os.path.exists(self.Geometric):
            os.mkdir(self.Geometric)
            
        self.TG = self.Geometric + '/Tessellation_Grid_Line_Plots'
        if not os.path.exists(self.TG):
            os.mkdir(self.TG)
        
        self.GVCM = self.Geometric + '/Geometric_Volume_Caustics_Line_Plots'
        if not os.path.exists(self.GVCM):
            os.mkdir(self.GVCM)
            
        self.LCF = self.Geometric + '/Tessellation_Total_Flip_Flops_Lagrangian_Line_Plots'
        if not os.path.exists(self.LCF):
            os.mkdir(self.LCF)
            
        self.ECF = self.Geometric + '/Tessellation_Total_Flip_Flops_Eulerian_Line_Plots'
        if not os.path.exists(self.ECF):
            os.mkdir(self.ECF)
            
        self.NAL = self.Geometric + '/Tessellation_Inverted_Tessellations_Lagrangian_Line_Plots'
        if not os.path.exists(self.NAL):
            os.mkdir(self.NAL)
            
        self.NAE = self.Geometric + '/Tessellation_Inverted_Tessellationss_Eulerian_Line_Plots'
        if not os.path.exists(self.NAE):
            os.mkdir(self.NAE)
    
    def angleDirectories(self):
        self.Angle = self.Output_Folder + '/Caustic_Collapse_Model_Figures'
        if not os.path.exists(self.Angle):
            os.mkdir(self.Angle)
        
        self.CAoGCL = self.Angle + '/Collapse_Angle_of_Geometric_Caustics_Lagrangian_Line_Plots'
        if not os.path.exists(self.CAoGCL):
            os.mkdir(self.CAoGCL)
            
        self.CAoGCE = self.Angle + '/Collapse_Angle_of_Geometric_Caustics_Eulerian_Line_Plots'
        if not os.path.exists(self.CAoGCE):
            os.mkdir(self.CAoGCE)
    
        self.TGAoCL = self.Angle + '/Lagrangian_Tessellation_Grid_Angle_of_Collapse'
        if not os.path.exists(self.TGAoCL):
            os.mkdir(self.TGAoCL)
            
        self.TGAoCE = self.Angle + '/Eulerian_Tessellation_Grid_Angle_of_Collapse'
        if not os.path.exists(self.TGAoCE):
            os.mkdir(self.TGAoCE)
            
        self.GDPAoCL = self.Angle + '/Lagrangian_Grid_Dot_Plots_Angle_of_Collapse'
        if not os.path.exists(self.GDPAoCL):
            os.mkdir(self.GDPAoCL)
            
        self.GDPAoCE = self.Angle + '/Eulerian_Grid_Dot_Plots_Angle_of_Collapse'
        if not os.path.exists(self.GDPAoCE):
            os.mkdir(self.GDPAoCE)
            
        self.DPAoCL = self.Angle + '/Lagrangian_Dot_Plots_Angle_of_Collapse'
        if not os.path.exists(self.DPAoCL):
            os.mkdir(self.DPAoCL)
            
        self.DPAoCE = self.Angle + '/Eulerian_Dot_Plots_Angle_of_Collapse'
        if not os.path.exists(self.DPAoCE):
            os.mkdir(self.DPAoCE)

    # '''
