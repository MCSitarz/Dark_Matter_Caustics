#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:40:45 2023
@author: michaelsitarz
"""
import sys
import numpy as np
import warnings
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class Jacobian_Based_Methods:
    def __init__(self, UP, B, C):
        self.B = B
        self.UP = UP
        self.C = C
        
        '''Importing Parameters'''
        self.q = self.B.q
        self.dqi = self.q[0][1,0] - self.q[0][0,0]
        self.dqj = self.q[1][0,1] - self.q[1][0,0]
        self.L = self.UP.L
        self.Np = self.UP.Np
        self.Mass_Resolution = self.L / self.Np
     
    def Jacobians(self, S):
        Q0_Minus = self.q[0] - self.dqi
        Q0_Plus  = self.q[0] + self.dqi
        Q1_Minus = self.q[1] - self.dqj
        Q1_Plus  = self.q[1] + self.dqj
                
        Q0 = []
        Q0.append(Q0_Minus[0])
        for i in range(len(S[0])):
            Q0.append(self.q[0][i])
        Q0.append(Q0_Plus[-1])
        Q0 = np.array(Q0)
        
        Q1 = []
        for i in range(len(S[0])):
            Grid_Column = np.insert(np.append(self.q[1][i], [Q1_Plus[i][-1]]), 0, Q1_Minus[i][0])
            Q1.append(np.array(Grid_Column))
        Q1 = np.array(Q1)
        
        dQ0_dqi_Interp = np.gradient(Q0, self.dqi, axis=0)
        dQ0_dqj_Interp = np.gradient(Q0, self.dqi, axis=1)
        dQ1_dqi_Interp = np.gradient(Q1, self.dqj, axis=0)
        dQ1_dqj_Interp = np.gradient(Q1, self.dqj, axis=1)
        
        dQ0_dqi = dQ0_dqi_Interp[1:-1]
        dQ0_dqj = dQ0_dqj_Interp[1:-1]
        dQ1_dqi = []
        dQ1_dqj = []
        for i in range(len(S[0])):
            dQ1_dqi.append(dQ1_dqi_Interp[i][1:-1])
            dQ1_dqj.append(dQ1_dqj_Interp[i][1:-1])
        dQ1_dqi = np.array(dQ1_dqi)
        dQ1_dqj = np.array(dQ1_dqj)
                        
        dDisi_dqi = np.gradient(S[0], self.dqi, axis=0)
        dDisi_dqj = np.gradient(S[0], self.dqj, axis=1)
        dDisj_dqi = np.gradient(S[1], self.dqi, axis=0)
        dDisj_dqj = np.gradient(S[1], self.dqj, axis=1)

        dXi_dQi = (dQ0_dqi * self.Mass_Resolution) + dDisi_dqi
        dXi_dQj = (dQ0_dqj * self.Mass_Resolution) + dDisi_dqj
        dXj_dQi = (dQ1_dqi * self.Mass_Resolution) + dDisj_dqi
        dXj_dQj = (dQ1_dqj * self.Mass_Resolution) + dDisj_dqj
        
        Jx = np.array([[[np.ravel(dXi_dQi)[i], np.ravel(dXi_dQj)[i]],[np.ravel(dXj_dQi)[i], np.ravel(dXj_dQj)[i]]] for i in range(len(np.ravel(self.q[0])))])
        
        return Jx 
            
    def Specific_Volume(self, Jx):
        Specific_Volumes = []
        for J in Jx:
            Specific_Volumes.append(np.linalg.det(J))
        Specific_Volumes = np.array(Specific_Volumes)
        return Specific_Volumes
    
    def Determinate_Sorting(self, Volumes):
        Positive = []
        Negative = []
        Zero = []
        for i in range(len(Volumes)):
            if Volumes[i] > 0:
                Positive.append(i)
            if Volumes[i] < 0:
                Negative.append(i)
            if Volumes[i] == 0:
                Zero.append(i)
        Sorted = [Positive, Negative, Zero]
        return Sorted
  
    def Total_Struct_Determinate_Sorting(self, Volumes, a):
            '''Sorting the specific volumes.''' 
            '''Going with 0 = Positive, 1 = Negative'''
            if a == 0.0:
                self.Previous_Sign = np.ones(len(Volumes))
                self.Current_Sign = np.ones(len(Volumes))
                self.Counts = np.zeros(len(Volumes))
            else:
                self.Previous_Sign = self.Current_Sign
                self.Current_Sign = np.ones(len(Volumes))
    
            Positive = []
            Negative = []
            Zero = []
            for i in range(len(Volumes)):
                if Volumes[i] > 0:
                    Positive.append(i)
                    self.Current_Sign[i] = 1 
                if Volumes[i] < 0:
                    Negative.append(i)
                    self.Current_Sign[i] = -1 
                if self.Current_Sign[i] != self.Previous_Sign[i]:
                    self.Counts[i] += 1
            Sorted = [Positive, Negative, Zero]
            return Sorted
    
    def Caustic_Connectivity(self, Grid_Volumes, Interpolated_Volumes, Flip_Counts, Particles, Sorted_Volumes):
        #------------------------------------------------------------------------------
        self.Flip_Counts = Flip_Counts
        self.Particles = Particles / (self.UP.L / self.UP.Np)
            
        q = np.indices((self.UP.Np, self.UP.Np)).astype(float)
        Q_Int_primer = q + 0.5
                        
        Filter_0 = np.where(np.ravel(Q_Int_primer[0]) < (self.UP.Np - 1))[0]
        Filter_1 = np.where(np.ravel(Q_Int_primer[1]) < (self.UP.Np - 1))[0]
        Filter = np.intersect1d(Filter_1, Filter_0)
                
        Q_Int = np.zeros((2, self.UP.Np - 1, self.UP.Np - 1))
        Q_Int[0] = np.ravel(Q_Int_primer[0])[Filter].reshape((self.UP.Np - 1, self.UP.Np - 1))
        Q_Int[1] = np.ravel(Q_Int_primer[1])[Filter].reshape((self.UP.Np - 1, self.UP.Np - 1))
        
        grid_q0 = q[0]
        grid_q1 = q[1]
        grid_q0_c = Q_Int[0]
        grid_q1_c = Q_Int[1]
        
        field = Grid_Volumes.reshape((self.UP.Np, self.UP.Np)) # simulated volumes # make sure it is in the correct structure
        field_c = Interpolated_Volumes[Filter].reshape((self.UP.Np - 1, self.UP.Np - 1))# coords. of box centers
        #------------------------------------------------------------------------------
        
        #------------------------------------------------------------------------------
        QT_cor_sorted, QH_cor_sorted = self.Shandarin_Line_Finder(self.UP.Np, self.UP.Np, grid_q0, grid_q1, field, grid_q0_c, grid_q1_c, field_c)
        # print(QT_cor_sorted)
        # print()
        # print(QH_cor_sorted)
        #------------------------------------------------------------------------------
        
        #------------------------------------------------------------------------------
        '''QH_cor_sorted (arrows) and QT_cor_sorted (lines) are the grid space sorted points'''
        '''With the grid space counting up then right'''
        Sorted_Starts = np.array(QH_cor_sorted[0])
        HIM = []
        for i in range(len(Sorted_Starts)):
            HIM.append([Sorted_Starts[i][1], Sorted_Starts[i][2]])
            
        Flip_Choices = np.unique(Flip_Counts)
        flipCategories = []
        for i in range(len(Flip_Choices)):
            if i == 0:
                continue
            else:
                flipCategories.append([Flip_Choices[i- 1], Flip_Choices[i]])
        
        curves = []
        if len(HIM) > 0:
            HIM = np.array(HIM)
            structBarriersSUMS = []
            structBarriers = []
            for point in HIM:
                ID1_ID2 = self.Line_Group_Finder(point[0], point[1])
                Flip_Count_1 = self.Flip_Count_Finder(ID1_ID2[0])
                Flip_Count_2 = self.Flip_Count_Finder(ID1_ID2[1])
                structBarriersSUMS.append(Flip_Count_1 + Flip_Count_2)
            for Category in flipCategories:
                targets = np.where(structBarriersSUMS == sum(Category))[0]
                v = HIM[targets]
                structBarriers.append(v)

            
            for i in range(len(structBarriers)):
                vertexHoldingCell = []
                curveLandingShuttle = []
                vertexHoldingCell = np.unique(structBarriers[i], axis=0)

                

                while len(vertexHoldingCell) > 1:
                    start = np.array(vertexHoldingCell[0])
                    vertexHoldingCell = np.delete(vertexHoldingCell, 0, axis=0)
                    index = np.argmin(np.sum((np.array(vertexHoldingCell) - np.array(start))**2, axis=1))
                    curveDraft = np.array([start, vertexHoldingCell[index]])
                    vertexHoldingCell = np.delete(vertexHoldingCell, index, axis=0)
                    while True:
                        distances = np.sum((np.array(vertexHoldingCell) - np.array(curveDraft[-1]))**2, axis=1)
                        if (len(distances) == 0) or (min(distances) > np.sqrt(2)):
                            break
                        index = np.argmin(distances)
                        curveDraft = np.append(curveDraft, [vertexHoldingCell[index]], axis=0)
                        vertexHoldingCell = np.delete(vertexHoldingCell, index, axis=0)
                    curveLandingShuttle.append(curveDraft)
            
                '''The else here only executes if the for runs all the way to the end 
                without ever encountering the condition that breaks out of it early.'''
                       
                while True:
                    for e in range(len(curveLandingShuttle) - 1):
                        alpha = curveLandingShuttle[e]
                        beta = curveLandingShuttle[e+1]
                        d1 = np.sqrt(np.square(beta[0][0] - alpha[0][0]) + np.square(beta[0][1] - alpha[0][1]))
                        d2 = np.sqrt(np.square(beta[-1][0] - alpha[0][0]) + np.square(beta[-1][1] - alpha[0][1]))
                        d3 = np.sqrt(np.square(beta[0][0] - alpha[-1][0]) + np.square(beta[0][1] - alpha[-1][1]))
                        d4 = np.sqrt(np.square(beta[-1][0] - alpha[-1][0]) + np.square(beta[-1][1] - alpha[-1][1]))
                        dn = [d1,d2,d3,d4]
                        d = np.argmin(dn)
                        if dn[d] <= np.sqrt(2):
                            print('Canidate:', d, dn, dn[d])
                            if d == 0:
                                delta = []
                                for f in reversed(alpha):
                                    delta.append(f)
                                for g in beta:
                                    delta.append(g)
                                curveLandingShuttle.append(delta)
                                curveLandingShuttle.pop(e+1)
                                curveLandingShuttle.pop(e)
                                break
                            if d == 1:
                                delta = []
                                for f in reversed(alpha):
                                    delta.append(f)
                                for g in reversed(beta):
                                    delta.append(g)
                                curveLandingShuttle.append(delta)
                                curveLandingShuttle.pop(e+1)
                                curveLandingShuttle.pop(e)
                                break
                            if d == 2:
                                delta = []
                                for f in alpha:
                                    delta.append(f)
                                for g in beta:
                                    delta.append(g)
                                curveLandingShuttle.append(delta)
                                curveLandingShuttle.pop(e+1)
                                curveLandingShuttle.pop(e)
                                break
                            if d == 3:
                                delta = []
                                for f in alpha:
                                    delta.append(f)
                                for g in reversed(beta):
                                    delta.append(g)
                                curveLandingShuttle.append(delta)
                                curveLandingShuttle.pop(e+1)
                                curveLandingShuttle.pop(e)
                                break
                    else:
                        break
                curves.append(curveLandingShuttle)
                
        if len(curves) > 0:
            return curves, flipCategories
        else:
            return [], flipCategories

    def Curve_Morphology_Sorting_Routine(self, Connected_Curves, Piecewise_Curves, Transition_Categories):
        CC_Transitions = []
        Morph_CC = []
        PC_Transitions = []
        Morph_PC = []
        for Curve in Connected_Curves:
            ID1_ID2 = self.Line_Group_Finder(Curve[0][0], Curve[0][1])
            Flip_Count_1 = self.Flip_Count_Finder(ID1_ID2[0])
            Flip_Count_2 = self.Flip_Count_Finder(ID1_ID2[1])
            CC_Transitions.append(Flip_Count_1 + Flip_Count_2)
        for Curve in Piecewise_Curves:
            ID1_ID2 = self.Line_Group_Finder(Curve[0][0], Curve[0][1])
            Flip_Count_1 = self.Flip_Count_Finder(ID1_ID2[0])
            Flip_Count_2 = self.Flip_Count_Finder(ID1_ID2[1])
            PC_Transitions.append(Flip_Count_1 + Flip_Count_2)
        
        for Category in Transition_Categories:
            CC_trans_idx = np.where(CC_Transitions == sum(Category))[0]
            Morph_CC.append(Connected_Curves[CC_trans_idx])
            PC_trans_idx = np.where(PC_Transitions == sum(Category))[0]
            Morph_PC.append(Piecewise_Curves[PC_trans_idx])
        return Morph_CC, Morph_PC
    
    def Line_Group_Finder(self, Start_x, Start_y):
        if Start_x in self.Q0bl_tl[:,0]:
            if Start_y in self.Q1bl_tl[:,0]:
                ID1_1 = [np.floor(Start_x), np.floor(Start_y)]
                ID1_2 = [np.floor(Start_x) + 0.5, np.floor(Start_y) + 0.5]
                return [ID1_1, ID1_2]
            
        if Start_x in self.Q0bl_tl[:,1]:
            if Start_y in self.Q1bl_tl[:,1]:
                ID2_1 = [np.floor(Start_x), np.ceil(Start_y)]
                ID2_2 = [np.floor(Start_x) + 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
        
        if Start_x in self.Q0tl_tr[:,0]:
            if Start_y in self.Q1tl_tr[:,0]:
                ID1_1 = [np.floor(Start_x), np.ceil(Start_y)]
                ID1_2 = [np.floor(Start_x) + 0.5, np.ceil(Start_y) - 0.5]
                return [ID1_1, ID1_2]
            
        if Start_x in self.Q0tl_tr[:,1]:
            if Start_y in self.Q1tl_tr[:,1]:
                ID2_1 = [np.ceil(Start_x), np.ceil(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
         
        if Start_x in self.Q0tr_br[:,0]:
            if Start_y in self.Q1tr_br[:,0]:
                ID1_1 = [np.ceil(Start_x), np.ceil(Start_y)]
                ID1_2 = [np.ceil(Start_x) - 0.5, np.ceil(Start_y) - 0.5]
                return [ID1_1, ID1_2]
            
        if Start_x in self.Q0tr_br[:,1]:
            if Start_y in self.Q1tr_br[:,1]:
                ID2_1 = [np.ceil(Start_x), np.floor(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
   
        if Start_x in self.Q0br_bl[:,0]:
            if Start_y in self.Q1br_bl[:,0]:
                ID1_1 = [np.ceil(Start_x), np.floor(Start_y)]
                ID1_2 = [np.ceil(Start_x) - 0.5, np.floor(Start_y) + 0.5]
                return [ID1_1, ID1_2]
        
        if Start_x in self.Q0br_bl[:,1]:
            if Start_y in self.Q1br_bl[:,1]:
                ID2_1 = [np.floor(Start_x), np.floor(Start_y)]
                ID2_2 = [np.floor(Start_x) + 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
        
        if Start_x in self.Q0l_bl[:,0]:
            if Start_y in self.Q1l_bl[:,0]:
                ID1_1 = [Start_x, np.floor(Start_y)]
                ID1_2 = [Start_x, np.ceil(Start_y)]
                return [ID1_1, ID1_2]
                 
        if Start_x in self.Q0l_bl[:,1]:
            if Start_y in self.Q1l_bl[:,1]:
                ID2_1 = [np.floor(Start_x), np.floor(Start_y)]
                ID2_2 = [np.floor(Start_x) + 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
                 
        if Start_x in self.Q0l_tl[:,0]:
            if Start_y in self.Q1l_tl[:,0]:
                ID1_1 = [Start_x, np.floor(Start_y)]
                ID1_2 = [Start_x, np.ceil(Start_y)]
                return [ID1_1, ID1_2]
                 
        if Start_x in self.Q0l_tl[:,1]:
            if Start_y in self.Q1l_tl[:,1]: 
                ID2_1 = [np.floor(Start_x), np.ceil(Start_y)]
                ID2_2 = [np.floor(Start_x) + 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
          
        if Start_x in self.Q0t_tl[:,0]:
            if Start_y in self.Q1t_tl[:,0]:
                ID1_1 = [np.floor(Start_x), Start_y] 
                ID1_2 = [np.ceil(Start_x), Start_y]
                return [ID1_1, ID1_2]
                   
        if Start_x in self.Q0t_tl[:,1]:
            if Start_y in self.Q1t_tl[:,1]:
                ID2_1 = [np.floor(Start_x), np.ceil(Start_y)] 
                ID2_2 = [np.floor(Start_x) + 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
                  
        if Start_x in self.Q0t_tr[:,0]:
            if Start_y in self.Q1t_tr[:,0]:
                ID1_1 = [np.floor(Start_x), Start_y]
                ID1_2 = [np.ceil(Start_x), Start_y]
                return [ID1_1, ID1_2]
                  
        if Start_x in self.Q0t_tr[:,1]:
            if Start_y in self.Q1t_tr[:,1]:
                ID2_1 = [np.ceil(Start_x), np.ceil(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
        
        if Start_x in self.Q0r_tr[:,0]:
            if Start_y in self.Q1r_tr[:,0]:
                ID1_1 = [Start_x, np.floor(Start_y)]
                ID1_2 = [Start_x, np.ceil(Start_y)]
                return [ID1_1, ID1_2]
                  
        if Start_x in self.Q0r_tr[:,1]:
            if Start_y in self.Q1r_tr[:,1]:
                ID2_1 = [np.ceil(Start_x), np.ceil(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.ceil(Start_y) - 0.5]
                return [ID2_1, ID2_2]
                 
        if Start_x in self.Q0r_br[:,0]:
            if Start_y in self.Q1r_br[:,0]:
                ID1_1 = [Start_x, np.floor(Start_y)] 
                ID1_2 = [Start_x, np.ceil(Start_y)]
                return [ID1_1, ID1_2]
                  
        if Start_x in self.Q0r_br[:,1]:
            if Start_y in self.Q1r_br[:,1]:
                ID2_1 = [np.ceil(Start_x), np.floor(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
          
        if Start_x in self.Q0b_br[:,0]:
            if Start_y in self.Q1b_br[:,0]:
                ID1_1 = [np.floor(Start_x), Start_y] 
                ID1_2 = [np.ceil(Start_x), Start_y]
                return [ID1_1, ID1_2]
                 
        if Start_x in self.Q0b_br[:,1]:
            if Start_y in self.Q1b_br[:,1]:
                ID2_1 = [np.ceil(Start_x), np.floor(Start_y)]
                ID2_2 = [np.ceil(Start_x) - 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
                 
        if Start_x in self.Q0b_bl[:,0]:
            if Start_y in self.Q1b_bl[:,0]:
                ID1_1 = [np.floor(Start_x), Start_y]
                ID1_2 = [np.ceil(Start_x), Start_y]
                return [ID1_1, ID1_2]
                 
        if Start_x in self.Q0b_bl[:,1]:
            if Start_y in self.Q1b_bl[:,1]:
                ID2_1 = [np.floor(Start_x), np.floor(Start_y)]
                ID2_2 = [np.floor(Start_x) + 0.5, np.floor(Start_y) + 0.5]
                return [ID2_1, ID2_2]
    
    def Flip_Count_Finder(self, ID):
        Find_1 = np.where(self.Particles[:,0] == ID[0])[0]
        Find_2 = np.where(self.Particles[:,1] == ID[1])[0]
        Idx = np.intersect1d(Find_1, Find_2)
        Grid_Point_Count = self.Flip_Counts[Idx] 
        return Grid_Point_Count
    
    def Curve_Connection_Function(self, Stitched_Curves):
        Connected_Curves = []
        Piecewise_Curves = []
        for i in reversed(Stitched_Curves):
            Subject = np.array(i[0])
            Curve_Vertices = [Subject[0][0], Subject[0][1]]
            for i in range(len(Subject)):
                if i == 0:
                    continue
                Curve_Vertices.append(Subject[i][1])
            
            '''Connect lines where endpoints are on the same border.'''
            # Starting_Point = Curve_Vertices[0]
            # Ending_Point = Curve_Vertices[-1]    
            # if (Starting_Point[0] < 0.5 and Ending_Point[0] < 0.5) or \
            #    (Starting_Point[0] > (self.UP.Np - 1.5) and Ending_Point[0] > (self.UP.Np - 1.5)) or \
            #    (Starting_Point[1] < 0.5 and Ending_Point[1] < 0.5) or \
            #    (Starting_Point[1] > (self.UP.Np - 1.5) and Ending_Point[1] > (self.UP.Np - 1.5)):
            #     Curve_Vertices.append(Starting_Point)
                
            '''Connect lines that lie over a corner.'''
            # if (Starting_Point[0] < 0.5 and Ending_Point[1] < 0.5) or \
            #    (Starting_Point[1] < 0.5 and Ending_Point[0] < 0.5):
            #     Curve_Vertices.append([0,0])
            #     Curve_Vertices.append(Starting_Point)
            # if (Starting_Point[0] < 0.5 and Ending_Point[1] > (self.UP.Np - 1.5)) or \
            #    (Starting_Point[1] > (self.UP.Np - 1.5) and Ending_Point[0] < 0.5):
            #     Curve_Vertices.append([0, self.UP.Np - 1])
            #     Curve_Vertices.append(Starting_Point)
            # if (Starting_Point[0] > (self.UP.Np - 1.5) and Ending_Point[1] > (self.UP.Np - 1.5)) or \
            #    (Starting_Point[1] > (self.UP.Np - 1.5) and Ending_Point[0] > (self.UP.Np - 1.5)):
            #     Curve_Vertices.append([self.UP.Np - 1, self.UP.Np - 1])
            #     Curve_Vertices.append(Starting_Point)
            # if (Starting_Point[0] > (self.UP.Np - 1.5) and Ending_Point[1] < 0.5) or \
            #    (Starting_Point[1] < 0.5 and Ending_Point[0] > (self.UP.Np - 1.5)):
            #     Curve_Vertices.append([self.UP.Np - 1, 0])
            #     Curve_Vertices.append(Starting_Point)
            
            if np.array_equal(Curve_Vertices[0], Curve_Vertices[-1]) == True:
                Connected_Curves.append(np.array(Curve_Vertices))
            else:
                Piecewise_Curves.append(Curve_Vertices)
                Piecewise_Curves.append(np.array(Curve_Vertices))
        
            if len(Piecewise_Curves) > 0:
                restart_Clause = True
                while restart_Clause:
                    restart_Clause = False
                    for i in range(len(Piecewise_Curves)):
                        Start_Point = Piecewise_Curves[i][0]
                        End_Point = Piecewise_Curves[i][-1]
                        for j in range(len(Piecewise_Curves)):
                            if np.array_equal(Piecewise_Curves[i], Piecewise_Curves[j]) == True:
                                continue
                            Starter_Distance_to_Start = abs(np.sqrt(np.square(Start_Point[0] - Piecewise_Curves[j][0][0]) + \
                                                                        np.square(Start_Point[1] - Piecewise_Curves[j][0][1])))
                            Starter_Distance_to_End = abs(np.sqrt(np.square(Start_Point[0] - Piecewise_Curves[j][-1][0]) + \
                                                            np.square(Start_Point[1] - Piecewise_Curves[j][-1][1])))
                            Ender_Distance_to_Start = abs(np.sqrt(np.square(End_Point[0] - Piecewise_Curves[j][0][0]) + \
                                                            np.square(End_Point[1] - Piecewise_Curves[j][0][1])))
                            Ender_Distance_to_End = abs(np.sqrt(np.square(End_Point[0] - Piecewise_Curves[j][-1][0]) + \
                                                            np.square(End_Point[1] - Piecewise_Curves[j][-1][1])))
    
                            if Starter_Distance_to_Start < 1.0:
                                New_Connection = []
                                for segment in Piecewise_Curves[i][::-1]:
                                    New_Connection.append(segment)
                                for segment in Piecewise_Curves[j]:
                                    New_Connection.append(segment)
                                Piecewise_Curves[i] = np.array(New_Connection)
                                Piecewise_Curves = np.delete(Piecewise_Curves, j, 0)
                                restart_Clause = True
                                break
                                
                            if Starter_Distance_to_End < 1.0:
                                New_Connection = []
                                for segment in Piecewise_Curves[i][::-1]:
                                    New_Connection.append(segment)
                                for segment in Piecewise_Curves[j][::-1]:
                                    New_Connection.append(segment)
                                Piecewise_Curves[i] = New_Connection
                                Piecewise_Curves = np.delete(Piecewise_Curves, j, 0)
                                restart_Clause = True
                                break
                            
                            if Ender_Distance_to_Start < 1.0:
                                New_Connection = []
                                for segment in Piecewise_Curves[i]:
                                    New_Connection.append(segment)
                                for segment in Piecewise_Curves[j]:
                                    New_Connection.append(segment)
                                Piecewise_Curves[i] = New_Connection
                                Piecewise_Curves = np.delete(Piecewise_Curves, j, 0)
                                restart_Clause = True
                                break
                                
                            if Ender_Distance_to_End < 1.0:
                                New_Connection = []
                                for segment in Piecewise_Curves[i]:
                                    New_Connection.append(segment)
                                for segment in Piecewise_Curves[j][::-1]:
                                    New_Connection.append(segment)
                                Piecewise_Curves[i] = New_Connection
                                Piecewise_Curves = np.delete(Piecewise_Curves, j, 0)
                                restart_Clause = True
                                break
                                
                        else:
                            continue  # only executed if the inner loop did NOT break
                        break  # only executed if the inner loop DID break
            # print('PC Insurance Banked', time.time() - startpc)        
            
            temp = Connected_Curves
            Connected_Curves = []
            for i in range(len(temp)):
                Connected_Curves.append(np.array(temp[i]))
            del(temp)
            
            temp = Piecewise_Curves
            Piecewise_Curves = []
            for i in range(len(temp)):
                if np.array_equal(temp[i][0], temp[i][-1]) == True:
                    Connected_Curves.append(np.array(temp[i]))
                else:
                    Piecewise_Curves.append(np.array(temp[i]))
            del(temp)
            # temp = Piecewise_Curves
            # for i in range(len(temp)):
            #     for j in range(i, len(temp)):
            #         if temp[i] == temp[j]:
                        
            
        return Connected_Curves, Piecewise_Curves
        
    def Sitarz_Inter_Bin_Stiching(self, Bin_Curves):                    
        '''Bins "i" can share lines with "i + 1" and "i + (N - 1)" '''
        '''Loop through Bins''' 
        for i in range(len(Bin_Curves)):
            '''Loop through Curves in the Bins'''
            for j in range(len(Bin_Curves[i])):
                
                '''If Bin_Curves[i][j] is enmpty can i put a continue here to skip all the checks? can then eliminate the length check'''
                if len(Bin_Curves[i][j]) == 0:
                    continue
                
                '''Create a temp variable to hold the current curve we look to connect things to. This is the curve that will be added on to.'''
                subject_curve = Bin_Curves[i][j]
                '''Create a holding array to store the index (i + 1 and/or i + (N-1)) of the curves that connect with the subject'''
                target_bin_index = []
                '''Create a bin to hold the curve index to cirrespond to the bin number above, identifying curve [k][a]'''
                target_curve_index = []
                '''Loop Through Possible Bins'''
                for k in (i + 1, i + (self.UP.Np - 1)):
                    '''Need to break the loop if the new bin id isnt allowed (id > N^1 - 1)'''
                    if k > ((self.UP.Np-1)**2 - 1):
                        break
                    '''Loop Through Curves in New Bin'''
                    for a in range(len(Bin_Curves[k])):
                        if len(Bin_Curves[k][a]) == 0:
                            continue
                        '''create a temp variable of the curve that is checked if it connects with the subject'''
                        possible_target = Bin_Curves[k][a]
        
                        '''Can i place a continue after the if statments if they are triggered? Is ther ever a scenario that two will be 
                        triggered in the same loop? If there is no [i][j] then 
                        cant I move to next [j]'''
               
                        '''Head of subject matches tail of target'''
                        if len(possible_target) != 0 and np.array_equal(subject_curve[0][0], possible_target[-1][1]) == True:
                            target_bin_index.append(k)
                            target_curve_index.append(a)
                            for b in reversed(possible_target):
                                subject_curve.insert(0, b)
                            possible_target = []
                        '''Tail of subject matches head of target'''
                        if len(possible_target) != 0 and np.array_equal(subject_curve[-1][1], possible_target[0][0]) == True:
                            target_bin_index.append(k)
                            target_curve_index.append(a)
                            for b in possible_target:
                                subject_curve.append(b)
                            possible_target = []
                        '''Head of subject matches head of target'''
                        if len(possible_target) != 0 and np.array_equal(subject_curve[0][0], possible_target[0][0]) == True:
                            target_bin_index.append(k)
                            target_curve_index.append(a)
                            for b in possible_target:
                                subject_curve.insert(0, b)
                            possible_target = []
                        '''Tail of subject matches tail of target'''
                        if len(possible_target) != 0 and np.array_equal(subject_curve[-1][1], possible_target[-1][1]) == True:
                            target_bin_index.append(k)
                            target_curve_index.append(a)
                            for b in reversed(possible_target):
                                subject_curve.append(b)
                            possible_target = []
                                
                '''after looping through each curve of the two possible bins, find the max bin index
                   that was triggered, place the newly connected curve in that bin, and empty the component curve bins.'''
                if len(target_bin_index) != 0:
                    if max(target_bin_index) == (i + 1):
                        Bin_Curves[i][j] = []
                        c = target_bin_index.index(i+1)
                        Bin_Curves[target_bin_index[c]][target_curve_index[c]] = subject_curve
                    if max(target_bin_index) == (i + (self.UP.Np - 1)):
                        Bin_Curves[i][j] = []
                        if (i + 1) in target_bin_index:
                            c = target_bin_index.index(i+1)
                            Bin_Curves[target_bin_index[c]][target_curve_index[c]] = []
                        d = target_bin_index.index(i+(self.UP.Np-1))
                        Bin_Curves[target_bin_index[d]][target_curve_index[d]] = subject_curve
        
        return Bin_Curves  
        
    def Sitarz_Intra_Bin_Stiching(self, Bin, Sorted):
        Bin_Curves_Index = []
        Bin_Curves = []
        '''Loop through each grid bin'''
        for i in range((self.UP.Np - 1)**2):
            '''Find the indices of the line segments in a single bin.'''
            idx = np.where(Bin == i)[0]
            Bin_Curves_Index.append(i)
            '''At Most, there can be four curves per box.'''     
            Bin_Curve_Order_1 = []
            Bin_Curve_Order_2 = []
            Bin_Curve_Order_3 = []
            Bin_Curve_Order_4 = []
            '''Loop through the line segments foudn to be in a single bin.'''
            '''Many if statments because onle one statment can be TRUE.'''
            for j in range(len(idx)):
                '''Curve 1'''
                if j == 0:
                    Bin_Curve_Order_1.append(Sorted[idx[j]])
                    continue
                '''Check to see if the point head matches the curve tail.'''
                if np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_1[-1][1]) == True:
                    Bin_Curve_Order_1.append(Sorted[idx[j]])
                    continue
                '''Check to see if the point tail matches the curve head.'''
                if np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_1[0][0]) == True:
                    Bin_Curve_Order_1.insert(0, Sorted[idx[j]])
                    continue
                '''If the point does not belong to the first curve, start another curve.'''
                if      np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_1[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_1[0][0]) == False \
                    and len(Bin_Curve_Order_2) == 0:
                    Bin_Curve_Order_2.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_2[-1][1]) == True:
                    Bin_Curve_Order_2.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_2[0][0]) == True:
                    Bin_Curve_Order_2.insert(0, Sorted[idx[j]])
                    continue
                '''If the point does not belong in 1 or 2, start 3.'''
                if      np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_1[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_1[0][0]) == False \
                    and np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_2[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_2[0][0]) == False \
                    and len(Bin_Curve_Order_3) == 0:
                    Bin_Curve_Order_3.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_3[-1][1]) == True:
                    Bin_Curve_Order_3.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_3[0][0]) == True:
                    Bin_Curve_Order_3.insert(0, Sorted[idx[j]])
                    continue
                '''If the point does not belong in 1, 2, or 3, start the final curve.'''
                if      np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_1[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_1[0][0]) == False \
                    and np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_2[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_2[0][0]) == False \
                    and np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_3[-1][1]) == False \
                    and np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_3[0][0]) == False \
                    and len(Bin_Curve_Order_4) == 0:
                    Bin_Curve_Order_4.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][0], Bin_Curve_Order_4[-1][1]) == True:
                    Bin_Curve_Order_4.append(Sorted[idx[j]])
                    continue
                if np.array_equal(Sorted[idx[j]][1], Bin_Curve_Order_4[0][0]) == True:
                    Bin_Curve_Order_4.insert(0, Sorted[idx[j]])
                    continue
            
            '''Finally, check to see if any of the four curves connect to eachother.'''
            '''Due to the original structure of heads and tails, we dont need to check if intital points
               match intial points or if tail points match tail points'''
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_2) != 0:
                '''Curve_1[0] = Curve_2[-1]'''
                if np.array_equal(Bin_Curve_Order_1[0][0], Bin_Curve_Order_2[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_2):
                        Bin_Curve_Order_1.insert(0, i)
                    Bin_Curve_Order_2 = Bin_Curve_Order_3
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
                '''Curve_1[-1] = Curve_2[0]'''
                if len(Bin_Curve_Order_2) != 0 and np.array_equal(Bin_Curve_Order_1[-1][1], Bin_Curve_Order_2[0][0]) == True:
                    for i in Bin_Curve_Order_2:
                        Bin_Curve_Order_1.append(i)
                    Bin_Curve_Order_2 = Bin_Curve_Order_3
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
            
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_3) != 0:
                '''Curve_1[0] = Curve_3[-1]'''
                if np.array_equal(Bin_Curve_Order_1[0][0], Bin_Curve_Order_3[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_3):
                        Bin_Curve_Order_1.insert(0, i)
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
                '''Curve_1[-1] = Curve_3[0]'''
                if len(Bin_Curve_Order_3) != 0 and np.array_equal(Bin_Curve_Order_1[-1][1], Bin_Curve_Order_3[0][0]) == True:
                    for i in Bin_Curve_Order_3:
                        Bin_Curve_Order_1.append(i)
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
            
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_4) != 0:
                '''Curve_1[0] = Curve_4[-1]'''
                if np.array_equal(Bin_Curve_Order_1[0][0], Bin_Curve_Order_4[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_4):
                        Bin_Curve_Order_1.insert(0, i)
                    Bin_Curve_Order_4 = []
                '''Curve_1[-1] = Curve_4[0]'''
                if len(Bin_Curve_Order_4) != 0 and np.array_equal(Bin_Curve_Order_1[-1][1], Bin_Curve_Order_4[0][0]) == True:
                    for i in Bin_Curve_Order_4:
                        Bin_Curve_Order_1.append(i)
                    Bin_Curve_Order_4 = []
            
            if len(Bin_Curve_Order_2) != 0 and len(Bin_Curve_Order_3) != 0:
                '''Curve_2[0] = Curve_3[-1]'''
                if np.array_equal(Bin_Curve_Order_2[0][0], Bin_Curve_Order_3[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_3):
                        Bin_Curve_Order_2.insert(0, i)
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
                '''Curve_2[-1] = Curve_3[0]'''
                if len(Bin_Curve_Order_3) != 0 and np.array_equal(Bin_Curve_Order_2[-1][1], Bin_Curve_Order_3[0][0]) == True:
                    for i in Bin_Curve_Order_3:
                        Bin_Curve_Order_2.append(i)
                    Bin_Curve_Order_3 = Bin_Curve_Order_4
                    Bin_Curve_Order_4 = []
                    
            if len(Bin_Curve_Order_2) != 0 and len(Bin_Curve_Order_4) != 0:
                '''Curve_2[0] = Curve_4[-1]'''
                if np.array_equal(Bin_Curve_Order_2[0][0], Bin_Curve_Order_4[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_4):
                        Bin_Curve_Order_2.insert(0, i)
                    Bin_Curve_Order_4 = []
                '''Curve_2[-1] = Curve_4[0]'''
                if len(Bin_Curve_Order_4) != 0 and np.array_equal(Bin_Curve_Order_2[-1][1], Bin_Curve_Order_4[0][0]) == True:
                    for i in Bin_Curve_Order_4:
                        Bin_Curve_Order_2.append(i)
                    Bin_Curve_Order_4 = []
            
            if len(Bin_Curve_Order_3) != 0 and len(Bin_Curve_Order_4) != 0:
                '''Curve_3[0] = Curve_4[-1]'''
                if np.array_equal(Bin_Curve_Order_3[0][0], Bin_Curve_Order_4[-1][1]) == True:
                    for i in reversed(Bin_Curve_Order_4):
                        Bin_Curve_Order_3.insert(0, i)
                    Bin_Curve_Order_4 = []
                '''Curve_3[-1] = Curve_4[0]'''
                if len(Bin_Curve_Order_4) != 0 and np.array_equal(Bin_Curve_Order_3[-1][1], Bin_Curve_Order_4[0][0]) == True:
                    for i in Bin_Curve_Order_4:
                        Bin_Curve_Order_3.append(i)
                    Bin_Curve_Order_4 = []
            
            
            '''Place the Bin Curves into the Large array according to how many were filled.'''
            if len(Bin_Curve_Order_1) == 0 and len(Bin_Curve_Order_2) == 0 \
                and len(Bin_Curve_Order_3) == 0 and len(Bin_Curve_Order_4) == 0:
                Bin_Curves.append([])
            
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_2) == 0 \
                and len(Bin_Curve_Order_3) == 0 and len(Bin_Curve_Order_4) == 0:
                Bin_Curves.append([Bin_Curve_Order_1])
                    
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_2) != 0 \
                and len(Bin_Curve_Order_3) == 0 and len(Bin_Curve_Order_4) == 0:
                Bin_Curves.append([Bin_Curve_Order_1, Bin_Curve_Order_2])
                    
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_2) != 0 \
                and len(Bin_Curve_Order_3) != 0 and len(Bin_Curve_Order_4) == 0:
                Bin_Curves.append([Bin_Curve_Order_1, Bin_Curve_Order_2, Bin_Curve_Order_3])
                
            if len(Bin_Curve_Order_1) != 0 and len(Bin_Curve_Order_2) != 0 \
                and len(Bin_Curve_Order_3) != 0 and len(Bin_Curve_Order_4) != 0:
                Bin_Curves.append([Bin_Curve_Order_1, Bin_Curve_Order_2, Bin_Curve_Order_3, Bin_Curve_Order_4])
        
        return Bin_Curves
        
    def Shandarin_Line_Finder(self, Ng0, Ng1, grid_q0, grid_q1, field, grid_q0_c, grid_q1_c, field_c):
        
        q0_c_neg, q1_c_neg, q0_c_pos, q1_c_pos = self.coords_neg_pos_vols(grid_q0_c,grid_q1_c, field_c)
        
        q0_neg,q1_neg, q0_pos,q1_pos = self.coords_neg_pos_vols(grid_q0,grid_q1, field)
        
        q0_zero_on_Q0, q1_zero_on_Q0 = self.crossing_q0(grid_q0,grid_q1, field)
        q0_zero_on_Q1, q1_zero_on_Q1 = self.crossing_q1(grid_q0,grid_q1, field)
        
        q0Hseg,q1Hseg,q0Vseg,q1Vseg,q0BLseg,q0BRseg,q1BLseg,q1TLseg = \
                                    self.grid_segments(grid_q0,grid_q1,grid_q0_c,grid_q1_c)
        #------------------------------------------------------------------------------
        diag_id = 'bl'
        q0_zero_bl, q1_zero_bl = self.crossing_diags(diag_id,field,grid_q0,grid_q1,field_c,grid_q0_c,grid_q1_c)
        diag_id = 'tl'
        q0_zero_tl, q1_zero_tl = self.crossing_diags(diag_id,field,grid_q0,grid_q1,field_c,grid_q0_c,grid_q1_c)
        diag_id = 'br'
        q0_zero_br, q1_zero_br = self.crossing_diags(diag_id,field,grid_q0,grid_q1,field_c,grid_q0_c,grid_q1_c)
        diag_id = 'tr'
        q0_zero_tr, q1_zero_tr = self.crossing_diags(diag_id,field,grid_q0,grid_q1,field_c,grid_q0_c,grid_q1_c)
        
        #----------------DIAGONAL SEGMENTS-----------------------------------------------------------
        
        ################# BL - TL 
        self.Q0bl_tl, self.Q1bl_tl = self.zero_line_segments(q0_zero_bl, q1_zero_bl, q0_zero_tl, q1_zero_tl)
        
        ################# TL - TR 
        self.Q0tl_tr, self.Q1tl_tr = self.zero_line_segments(q0_zero_tl, q1_zero_tl, q0_zero_tr, q1_zero_tr)
        
        ################# TR - BR 
        self.Q0tr_br, self.Q1tr_br = self.zero_line_segments(q0_zero_tr, q1_zero_tr, q0_zero_br, q1_zero_br)
        
        ################# BR - BL 
        self.Q0br_bl, self.Q1br_bl = self.zero_line_segments(q0_zero_br, q1_zero_br, q0_zero_bl, q1_zero_bl)
        
        #-----------------------------------------------------------------------------
        ind_q0_zero_on_Q1 = np.int32(q0_zero_on_Q1)
        ind_ind_q0_zero_on_l = np.where(ind_q0_zero_on_Q1 < Ng0-1)
        q0_zero_on_l = q0_zero_on_Q1[ind_ind_q0_zero_on_l]
        q1_zero_on_l = q1_zero_on_Q1[ind_ind_q0_zero_on_l]
        
        ################# L - BL
        self.Q0l_bl, self.Q1l_bl = self.zero_line_segments(q0_zero_on_l, q1_zero_on_l, q0_zero_bl, q1_zero_bl) 
        ################# L - TL
        self.Q0l_tl, self.Q1l_tl = self.zero_line_segments(q0_zero_on_l, q1_zero_on_l, q0_zero_tl, q1_zero_tl)  
        
        #-----------------------------------------------------------------------------
        ind_q1_zero_on_Q0 = np.int32(q1_zero_on_Q0)
        ind_ind_q1_zero_on_t = np.where(ind_q1_zero_on_Q0 > 0)
        q0_zero_on_t = q0_zero_on_Q0[ind_ind_q1_zero_on_t]
        q1_zero_on_t = q1_zero_on_Q0[ind_ind_q1_zero_on_t]
        
        ################# T - TL
        self.Q0t_tl, self.Q1t_tl = self.zero_line_segments(q0_zero_on_t, q1_zero_on_t, q0_zero_tl, q1_zero_tl, cor_ind = 'T') 
        ################# T - TR 
        self.Q0t_tr, self.Q1t_tr = self.zero_line_segments(q0_zero_on_t, q1_zero_on_t, q0_zero_tr, q1_zero_tr, cor_ind = 'T')
        
        ind_q0_zero_on_Q1 = np.int32(q0_zero_on_Q1)
        ind_ind_q0_zero_on_r = np.where(ind_q0_zero_on_Q1 > 0)
        q0_zero_on_r = q0_zero_on_Q1[ind_ind_q0_zero_on_r]
        q1_zero_on_r = q1_zero_on_Q1[ind_ind_q0_zero_on_r]
        
        ################# R - TR 
        self.Q0r_tr, self.Q1r_tr = self.zero_line_segments(q0_zero_on_r, q1_zero_on_r, q0_zero_tr, q1_zero_tr, cor_ind = 'R') 
        ################# R - BR 
        self.Q0r_br, self.Q1r_br = self.zero_line_segments(q0_zero_on_r, q1_zero_on_r, q0_zero_br, q1_zero_br, cor_ind = 'R')
        
        ind_q1_zero_on_Q0 = np.int32(q1_zero_on_Q0)
        ind_ind_q1_zero_on_b = np.where(ind_q1_zero_on_Q0 < Ng1-1)
        q0_zero_on_b = q0_zero_on_Q0[ind_ind_q1_zero_on_b]
        q1_zero_on_b = q1_zero_on_Q0[ind_ind_q1_zero_on_b]
        
        ################# B - BR 
        self.Q0b_br, self.Q1b_br = self.zero_line_segments(q0_zero_on_b, q1_zero_on_b, q0_zero_br, q1_zero_br) 
        ################# B - BL  Q0-th
        self.Q0b_bl, self.Q1b_bl = self.zero_line_segments(q0_zero_on_b, q1_zero_on_b, q0_zero_bl, q1_zero_bl)
        #----------
        
        Q0_th = np.vstack((self.Q0bl_tl, self.Q0tl_tr, self.Q0tr_br, self.Q0br_bl,
                self.Q0l_bl, self.Q0l_tl,  self.Q0t_tl, self.Q0t_tr,  self.Q0r_tr, self.Q0r_br,  self.Q0b_br, self.Q0b_bl))
        Q1_th = np.vstack((self.Q1bl_tl, self.Q1tl_tr, self.Q1tr_br, self.Q1br_bl,
                self.Q1l_bl, self.Q1l_tl,  self.Q1t_tl, self.Q1t_tr,  self.Q1r_tr, self.Q1r_br,  self.Q1b_br, self.Q1b_bl))
        
        # fig = plt.figure()
        # plt.gca().set_aspect('equal')
        # for x in range(len(Q0_th)):
        #     plt.plot([Q0_th[x][0], Q0_th[x][1]], [Q1_th[x][0], Q1_th[x][1]])
        # plt.xlim(40, 60)
        # plt.ylim(40, 60)
        # plt.show()
        # plt.close()
        # plt.clf()
        
        # fig = plt.figure()
        # plt.gca().set_aspect('equal')
        # for x in range(len(Q0_th)):
        #     plt.plot([Q0_th[x][0], Q0_th[x][1]], [Q1_th[x][0], Q1_th[x][1]])
        # plt.xlim(0, self.UP.Np)
        # plt.ylim(0, self.UP.Np)
        # plt.show()
        # plt.close()
        # plt.clf()
        
        QT = np.dstack((Q0_th[:,0].transpose(),Q1_th[:,0].transpose()))[0]
        QH = np.dstack((Q0_th[:,1].transpose(),Q1_th[:,1].transpose()))[0]
        
        # 2D box indecies i = box_ij[0] and j = box_ij[1]
        box_ind0 = np.int32((Q0_th[:,0] + Q0_th[:,1])/2)
        box_ind1 = np.int32((Q1_th[:,1] + Q1_th[:,1])/2)
        box_ind01 = np.stack((box_ind0, box_ind1))
        box_ind_1d = box_ind0*(Ng1-1) + box_ind1
        #-----------------------------------------------------------------------------
        box_c_q01 = np.float32(box_ind01 + 0.5)                              # FLOAT16
        
        field_c_1d = field_c.ravel()    # field at centers of boxes (1D array)
        
                                        # random orientation
        QT0 = np.float32(QT[:,0]); QT1 = np.float32(QT[:,1])    # TAIL COORDINATES 
        QH0 = np.float32(QH[:,0]); QH1 = np.float32(QH[:,1])    # HEAD COORDINATES
        #-----------------------------------------------------------------------------
        # This is from "Orientation,Simplicity ..." paper p. 596
        #                     | qc0 qc1 1 |   qc0 = box_c_q0q1[0], etc
        #trngl_sign = np.sign | qt0 qt1 1 |   #
        #                     | qh0 qh1 1 |
        #
        # print('hi')
        trngl_sign = np.sign(- ( box_c_q01[0] * QH1 - box_c_q01[1] * QH0) 
                     +(QT0 * QH1) - (QT1 * QH0)
                     +(box_c_q01[0] * QT1 - box_c_q01[1] * QT0))
        
        '''Mike's Addition to bypass an error where an index > size'''
        box_ind_1d = np.delete(box_ind_1d, box_ind_1d > len(field_c_1d) - 1, axis=0)
        # print((self.UP.Np - 1)**2)
        # print(len(box_ind_1d), len(field_c_1d), len(trngl_sign), len(field_c_1d[box_ind_1d]))
        # print(np.where(box_ind_1d > len(box_ind_1d))[0])
        # if len(box_ind_1d) != 0:
        #     print('Um, Error?', max(box_ind_1d))
        # print('hey')
        cond = trngl_sign * field_c_1d[box_ind_1d] # condition to correct contour orientatons
        # print('ho')
                                  
        # coordinates of Tail and Head points Correctly Oriented  (co)
        QH0cor = np.where(cond < 0, QH0, QT0)
        QH1cor = np.where(cond < 0, QH1, QT1)
        QT0cor = np.where(cond < 0, QT0, QH0)
        QT1cor = np.where(cond < 0, QT1, QH1)
        

        ind_sort_box_ind_1d = np.argsort(box_ind_1d)
        box_ind_1d_sorted  = box_ind_1d[ind_sort_box_ind_1d]
        QT0_cor_sorted = QT0cor[ind_sort_box_ind_1d]    
        QT1_cor_sorted = QT1cor[ind_sort_box_ind_1d]
        QT_cor_sorted = np.float32(np.dstack((box_ind_1d_sorted, QT0_cor_sorted, 
                                              QT1_cor_sorted)))
        
        QH0_cor_sorted = QH0cor[ind_sort_box_ind_1d]
        QH1_cor_sorted = QH1cor[ind_sort_box_ind_1d]
        QH_cor_sorted = np.float32(np.dstack((box_ind_1d_sorted, QH0_cor_sorted, 
                                              QH1_cor_sorted)))
        
        return QT_cor_sorted, QH_cor_sorted

    def coords_neg_pos_vols(self, grid_q0,grid_q1, field):
        """Returns four 1d arrays: 
        two L coordinats where volumes < 0 and two where volumes >= 0
        """
        ind_neg = np.where(field < 0)
        q0_neg = grid_q0[ind_neg]; q1_neg = grid_q1[ind_neg]
    
        ind_pos = np.where(field >=  0)
        q0_pos = grid_q0[ind_pos]; q1_pos = grid_q1[ind_pos] 
        return q0_neg,q1_neg,q0_pos,q1_pos

    def crossing_q0(self, grid_q0, grid_q1, field):
        """ Returns L coordinates where volumes = 0 (interpoated) on q0 lines (q1=const)
        """
        vols_in_low_on_Q0 = field[0:-1,:]
        q0_low_on_Q0 = grid_q0[0:-1,:]
    #    q1_low_on_Q0 = grid_q1[0:-1,:]
    
        vols_in_high_on_Q0 = field[1:,:]
        q0_high_on_Q0 = grid_q0[1:,:]
        q1_high_on_Q0 = grid_q1[1:,:]
    
        neg_prod_on_Q0 = vols_in_low_on_Q0 * vols_in_high_on_Q0
        ind_neg_prod_Q0 = np.where(neg_prod_on_Q0 < 0)
    
        x1 = q0_low_on_Q0[ind_neg_prod_Q0];      x2 = q0_high_on_Q0[ind_neg_prod_Q0]
        u1 = vols_in_low_on_Q0[ind_neg_prod_Q0]; u2 = vols_in_high_on_Q0[ind_neg_prod_Q0]
        #-------------------------------coordinates of vols=0 q0 lines
        q0_zero_on_Q0 = (x1*u2 - x2*u1)/(u2 - u1)
        q1_zero_on_Q0 = q1_high_on_Q0[ind_neg_prod_Q0]
        return q0_zero_on_Q0, q1_zero_on_Q0

    def crossing_q1(self, grid_q0, grid_q1, field):
        """ Returns L coordinates where volumes = 0 (interpoated) on q1 lines (q0=const)
        """
        vols_in_low_on_Q1 = field[:, 0:-1]
    #    q0_low_on_Q1 = grid_q0[:, 0:-1]
        q1_low_on_Q1 = grid_q1[:, 0:-1]
    
        vols_in_high_on_Q1 = field[:,1:]
        q0_high_on_Q1 = grid_q0[:,1:]
        q1_high_on_Q1 = grid_q1[:,1:]
    
        neg_prod_on_Q1 = vols_in_low_on_Q1 * vols_in_high_on_Q1
        ind_neg_prod_Q1 = np.where(neg_prod_on_Q1 < 0)
    #    ind_pos_prod_Q1 = np.where(neg_prod_on_Q1 >= 0)
    
        y1 = q1_low_on_Q1[ind_neg_prod_Q1];      y2 = q1_high_on_Q1[ind_neg_prod_Q1]
        v1 = vols_in_low_on_Q1[ind_neg_prod_Q1]; v2 = vols_in_high_on_Q1[ind_neg_prod_Q1]
    
        q1_zero_on_Q1 = (y1*v2 - y2*v1)/(v2 - v1)
        q0_zero_on_Q1 = q0_high_on_Q1[ind_neg_prod_Q1]
        return q0_zero_on_Q1, q1_zero_on_Q1
#==============================================================================       
    def grid_segments(self, grid_q0, grid_q1, grid_q0_c, grid_q1_c):
        """ Returns the arrays of end L coordinates of horizontal (xh,yh), vertical(xv,yv) 
            and diagonal (xbl,xbr, ybl, ytl) segments (bottom-left,bottom-right, )
            important xtl = xbl; ybr=ybl; xtr=xbr; ytr=ytl
        """
        # horizontal segments
        q0h=np.column_stack((grid_q0[:-1].ravel(),grid_q0[1:].ravel().ravel()))
        q1h=np.column_stack((grid_q1[:-1].ravel(),grid_q1[1:].ravel()))
        # vertical segments
        q0v=np.column_stack((grid_q0[:,:-1].ravel(),grid_q0[:,1:].ravel().ravel()))
        q1v=np.column_stack((grid_q1[:,:-1].ravel(),grid_q1[:,1:].ravel()))
        # xbl xbr diagonal segments
        q0bl=np.column_stack((grid_q0[:-1,:-1].transpose().ravel(),grid_q0_c.transpose().ravel()))
        q0br=np.column_stack((grid_q0[1:,:-1]. transpose().ravel(),grid_q0_c.transpose().ravel()))
        # ybl ytl diagonal segments
        q1bl=np.column_stack((grid_q1[:-1,:-1].transpose().ravel(),grid_q1_c.transpose().ravel()))
        q1tl=np.column_stack((grid_q1[:-1,1:]. transpose().ravel(),grid_q1_c.transpose().ravel()))    
        return q0h,q1h,q0v,q1v,q0bl,q0br,q1bl,q1tl

    def crossing_diags(self, diag_id, field, grid_q0, grid_q1, vol_c, grid_q0_c, grid_q1_c):
        """Reterns L coordinates of points on b-l or t-l or b-r or t-r diagonals
        """
        v2 = vol_c.ravel()
        if diag_id =='bl':
            v1 = field[:-1,:-1].ravel()
            ind = np.where(v1 * v2 < 0)
            v1 = field[:-1,:-1].ravel()[ind]
            x1 = grid_q0[:-1,:-1].ravel()[ind]
            y1 = grid_q1[:-1,:-1].ravel()[ind]    
        elif diag_id =='tl':
            v1 = field[:-1,1:].ravel()
            ind = np.where(v1 * v2 < 0)
            v1 = field[:-1,1:].ravel()[ind]
            x1 = grid_q0[:-1,1:].ravel()[ind]    
            y1 = (grid_q1[:-1,1:].ravel())[ind]
        elif diag_id =='br':
            v1 = field[1:,:-1].ravel()
            ind = np.where(v1 * v2 < 0)        
            v1 = field[1:,:-1].ravel()[ind]
            x1 = grid_q0[1:,:-1].ravel()[ind]
            y1 = (grid_q1[1:,:-1].ravel())[ind]
        elif diag_id =='tr':
            v1 = field[1:,1:].ravel()
            ind = np.where(v1 * v2 < 0)
            v1 = field[1:,1:].ravel()[ind]     
            x1 = grid_q0[1:,1:].ravel()[ind]    
            y1 = (grid_q1[1:,1:].ravel())[ind]
        else:
            print ('wrong id_diag')
        v2= vol_c.ravel()[ind]            
        x2 = grid_q0_c.ravel()[ind];         y2 = grid_q1_c.ravel()[ind]
        q0_zero_diag = (x1*v2- x2*v1)/(v2-v1); q1_zero_diag = (y1*v2- y2*v1)/(v2-v1)            
        return q0_zero_diag, q1_zero_diag

    def zero_line_segments(self, q0_zeros_1, q1_zeros_1, q0_zeros_2, q1_zeros_2, cor_ind = None):
        """
        Parameters: q0_zeros_1 , q1_zeros_1 , q0_zeros_2 , q1_zeros_2, cor_ind (when needed)
        Returns:  coords. of segment ends: Q1 -one end , Q2 -the other end
        """
        ind_q0_zeros_1 = np.int32(q0_zeros_1)
        ind_q1_zeros_1 = np.int32(q1_zeros_1)
        if cor_ind == 'T':   #Top segment inds change from [1:Ng1] to [0:Ng1-1]
            ind_q1_zeros_1 -=1
        ind_q0_zeros_2 = np.int32(q0_zeros_2) 
        if cor_ind == 'R':  #Right segment inds change from [1:Ng0] to [0:Ng0-1]        
            ind_q0_zeros_1 -=1
        ind_q1_zeros_2 = np.int32(q1_zeros_2)
        sq_ind_1 = (self.UP.Np-1)*ind_q0_zeros_1 + ind_q1_zeros_1
        sq_ind_2 = (self.UP.Np-1)*ind_q0_zeros_2 + ind_q1_zeros_2
        common_val_ind=np.intersect1d(sq_ind_1,sq_ind_2)
        n_zer = len(common_val_ind)
        
        Q1 = np.zeros((n_zer,2), dtype=np.float32)
        Q2 = np.zeros((n_zer,2), dtype=np.float32)
        if n_zer > 0:      
            for i_zer in range(n_zer):
                # print ('i_zer=', i_zer, 'common_val_ind[i_zer]=', common_val_ind[i_zer])
                Q1[i_zer,0]=q0_zeros_1[np.where(sq_ind_1 == common_val_ind[i_zer])]
                Q1[i_zer,1]=q0_zeros_2[np.where(sq_ind_2 == common_val_ind[i_zer])]
                Q2[i_zer,0]=q1_zeros_1[np.where(sq_ind_1 == common_val_ind[i_zer])]
                Q2[i_zer,1]=q1_zeros_2[np.where(sq_ind_2 == common_val_ind[i_zer])]
            #print('exit 1 n_zer=', n_zer)    
            # return Q1, Q2
        # else:
            # print('zero_line_segments: exit 2 n_zer=', n_zer) 
        return   Q1,Q2
    
    def dist(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dis = ((x1-x2)**2 + (y1-y2)**2)**0.5
        return dis
    
    def offices_to_merge(self, points):
        current_minimum = float('inf')
        min_p1 = -1
        min_p2 = -1
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dis = self.dist(points[i], points[j])
                if dis < current_minimum:
                   min_p1 = i
                   min_p2 = j
                   current_minimum = dis
        return (min_p1, min_p2), current_minimum

    
    