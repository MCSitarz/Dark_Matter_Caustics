#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:49:39 2023

@author: michaelsitarz
"""

class Topology_Based_Methods:
    def __init__(self, UP):
        self.UP = UP
#==============================================================================     
    def define_Jacobian_Connection_Minkowski_Functional(self, Connected_Curves, Piecewise_Curves, a, Tag): 
        '''Calculate Minkowski Functionals'''
        '''Single Curve[0] == Single Curve[-1]'''
        
        '''Minkowski Functional: Genus'''
        Euler_Numbers = []
        External_Curves = []
        Internal_Curves = []
        File_Euler_Numbers = []
        for i in range(len(Connected_Curves)):
            Suspect_1 = Connected_Curves[i]
            Euler_Number_Curve = 0
            Encompassed_Curves_for_1 = []
            for j in range(len(Connected_Curves)):
                if np.array_equal(Connected_Curves[i], Connected_Curves[j]) == True:
                    continue 
                Suspect_2 = Connected_Curves[j]
                        
                mpl_Curve_Codes_1 = np.empty(len(Suspect_1), dtype='object')
                mpl_Curve_Codes_1[:] = Path.LINETO
                mpl_Curve_Codes_1[0] = Path.MOVETO
                mpl_Curve_Codes_1[-1] = Path.CLOSEPOLY
                mpl_Curve_Codes_2 = np.empty(len(Suspect_2), dtype='object')
                mpl_Curve_Codes_2[:] = Path.LINETO
                mpl_Curve_Codes_2[0] = Path.MOVETO
                mpl_Curve_Codes_2[-1] = Path.CLOSEPOLY
                
                mpl_Curve_1 = Path(Suspect_1, mpl_Curve_Codes_1)
                
                Suspect_1_Booleans = []
                
                for point in Suspect_2:
                    Suspect_1_Booleans.append(mpl_Curve_1.contains_point(point))
                
                if np.all(Suspect_1_Booleans) == True:
                    Euler_Number_Curve += 1
                    Encompassed_Curves_for_1.append(Suspect_2)
                
            Euler_Numbers.append(Euler_Number_Curve)
            External_Curves.append(Suspect_1)
            Internal_Curves.append(Encompassed_Curves_for_1)
            File_Euler_Numbers.append([len(Suspect_1) - 1, Euler_Number_Curve])
            
        '''Euler Characteristic'''
        '''chi = V - E + F'''
        File_Euler_Characteristics = []
        for n in range(len(Euler_Numbers)):
            if Euler_Numbers[n] == 0:
                Ordered_Vertices = np.delete(External_Curves[n], -1, 0)
                External_Vertices = len(Ordered_Vertices)
                External_Edges = len(Ordered_Vertices)
                
                Total_Vertices = External_Vertices 
                Faces = 1
                Total_Edges = External_Edges 
                Euler_Characteristic = Total_Vertices - Total_Edges + Faces
                
            if Euler_Numbers[n] > 0:
                Ordered_Vertices = np.delete(External_Curves[n], -1, 0)
                External_Vertices = len(Ordered_Vertices)
                External_Edges = len(Ordered_Vertices)
                Internal_Vertices = 0
                Internal_Edges = 0
                for i in range(len(Internal_Curves[n])):
                    Ordered_Vertices = np.delete(Internal_Curves[n][i], -1, 0)
                    Internal_Edges += len(Ordered_Vertices)
                    Internal_Vertices += len(Ordered_Vertices)
                Total_Vertices = External_Vertices + Internal_Vertices
                Faces = 1
                Total_Edges = External_Edges + Internal_Edges
                Euler_Characteristic = Total_Vertices - Total_Edges + Faces
            File_Euler_Characteristics.append([len(External_Curves[n]) - 1, Total_Vertices, Total_Edges, Faces, Euler_Characteristic])
        
        '''Minkowski Functional: Perimeter'''
        File_Perimeter_Data = []
        for i in range(len(Connected_Curves)):
            Perimeter = 0
            for j in range(len(Connected_Curves[i])):
                Perimeter += np.sqrt(np.square(Connected_Curves[i][0][0] - Connected_Curves[i][1][0]) + np.square(Connected_Curves[i][0][1] - Connected_Curves[i][1][1]))
            File_Perimeter_Data.append([Perimeter])

            
        '''Minkowski Functional: Area - Shoelace Formula'''  
        Shoelace_Area_Data = []
        for i in range(len(External_Curves)):
            if len(Internal_Curves[i]) == 0:
                Ordered_Vertices = np.delete(External_Curves[i], -1, 0)
                X_Vertices = Ordered_Vertices[:,0]
                Y_Vertices = Ordered_Vertices[:,1]
                Total_Area = 0.5 * np.abs(np.dot(X_Vertices, np.roll(Y_Vertices,1)) - np.dot(Y_Vertices, np.roll(X_Vertices,1)))
            if len(Internal_Curves[i]) != 0:
                Ordered_Vertices = np.delete(External_Curves[i], -1, 0)
                X_Vertices = Ordered_Vertices[:,0]
                Y_Vertices = Ordered_Vertices[:,1]
                External_Area = 0.5 * np.abs(np.dot(X_Vertices, np.roll(Y_Vertices,1)) - np.dot(Y_Vertices, np.roll(X_Vertices,1)))
                Internal_Area = 0
                for j in range(len(Internal_Curves[i])):
                    Ordered_Vertices = np.delete(Internal_Curves[i][j], -1, 0)
                    X_Vertices = Ordered_Vertices[:,0]
                    Y_Vertices = Ordered_Vertices[:,1]
                    Internal_Area += 0.5 * np.abs(np.dot(X_Vertices, np.roll(Y_Vertices,1)) - np.dot(Y_Vertices, np.roll(X_Vertices,1)))
                Total_Area = External_Area - Internal_Area
            Shoelace_Area_Data.append([Total_Area])
    
        '''Curvature of Closed Irregular Polygon'''
        '''Starting Point of Curve == Ending Point of Curve'''
        File_Curvature_Data = []
        print('Number of Connected Curves:', len(Connected_Curves))
        for IrrPoly in Connected_Curves:
            Curve_Curvature = []
            Irregular_Ploygon = np.delete(IrrPoly, -1, 0)
            for i in range(len(Irregular_Ploygon)):
                k_m_1 = Irregular_Ploygon[i - 1]
                k = Irregular_Ploygon[i]
                if i == len(Irregular_Ploygon) - 1:
                    k_p_1 = Irregular_Ploygon[0]
                else:
                    k_p_1 = Irregular_Ploygon[i + 1]
                P_k_m_1_k = k_m_1 - k
                P_k_k_p_1 = k - k_p_1
                Vector_Dot = (P_k_m_1_k[0] * P_k_k_p_1[0]) + (P_k_m_1_k[1] * P_k_k_p_1[1])
                l_k_m_1 = np.sqrt(np.square(k_m_1[0] - k[0]) + np.square(k_m_1[1] - k[1]))
                l_k = np.sqrt(np.square(k[0] - k_p_1[0]) + np.square(k[0] - k_p_1[1]))
                alph_k_pi = np.arccos((Vector_Dot)/(l_k * l_k_m_1))
                rho_k = (l_k_m_1 + l_k) / (2 * alph_k_pi)
                Curve_Curvature.append(rho_k)
            File_Curvature_Data.append(Curve_Curvature)
    
        Minkowski_Snapshot = os.path.join(self.UP.Minkowski_Snapshots, Tag + '_Minkowski_Snapshot_' + str(round(a, 2)) + '.txt')
    
        if len(Piecewise_Curves) == 0:
            Piecewise_Curves = []
            for i in range(len(Connected_Curves)):
                Piecewise_Curves.append([0])
        else:
            temp = Piecewise_Curves
            Piecewise_Curves = []
            for i in range(len(temp)):
                curve_i = []
                for j in range(len(temp[i])):
                    point_j = []
                    for k in range(len(temp[i][j])):
                        point_j.append(str(temp[i][j][k]))
                    curve_i.append(point_j)
                Piecewise_Curves.append(curve_i)
        
        temp = Connected_Curves
        Connected_Curves = []
        for i in range(len(temp)):
            curve_i = []
            for j in range(len(temp[i])):
                point_j = []
                for k in range(len(temp[i][j])):
                    point_j.append(str(temp[i][j][k]))
                curve_i.append(point_j)
            Connected_Curves.append(curve_i)
        
        data = {        
                "Connected_Curves" : Connected_Curves,
                "Piecewise_Wise" : Piecewise_Curves,
                "Curve_Length" : list([i[0] for i in File_Euler_Numbers]),
                "Euler_Number" : list([i[1] for i in File_Euler_Numbers]),
                "Vertex_Number": list([i[1] for i in File_Euler_Characteristics]),
                "Edge_Number"  : list([i[2] for i in File_Euler_Characteristics]),
                "Face_Number"  : list([i[3] for i in File_Euler_Characteristics]),
                "Euler_Characteristic" : list([i[4] for i in File_Euler_Characteristics]),
                "Perimeter_Estimate" : list([i[0] for i in File_Perimeter_Data]),
                "Shoelace_Area" : list([i[0] for i in Shoelace_Area_Data]),
                "Curvature_Per_Vertex": list([i[0] for i in File_Curvature_Data])
                }
        
        with open(Minkowski_Snapshot, 'w') as f:
                    json.dump(data, f)