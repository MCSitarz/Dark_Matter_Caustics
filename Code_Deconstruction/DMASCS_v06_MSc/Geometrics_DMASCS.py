#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:43:50 2023
@author: michaelsitarz
"""
import os
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter

class Geometric_Caustic_Based_Methods:
    def __init__(self, UP, B, PCH):
        self.UP = UP
        self.B = B
        self.PCH = PCH
            
    def Tessellation_Studies(self, Particles, a):
        self.Triangles = self.B.Triangles
        
        if a == 0.0:
            self.Vecs = [] # Array for the line-edges of the tessellation.
            self.Tris = [] # Array for pairs of neighboring triangles.
            self.previousSign = []
            self.currentSign = []
            self.Triangle_Geometry(Particles)
            
            self.prevParticles = Particles
            self.causticCollapser = []
            self.collapseTheta = []
            self.causticTessIdx = []

        self.Eulerian_Geometery(Particles)
        self.Caustic_Formation(Particles)
        self.flipflopCounts(a, Particles)
        
        self.collapseAngle(a, Particles)
        
        self.prev_xAxisVelocity = (Particles[:,2] - self.prevParticles[:,2]) / self.UP.da
        self.prev_yAxisVelocity = (Particles[:,3] - self.prevParticles[:,3]) / self.UP.da
        self.prevParticles = Particles
        return self.triangleMorph, self.causticVector, self.causticParticle, self.totalFlips, self.currentSign, self.previousSign, self.uniqueFlips, np.array(self.collapseTheta), self.eulerianVolumes
            
    def Triangle_Geometry(self, Particles):
        for i in range(len(self.Triangles)):
            L = self.Triangles[i][0]
            T = self.Triangles[i][1]
            R = self.Triangles[i][2]
            for j in [i + 1, i + 2, i + 3, i + (self.UP.Np * 2)]:
                if j <= len(self.Triangles) - 1:
                    nL = self.Triangles[j][0]
                    nT = self.Triangles[j][1]
                    nR = self.Triangles[j][2]
                    Combo = [L, T, R, nL, nT, nR]
                    Same = [item for item, count in Counter(Combo).items() if count > 1]
                    if len(Same) == 2:
                        self.Vecs.append(Same)
                        self.Tris.append([i,j])
        self.totalFlips = np.zeros(len(self.Triangles))
    
    def Eulerian_Geometery(self, Particles):
        self.eulerianVolumes = []
        for tri in self.Triangles:
            self.eulerianVolumes.append(
                Particles[tri[0]][2]*Particles[tri[1]][3] + Particles[tri[1]][2]*Particles[tri[2]][3] + Particles[tri[2]][2]*Particles[tri[0]][3] - \
                Particles[tri[0]][2]*Particles[tri[2]][3] - Particles[tri[1]][2]*Particles[tri[0]][3] - Particles[tri[2]][2]*Particles[tri[1]][3])
        self.eulerianVolumes = list(np.array(self.eulerianVolumes) * -1)

    def Caustic_Formation(self, Particles):
        self.triangleMorph = np.zeros_like(self.Triangles)
        self.causticVector = []
        self.causticParticle = []
        for T in range(len(self.Tris)):
            if (self.eulerianVolumes[self.Tris[T][0]] > 0) != (self.eulerianVolumes[self.Tris[T][1]] > 0):
                if Particles[self.Vecs[T][1]][2] < 90 and Particles[self.Vecs[T][1]][2] > 10 and \
                   Particles[self.Vecs[T][0]][2] < 90 and Particles[self.Vecs[T][0]][2] > 10 and \
                   Particles[self.Vecs[T][1]][3] < 90 and Particles[self.Vecs[T][1]][3] > 10 and \
                   Particles[self.Vecs[T][0]][3] < 90 and Particles[self.Vecs[T][0]][3] > 10:
                    if np.sqrt(np.power((Particles[self.Vecs[T][1]][2] - Particles[self.Vecs[T][0]][2]),2) + \
                       np.power((Particles[self.Vecs[T][1]][3] - Particles[self.Vecs[T][0]][3]),2)) < self.UP.L/2:
                        
                        if self.eulerianVolumes[self.Tris[T][0]] < 0:
                            Indx = self.Tris[T][0]
                        if self.eulerianVolumes[self.Tris[T][1]] < 0:
                            Indx =self. Tris[T][1]
                        
                        crosser = set(self.Triangles[Indx]) ^ set(self.Vecs[T])
                        self.triangleMorph[Indx] += 1
                        self.causticVector.append(self.Vecs[T])
                        self.causticParticle.append(crosser.pop())
                        self.causticTessIdx.append(Indx)

    def flipflopCounts(self, a, Particles):
            if a == 0.0:
                self.currentSign = np.sign(self.eulerianVolumes)
                self.previousSign = np.sign(self.eulerianVolumes)
            if a != 0.0:
                self.previousSign = self.currentSign
                self.currentSign = np.sign(self.eulerianVolumes)
                for i in range(len(self.currentSign)):
                    if np.sign(self.currentSign[i]) != np.sign(self.previousSign[i]):
                        self.totalFlips[i] += 1
            self.uniqueFlips = np.unique(self.totalFlips)

    def collapseAngle(self, a, Particles):
        for c in range(len(self.causticParticle)):
            if self.causticParticle[c] not in self.causticCollapser:
                QX1 = self.prevParticles[self.causticVector[c][0]][2]
                QY1 = self.prevParticles[self.causticVector[c][0]][3]
                VX1 = self.prev_xAxisVelocity[self.causticVector[c][0]]
                VY1 = self.prev_yAxisVelocity[self.causticVector[c][0]]
                
                QX2 = self.prevParticles[self.causticParticle[c]][2]
                QY2 = self.prevParticles[self.causticParticle[c]][3]
                VX2 = self.prev_xAxisVelocity[self.causticParticle[c]]
                VY2 = self.prev_yAxisVelocity[self.causticParticle[c]]
                
                QX3 = self.prevParticles[self.causticVector[c][1]][2]
                QY3 = self.prevParticles[self.causticVector[c][1]][3]
                VX3 = self.prev_xAxisVelocity[self.causticVector[c][1]]
                VY3 = self.prev_yAxisVelocity[self.causticVector[c][1]]
                
                P1xtC, P1ytC, P2xtC, P2ytC, P3xtC, P3ytC, R12 = self.collapseTime(QX1, QY1, VX1, VY1, QX2, QY2, VX2, VY2, QX3, QY3, VX3, VY3, a)

                self.collapseTheta.append([int(self.causticTessIdx[c]), int(self.causticParticle[c]), int(self.causticVector[c][0]), int(self.causticVector[c][1]), self.ang([[P2xtC, P2ytC], [QX2, QY2]], [[P1xtC, P1ytC], [P3xtC, P3ytC]]), P1xtC, P1ytC, P2xtC, P2ytC, P3xtC, P3ytC, a])

    def collapseTime(self, QX1, QY1, VX1, VY1, QX2, QY2, VX2, VY2, QX3, QY3, VX3, VY3, a):
        t0 = ((QY3 - QY2)*(QX2 - QX1)) - ((QX3 - QX2)*(QY2 - QY1))
        t1 = ((QY3 - QY2)*(VX2 - VX1)) + ((VY3 - VY2)*(QX2 - QX1)) - ((QX3 - QX2)*(VY2 - VY1)) - ((VX3 - VX2)*(QY2 - QY1))
        t2 = ((VY3 - VY2)*(VX2 - VX1)) - ((VX3 - VX2)*(VY2 - VY1))
        R12 = np.roots([t2, t1, t0])
        
        R1 = R12[0]
        R2 = R12[1]

        if R1 > 0 and R2 < 0:
            R = R1.real
        if R1 < 0 and R2 > 0:
            R = R2.real
        if R1 > 0 and R2 > 0:
            R = min(R1, R2)
        if R1 < 0 and R2 < 0:
            if round(R1, 2) == 0:
                R = R1.real
            if round(R2, 2) == 0:
                R = R2.real
            else:
                if round(R1, 1) == 0:
                    R = R1.real
                if round(R2, 1) == 0:
                    R = R2.real
                else:   # For the sake of a full run and showing it is iffy on late stage, don't exit.
                    print('MOO', R12, round(R1,1), round(R2,1))
                    R = min(R1.real, R2.real)
                    
        P1xtC = QX1 + (VX1 * (R))
        P1ytC = QY1 + (VY1 * (R))
        P2xtC = QX2 + (VX2 * (R))
        P2ytC = QY2 + (VY2 * (R))
        P3xtC = QX3 + (VX3 * (R))
        P3ytC = QY3 + (VY3 * (R))
        
        return P1xtC, P1ytC, P2xtC, P2ytC, P3xtC, P3ytC, R12

    def dot(self, vA, vB):
        return vA[0]*vB[0]+vA[1]*vB[1]

    def ang(self, lineA, lineB):
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        dot_prod = self.dot(vA, vB)
        magA = self.dot(vA, vA)**0.5
        magB = self.dot(vB, vB)**0.5
        cos_ = dot_prod.real/magB.real/magA.real
        while cos_ < -1.0 or cos_ > 1.0:
            if cos_ < -1.0:
                cos_ += 1.0
            if cos_ > 1.0:
                cos_ -= 1.0
        angle = math.acos(cos_)
        ang_deg = math.degrees(angle)%360
        if ang_deg-180>=0:
            return 360 - ang_deg
        else:
            return ang_deg
        
