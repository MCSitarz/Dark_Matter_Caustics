#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:09:01 2024
@author: michaelsitarz
DMASCS_v06: Dark Matter Analytical Simulation for Caustic Studies Version 06
"""

from Box_DMASCS import Box
from Snapshots_DMASCS import Snapshots
from Computations_DMASCS import Computations
from Interpolations_DMASCS import Interpolation
from Topology_DMASCS import Topology_Based_Methods
from Jacobians_DMASCS import Jacobian_Based_Methods
from Plotting_Call_Hub_DMASCS import Plotting_Call_Hub
from Plotting_Functions_DMASCS import Plotting_Functions
from Universal_Parameters_DMASCS import Universal_Parameters
from Geometrics_DMASCS import Geometric_Caustic_Based_Methods

def main():
    UP = Universal_Parameters()
    C = Computations(UP)
    B = Box(UP, C)
    PF = Plotting_Functions(UP, B, C)
    PCH = Plotting_Call_Hub(UP, B, C, PF)
    JBM = Jacobian_Based_Methods(UP, B, C)
    IH = Interpolation(UP, C, JBM, B)
    GCBM = Geometric_Caustic_Based_Methods(UP, B, PCH)
    TOP = Topology_Based_Methods(UP)
    Snapshots(UP, B, C, PCH, IH, JBM, GCBM, TOP)


import time
start_time  = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))#==============================================================================
