#!/usr/bin/env python

# -*- coding: UTF-8 -*-



"""

Test the tract space and tracts. 

"""
from DLITE.cell_describe import cell_describe 
from cell_describe import node, edge, cell, colony
from DLITE.ManualTracingMultiple import ManualTracingMultiple
from DLITE.SurfaceEvolver import SurfaceEvolver


os.chdir(r'./data')
timepoints = [int(i)/10 for i in np.linspace(10, 1, 9)]
name_first = 'voronoi_very_small_tension_edges_20_30_'
name_end = '.fe.txt'

SurfaceEvolverInstance = SurfaceEvolver(name_first, name_end)

# %%prun
colonies = SurfaceEvolverInstance.computation_based_on_prev_surface_evolver(timepoints, colonies = None, index = None, 
                                          old_dictionary = None, maxiter = 60*1000, solver = 'CellFIT')
