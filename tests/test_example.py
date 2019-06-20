#!/usr/bin/env python

# -*- coding: UTF-8 -*-
from DLITE.cell_describe import node, edge, cell, colony
from DLITE.ManualTracingMultiple import ManualTracingMultiple
from DLITE.SurfaceEvolver import SurfaceEvolver
import os
import numpy as np

os.chdir(r'./tests/data')

timepoints = [int(i)/10 for i in np.linspace(20, 1, 20)]
name_first = 'voronoi_very_small_tension_edges_20_30_'
name_end = '.fe.txt'
SurfaceEvolverInstance = SurfaceEvolver(name_first, name_end)
colonies = SurfaceEvolverInstance.computation_based_on_prev_surface_evolver(timepoints, colonies = None, index = None, 
                          old_dictionary = None, maxiter = 60*1000, solver = 'CellFIT')

class TestColony:
	def test_no_NaN(self):
		"""
		No NaN's in tension and pressure solution
		"""

		for i in range(len(colonies)):
			tt = [e.tension for e in colonies[str(i)].tot_edges]
			assert ~np.isnan(tt).any()
			cc = [c.pressure for c in colonies[str(i)].cells]
			assert ~np.isnan(cc).any()

	def test_normalization(self):
		""" 
		Average tension is 1, 
		average pressure is 0
		"""

		for i in range(len(colonies)):
			assert abs(np.mean([e.tension for e in colonies[str(i)].tot_edges]) - 1) < 0.00001
			assert abs(np.mean([c.pressure for c in colonies[str(i)].cells])) < 0.00001

	def test_connectivity(self):
		"""
		Check that no node is connected to 2 edges 
		(Impossible to do a force balance)
		"""

		for i in range(len(colonies)):
			assert [len(n.edges) for n in colonies[str(i)].tot_nodes][0] != 2

