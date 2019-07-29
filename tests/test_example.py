#!/usr/bin/env python

# -*- coding: UTF-8 -*-
from DLITE.cell_describe import node, edge, cell, colony
from DLITE.ManualTracingMultiple import ManualTracingMultiple
from DLITE.SurfaceEvolver import SurfaceEvolver
from DLITE.PlottingFunctions import PlottingFunctions
from DLITE.Lloyd_relaxation_class import Atlas
from DLITE.SaveSurfEvolFile import SaveFile
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
	def test_make_Voronoi(self):
		"""
		Check that you can generate and save a Voronoi tessellation in Surface Evolver format
		"""
		P = np.random.random((6000,2))
		gg = Atlas(points = P, dimensions = (6000,2))
		vor = gg.generate_voronoi()
		vor2 = gg.relax_points(times = 180)
		savefile = SaveFile('test.fe', vor)
		savefile.save()

	# def test_ZO_1_data(self):
	# 	timepoints = [int(i) for i in np.linspace(0, 10, 11)]
	# 	name_first = 'MAX_20170123_I01_003-Scene-4-P4-split_T'
	# 	name_last = '.ome.txt'
	# 	ManualTracingMultipleInstance = ManualTracingMultiple(timepoints, name_first = name_first,
 #                                                           name_last = name_last, type=None)
	# 	colonies = ManualTracingMultipleInstance.main_computation_based_on_prev(timepoints, colonies = None, index = None, 
 #                                          old_dictionary = None, solver = 'CellFIT', maxiter = 60*1000)

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

	# def test_make_dataframes(self):
	# 	"""
	# 	Check that no errors while making nodes, edges and cells dataframes
	# 	"""
	# 	PlottingFunctionsInstance = PlottingFunctions()
	# 	common_edge_labels = PlottingFunctionsInstance.get_repeat_edge(colonies)
	# 	common_cell_labels = PlottingFunctionsInstance.get_repeat_cell(colonies)
	# 	# Make the dataframes
	# 	edges_dataframe, cells_dataframe = PlottingFunctionsInstance.seaborn_plot(None, colonies,
 #                                                                                  common_edge_labels,
 #                                                                                  common_cell_labels,
 #                                                                                  ground_truth = True)
	# 	nodes_dataframe = PlottingFunctionsInstance.seaborn_nodes_dataframe(colonies, None)

	# def test_make_video(self):
	# 	"""
	# 	Check that the correct version of ffmpeg is installed so that you can save videos
	# 	"""
	# 	PlottingFunctionsInstance = PlottingFunctions()

	# 	fig, (ax, ax1, ax3) = plt.subplots(3, 1, figsize=(5.5, 15))

	# 	PlottingFunctionsInstance.plot_single_edges(fig, ax, ax1, ax3, colonies, 0, 30)



