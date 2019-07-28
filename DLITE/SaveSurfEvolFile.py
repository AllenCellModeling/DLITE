import pandas as pd
import numpy as np

class SaveFile:
	def __init__(self, filename, voronoi):
		"""
		Class to save a voronoi tesselation as a .fe file 
		that is readable by Surface Evolver
		Inputs
		----------
		Filename: Name of file to save
		"""
		self.filename = filename
		self.voronoi = voronoi

	def save(self):
		"""
		Save method that saves a given filename and Voronoi
		tessellation in Surface Evolver format
		"""

		# Write file header
		with open(self.filename, 'a') as f:
			f.write('STRING' + '\n')
			f.write('space_dimension' + ' ' + '2' + '\n' + '\n')
			f.write('vertices' + '\n')

    	# Write file vertices
		vertices_data = {'label': [], 'x': [], 'y': []}
		count = 0
		for j, a in enumerate(self.voronoi.vertices):		    
		    if count != 0:
		        with open(self.filename, 'a') as f:
		            xx = []
		            xx.append(str(count))
		            xx.append(str(a[0]*1000))
		            xx.append(str(a[1]*1000))
		            this_row = ' '.join(str(x) for x in xx)
		            f.write(this_row + '\n')   
		        vertices_data['label'].append(count )
		        vertices_data['x'].append(a[0]*1000)
		        vertices_data['y'].append(a[1]*1000)
		    count += 1

		vertices_dataframe = pd.DataFrame(vertices_data)
		vertices_dataframe.set_index(['label'], inplace = True)

		# Write self.filename edges
		with open(self.filename, 'a') as f:
			f.write('\n' + 'edges' + '\n')

		edges_data = {'label': [], 'v1': [], 'v2': []}
		count = 1
		ridge_vertices = []
		for a in self.voronoi.ridge_vertices:
		    if a[0] > 0 and a[1] > 0:
		        with open(self.filename, 'a') as f:
		            #print(count ,a[0] , a[1])
		            xx = []
		            xx.append(str(count))
		            xx.append(str(a[0]))
		            xx.append(str(a[1]))
		            this_row = ' '.join(str(x) for x in xx)
		            #print(this_row)
		            f.write(this_row + '\n')  
		        edges_data['label'].append(count)
		        edges_data['v1'].append(a[0] )
		        edges_data['v2'].append(a[1])
		        ridge_vertices.append(np.array([a[0], a[1]]))
		        count += 1
		edges_dataframe = pd.DataFrame(edges_data)
		edges_dataframe.set_index(['label'], inplace = True)


		# Write file facets
		all_edges_list = []
		for i in self.voronoi.regions:
		    if all([a > 0 for a in i ]) and len(i) != 0:
		        edge_list = []
		        for ff in range(len(i)):
		            try:
		                p1 = i[ff]
		                p2 = i[ff + 1]
		                count = 0
		                for jj, j in enumerate(ridge_vertices):
		                    count += 1
		                    if j[0] == p1 and j[1] == p2:
		                        if len(edge_list) == 0:
		                            edge_list.append(count)
		                        else:
		                            edge_list.append(count)
		                    elif j[1] == p1 and j[0] == p2:
		                        if len(edge_list) == 0:
		                            edge_list.append(-count)
		                        else:
		                            edge_list.append(-count)
		            except:
		                p1 = i[ff]
		                p2 = i[0]
		                count = 0
		                for jj, j in enumerate(ridge_vertices):
		                    count += 1
		                    if j[0] == p1 and j[1] == p2:
		                        if len(edge_list) == 0:
		                            edge_list.append(count)
		                        else:
		                            edge_list.append(count)

		                    elif j[1] == p1 and j[0] == p2:
		                        if len(edge_list) == 0:
		                            edge_list.append(-count)
		                        else:
		                            edge_list.append(-count)
		        all_edges_list.append(edge_list)

		with open(self.filename, 'a') as f:
			f.write('\n' + 'faces' + '\n')

		ars = []
		ares = []
		tots = []
		tot_area = []

		for j, i in enumerate(all_edges_list):
		    with open(self.filename, 'a') as f:
		        f.write(str(j + 1) + ' ')
		    points = []
		    for k, ii in enumerate(i):
		        if ii > 0:
		            v1 = edges_dataframe.at[ii, 'v1']
		            v2 = edges_dataframe.at[ii, 'v2']
		            x1, y1 = vertices_dataframe.at[v1, 'x'], vertices_dataframe.at[v1, 'y']
		            x2, y2 = vertices_dataframe.at[v2, 'x'], vertices_dataframe.at[v2, 'y']
		            cw_or_not = -(x2 - x1) /(y2 + y1)  
		            area  = x1*y2 - x2*y1
		            ars.append(cw_or_not)
		            tots.append(area)
		            with open(self.filename, 'a') as f:
		                f.write(str(ii) + ' ')
		            if k ==0:
		                points.append(v1)
		                points.append(v2)
		            else:
		                if v1 not in points:
		                    points.append(v1)
		                elif v2 not in points:
		                    points.append(v2)
		        else:
		            v2 = edges_dataframe.at[-ii, 'v1']
		            v1 = edges_dataframe.at[-ii, 'v2']
		            x1, y1 = vertices_dataframe.at[v1, 'x'], vertices_dataframe.at[v1, 'y']
		            x2, y2 = vertices_dataframe.at[v2, 'x'], vertices_dataframe.at[v2, 'y']
		            cw_or_not = -(x2 - x1) /(y2 + y1) 
		            area  = x1*y2 - x2*y1
		            ars.append(cw_or_not)
		            tots.append(area)
		            with open(self.filename, 'a') as f:
		                f.write(str(ii) + ' ')
		            if k ==0:
		                points.append(v1)
		                points.append(v2)
		            else:
		                if v1 not in points:
		                    points.append(v1)
		                elif v2 not in points:
		                    points.append(v2)
		    
		    ares.append(np.sum(ars))
		    tot_area.append(np.sum(tots))
		    ars = []
		    tots = []   
		    with open(self.filename, 'a') as f:
		        f.write('\n')

		
		# Write bodies
		with open(self.filename, 'a') as f:
			f.write('\n' + 'bodies' + '\n')

		for j, i in enumerate(range(len(all_edges_list))):
		    with open(self.filename, 'a') as f:
		        if ares[j] < 0:
		            #f.write(str(i+1) + ' ' + str(i + 1) + ' ' + 'VOLUME' + ' ' + str(abs(tot_area[j])) + '\n')
		            f.write(str(i+1) + ' ' + str(i + 1) + ' ' + 'VOLUME' + ' ' + str(500) + '\n')
		        if ares[j] > 0:
		            #f.write(str(i+1) + ' ' + str(-(i + 1)) + ' ' + 'VOLUME' + ' ' + str(abs(tot_area[j]*1)) + '\n')
		            f.write(str(i+1) + ' ' + str(-(i + 1)) + ' ' + 'VOLUME' + ' ' + str(500) + '\n')

		# Write gogo function
		with open(self.filename, 'a') as f:
		    f.write('\n' + 'read' + '\n' + '\n' + 'gogo := { g 2;' + '\n' 
		           + '      o;' + '\n' + 
		            '      g 5;'+ '\n' +
		            '      r;' + '\n'
		            + '      g 20;' + '\n' +
		            '      r;' + '\n' + 
		            '      g 20;' + '\n' + 
		           '      V 3;' + '\n'  + 
		           '      r;' + '\n'  + 
		           '      g 20;' + '\n'  +
		           '      };' + '\n' + '\n' )

	def add_gogo3(self, tension, begin_id, end_id):
		"""
		Function to add gogo3 function to SE txt file
		It sets a tension 'tension' to all edges with labels/ID
		within a specified range between begin_id and end_id
		""" 
		numbers = np.linspace(begin_id, end_id, end_id - begin_id + 1)

		with open(self.filename, 'a') as f:
			f.write('gogo3 := {set edge tension %.2f where id == %d;' % (tension, numbers[0]) + '\n')
		for j, i in enumerate(numbers):
		    if j > 0:
		        with open(self.filename, 'a') as f:
		            #print('      set edge tension %.2f where id == %d; \n' % (tension, i), end = '')
		            f.write('      set edge tension %.2f where id == %d;' % (tension, i) + '\n')

		with open(self.filename, 'a') as f:
		    #print('      gogo; \n', end = '')
		    f.write('      gogo;' + '\n')
		    #print('      dump "test_automated_%.1f.fe.txt"; \n'% num, end = '')
		    f.write('      dump "%s_%.1f.fe.txt";'% (self.filename[:-3] , tension) + '\n')
		    #print('      }')
		    f.write('      }')










