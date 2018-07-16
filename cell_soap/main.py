from cell_describe import cell
def main(edge, angles1, con_edges0, cells, type, num):
		    # to form a cycle, need the largest negative from one node and smallest positive angle from other node and vice versa
#     try:
    # max negative number from node 0
    # type -- 0 for choosing only the minimum positive angles
         # -- 1 for choosing only the maximum negative angles
    # num -- 0 for choosing edge.node_a, 1 for choosing edge.node_b (depends on which node you start from)

    # initialize the final node that we want to match
	final_node = [edge.node_a, edge.node_b]
    
	if type == 1:
		angle_node0 = max([n for n in angles1 if n<0], default = 9000)
	else:
    # min positive number from node 0
		angle_node0 = min([n for n in angles1 if n>0], default = 9000)

    
	if angle_node0 != 9000:
        # find edge corresponding to the angle 
		edge1 = [e for e in con_edges0 if edge.edge_angle(e) == angle_node0]
        
		if set(edge1[0].nodes) == set(edge.nodes):
            # two edge cell
			cells.append(cell(list(edge.nodes),[edge, edge1]))
		else:

            # find the index of the non common node (with original edge) on this edge
			common_node = edge1[0].nodes.intersection(edge.nodes).pop()
			other_node = edge1[0].nodes.difference([common_node]).pop()
			# = list(edge1[0].nodes).index(other_node) # has error because set reorders things
			if edge1[0].node_a == other_node:
				i = 0
			else:
				i = 1


            # find the edges connected to this edge and find all the angles that these connected edges make with this edge
			edge1_con = edge1[0].connected_edges[i]
			angles_edge1 = [edge1[0].edge_angle(e2) for e2 in edge1_con]

			# find maximum negative or minimum positive angle depending on the type 
			if type == 1:
				max_angles_edge1 = max([n for n in angles_edge1 if n<0], default = 9000)
			else:
				max_angles_edge1 = min([n for n in angles_edge1 if n>0], default = 9000)

			# check that angle doesnt exist

			if max_angles_edge1 != 9000:

				# find the edge corresponding to the angle
				edge2 = [e for e in edge1_con if edge1[0].edge_angle(e) == max_angles_edge1]

				# find index of non common node
				common_node2 = other_node
				other_node2 = edge2[0].nodes.difference([common_node2]).pop()
				#j = list(edge2[0].nodes).index(other_node2)
				if edge2[0].node_a == other_node2:
					j = 0
				else:
					j = 1
                
                # check that non common node matches the other node on the original edge
				if other_node2 == final_node[num]:
					# found a 3 edge cell
					cell_nodes = [edge.node_a, common_node2, edge.node_b]
					cell_edges = [edge, edge1[0], edge2[0]]

					# check for any repeated cells
					# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_three]):
					#     pass
					# else:
					#     cell_three.append(cell(cell_nodes, cell_edges))
					check = 0
					for c in cells:
						if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
							check = 1
					if check == 0:
						cells.append(cell(cell_nodes, cell_edges))

				else:
					# didnt find a 3 edge cell. find the connected edges to this edge
					edge2_con = edge2[0].connected_edges[j]

					# find angles that all these connected edges make with this edge
					angles_edge2 = [edge2[0].edge_angle(e2) for e2 in edge2_con]

					# find max or min depending on type
					if type == 1:
						max_angles_edge2 = max([n for n in angles_edge2 if n<0], default = 9000)
					else:
						max_angles_edge2 = min([n for n in angles_edge2 if n>0], default = 9000)

					if max_angles_edge2 != 9000:
						# if angle exists, find that edge
						edge3 = [e for e in edge2_con if edge2[0].edge_angle(e) == max_angles_edge2]

					    # find index of non common node
						common_node3 = other_node2
						other_node3 = edge3[0].nodes.difference([common_node3]).pop()
						#k = list(edge3[0].nodes).index(other_node3)
						if edge3[0].node_a == other_node3:
							k = 0
						else:
							k = 1

					    # check if non common node matches the other node in original edge
					    
						if other_node3 == final_node[num]:
					        # found a 4 corner cell
							cell_nodes = [edge.node_a, common_node2, common_node3, edge.node_b ]
							cell_edges = [edge, edge1[0], edge2[0], edge3[0]]
					        
					        # check for repeats
							# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_four]):
							# 	pass
							# else:
							# 	cell_four.append(cell(cell_nodes, cell_edges))
							check = 0
							for c in cells:
								if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
									check = 1
							if check == 0:
								cells.append(cell(cell_nodes, cell_edges))
						else:
					        # find the connected edges to this edge and angles
							edge3_con = edge3[0].connected_edges[k]
							angles_edge3 = [edge3[0].edge_angle(e2) for e2 in edge3_con]
					        
							if type == 1:
								max_angles_edge3 = max([n for n in angles_edge3 if n<0], default = 9000)
							else:
								max_angles_edge3 = min([n for n in angles_edge3 if n>0], default = 9000)
					        
							if max_angles_edge3 != 9000:

					        	# find edge corresponding to this angle
								edge4 = [e for e in edge3_con if edge3[0].edge_angle(e) == max_angles_edge3]

					            # find index of non common node
								common_node4 = other_node3
								other_node4 = edge4[0].nodes.difference([common_node4]).pop()
								#m = list(edge4[0].nodes).index(other_node4)
								if edge4[0].node_a == other_node4:
									m = 0
								else:
									m = 1

					            # check that non common node matches other node on original edge
									 
								if other_node4 == final_node[num]:
									# found a 5 corner cell
									cell_nodes = [edge.node_a, common_node2, common_node3, common_node4,  edge.node_b]
									cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0]]
									# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_five]):
									# 	pass
									# else:
									# 	cell_five.append(cell(cell_nodes, cell_edges))
									check = 0
									for c in cells:
										if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
											check = 1
									if check == 0:
										cells.append(cell(cell_nodes, cell_edges))

								else:
									# find connected edges to this edge
									edge4_con = edge4[0].connected_edges[m]
									angles_edge4 = [edge4[0].edge_angle(e2) for e2 in edge4_con]

									if type == 1:
										max_angles_edge4 = max([n for n in angles_edge4 if n<0], default = 9000)
									else:
										max_angles_edge4 = min([n for n in angles_edge4 if n>0], default = 9000)

									if max_angles_edge4 != 9000:
										# find that edge
										edge5 = [e for e in edge4_con if edge4[0].edge_angle(e) == max_angles_edge4]

										# find index of non common node
										common_node5 = other_node4
										other_node5 = edge5[0].nodes.difference([common_node5]).pop()
										#n = list(edge5[0].nodes).index(other_node5)
										if edge5[0].node_a == other_node5:
											n = 0
										else:
											n = 1

										if other_node5 == final_node[num]:
											cell_nodes = [edge.node_a, common_node2, common_node3, common_node4, common_node5, edge.node_b]
											cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0], edge5[0]]

											# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_six]):
											# 	pass
											# else:
											# 	cell_six.append(cell(cell_nodes, cell_edges))
											check = 0
											for c in cells:
												if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
													check = 1
											if check == 0:
												cells.append(cell(cell_nodes, cell_edges))
										else:
											# check for seven edge cell
											edge5_con = edge5[0].connected_edges[n]
											angles_edge5 = [edge5[0].edge_angle(e2) for e2 in edge5_con]

											if type == 1:
												max_angles_edge5 = max([n for n in angles_edge5 if n<0], default = 9000)
											else:
												max_angles_edge5 = min([n for n in angles_edge5 if n>0], default = 9000)

											if max_angles_edge5 != 9000:
												edge6 = [e for e in edge5_con if edge5[0].edge_angle(e) == max_angles_edge5]

												# find index of non common node
												common_node6 = other_node5
												other_node6 = edge6[0].nodes.difference([common_node6]).pop()
												if edge6[0].node_a == other_node6:
													n = 0
												else:
													n = 1

												if other_node6 == final_node[num]:
													cell_nodes = [edge.node_a, common_node2, common_node3, common_node4, common_node5, common_node6, edge.node_b]
													cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0], edge5[0], edge6[0]]

													# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_six]):
													# 	pass
													# else:
													# 	cell_six.append(cell(cell_nodes, cell_edges))
													check = 0
													for c in cells:
														if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
															check = 1
													if check == 0:
														cells.append(cell(cell_nodes, cell_edges))
												else:
													# check for eight edge cell
													edge6_con = edge6[0].connected_edges[n]
													angles_edge6 = [edge6[0].edge_angle(e2) for e2 in edge6_con]

													if type == 1:
														max_angles_edge6 = max([n for n in angles_edge6 if n<0], default = 9000)
													else:
														max_angles_edge6 = min([n for n in angles_edge6 if n>0], default = 9000)

													if max_angles_edge6 != 9000:
														edge7 = [e for e in edge6_con if edge6[0].edge_angle(e) == max_angles_edge6]

														# find index of non common node
														common_node7 = other_node6
														other_node7 = edge7[0].nodes.difference([common_node7]).pop()
														if edge7[0].node_a == other_node7:
															n = 0
														else:
															n = 1

														if other_node7 == final_node[num]:
															cell_nodes = [edge.node_a, common_node2, common_node3, common_node4, common_node5, common_node6, common_node7,edge.node_b]
															cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0], edge5[0], edge6[0], edge7[0]]

															# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_six]):
															# 	pass
															# else:
															# 	cell_six.append(cell(cell_nodes, cell_edges))
															check = 0
															for c in cells:
																if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
																	check = 1
															if check == 0:
																cells.append(cell(cell_nodes, cell_edges))
														else:
															# check for 9 edge cells
															edge7_con = edge7[0].connected_edges[n]
															angles_edge7 = [edge7[0].edge_angle(e2) for e2 in edge7_con]

															if type == 1:
																max_angles_edge7 = max([n for n in angles_edge7 if n<0], default = 9000)
															else:
																max_angles_edge7 = min([n for n in angles_edge7 if n>0], default = 9000)

															if max_angles_edge7 != 9000:
																edge8 = [e for e in edge7_con if edge7[0].edge_angle(e) == max_angles_edge7]

																# find index of non common node
																common_node8 = other_node7
																other_node8 = edge8[0].nodes.difference([common_node8]).pop()
																if edge8[0].node_a == other_node8:
																	n = 0
																else:
																	n = 1

																if other_node8 == final_node[num]:
																	cell_nodes = [edge.node_a, common_node2, common_node3, common_node4, common_node5, common_node6, common_node7,common_node8, edge.node_b]
																	cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0], edge5[0], edge6[0], edge7[0], edge8[0]]

																	# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_six]):
																	# 	pass
																	# else:
																	# 	cell_six.append(cell(cell_nodes, cell_edges))
																	check = 0
																	for c in cells:
																		if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
																			check = 1
																	if check == 0:
																		cells.append(cell(cell_nodes, cell_edges))

																else:
																	# check for 10 edge cells

																	edge8_con = edge8[0].connected_edges[n]
																	angles_edge8 = [edge8[0].edge_angle(e2) for e2 in edge8_con]
																	if type == 1:
																		max_angles_edge8 = max([n for n in angles_edge8 if n<0], default = 9000)
																	else:
																		max_angles_edge8 = min([n for n in angles_edge8 if n>0], default = 9000)

																	if max_angles_edge8 != 9000:
																		edge9 = [e for e in edge8_con if edge8[0].edge_angle(e) == max_angles_edge8]

																		# find index of non common node
																		common_node9 = other_node8
																		other_node9 = edge9[0].nodes.difference([common_node9]).pop()
																		if edge9[0].node_a == other_node9:
																			n = 0
																		else:
																			n = 1

																		if other_node9 == final_node[num]:
																			cell_nodes = [edge.node_a, common_node2, common_node3, common_node4, common_node5, common_node6, common_node7,common_node8, common_node9, edge.node_b]
																			cell_edges = [edge, edge1[0], edge2[0], edge3[0], edge4[0], edge5[0], edge6[0], edge7[0], edge8[0], edge9[0]]

																			# if any([set(cell(cell_nodes, cell_edges).edges).intersection(set(c.edges)) for c in cell_six]):
																			# 	pass
																			# else:
																			# 	cell_six.append(cell(cell_nodes, cell_edges))
																			check = 0
																			for c in cells:
																				if set(cell(cell_nodes, cell_edges).edges) == set(c.edges):
																					check = 1
																			if check == 0:
																				cells.append(cell(cell_nodes, cell_edges))

																		else:
																			# look for 11 edge cells
																			pass







	return cells
		 