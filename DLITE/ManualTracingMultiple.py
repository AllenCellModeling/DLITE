import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as la
import collections
from scipy import stats
from matplotlib import cm
import os
import matplotlib.patches as mpatches
from collections import defaultdict
import pylab
import pandas as pd
from .cell_describe import colony
from .ManualTracing import ManualTracing


class ManualTracingMultiple:
    def __init__(self, numbers, name_first=None, name_last=None, type=None):
        """
        Class to handle colonies at mutliple time points that have been manually traced out
        using NeuronJ
        Numbers is a list containing start and stop index e.g [2,4]
        of the files labeled as -
        'MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt'
                                                ^ this number changes - 0,1,2,3,..30
        """

        if not type:
            self.name_first = 'MAX_20170123_I01_003-Scene-4-P4-split_T'
            self.name_last = '.ome.txt'
        elif type == 'small_v2':
            self.name_first = '20170123_I01_003.czi - 20170123_I01_00300'
            self.name_last = '.txt'
        elif type == 'small_v3':
            self.name_first = '7_20170123_I01_003.czi - 20170123_I01_00300'
            self.name_last = '.txt'
        elif type == 'small_v4':
            self.name_first = '002_20170123_I01_002.czi - 20170123_I01_00200'
            self.name_last = '.txt'
        elif type == 'large_v1':
            self.name_first = 'AVG_20170124_I07_001-Scene-4-P2-100'
            self.name_last = '.txt'
        else:
            self.name_first = name_first
            self.name_last = name_last

        names = []
        for i in range(numbers[0], numbers[-1], 1):
            names.append(self.name_first + str(i) + self.name_last)
        names.append(self.name_first + str(numbers[-1]) + self.name_last)

        self.names = names

    def get_x_y_data(self, number):
        """
        Retrieve X and Y co-ordinates of a colony at a time point specified by number
        """

        if number < 10 and self.name_first == 'AVG_20170124_I07_001-Scene-4-P2-100':
            file = self.name_first + str(0) + str(number) + self.name_last
        elif number < 10 and self.name_first == '20170123_I01_003.czi - 20170123_I01_00300':
            file = self.name_first + str(0) + str(number) + self.name_last
        elif number < 10 and self.name_first == '7_20170123_I01_003.czi - 20170123_I01_00300':
            file = self.name_first + str(0) + str(number) + self.name_last
        elif number < 10 and self.name_first == '002_20170123_I01_002.czi - 20170123_I01_00200':
            file = self.name_first + str(0) + str(number) + self.name_last
        else:
            file = self.name_first + str(number) + self.name_last

        with open(file, 'r') as f:
            a = [l.split(',') for l in f]

        x, y, X, Y = [], [], [], []

        for num in a:
            if len(num) == 2:
                x.append(int(num[0]))
                y.append(int(num[1].strip('\n')))
            if len(num) == 1:
                X.append(x)
                Y.append(y)
                x = []
                y = []
        X.append(x)
        Y.append(y)
        X.pop(0)
        Y.pop(0)

        return X, Y

    def get_nodes_edges_cells(self, number):
        """
        Get nodes, edges and cells at time point number
        these nodes, edges and cells do not have any labels
        CHECK - cutoff values used, need to go to every frame and check to see if cutoff works or not
        """

        x, y = self.get_x_y_data(number)

        ex = ManualTracing(x, y)

        if self.name_first == 'MAX_20170123_I01_003-Scene-4-P4-split_T':
            cutoff = 14 if 0 <= number <= 3 \
                           or 5 <= number <= 7 \
                           or 10 <= number <= 11 \
                           or number == 13 or number == 17 else \
                16 if 14 <= number <= 15 \
                      or 18 <= number <= 20 or 22 <= number <= 30 else \
                    17 if 8 <= number <= 9 or number == 16 \
                          or number == 16 else \
                        20 if number == 4 else 12
        elif self.name_first == '20170123_I01_003.czi - 20170123_I01_00300':
            cutoff = 20
        elif self.name_first == '7_20170123_I01_003.czi - 20170123_I01_00300':
            cutoff = 15
        elif self.name_first == '002_20170123_I01_002.czi - 20170123_I01_00200':
            cutoff = 15
        else:
            cutoff = 10

        print('File %d used a Cutoff value ------> %d' % (number, cutoff))

        nodes, edges, new = ex.cleanup(cutoff)

        cells = ex.find_cycles(edges)

        return nodes, edges, cells

    def initial_numbering(self, number0):
        """
        Assign random labels to nodes, edges and cells in the colony specified by number0
        Returns labeled nodes, edges and cells.
        Also returns a dictionary defined as
        {node.label: edges connected to node label, vectors of edges connected to node label}
        """

        # Get the list of nodes for name0
        temp_nodes, edges, initial_cells = self.get_nodes_edges_cells(number0)

        def func(p, common_node):
            # This function outputs the absolute angle (0 to 360) that the edge makes with the horizontal
            if p.node_a == common_node:
                this_vec = np.subtract(p.node_b.loc, p.node_a.loc)
            else:
                this_vec = np.subtract(p.node_a.loc, p.node_b.loc)
            return this_vec

        # Create an empty dictionary
        old_dictionary = defaultdict(list)
        for j, node in enumerate(temp_nodes):
            # Give every node a label -> in this case we're arbitrarily givig labels as we loop through
            node.label = j

            sort_edges = node.edges

            this_vec = [func(p, node) for p in sort_edges]

            old_dictionary[node.label].append(sort_edges)
            old_dictionary[node.label].append(this_vec)

        for k, c in enumerate(initial_cells):
            c.label = k

        for p, ed in enumerate(edges):
            ed.label = p

        return temp_nodes, old_dictionary, initial_cells, edges

    def track_timestep(self, old_colony, old_dictionary, number_now):
        """
        We want to output a dictionary that contains a list of edges in number_now that is the
        same (almost) as the edges in old_colony
        -----------
        Parameters
        -----------
        old_colony - colony instance for the previous time step
        old_dictionary - dictionary for the colony instance in old_colony {node.label: edges, vectors}
        number_now - number of current time point
        """

        def func(p, common_node):
            # This function outputs the absolute angle (0 to 360) that the edge makes with the horizontal
            if p.node_a == common_node:
                this_vec = np.subtract(p.node_b.loc, p.node_a.loc)
            else:
                this_vec = np.subtract(p.node_a.loc, p.node_b.loc)
            return this_vec

        def py_ang(v1, v2):
            """ Returns the angle in degrees between vectors 'v1' and 'v2'    """
            cosang = np.dot(v1, v2)
            sinang = la.norm(np.cross(v1, v2))
            return np.rad2deg(np.arctan2(sinang, cosang))

        # Get list of nodes and edges for every time point
        old_nodes = old_colony.tot_nodes
        old_edges = old_colony.tot_edges
        old_cells = old_colony.cells

        # Get list of nodes and edges for names_now that don't have labels
        now_nodes, now_edges, now_cells = self.get_nodes_edges_cells(number_now)

        # Find the node in now_nodes that is closest to a node in old_nodes and give same label
        for j, prev_node in enumerate(old_nodes):
            # Give the same label as the previous node
            closest_new_node = min([node for node in now_nodes],
                                   key=lambda p: np.linalg.norm(np.subtract(prev_node.loc, p.loc)))

            # Check that the edge vectors on this node are similar to the edge vectors on the prev node
            if len(closest_new_node.edges) == 1:
                # Want to check that angles are similar
                if py_ang(closest_new_node.tension_vectors[0], prev_node.tension_vectors[0]) < 15:
                    closest_new_node.label = prev_node.label
            else:
                # If its connected to 3 edges, closest node is fine. only single edge nodes are problematic
                closest_new_node.label = prev_node.label

        upper_limit = max([n.label for n in now_nodes if n.label != []])
        upper_edge_limit = max([e.label for e in old_edges if e.label != []])

        new_dictionary = defaultdict(list)
        total_now_edges = []
        special_labels = []

        for node in now_nodes:
            if node.label:
                if node.label < upper_limit + 1:
                    try:
                        old_edges_node = old_dictionary[node.label][0]
                        old_angles = old_dictionary[node.label][1]
                        temp_edges = []
                        new_vec = [func(p, node) for p in node.edges]
                        if len(old_angles) == len(node.edges):
                            for old_e in old_angles:
                                v1_v2_angs = [py_ang(old_e, nw) for nw in new_vec]
                                min_ang = min(v1_v2_angs)
                                for ed in node.edges:
                                    vec = func(ed, node)
                                    if py_ang(old_e, vec) == min_ang:
                                        closest_edge = ed
                                        temp_edges.append(closest_edge)
                                        if closest_edge not in total_now_edges:
                                            total_now_edges.append(closest_edge)
                        if len([item for item, count in collections.Counter(temp_edges).items() if count > 1]) != 0:
                            temp_edges = []

                        new_vecs = [func(p, node) for p in temp_edges]
                        for k, p in zip(old_edges_node, temp_edges):
                            if not p.label:
                                labels = [e.label for e in total_now_edges]
                                if k.label not in labels:
                                    p.label = k.label
                        new_dictionary[node.label].append(temp_edges)
                        new_dictionary[node.label].append(new_vecs)

                        if len([e.label for e in old_edges_node]) != len([e.label for e in node.edges]):
                            special_labels.append(node.label)
                            # print('Possible change in topology')
                            # print('Node label', node.label, 'old edge labels',
                            #       [e.label for e in old_edges_node], 'New edge labels',
                            #       [e.label for e in node.edges], end='  ')
                    except:
                        pass

        if upper_limit < 1000:
            count = upper_limit + 1
        else:
            count = upper_limit + 1
        if upper_edge_limit < 1000:
            count_edge = upper_edge_limit + 1
        else:
            count_edge = upper_edge_limit + 1

        for node in now_nodes:
            check = 0
            if not node.label and node.label != 0:
                node.label = count
                count += 1
                check = 1
            for e in node.edges:
                if not e.label and e.label != 0:
                    e.label = count_edge
                    count_edge += 1
                    check = 1
            if check == 1:
                temp_edges = node.edges
                new_vecs = [func(p, node) for p in temp_edges]
                if len(new_dictionary[node.label]) == 2:
                    new_dictionary[node.label].pop()
                    new_dictionary[node.label].pop()
                new_dictionary[node.label].append(temp_edges)
                new_dictionary[node.label].append(new_vecs)
            if node.label in special_labels:
                temp_edges = node.edges
                new_vecs = [func(p, node) for p in temp_edges]
                new_dictionary[node.label].pop()
                new_dictionary[node.label].pop()
                new_dictionary[node.label].append(temp_edges)
                new_dictionary[node.label].append(new_vecs)

        set1 = set(old_dictionary)
        set2 = set(new_dictionary)

        combined_dict = defaultdict(list)
        # Find the labels that are common between the 2 lists and return a dictionary of the form
        # {label, old_edges, new_edges}
        for label in set1.intersection(set2):
            if old_dictionary[label] != [] and new_dictionary[label] != []:
                if len(old_dictionary[label][0]) == len(new_dictionary[label][0]):
                    combined_dict[label].append(old_dictionary[label][0])
                    combined_dict[label].append(new_dictionary[label][0])

        # Label cells
        now_cells = self.label_cells(old_cells, now_cells)

        # Define a colony with labeled nodes, edges and cells
        edges2 = [e for e in now_edges if e.radius is not None]
        now_nodes, now_cells, edges2 = self.assign_intial_guesses(now_nodes, now_cells, old_cells, edges2, old_edges)
        new_colony = colony(now_cells, edges2, now_nodes)

        return new_colony, new_dictionary

    def label_cells(self, old_cells, now_cells):
        """
        Now_nodes is the list of nodes at the current time step
        These nodes have labels based on previous time steps
        old_cells - cells from prev time step
        now_cells - cells in current time step
        """
        for j, cc in enumerate(old_cells):
            closest_new_cell = min([c for c in now_cells],
                                   key=lambda p: np.linalg.norm(np.subtract(cc.centroid(), p.centroid())))
            if not closest_new_cell.label:
                if np.linalg.norm(np.subtract(cc.centroid(), closest_new_cell.centroid())) < 100:
                    closest_new_cell.label = cc.label

        max_label = max([c.label for c in now_cells if c.label != []])
        if max_label > 999:
            count = max_label + 1
        else:
            count = 1000

        for j, cc in enumerate(now_cells):
            if not cc.label and cc.label != 0:
                now_cells[j].label = count
                count += 1
        return now_cells

    def assign_intial_guesses(self, now_nodes, now_cells, old_cells, edges2, old_edges):
        """
        Process -
        (1) Call track_timestep - this assigns a generic labeling of nodes and cells to
        number_old (by calling initial_numbering) and also retrieves a dictionary relating
        each labelled node to its connected edge vectors. It then assigns labels
        to nodes, edges and cells in number_now based on this older timestep.
        Summary
        This function gives you labelled nodes and cells for number_now and number_old (edges saved in dictionary - combined_dict)
        (2) Assign 'guess tension/pressure' (for number_now) based on the 'true tension/pressure'
        (for number_old) by matching labels
        """

        for cc in now_cells:
            if cc.label:
                if cc.label in [old.label for old in old_cells]:
                    match_old_cell = [old for old in old_cells if old.label == cc.label][0]
                    cc.guess_pressure = match_old_cell.pressure

        for ed in old_edges:
            label = ed.label
            for new_ed in edges2:
                if new_ed.label == label:
                    if not new_ed.guess_tension:
                        new_ed.guess_tension = ed.tension

        return now_nodes, now_cells, edges2

    def first_computation(self, number_first, solver=None, type=None, **kwargs):
        """
        Main first computation
        Retuns a colony at number_first
        Parameters
        -------------
        number_first - number specifying first time step
        """

        colonies = {}

        nodes, dictionary, cells, edges = self.initial_numbering(number_first)

        edges2 = [e for e in edges if e.radius is not None]

        if type == 'Jiggling':
            name = str(0)
        else:
            name = str(number_first)

        colonies[name] = colony(cells, edges2, nodes)

        tensions, p_t, a_mat = colonies[name].calculate_tension(solver=solver, **kwargs)
        pressures, p_p, b_mat = colonies[name].calculate_pressure(solver=solver, **kwargs)

        colonies[name].tension_matrix = a_mat
        colonies[name].pressure_matrix = b_mat

        return colonies, dictionary

    def main_computation_based_on_prev(self, numbers, colonies=None, index=None, old_dictionary=None,
                                       solver=None, **kwargs):
        """
        Recursive loop that cycles through all the time steps
        Steps -
        (1) Call self.first_computation() - returns first colony with generic labeling
        (2) Call self.track_timestep() - returns new colony that used info in the old colony to assign some initial guesses
        for tensions and pressures. saved in edge.guess_tension and cell.guess_pressure
        (3) Calculate tension and pressure on the new colony
        (4) Call this function again. Keep doing this till we reach the max number
        (5) Return colonies
        """
        if not colonies:
            colonies, old_dictionary = self.first_computation(numbers[0],
                                                              solver, type='Jiggling', **kwargs)
            print('Solver is', solver)
            print('First colony', colonies)
            colonies[str(0)].dictionary = old_dictionary
            index = 0

        colonies[str(index + 1)], new_dictionary = self.track_timestep(colonies[str(index)],
                                                                       old_dictionary, numbers[index + 1])
        colonies[str(index + 1)].dictionary = new_dictionary
        tensions, p_t, a_mat = colonies[str(index + 1)].calculate_tension(solver=solver, **kwargs)
        pressures, p_p, b_mat = colonies[str(index + 1)].calculate_pressure(solver=solver, **kwargs)

        # Save tension and pressure matrix
        colonies[str(index + 1)].tension_matrix = a_mat
        colonies[str(index + 1)].pressure_matrix = b_mat
        print('Next colony number', str(index + 1))

        index = index + 1
        if index < len(numbers) - 1:
            colonies = self.main_computation_based_on_prev(numbers,
                                                           colonies, index, new_dictionary, solver, **kwargs)
        return colonies

    def alternative_computation_based_on_prev(self, numbers, colonies=None, index=None,
                                              old_dictionary=None, solver=None, **kwargs):
        """
        Same as above, except can store colonies in a dictionary numbered as their input time points
        """
        if not colonies:
            colonies, old_dictionary = self.first_computation(numbers[0], solver, **kwargs)
            colonies[str(numbers[0])].dictionary = old_dictionary
            index = 0

        if numbers[index + 1] == numbers[index]:
            colonies[str(index + 1)], new_dictionary = self.track_timestep(colonies[str(numbers[index])],
                                                                           old_dictionary, numbers[index + 1])
            colonies[str(index + 1)].dictionary = new_dictionary
            tensions, p_t, a_mat = colonies[str(index + 1)].calculate_tension(solver=solver, **kwargs)
            pressures, p_p, b_mat = colonies[str(index + 1)].calculate_pressure(solver=solver, **kwargs)

            # Save tension and pressure matrix
            colonies[str(index + 1)].tension_matrix = a_mat
            colonies[str(index + 1)].pressure_matrix = b_mat
            print('Next colony number', str(index + 1))
        else:
            colonies[str(numbers[index + 1])], new_dictionary = self.track_timestep(colonies[str(numbers[index])],
                                                                                    old_dictionary, numbers[index + 1])
            colonies[str(numbers[index + 1])].dictionary = new_dictionary
            tensions, p_t, a_mat = colonies[str(numbers[index + 1])].calculate_tension(solver=solver, **kwargs)
            pressures, p_p, b_mat = colonies[str(numbers[index + 1])].calculate_pressure(solver=solver, **kwargs)

            # Save tension and pressure matrix
            colonies[str(numbers[index + 1])].tension_matrix = a_mat
            colonies[str(numbers[index + 1])].pressure_matrix = b_mat
            print('Next colony number', str(numbers[index + 1]))

        index = index + 1

        if index < len(numbers) - 1:
            colonies = self.alternative_computation_based_on_prev(numbers, colonies, index, new_dictionary, solver,
                                                                  **kwargs)

        return colonies