import numpy as np
import numpy.linalg as la
import collections
from collections import defaultdict
import pandas as pd
from .cell_describe import colony
from .ManualTracing import ManualTracing


class SurfaceEvolver:
    def __init__(self, name_first, name_end):
        """
        Class for reading files generated by Surface Evolver and computing tensions
        and pressures
        Parameters
        --------------
        name_first, name_end
        Names for txt files should be of the form
        name_first + number + name_end
        Example - '5pb_edges[3]_tension_' + str(number0) + '.fe.txt'
        The number in the middle should be iterable (e.g. 1,2,3...etc.) so that we can loop through
        """
        self.name_first = name_first
        self.name_end = name_end

    def compute(self, name, solver='KKT'):

        nodes, edges, cells = self.get_X_Y_data_only_junction(name)
        edges2 = [e for e in edges if e.radius is not None]
        col1 = colony(cells, edges2, nodes)
        tensions, p_t, a_mat = col1.calculate_tension(solver=solver)

        pressures, p_p, b_mat = col1.calculate_pressure(solver=solver)

        return col1, tensions, pressures

    def get_X_Y_data_only_junction(self, name):
        # Open file
        with open(name, 'r') as f:
            a = [l.split('\t') for l in f]

        vertices = {'id': [], 'x': [], 'y': []}
        edges = {'id': [], 'v1': [], 'v2': [], 'tension': []}
        faces = {'id': [], 'ed_id': []}
        bodies = {'id': [], 'pressure': []}

        for j, num in enumerate(a):
            try:
                if num[0].split()[0] == 'vertices':
                    for vv in a[j + 1:-1]:
                        if len(vv) != 0:
                            v_list = vv[0].split()
                            vertices['id'].append(v_list[0])
                            vertices['x'].append(v_list[1])
                            vertices['y'].append(v_list[2])
                        else:
                            break
                if num[0].split()[0] == 'edges':
                    for vv in a[j + 1:-1]:
                        if len(vv) != 0:
                            v_list = vv[0].split()
                            edges['id'].append(v_list[0])
                            edges['v1'].append(v_list[1])
                            edges['v2'].append(v_list[2])
                            if len(v_list) > 3:
                                if v_list[3] == 'density':
                                    edges['tension'].append(float(v_list[4]))
                                else:
                                    edges['tension'].append(1)
                            else:
                                edges['tension'].append(1)
                        else:
                            break
                if num[0].split()[0] == 'faces':
                    for jj, vv in enumerate(a[j + 1:-1]):
                        if vv[0] != '\n':
                            v_list = vv[0].split()
                            if any(a == '/*area' for a in v_list):
                                if jj == 0:
                                    temp_id = v_list[0]
                                for vf in v_list[0:-1]:
                                    if vf != '/*area':
                                        if vf != temp_id:
                                            faces['id'].append(temp_id)
                                            faces['ed_id'].append(vf)
                                    else:
                                        break

                                temp_id = a[j + jj + 2][0].split()[0]
                            else:
                                if jj == 0:
                                    temp_id = v_list[0]
                                for vf in v_list[0:-1]:
                                    if vf != temp_id:
                                        faces['id'].append(temp_id)
                                        faces['ed_id'].append(vf)
                        else:
                            break
                if num[0].split()[0] == 'bodies':
                    for jj, vv in enumerate(a[j + 1:-1]):
                        bodies['id'].append(vv[0].split()[0])
                        bodies['pressure'].append(vv[0].split()[7])
            except:
                pass

        vertices_data = pd.DataFrame(vertices)
        vertices_data.set_index(['id'], inplace=True)
        edges_data = pd.DataFrame(edges)
        edges_data.set_index(['id'], inplace=True)
        faces_data = pd.DataFrame(faces)
        unique_faces = sorted(list(set((faces_data.id))), key=lambda x: int(x))
        if len(bodies['pressure']) != 0:
            bodies_data = pd.DataFrame(bodies)
            bodies_data.set_index(['id'], inplace=True)

        # First loop through all face ids
        X, Y = [], []
        all_ten, all_pressures = [], []

        for face in unique_faces:
            x, y = [], []
            tensions = []

            for k, v in faces_data.iterrows():
                # v[0] is the face id, v[1] is the edge id , i set it up as a one to one mapping, but multiple edge ids
                # assigned to one face id
                # Now we look at the values that have v[0] (face id) == to the unique face id we are looking at now
                if v[0] == face:
                    # v[1] is the edge id, go to edges_data to get the vertices assigned to it
                    if int(v[1]) > 0:
                        v1_id = edges_data.at[v[1], 'v1']
                        v2_id = edges_data.at[v[1], 'v2']
                        t1 = edges_data.at[v[1], 'tension']
                    else:
                        v2_id = edges_data.at[str(-int(v[1])), 'v1']
                        v1_id = edges_data.at[str(-int(v[1])), 'v2']
                        t1 = edges_data.at[str(-int(v[1])), 'tension']

                    # go to vertices_data to get x and y co-ords of that vertex
                    x1 = float(vertices_data.at[v1_id, 'x'])
                    y1 = float(vertices_data.at[v1_id, 'y'])
                    x2 = float(vertices_data.at[v2_id, 'x'])
                    y2 = float(vertices_data.at[v2_id, 'y'])

                    tensions.append(t1)
                    x.append(x1)
                    y.append(y1)
            x.append(x2)
            y.append(y2)
            X.append(x)
            Y.append(y)
            all_ten.append(tensions)
            if len(bodies['pressure']) != 0:
                all_pressures.append(bodies_data.at[face, 'pressure'])

        # ADD BACK IF NEEDED
        # for a, b in zip(X, Y):
        #     a.pop(0)
        #     b.pop(0)

        # Get face-face junction co-ordinates
        for a in all_ten:
            a.append(a[0])

        new_x, new_y, new_ten = [], [], []

        for j, num in enumerate(unique_faces):
            cur_x = X[j]
            cur_y = Y[j]
            cur_ten = all_ten[j]
            for j, num in enumerate(unique_faces):
                int_x = [a for a in cur_x if a in X[j]]
                int_y = [a for a in cur_y if a in Y[j]]

                ind_int_x = [cur_x.index(a) for a in int_x]
                int_ten = [cur_ten[j] for j in ind_int_x]

                if len(int_x) != len(cur_x) and len(int_x) != 0:
                    check = 0
                    for a in new_x:
                        if len(set(a).intersection(set(int_x))) == len(set(int_x)):
                            check = 1
                    if check == 0:
                        new_x.append(int_x)

                        new_y.append(int_y)
                        new_ten.append(int_ten)
        count = 0

        for a, b in zip(new_x, new_y):
            if a[0] == a[-1]:
                c1 = sorted(a[0:-1])
                c2 = sorted(a[0:-1], reverse=True)
                if c1 == a[0:-1] or c2 == a[0:-1]:
                    a.pop()
                    b.pop()
                    new_ten[count].pop()
                elif c1 == a[1:] or c2 == a[1:]:
                    a.pop(0)
                    b.pop(0)
                    new_ten[count].pop(0)
                else:
                    g1 = sorted(b[0:-1])
                    g2 = sorted(b[0:-1], reverse=True)
                    if g1 == b[0:-1] or g2 == b[0:-1]:
                        a.pop()
                        b.pop()
                        new_ten[count].pop()
                    elif g1 == b[1:] or g2 == b[1:]:
                        a.pop(0)
                        b.pop(0)
                        new_ten[count].pop(0)
                count += 1

        ex = ManualTracing(new_x, new_y, new_ten)
        nodes, edges, new = ex.cleanup(0.5)

        print('Number of fit edges:', len(edges))

        mean_gt = np.mean([e.ground_truth for e in edges])
        for e in edges:
            e.ground_truth = e.ground_truth / mean_gt

        cells = ex.find_cycles(edges)

        if len(bodies['pressure']) != 0:
            C_X, C_Y = [], []
            for c in cells:
                c_x, c_y = [], []
                for e in c.edges:
                    c_x.append(e.co_ordinates[0])
                    c_y.append(e.co_ordinates[1])
                c_x = [item for sublist in c_x for item in sublist]
                c_y = [item for sublist in c_y for item in sublist]
                C_X.append(c_x)
                C_Y.append(c_y)

            c_presses = []
            for c in C_X:
                count = 1
                for a in X:
                    if set(a) == set(c):
                        c_presses.append(float(bodies_data.at[str(count), 'pressure']))
                    count += 1
            print('Length of cpresses', len(c_presses), len(cells))

            mean_pres_c = np.mean(c_presses)
            for j, c in enumerate(cells):
                try:
                    c.ground_truth_pressure = c_presses[j] / mean_pres_c - 1
                    # c.ground_truth_pressure = c_presses[j]
                except:
                    c.ground_truth_pressure = np.NaN

        print('ground_truth', [e.ground_truth for e in edges])
        print('Number of cells:', len(cells))
        return nodes, edges, cells

    def initial_numbering(self, number0):
        """
        Assign random labels to nodes and cells in the colony specified by number0
        Returns labeled nodes and cells.
        Also returns a dictionary defined as {node.label: edges connected to node label, vectors of edges connected to node label}
        Also returns the edge list (not labeled)
        """

        # Get the list of nodes for name0
        # name =  'ritvik_5pb_edges[3]_tension_' + str(number0) + '.fe.txt'
        name = self.name_first + str(number0) + self.name_end
        print('Name is', name)
        temp_nodes, edges, initial_cells = self.get_X_Y_data_only_junction(name)

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
            # Give every node a label -> in this case we're arbitrarily giving labels as we loop through
            node.label = j
            sort_edges = node.edges
            this_vec = [func(p, node) for p in sort_edges]
            # Add these sorted edges to a dictionary associated with the node label
            old_dictionary[node.label].append(sort_edges)
            old_dictionary[node.label].append(this_vec)

        for k, cc in enumerate(initial_cells):
            cc.label = k

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

        # Get list of nodes and edges for names_now
        # No labelling
        print('Number now is', number_now)
        # name_now = 'ritvik_5pb_edges[3]_tension_' + str(number_now) + '.fe.txt'
        name_now = self.name_first + str(number_now) + self.name_end

        now_nodes, now_edges, now_cells = self.get_X_Y_data_only_junction(name_now)

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
                    closest_new_node.previous_label = prev_node.label
                    closest_new_node.velocity_vector = np.array(
                        (closest_new_node.loc[0] - prev_node.loc[0], closest_new_node.loc[1] - prev_node.loc[1]))
            else:
                # If its connected to 3 edges, closest node is fine. only single edge nodes had problems
                closest_new_node.label = prev_node.label
                closest_new_node.velocity_vector = np.array((closest_new_node.loc[0]
                                                             - prev_node.loc[0], closest_new_node.loc[1]
                                                             - prev_node.loc[1]))

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
                            # print('Possible topological change')
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
            if not node.label:
                node.label = count
                count += 1
                check = 1
            for e in node.edges:
                if not e.label:
                    e.label = count_edge
                    count_edge += 1
                    check = 1
            if check == 1:
                temp_edges = node.edges
                new_vecs = [func(p, node) for p in temp_edges]
                if len(new_dictionary[node.label]) != 0:
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

        now_cells = self.label_cells(old_cells, now_cells)

        # Define a colony
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
            if not cc.label:
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

    def computation_based_on_prev_surface_evolver(self, numbers, colonies=None, index=None, old_dictionary=None,
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
            colonies, old_dictionary = self.first_computation(numbers[0], solver,
                                                              type='Jiggling', **kwargs)
            colonies[str(0)].dictionary = old_dictionary
            print('Solver is', solver)
            print('First colony', colonies)
            index = 0

        if numbers[index + 1] == numbers[index]:
            colonies[str(index + 1)], new_dictionary = self.track_timestep(colonies[str(index)],
                                                                           old_dictionary, numbers[index + 1])
            colonies[str(index + 1)].dictionary = new_dictionary
            tensions, p_t, a_mat = colonies[str(index + 1)].calculate_tension(solver=solver, **kwargs)
            pressures, p_p, b_mat = colonies[str(index + 1)].calculate_pressure(solver=solver, **kwargs)
            # Save tension and pressure matrix
            colonies[str(index + 1)].tension_matrix = a_mat
            colonies[str(index + 1)].pressure_matrix = b_mat
            print('Next colony number', str(index + 1))
        else:
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
            colonies = self.computation_based_on_prev_surface_evolver(numbers, colonies, index, new_dictionary, solver,
                                                                      **kwargs)

        return colonies