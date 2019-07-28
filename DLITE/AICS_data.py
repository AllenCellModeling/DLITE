import numpy as np
from .cell_describe import node
from .AICS_data import data


class ManualTracing(data):
    def __init__(self, x, y, ground_truth_tensions=None):
        """
        Class for a single frame of manual tracing that has been traced out using NeuronJ
        Manual tracing that outputs an array of X and Y co-ordinates
        length(X) == number of edges
        length(X[0]) == X co-ordinates on edge 0
        length(Y[0]) == Y co-ordinates on edge 0
        """
        self.x = x
        self.y = y
        self.ground_truth_tensions = ground_truth_tensions
        self.length = len(self.x)

    def co_ordinates(self, edge_num):
        """
        Get X and Y co-ordinates for specified edge number
        """
        return self.x[edge_num], self.y[edge_num]

    def fit_X_Y(self, edge_num):
        """
        Fit a circular arc to an edge.
        Call self.fit - .fit is a function in the data class
        """
        r_2, xc_2, yc_2, residu_2 = self.fit(self.x[edge_num], self.y[edge_num])
        return r_2, xc_2, yc_2, residu_2

    def cleanup(self, cutoff):
        """
        post process the data to merge nodes that are within a distance
        specified as 'cutoff'. Also calls functions
        (1) remove_dangling_edges
        (2) remove_two_edge_connections
        ----------
        Parameters
        ------------
        cutoff - distance within which we merge nodes
        num (optional) - Number of branches to consider (default - all of them)
        """
        nodes, edges = [], []

        for index in range(self.length):
            # Add 2 nodes at both ends of the branch
            node_a = node((self.x[index][0], self.y[index][0]))
            node_b = node((self.x[index][-1], self.y[index][-1]))

            dist_a, dist_b = [], []
            for n in nodes:
                # Find distance of all nodes in the list (nodes) from current node_a
                dist_a.append(np.linalg.norm(np.subtract(n.loc, node_a.loc)))

            if not dist_a:
                # If dist = [], then nodes was empty -> add node_a to list
                nodes.append(node_a)
            else:
                # If all values in dist are larger than a cutoff, add the node
                if all(i >= cutoff for i in dist_a):
                    nodes.append(node_a)
                else:
                    # Find index of minimum distance value. Replace node_a with the node at that point
                    ind = dist_a.index(min(dist_a))
                    node_a = nodes[ind]

            # Have to do this separately and not with node_a because
            # we want to check that distance between node_a and node_b is very small too
            # So if node_b is so close to node_a that we replace node_b with node_a, have to not add that edge
            for n in nodes:
                dist_b.append(np.linalg.norm(np.subtract(n.loc, node_b.loc)))

            if not dist_b:
                # If dist = [], then nodes was empty -> add node_a to list
                nodes.append(node_b)
            else:
                # If all values in dist are larger than a cutoff, add the node
                if all(i >= cutoff for i in dist_b):
                    nodes.append(node_b)
                else:
                    # Find index of minimum distance value. Replace node_a with the node at that point
                    ind = dist_b.index(min(dist_b))
                    node_b = nodes[ind]

            if node_a.loc != node_b.loc:
                ed = self.add_edge(node_a, node_b, None, self.x[index], self.y[index])
                if self.ground_truth_tensions is not None:
                    mean_ground_truth = np.mean(self.ground_truth_tensions[index])
                    ed.ground_truth = mean_ground_truth
                edges.append(ed)
            else:
                print('node a = node b, possible topological change')

        # Remove dangling edges (single edges connected to an interface at nearly 90 deg angle)
        new_edges = []

        # Possible to add cleanup in manual tracing, but not necessary

        # Below are the next 3 post processing steps

        # Step 1 - remove small stray edges (nodes connected to 1 edge)
        # nodes, edges = self.remove_dangling_edges(nodes, edges)

        # Step 2 - remove small cells
        # nodes, edges, new_edges = self.remove_small_cells(nodes, edges)

        # Step 3 - remove nodes connected to 2 edges
        # nodes, edges = self.remove_two_edge_connections(nodes, edges)

        return nodes, edges, new_edges