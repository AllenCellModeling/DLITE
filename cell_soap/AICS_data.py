import numpy as np
from scipy import ndimage, optimize
import collections
from cell_describe import node, edge, colony


class data:
    def __init__(self, v, t):
        """
        Data class made specifically for the pickle file format used at the
        Allen Institute for Cell Science
        Parameters
        ---------
        v is data structure obtained after loading the pickle file
        t is time step
        ---------
        """
        self.v = v
        self.t = t
        self.length = len(self.v[2][self.t])

    def x(self, index, f_or_l):
        """
        Returns x co-ordinate of branch number "index" at branch end "f_or_l"
        If f_or_l (str value - 'first', or 'last') not specified,
         returns all x co-ordinates along the branch
        -------------
        Parameters
        -------------
        index - index of branch in the list of branches in the data
        f_or_l - either first or last index on a branch

        """
        if f_or_l == "first":
            return self.v[1][self.t][self.v[2][self.t][index][0], 1]
        elif f_or_l == "last":
            loc = len(self.v[2][self.t][index][:]) - 1
            return self.v[1][self.t][self.v[2][self.t][index][loc], 1]
        else:
            return self.v[1][self.t][self.v[2][self.t][index][:], 1]

    def y(self, index, f_or_l):
        """
        Returns y co-ordinate of branch number "index" at branch end "f_or_l"
        If f_or_l (str value - 'first', or 'last') not specified,
        returns all y co-ordinates along the branch
        -------------
        Parameters
        -------------
        index - index of branch in the list of branches in the data
        f_or_l (str) - either "first" or "last" index on a branch

        """
        if f_or_l == "first":
            return self.v[1][self.t][self.v[2][self.t][index][0], 0]
        elif f_or_l == "last":
            loc = len(self.v[2][self.t][index][:]) - 1
            return self.v[1][self.t][self.v[2][self.t][index][loc], 0]
        else:
            return self.v[1][self.t][self.v[2][self.t][index][:], 0]

    def add_node(self, index, f_or_l):
        """
        Define a node on branch "index" and location on branch "f_or_l" (str)
        """
        return node((self.x(index, f_or_l), self.y(index, f_or_l)))

    def add_edge(self, node_a, node_b, index=None, x=None, y=None):
        """
        Define an edge given a branch index and end nodes of class node
        Calls fit() to fit a curve to the data set
        -----------
        Parameters
        -----------
        node_a - node object at one end of the edge
        node_b - node object at other end of the edge
        index = Branch location. If this is specified, can get the x and y co-ordinates
        of this branch location from data
        If index is not specified, x and y need to be provided.
        x - x co-ordinates along the edge
        y - y co-ordinates along the edge
        """

        # Get all co-ordinates along the branch
        if index is not None:
            x = self.x(index, None)
            y = self.y(index, None)

        # we want to fit a curve to this. Use least squares fitting.
        # output is radius and x,y co-ordinates of the centre of circle
        radius, xc, yc = self.fit(x, y)

        # Check the direction of the curve. Do this by performing cross product
        x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]

        v1 = [x1 - xc, y1 - yc]
        v2 = [x2 - xc, y2 - yc]

        cr = np.cross(v1, v2)

        a = 0.5 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]))  # dist to midpoint

        # Check if radius is 0
        if radius > 0:
            # Check for impossible arc
            if a < radius:
                # if cross product is negative, then we want to go from node_a to node_b
                # if positive, we want to go from node_b to node_a
                if cr > 0:  # correct is cr > 0
                    ed = edge(node_b, node_a, radius, None, None, x, y)
                else:
                    ed = edge(node_a, node_b, radius, None, None, x, y)
            else:
                # Add rnd to radius, since its fitting an impossible arc.
                # Can also choose to exit/break code if we enter this block,
                # since arc-fit doesn't work.
                rnd = a - radius + 5
                if cr > 0:
                    ed = edge(node_b, node_a, radius + rnd, None, None, x, y)
                else:
                    ed = edge(node_a, node_b, radius + rnd, None, None, x, y)
        else:
            # if no radius, leave as None
            ed = edge(node_a, node_b, None, None, None, x, y)

        ed.center_of_circle = [xc, yc]

        return ed

    def fit(self, x, y):
        """
        Fit a circular arc to a list of co-ordinates
        -----------
        Parameters
        -----------
        x, y
        """

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            ri = calc_R(*c)
            return ri - ri.mean()

        x_m = np.mean(x)
        y_m = np.mean(y)

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        ri_2 = calc_R(*center_2)
        r_2 = ri_2.mean()
        residu_2 = np.sum((ri_2 - r_2) ** 2)

        theta1 = np.rad2deg(np.arctan2(y[np.argmax(x)] - yc_2, x[np.argmax(x)] - xc_2))  # starting angle
        theta2 = np.rad2deg(np.arctan2(y[np.argmin(x)] - yc_2, x[np.argmin(x)] - xc_2))

        return r_2, xc_2, yc_2

    def post_processing(self, cutoff, num=None):
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

        if not num:
            num = self.length

        for index in range(num):
            # Add 2 nodes at both ends of the branch
            node_a = self.add_node(index, "first")
            node_b = self.add_node(index, "last")

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
                ed = self.add_edge(node_a, node_b, index)
                edges.append(ed)

        # Below are the next 3 post processing steps

        # Step 1 - remove small stray edges (nodes connected to 1 edge)
        # These are not stray edges along the boundary, but rather small
        # stray edges sticking out from another edge that shouldn't be there.
        # Sometimes, these could be edges that are not connected to any other edge.
        # This is a common error in automated skeletonization
        nodes, edges = self.remove_dangling_edges(nodes, edges)

        # Step 2 - remove small cells
        # Small cells occur when automated skeletonization says there should be multiple nodes
        # at a location where there should only be one node. We want to merge these nodes/cells
        # into a single node
        nodes, edges, new_edges = self.remove_small_cells(nodes, edges)

        # Step 3 - remove nodes connected to 2 edges
        # Finally, we want to check that there are no nodes connected to 2 edges as
        # we cannot perform a force balance at such a node
        nodes, edges = self.remove_two_edge_connections(nodes, edges)

        return nodes, edges, new_edges

    def remove_dangling_edges(self, nodes, edges):
        """
        Clean up nodes connected to 1 edge
        Do this by -
        Removing edges that are really small and connected to 2 other edges at a nearly 90 deg angle
        Also remove edges that are connected to nobody else
        """
        # Get nodes connected to 1 edge
        n_1_edges = [n for n in nodes if len(n.edges) == 1]
        # Get those edges
        if len(n_1_edges) > 0:
            e_1 = [e for f in n_1_edges for e in f.edges]
            # Get the other node on these edges
            n_1_edges_b = [n for j, f in enumerate(e_1) for n in f.nodes if n != n_1_edges[j]]

        for j, e in enumerate(e_1):
            # Check that this edge is really small
            if e.straight_length < 3:
                # Get all the edges on node_b of the edge e
                other_edges = [a for a in n_1_edges_b[j].edges if a != e]
                # Get the angle and edge of the edges that are perpendicular (nearly) to e
                perps = []
                perps = [b for b in other_edges if 85 < abs(e.edge_angle(b)) < 95]  # 85 - 95, 40 - 140
                # If there is such a perpendicular edge, we want to delete e
                if perps:
                    other_node = [n for n in e.nodes if n != n_1_edges_b[j]][0]
                    e.kill_edge(n_1_edges_b[j])
                    if e in edges:
                        edges.remove(e)
                    nodes.remove(other_node)

        # Check for special case -> 2 nodes connected to single edge which they share - so connected to each other
        repeated_edges = [item for item, count in collections.Counter(e_1).items() if count > 1]

        for e in repeated_edges:
            edges.remove(e)
            nodes.remove(e.node_a)
            nodes.remove(e.node_b)

        return nodes, edges

    def remove_two_edge_connections(self, nodes, edges):
        """
        Clean up nodes connected to 2 edges
        """
        # Get nodes connected to 2 edges
        n_2 = [n for n in nodes if len(n.edges) == 2]
        # If there is such a node
        if len(n_2) > 0:
            for n in n_2:
                angle = n.edges[0].edge_angle(n.edges[1])
                if 0 < abs(angle) < 180 or 0 < abs(angle) < 60:
                    # Get non common node in edge 0
                    node_a = [a for a in n.edges[0].nodes if a != n][0]
                    # Get non common node in edge 1
                    node_b = [a for a in n.edges[1].nodes if a != n][0]
                    # Remove edge 0 from node_a and edge 1 from node_b
                    # Remove corresponding tension vectors saved in node_a and node_b
                    ind_a = node_a.edges.index(n.edges[0])
                    ind_b = node_b.edges.index(n.edges[1])
                    node_a.tension_vectors.pop(ind_a)
                    node_b.tension_vectors.pop(ind_b)
                    node_a.edges.remove(n.edges[0])
                    node_b.edges.remove(n.edges[1])

                    # Get co-ordinates  of edge 0 and edge 1
                    x1, y1 = n.edges[0].co_ordinates
                    x2, y2 = n.edges[1].co_ordinates

                    # Extend the list x1, y1 to include x2 and y2 values
                    if x1[-1] == x2[0]:
                        new_x = np.append(x1, x2)
                        new_y = np.append(y1, y2)
                    else:
                        new_x = np.append(x1, x2[::-1])
                        new_y = np.append(y1, y2[::-1])

                    # Define a new edge with these co-ordinates
                    try:
                        new_edge = self.add_edge(node_a, node_b, None, new_x, new_y)
                        # Finish cleanup. remove edge 0 and edge 1 from node n and then remove node n
                        # Add a new edge to the list
                        edges.remove(n.edges[0])
                        edges.remove(n.edges[1])
                        nodes.remove(n)
                        edges.append(new_edge)
                    except AssertionError:
                        pass

        return nodes, edges

    def remove_small_cells(self, nodes, edges):
        """
        Clean up small cells that have a small perimeter
        """
        # Get unique cells
        cells = self.find_cycles(edges)

        # Define a cutoff perimeter. We use 150, an arbitrary small value
        # that works for cleaning up most skeletonization errors in AICS dataset
        cutoff_perim = 150

        small_cells = [cell for cell in cells if cell.perimeter() < cutoff_perim]
        new_edges = []

        for cell in small_cells:
            # Delete the edges and tension vector saved in the nodes that are part of this cell
            for ed in cell.edges:
                if ed in ed.node_a.edges:
                    ed.kill_edge(ed.node_a)
                if ed in ed.node_b.edges:
                    ed.kill_edge(ed.node_b)
                # Also remove this edge from the list of edges
                if ed in edges:
                    edges.remove(ed)

            # Make a new node
            all_loc = [cell.nodes[i].loc for i in range(len(cell.nodes))]
            x, y = [i[0] for i in all_loc], [i[1] for i in all_loc]
            new_x, new_y = np.mean(x), np.mean(y)
            new_node = node((new_x, new_y))

            # Now we defined a new node, have to add new edges
            # Lets add a new edge with the first edge on a node - node.edges[0]
            # Old edge is node.edges[0]. want to replace it with a new_edge

            for n in cell.nodes:
                if len(n.edges) == 0:
                    if n in nodes:
                        nodes.remove(n)
            for n in cell.nodes:
                if len(n.edges) > 0:
                    for ned in n.edges:
                        node_b = [a for a in ned.nodes if a != n][0]

                        x1, y1 = ned.co_ordinates
                        new_x1, new_y1 = np.append(x1, new_x), np.append(y1, new_y)
                        ned.kill_edge(n)
                        ned.kill_edge(node_b)

                        # Finish cleanup
                        # Delete memory of the old edge from the nodes and then remove it from the list of edges
                        if ned in edges:
                            edges.remove(ned)
                        # Add new edge
                        new_edge = self.add_edge(node_b, new_node, None, new_x1, new_y1)
                        new_edges.append(new_edge)
                        edges.append(new_edge)
                    if n in nodes:
                        nodes.remove(n)
            nodes.append(new_node)

        # Check for nodes that are not connected to any edges
        for n in nodes:
            if len(n.edges) == 0:
                nodes.remove(n)

        return nodes, edges, new_edges

    @staticmethod
    def find_cycles(edges):
        """
        Find cycles given a list of edges.
        Takes a list of edges and for every edge, gives a maximum of 2 cells that its connected to
        This method calls which_cell which in turn calls recursive_cycle_finder
        """

        # Set max iterations for cycle finding
        max_iter = 300
        # Set initial cells
        cells = []

        for e in edges:
            cell = e.which_cell(edges, 0, max_iter)
            check = 0
            if cell:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    for edge in cell.edges:
                        edge.cells = cell
                    cells.append(cell)

            cell = e.which_cell(edges, 1, max_iter)
            check = 0
            if cell:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    for edge in cell.edges:
                        edge.cells = cell
                    cells.append(cell)
        return cells

    def compute(self, cutoff, nodes=None, edges=None):
        """
        Computation process. Steps ->
        (1) Call post_processing() -> returns nodes and edges
        (2) Call which_cell() for each edge -> returns cells
        (3) Define colony
        (4) Call calculate_tension() - find tensions
            (1) If any bad edges (> 3 std away, call remove_outliers() and repeat computation)
        (5) Call calculate_pressure() - find pressure
        """
        # Get nodes, edges
        if nodes is None and edges is None:
            nodes, edges, _ = self.post_processing(cutoff, None)

        # Get unique cells
        cells = self.find_cycles(edges)

        # Get tension and pressure
        edges2 = [e for e in edges if e.radius is not None]
        col1 = colony(cells, edges2, nodes)
        tensions, p_t, a_mat = col1.calculate_tension()

        # Check for bad tension values
        # Find mean and std
        mean = np.mean(tensions)
        sd = np.std(tensions)

        # Possibly recompute tensions by deleting edges if there are any poorly scaled tensions

        # Find tensions more than 3 standard deviations away
        # bad_tensions = [x for x in tensions if (x < mean - 3 * sd) or (x > mean + 3 * sd)]

        # if len(bad_tensions) > 0:
        #     new_nodes, new_edges = col1.remove_outliers(bad_tensions, tensions)
        #     col1, tensions, _, P_T, _, A, _ =  self.compute(cutoff, new_nodes, new_edges)

        pressures, p_p, b_mat = col1.calculate_pressure()

        return col1, tensions, pressures, p_t, p_p, a_mat, b_mat

    def plot(self, ax, type=None, num=None, **kwargs):
        """
        Plot the data set
        ----------------
        Parameters
        ----------------
        ax - axes to be plotted on
        type - "edge_and_node", "node", "edge", "image" - specifying what you want to plot
        num - number of branches to be plotted
        """
        if not num:
            num = self.length
        else:
            pass

        if type == "edge_and_node" or type == "edges_and_nodes" or type == "edge_and_nodes" or type == "edges_and_node":
            for i in range(num):
                ax.plot(self.x(i, None), self.y(i, None), **kwargs)
                ax.plot(self.x(i, "first"), self.y(i, "first"), 'ok')
                ax.plot(self.x(i, "last"), self.y(i, "last"), 'ok')
        elif type == "node" or type == "nodes":
            for i in range(num):
                ax.plot(self.x(i, "first"), self.y(i, "first"), 'ok')
                ax.plot(self.x(i, "last"), self.y(i, "last"), 'ok')
        elif type == "edge" or type == "edges":
            for i in range(num):
                ax.plot(self.x(i, None), self.y(i, None), **kwargs)
        elif type == "image" or type == "images":
            # plot image
            img = ndimage.rotate(self.v[0][self.t] == 2, 0)
            # plot the image with origin at lower left
            ax.imshow(img, origin='lower')

        ax.set(xlim=[0, 1000], ylim=[0, 1000], aspect=1)