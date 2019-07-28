import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as linalg 
from scipy.optimize import minimize
from matplotlib import cm
import math
import matplotlib.patches as mpatches
import random


class node:
    def __init__(self, loc):
        """
        loc is the (x,y) location of the node
        """
        self.loc = loc
        self._edges = []
        self._tension_vectors = []
        self._previous_tension_vectors = []
        self._horizontal_vectors = []
        self._vertical_vectors = []
        self._label = []
        self._residual_vector = []
        # edge indices in colony.tot_edges
        self._edge_indices = []
        self._previous_label = []
        self._velocity_vector = []

    @property
    def x(self):
        """
        x co-ordinate of node
        """
        return self.loc[0]

    @property
    def y(self):
        """
        y co-ordinate of node
        """
        return self.loc[1]
    
    @property
    def edges(self):
        """
        List of edges connected to this node
        """

        return self._edges

    @edges.setter
    def edges(self, edge):
        """
        Sets list of edges -- make sure no repeat edges
        """
        if edge not in self._edges:
            self._edges.append(edge)

    def remove_edge(self, edge):
        """Remove an edge and tension vector connected to this node """

        ind = self._edges.index(edge) 
        self._edges.pop(ind)
        self._tension_vectors.pop(ind)

    @property
    def tension_vectors(self):
        """ Tension vectors connected to this node"""
        return self._tension_vectors

    @tension_vectors.setter
    def tension_vectors(self, vector):
        """ Setter for tension_vectors, make sure no repeat tension_vectors """
        if vector not in self._tension_vectors:
            self._tension_vectors.append(vector)

    @property
    def previous_tension_vectors(self):
        """ Tension vectors connected to this node at previous time point"""
        return self._previous_tension_vectors

    @previous_tension_vectors.setter
    def previous_tension_vectors(self, vector):
        """
        Setter for tension_vectors, make sure no repeat tension_vectors
        """
        if vector not in self._previous_tension_vectors:
            self._previous_tension_vectors.append(vector)
    @property
    def horizontal_vectors(self):
        """ List of x-component of tension vectors connected to this node """
        return self._horizontal_vectors

    @horizontal_vectors.setter
    def horizontal_vectors(self, hor_vector):
        self._horizontal_vectors = hor_vector

    @property
    def vertical_vectors(self):
        """ List of y-component of tension vectors connected to this node """
        return self._vertical_vectors

    @vertical_vectors.setter
    def vertical_vectors(self, ver_vector):
        self._vertical_vectors = ver_vector

    @property
    def edge_indices(self):
        """ Indices of edges connected to this node in the list of all edges in colony """
        return self._edge_indices

    @edge_indices.setter
    def edge_indices(self, indices):
        self._edge_indices = indices      

    @property    
    def label(self):
        """
        Give a label to a node so we can track it over time
        """
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def previous_label(self):
        """ Label of this node at previous time point """
        return self._previous_label

    @previous_label.setter
    def previous_label(self, prev_label):
        self._previous_label = prev_label

    @property
    def velocity_vector(self):
        """ Velocity vector computed from distance traveled in time from previous time point"""
        return self._velocity_vector
    
    @velocity_vector.setter
    def velocity_vector(self, velocity):
        self._velocity_vector = velocity    

    @property    
    def residual_vector(self):
        """
        Give a label to a node so we can track it over time
        """
        return self._residual_vector

    @residual_vector.setter
    def residual_vector(self, residual):
        """
        Give a label to a node so we can track it over time
        """
        self._residual_vector = residual

    def plot(self, ax, **kwargs):
        """ Plot node as a point """
        ax.plot(self.loc[0], self.loc[1], ".", **kwargs)


class edge:
    def __init__(self, node_a, node_b, radius=None, xc=None, yc=None, x_co_ords=None, y_co_ords=None):
        """
        Define an edge clockwise between node_a and node_b.
        -------------
        Parameters
        -------------
        Node_a, Node_b - nodes at the end of the branch 
        radius - fitted radius of a circular arc
        -------------
        Notes ->
        Calling edge automatically makes the following changes to the node class:
        (1) Adds this edge to the edge property in node_a and node_b 
        (2) Calculates tension vectors at node_a and node_b (perp_a, perp_b)
        and adds these to the tension_vectors property in node_a and node_b
        """
        self.node_a = node_a
        self.node_b = node_b
        self.radius = radius
        self.xc = xc
        self.yc = yc 

        # Save x and y co-ord info from the image
        self.x_co_ords = x_co_ords
        self.y_co_ords = y_co_ords

        node_a.edges = self
        node_b.edges = self

        perp_a, perp_b = self.unit_vectors()
        perp_a = list(perp_a.reshape(1, -1)[0])
        perp_b = list(perp_b.reshape(1, -1)[0])

        node_a.tension_vectors = perp_a
        node_b.tension_vectors = perp_b

        self._cells = []
        self._tension = []
        self._ground_truth = []
        self._guess_tension = []
        self._label = []
        self._center_of_circle = []
        self._cell_indices = []
        self._cell_coefficients = []
        self._cell_rhs = []
        self._previous_label = []

    @property
    def co_ordinates(self):
        """
        List of co-ordinates along this edge
        """
        return self.x_co_ords, self.y_co_ords
    
    @property
    def cells(self):
        """
        List of cells that this edge is a part of
        """
        return self._cells

    @cells.setter
    def cells(self, cc):
        """
        Check no repeat cells by checking if the edges in this new cell dont match edges in a cell that already exists in the list
        """
        check = 0
        if cc not in self._cells:
            for c in self._cells:
                if len(set(c.edges).intersection(set(cc.edges))) == len(set(cc.edges)):
                    check = 1
            if check == 0:
                self._cells.append(cc)

    @property
    def cell_indices(self):
        """
        Make a list of cell indices from list of cells in colony - set during pressure matrix calculation
        """
        return self._cell_indices

    @cell_indices.setter
    def cell_indices(self, indices):
        """
        Setter for cell indices
        """
        self._cell_indices = indices

    @property
    def cell_coefficients(self):
        """
        List of cell co-efficents - not sure where i used this
        """
        return self._cell_coefficients

    @cell_coefficients.setter
    def cell_coefficients(self, coeff):
        self._cell_coefficients = coeff

    @property
    def cell_rhs(self):
        """
        RHS of pressure matrix equation, i.e tension/radius 
        """
        return self._cell_rhs

    @cell_rhs.setter
    def cell_rhs(self, rhs):
        self._cell_rhs = rhs    

    @property    
    def label(self):
        """
        Give a label to a node so we can track it over time
        """
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def previous_label(self):
        return self._previous_label

    @previous_label.setter
    def previous_label(self, previous_label):
        self._previous_label = previous_label

    def kill_edge(self, n):
        """
        Kill edge for a specified node i.e kill the edge and tension vector 
        associated with the specified node
        Parameters - n (a node)
        """

        if n == self.node_a:
            self.node_a.remove_edge(self)
        if n == self.node_b:
            self.node_b.remove_edge(self)

    @property
    def center_of_circle(self):
        """
        Center of circle that sets the radius of this edge
        """
        return self._center_of_circle

    @center_of_circle.setter
    def center_of_circle(self, center):
        self._center_of_circle = center

    @property
    def tension(self):
        """
        Tension of this edge
        """
        return self._tension

    @tension.setter
    def tension(self, tension):
        self._tension = tension

    @property
    def ground_truth(self):
        """
        Tension of this edge
        """
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, ground_truth):
        self._ground_truth = ground_truth

    @property
    def guess_tension(self):
        """
        Initial guess for tension based on the same edge at a previous time point
        """
        return self._guess_tension

    @guess_tension.setter
    def guess_tension(self, guess_tension):
        self._guess_tension = guess_tension

    @property
    def straight_length(self):
        """The distance from node A to node B"""
        return np.linalg.norm(np.subtract(self.node_a.loc, self.node_b.loc))

#    @staticmethod
#    def _circle_arc_center(point1, point2, radius):
    def _circle_arc_center(self, point1, point2, radius):
        """Get the center of a circle from arc endpoints and radius"""
        x1, y1 = point1
        x2, y2 = point2
        x0, y0 = 0.5*np.subtract(point2, point1)+point1  # midpoint
        a = 0.5*np.linalg.norm(np.subtract(point2, point1))  # dist to midpoint
        # assert a<radius, "Impossible arc asked for, radius too small"
        if a > radius:
            self.radius = radius + a - radius + 1
            radius = radius + a - radius + 1
        b = np.sqrt(radius**2-a**2)  # midpoint to circle center
        xc = x0 + (b*(y0-y1))/a  # location of circle center
        yc = y0 - (b*(x0-x1))/a
        return xc, yc

    def arc_translation(self, point1, point2, radius):
        """Get arc center and angles from endpoints and radius
        We want to be able to plot circular arcs on matplotlib axes.
        Matplotlib only supports plotting such by giving the center,
        starting angle, and stopping angle for such. But we want to
        plot using two points on the circle and the radius.
        For explanation, check out https://tinyurl.com/ya7wxoax
        """
        x1, y1 = point1
        x2, y2 = point2
        if self.xc is None or self.yc is None:
            xc, yc = self._circle_arc_center(point1, point2, radius)
        else:
            xc, yc = self.xc, self.yc
        theta1 = np.rad2deg(np.arctan2(y2-yc, x2-xc))  # starting angle
        theta2 = np.rad2deg(np.arctan2(y1-yc, x1-xc))  # stopping angle
        return (xc, yc), theta1, theta2

    def plot(self, ax, **kwargs):
        """
        Plot edges using matplotlib.patches.arc, requires a circle center, radius of arc and starting and ending angle
        """
        a, b = self.node_a, self.node_b
        if self.radius is not None:

            center, th1, th2 = self.arc_translation(a.loc, b.loc, self.radius)
            patch = matplotlib.patches.Arc(center, 2*self.radius, 2*self.radius,
                                           0, th1, th2, **kwargs)
            ax.add_patch(patch)
        else:
            ax.plot([a.x, b.x], [a.y, b.y], lw=4)

    def plot_fill(self, ax, resolution=50, **kwargs):
        """
        Similar to plot, only difference is arc is now filled, needed for pressure colormap diagram
        """
        a, b = self.node_a, self.node_b
        if self.radius is not None:
            center, th1, th2 = self.arc_translation(a.loc, b.loc, self.radius)
            old_th1, old_th2 = th1, th2
            th1, th2 = (th1 + 720) % 360, (th2 + 720) % 360
            if abs(th2 - th1) > 180:
                th2, th1 = old_th2, old_th1

            theta = np.linspace(np.radians(th1), np.radians(th2), resolution)
            points = np.vstack((self.radius*np.cos(theta) + center[0], self.radius*np.sin(theta) + center[1]))
            # build the polygon and add it to the axes
            poly = mpatches.Polygon(points.T, closed=True, **kwargs)
            ax.add_patch(poly)
            return poly

    @property
    def connected_edges(self):
        """The edges connected to nodes a and b"""
        edges_a = [e for e in self.node_a.edges if e is not self]
        # edges_b = [[i, e] for i, e in enumerate(self.node_b.edges) if e is not self]
        edges_b = [e for e in self.node_b.edges if e is not self]
        return edges_a, edges_b

    @property
    def nodes(self):
        """
        Set of nodes that comprise this edge
        """
        return set((self.node_a, self.node_b))

    def edge_angle(self, other_edge):
        """What is the angle between this edge and another edge connected
        at a node?
        """
        # find the common node between the two edges
        try:
            common_node = self.nodes.intersection(other_edge.nodes).pop()
            this_other_node = self.nodes.difference([common_node]).pop()
            other_other_node = other_edge.nodes.difference([common_node]).pop()
            # find edge vectors from common node
            this_vec = np.subtract(this_other_node.loc, common_node.loc)
            other_vec = np.subtract(other_other_node.loc, common_node.loc)

            return np.degrees(np.math.atan2(np.linalg.det([this_vec, other_vec]), np.dot(this_vec, other_vec)))
        except:
            return []

    def unit_vectors(self):
        """What are the unit vectors of the angles the edge makes as it goes 
        into nodes A and B? Unlike _edge_angle, this accounts for the curvature 
        of the edge.
        """
        a, b = self.node_a.loc, self.node_b.loc

        if self.radius is not None:
            if self.xc is None or self.yc is None:
                center = self._circle_arc_center(a, b, self.radius)
            else:
                center = self.xc, self.yc
            perp_a = np.array((a[1]-center[1], -(a[0]-center[0])))
            perp_a = perp_a/np.linalg.norm(perp_a)
            perp_b = np.array((-(b[1]-center[1]), b[0]-center[0]))
            perp_b = perp_b/np.linalg.norm(perp_b)
        else:
            center = 0.5*np.subtract(b, a)+a  # midpoint
            perp_a = np.array((a[0]-center[0], (a[1]-center[1])))
            np.seterr(divide='ignore', invalid='ignore')
            perp_a = perp_a/np.linalg.norm(perp_a)
            perp_b = np.array(((b[0]-center[0]), b[1]-center[1]))
            np.seterr(divide='ignore', invalid='ignore')
            perp_b = perp_b/np.linalg.norm(perp_b)
        return perp_a, perp_b

    def convex_concave(self, cell1, cell2):
        """
        Is the edge convex with respect to cell1 or cell2? 
        ----------
        Parameters
        ----------
        cell1, cell2

        Returns which of the 2 cells the edge (self) is curving out of
        """

        centroid_cell1 = list(cell1.centroid())
        centroid_cell2 = list(cell2.centroid())

        distance1 = math.sqrt(((centroid_cell1[0]-self.center_of_circle[0])**2) +
                              ((centroid_cell1[1]-self.center_of_circle[1])**2))
        distance2 = math.sqrt(((centroid_cell2[0]-self.center_of_circle[0])**2) +
                              ((centroid_cell2[1]-self.center_of_circle[1])**2))

        if distance1 > distance2:
            return cell2
        elif distance2 > distance1:
            return cell1
        else:
            print('Warning: Adjacent cells are equidistant')

    def which_cell(self, list_of_edges, ty, max_iter):
        """
        Find which cell the current edge is a part of. 
        Algorithm starts from node_a and looks for a cycle that reaches node_b

        Parameters
        ----------
        List_of_edges - List of all the edges in the colony
        ty -- 0 for choosing only the minimum positive angles
             -- 1 for choosing only the maximum negative angles
        Return -
        ----------
        Cell -> smallest cell that self is a part of    
        """

        cells = []
        # Find connected edges to current edge at node_a
        con_edges0 = self.connected_edges[0]
        # Find angles of these connected edges
        angles1 = [self.edge_angle(e2) for e2 in con_edges0]
        # Find either a max negative angle or min positive angle
        if ty == 1:
            angle_node0 = max([n for n in angles1 if n < 0], default=9000)
        else:
            # min positive number from node 0
            angle_node0 = min([n for n in angles1 if n > 0], default=9000)
        # Check that its not 9000
        if angle_node0 != 9000:
            # find edge corresponding to the smallest angle
            edge1 = [e for e in con_edges0 if self.edge_angle(e) == angle_node0]
            # Find common and other node
            common_node = edge1[0].nodes.intersection(self.nodes).pop()
            other_node = edge1[0].nodes.difference([common_node]).pop()
            # Make sure other node not self.node_b because then its a 2 edge cell
            if other_node != self.node_b:
                # Make a list of all nodes and edges found so far
                cell_nodes = [self.node_b, common_node, other_node]
                cell_edges = [self, edge1[0]]
                # Call recursion algorithm
                p = 1
                cells = self.recursive_cycle_finder([self], edge1, ty, cell_nodes, cell_edges, p, max_iter)
            else:
                # two edge cell
                cells = cell(list(self.nodes), [edge, edge1])

        return cells

    def recursive_cycle_finder(self, edge1, edge2, ty, cell_nodes, cell_edges, p, max_iter):
        """
        Apply a recursion algorithm until we find a cycle 
        This function is called by which_cell() 
        ----------
        Parameters
        ----------
        edge1 and edge2 - 2 edges connected by a common node
        ty - 0 or 1 - decides which direction to search (0 for max negative angle, 1 for min positive)
        cell_nodes - list of current cell nodes
        cell_edges - list of current cell edges
        p - A count of iterations - only for use in this function
        max_iter - A specified value of max iterations to be performed after which we stop recursion
        """
        cells = []

        # Set final node
        final_node = self.node_b
        if p > max_iter:
            print('Warning: Cycle iterator going past max allowed iterations')
            return []
        else:
            # find the index of common and non common nodes between edge1 and edge2
            common_node = edge1[0].nodes.intersection(edge2[0].nodes).pop()
            other_node = edge2[0].nodes.difference([common_node]).pop()
            # Check whether node_a or node_b in edge2 corresponds to other_node
            if edge2[0].node_a == other_node:
                i = 0
            else:
                i = 1
            # Find connected edges to edge2 at node_a
            con_edges0 = edge2[0].connected_edges[i]
            # Find angles of those edges
            angles1 = [edge2[0].edge_angle(e2) for e2 in con_edges0]

            # Find max negative or min positive angle
            if ty == 1:
                angle_node0 = max([n for n in angles1 if n < 0], default=9000)
            else:
                # min positive number from node 0
                angle_node0 = min([n for n in angles1 if n > 0], default=9000)

            # BEGIN BLOCK TO USE IF ANY CYCLES MISSED

            # if all(i < 0 for i in angles1) and ty == 0 and len(angles1) != 0:
            #     if angles1[0] == -177.92477691580484:
            #         angle_node0 = angles1[0]
            #     else:
            #         angle_node0 = angles1[1]

            # if all(i > 0 for i in angles1) and ty == 1 and len(angles1) != 0:
            #     angle_node0 = angles1[1]

            # END BLOCK TO USE IF ANY CYCLES MISSED

            # CHeck that its not 9000, if it is, stop recursion - no cells
            if angle_node0 != 9000:
                # find edge corresponding to the angle 
                edge3 = [e for e in con_edges0 if edge2[0].edge_angle(e) == angle_node0]
                # Fine index of common and non-common node between the new edge (edge3) and 
                # its connected edge (edge2)
                common_node = edge3[0].nodes.intersection(edge2[0].nodes).pop()
                other_node = edge3[0].nodes.difference([common_node]).pop()

                # check if non-common node is final node
                if other_node == final_node:
                    # found a cell
                    cell_edges.append(edge3[0])
                    cells = cell(cell_nodes, cell_edges)

                    return cells
                else:
                    # Add other_node (the only new node) to the list of cell_nodes
                    p = p + 1

                    cell_nodes.append(other_node)
                    # Add edge3 (the only new edge) to the list of cell_edges
                    cell_edges.append(edge3[0])

                    # Call the function again with the edge2 and edge3. Repeat until cycle found
                    cells = self.recursive_cycle_finder(edge2, edge3, ty, cell_nodes, cell_edges, p, max_iter)

        return cells
        

class cell:
    def __init__(self, nodes, edges):
        """
        Parameters
        ----------
        nodes: list of nodes
            Nodes that make up vertices of the cell
        edges: list of edges
            Directed edges that compose the cell
        ---------------
        Notes -> 
        ---------------
        Calling cell automatically makes the following changes to the edge class
        (1) Add this cell to every edge in self.edges
        """
        self.nodes = nodes
        self.edges = edges
        self._colony_cell = []
        self._pressure = []
        self._guess_pressure = []
        self._label = []
        self._ground_truth_pressure = []

    def plot(self, ax, **kwargs):

        """Plot the cell on a given axis"""
        [e.plot(ax, **kwargs) for e in self.edges]
        [n.plot(ax, **kwargs) for n in self.nodes]

    @property
    def colony_cell(self):
        """
        The colony that this cell is a part of (only one colony, this is not really useful)
        """
        return self._colony_cell

    @colony_cell.setter
    def colony_cell(self, col):
        if col not in self._colony_cell:
            self._colony_cell.append(col)

    def perimeter(self):
        """
        Perimeter of this cell (calculated as sum of straight lengths of every edge, need to add curvature of this edge)
        """
        return sum([e.straight_length for e in self.edges])

    def centroid(self):
        """
        Centroid of this cell, calculated as a mean of co-ordinates of all nodes that make up this cell
        """
        x = [n.loc[0] for n in self.nodes]
        y = [n.loc[1] for n in self.nodes]
        return (np.mean(x), np.mean(y))

    def area(self):
        """
        Calculate area of this cell, got this online
        """
        vertices = [n.loc for n in self.nodes]
        n = len(vertices)  # of corners
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += abs(vertices[i][0] * vertices[j][1]-vertices[j][0] * vertices[i][1])
        result = a / 2.0
        return result
    
    @property
    def pressure(self):
        """
        Pressure of this cell
        """
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        self._pressure = pres

    @property
    def ground_truth_pressure(self):
        return self._ground_truth_pressure

    @ground_truth_pressure.setter
    def ground_truth_pressure(self, ground_truth_pressure):
        self._ground_truth_pressure = ground_truth_pressure
    
    @property
    def label(self):
        """
        Assign a label to this cell to track it over time
        """
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

    @property
    def guess_pressure(self):
        """
        Guess pressure used as an initial condition during optimization
        """
        return self._guess_pressure

    @guess_pressure.setter
    def guess_pressure(self, guess_pres):
        self._guess_pressure = guess_pres


class colony:
    def __init__(self, cells, edges, nodes):
        """
        Parameters
        ----------
        cells: list of cells
        edges: total list of edges (including those not part of cells)
        nodes: total list of nodes (including those not part of cells)
        ---------------
        Notes ->
        ---------------
        Calling colony automatically makes the following changes to the cell class
        (1) Adds colony to all cells in self.cells1
        """
        self.cells = cells
        self.tot_edges = edges 
        self.tot_nodes = nodes
        self._dictionary = {}
        self._tension_matrix = []
        self._pressure_matrix = []
        self._pressure_rhs = []

        for cc in cells:
            cc.colony_cell = self

    def plot(self, ax):
        """
        plot the colony on a given axis
        """
        [e.plot(ax) for e in self.cells]

    def add_cell(self, c):
        """
        Add a cell to the colony, dont really use this anywhere
        """
        self.cells.append(c)

    @property
    def tension_matrix(self):
        """
        Store the calculated tension matrix as a property
        """
        return self._tension_matrix

    @tension_matrix.setter
    def tension_matrix(self, a):
        self._tension_matrix = a

    @property
    def pressure_matrix(self):
        """
        Store the calculated pressure matrix as a property
        """
        return self._pressure_matrix

    @pressure_matrix.setter
    def pressure_matrix(self, b):
        self._pressure_matrix = b

    @property
    def pressure_rhs(self):
        """
        Save the RHS of pressure equation as a property
        """
        return self._pressure_rhs

    @pressure_rhs.setter
    def pressure_rhs(self, rhs):
        self._pressure_rhs = rhs
    
    @property
    def dictionary(self):
        """
        Dictionary of the form {node label: edges connected to node label: edge vector with horizontal}
        """
        return self._dictionary
    
    @dictionary.setter
    def dictionary(self, dic):
        self._dictionary = dic

    @property
    def edges(self):
        """
        Edges is not the total list of edges - that is tot_edges. This gives list of edges that make up cells
        """
        edges = []
        [edges.append(x) for cc in self.cells for x in cc.edges if x not in edges]
        return edges

    @property
    def nodes(self):
        """
        Nodes is not the total list of nodes - that is tot_nodes. This gives list of nodes that make up cells
        """
        nodes = []
        [nodes.append(x) for cc in self.cells for ee in cc.edges for x in ee.nodes if x not in nodes]
        return nodes

    @staticmethod
    def solve_constrained_lsq(a, t_or_p, b=None):
        """
        Solve constrained least square system PX = Q. 

        Parameters
        ----------
        A is an M * N matrix comprising M equations and N unknowns 
        B is an N * 1 matrix comprising RHS of the equations
        Type specifies the Lagrange multiplier we use - 0 or 1
        For type 0 -> (For tension)
        Define a matrix P = [[A^TA, C^T],[C, 0]]  - > (N + 1) * (N + 1) matrix
                        Q = [0..., N]  -> (N + 1) * 1 matrix
        For type 1 -> (For pressure)
        Define a matrix P = [[A^TA, C^T],[C, 0]]  - > (N + 1) * (N + 1) matrix
                        Q = [B, 0]  -> (N + 1) * 1 matrix
        """
        
        # Get N - number of columns
        n = np.shape(a)[1]

        # Define matrix of ones
        p = np.ones((n + 1, n + 1))

        # Get A^TA
        a1 = np.dot(a.T, a)

        # Plug this into appropriate place in P 
        p[0:n, 0:n] = a1

        # Set last element in N,N position to 0 
        p[n, n] = 0

        if t_or_p == 0:
            # Define Q
            q = np.zeros((n+1, 1))

            # Set last element to N (number of edges) 
            # Effectively says average value is 1
            q[n, 0] = n

        if t_or_p == 1:
            # Define Q
            q = np.zeros((n+1, 1))
            c = np.dot(a.T, b)
            c = c.reshape((len(c), ))

            if len(c) >= len(q[0:n,0]):
                q[0:n, 0] = c[0:n]
            else:
                q[0:len(c), 0] = c[0:len(c)]
            # Effectively says average is 0 
            q[n, 0] = 0

        # Solve PX = Q
        try:
            # By QR decomposition
            r1, r2 = linalg.qr(p)  # QR decomposition with qr function
            y = np.dot(r1.T, q)  # Let y=R1'.Q using matrix multiplication
            x = linalg.solve(r2, y)  # Solve Rx=y

            # By least squares - gives same result
            # x = linalg.lstsq(R2, y)

            # By LU decomposition - Both give same results        
            # L, U = scipy.linalg.lu_factor(P)
            # x = scipy.linalg.lu_solve((L, U), Q)

        except np.linalg.LinAlgError as err:
            if 'Matrix is singular' in str(err):
                return None, p
            else:
                raise

        return x[0:n][:, 0], p  # use this if solved using linalg.solve
        # return x[0][0:N], P # use this if solved using linalg.lstsq

    def make_tension_matrix(self, nodes=None, edges=None):
        """
        Makes a tension matrix A 
        A is the coefficient vectors of all the edges coming into a node
        A is m * n matrix. 
        m is number of equations, which is 2 * number
        of nodes that have atleast 3 edges coming into it (2 * because both horizontal
        and vertical force balance). 
        n is number of edges. This includes stray edges (not part of a cell). Called
        by self.tot_edges
        """
        # get the list of nodes and edges in the colony
        # nodes = self.nodes
        if nodes is None:
            nodes = self.tot_nodes

        if edges is None:
            edges = self.tot_edges
        # change above to self.nodes and self.edges to plot tensions only in cells found

        # We want to solve for AX = 0 where A is the coefficient matrix - 
        # A is m * n and X is n * 1 where n is the number of the edges
        # m can be more than n
        # we initialize A as n * m (n*1 plus appends as we loop) because I couldnt figure out how to append rows
        a = np.zeros((len(edges), 1))

        for node in nodes:
            # only add a force balance if more than 2 edge connected to a node
            if len(node.edges) > 2:
                # create a temporary list of zeros
                temp = np.zeros((len(edges), 1))

                # node.edges should give the same edge as the edge
                # corresponding to node.tension_vectors since they are added together
                # only want to add indices of edges that are part of colony edge list
                # indices = np.array([edges.index(x) for x in node.edges if x in edges if x.radius is not None])
                # Use this for the Networkx plot
                indices = np.array([edges.index(x) for x in node.edges if x in edges])
                node.edge_indices = indices

                # similarly, only want to consider horizontal vectors that are a part of the colony edge list
                # horizontal_vectors = np.array([x[0] for x in node.tension_vectors
                # if node.edges[node.tension_vectors.index(x)] in edges if
                # node.edges[node.tension_vectors.index(x)].radius is not None])[np.newaxis]
                # Use this for networkx plot
                horizontal_vectors = np.array([x[0] for x in node.tension_vectors if
                                               node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                
                node.horizontal_vectors = horizontal_vectors[0]
                # add the horizontal vectors to the corresponding indices in temp
                temp[indices] = horizontal_vectors.T

                # append this list to A. This column now has the
                # horizontal force balance information for the node being looped
                a = np.append(a, temp, axis=1)

                # repeat the process for the vertical force balance
                temp = np.zeros((len(edges), 1))
                vertical_vectors = np.array([x[1] for x in node.tension_vectors if
                                             node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]

                node.vertical_vectors = vertical_vectors[0]

                temp[indices] = vertical_vectors.T
                a = np.append(a, temp, axis=1)

        # A is the coefficient matrix that contains all horizontal and vertical force balance information of all nodes.
        # its generally overdetermined and homogenous
        # Transpose the matrix because we want it of the form AX = 0
        # where A is m * n and X is n * 1 and n is number of edges
        a = a.T
        a = np.delete(a, (0), axis=0)

        self.tension_matrix = a
        return a

    def calculate_tension(self, nodes=None, edges=None, solver=None, **kwargs):
        """
        Calls a solver to calculate tension. Cellfit paper used 
        (1) self.solve_constrained_lsq
        We use 
        (2) self.scipy_opt_minimize
        This optimization is slower but allows us to set initial conditions, bounds, constraints
        """

        if nodes is None:
            nodes = self.tot_nodes
        if edges is None:
            edges = self.tot_edges

        # Solver used in CellFIT paper
       
        if solver == 'KKT' or solver == 'CellFIT':
            a_mat = self.make_tension_matrix(nodes, edges)
            tensions, p = self.solve_constrained_lsq(a_mat, 0, None)

        # Call DLITE optimizer
        if solver is None or solver == 'DLITE':
            a_mat = self.make_tension_matrix(nodes, edges)
            sol = self.scipy_opt_minimze(edges, **kwargs)

            tensions = sol.x

            # No Output KKT matrix in DLITE
            p = []

        # Add tensions to edge
        for j, e in enumerate(edges):
            e.tension = tensions[j]

        mean_ten = np.mean([e.tension for e in edges])

        for j, nod in enumerate(nodes):
            n_vec = nod.tension_vectors
            n_ten = [e.tension for e in nod.edges]/mean_ten
            residual = 0
            for a, b in zip(n_ten, n_vec):
                residual = residual + np.array(a) * np.array(b)
            nod.residual_vector = residual

        return tensions, p, a_mat

    def scipy_opt_minimze(self, edges, i=[0], **kwargs):
        """
        Calls minimize function from scipy optimize. 
        Parameters:
        ----------------------
        edges - really either edges (for tension calculation)
        or cells (for pressure calculation). Just give the solver a list of 
        variables and it will give a solution
        """

        i[0] += 1  # mutable variable get evaluated ONCE

        bnds = self.make_bounds(edges)

        x0 = self.initial_conditions(edges)

        if type(edges[0]) == edge:
            # Tension calculations
            # Equality constraint that can be used in the SLSQP optimizer
            cons = [{'type': 'eq', 'fun' : self.equality_constraint_tension}]

            # Check if the first element is empty
            if not edges[0].guess_tension:
                pass
            else:
                # Assign constant initial values if needed

                # for k, xxx in enumerate(x0):
                #     x0[k] = 0.2

                # If all elements are the same, run basinhopping with random initial guess
                if x0.count(x0[0]) == len(x0):

                    # Assign random initial guesses in range 0-1 upto 2 digits
                    for k, xxx in enumerate(x0):
                        x0[k] = random.randint(0, 101)/100

                    # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds} # used only BFGS and no bounds before
                    # sol = basinhopping(self.objective_function_tension, x0, T=0.5, interval=10,
                    #                    minimizer_kwargs=minimizer_kwargs, niter=100, stepsize=0.05, disp=True)

                    # Or, choose to run L-BFGS
                    sol = minimize(self.objective_function_tension, x0,
                                   method='L-BFGS-B', bounds=bnds, options={**kwargs}, tol=1e-9)

                    # Or, choose to run constrained SLSQP solver, similar to CellFIT
                    # sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', constraints = cons)
                else:
                    # Can choose to always have a random initial guess
                    # for k, xxx in enumerate(x0):
                    #     x0[k] = random.randint(0,101)/100

                    # Run L-BFGS
                    sol = minimize(self.objective_function_tension, x0, method='L-BFGS-B',
                                   bounds=bnds, options={**kwargs})

                    # Or, Run SLSQP
                    # sol = minimize(self.objective_function_tension, x0, method='SLSQP', constraints=cons)
        else:
            # Pressure calculations
            # Equality constraint for SLSQP optimizer
            cons = [{'type': 'eq', 'fun':self.equality_constraint_pressure}]

            # Check for empty element
            if not edges[0].guess_pressure and edges[0].guess_pressure != 0:
                pass
            else:
                # Assign constant initial guesses if needed
                # for k, xxx in enumerate(x0):
                #     x0[k] = 0.001
                if x0.count(x0[0]) == len(x0):

                    # If all initial guesses are the same, use random initial guesses
                    # for k, xxx in enumerate(x0):
                    #     x0[k] = random.randint(0,101)/10000

                    # minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : bnds}
                    # sol = basinhopping(self.objective_function_pressure, x0,
                    #                    minimizer_kwargs=minimizer_kwargs, niter=50, disp=True)

                    # Or, run L-BFGSB
                    sol = minimize(self.objective_function_pressure,
                                   x0, method='L-BFGS-B', bounds=bnds, options={**kwargs})

                    # Or, run SLSQP
                    # sol = minimize(self.objective_function_pressure,
                    # x0, method = 'SLSQP', constraints = cons)

                else:
                    # i[0] says how many times this function has been called.
                    # If it is 2, this is the first pressure iteration. Use basinhopping
                    if i[0] == 2:
                        # Run Basinhopping
                        # minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bnds}
                        # sol = basinhopping(self.objective_function_pressure, x0,
                        #                    minimizer_kwargs=minimizer_kwargs, niter=50, disp=True)

                        # Or, run SLSQP
                        # sol = minimize(self.objective_function_pressure, x0, method='SLSQP', constraints=cons)

                        # Or, run L-BFGSB
                        sol = minimize(self.objective_function_pressure, x0, method='L-BFGS-B',
                                       bounds=bnds, options={**kwargs})
                    else:
                        # Run L-BFGSB
                        sol = minimize(self.objective_function_pressure, x0,
                                       method='L-BFGS-B', bounds=bnds, options={**kwargs})

                        # Or, run SLSQP
                        # sol = minimize(self.objective_function_pressure,
                        #                x0, method='SLSQP', constraints=cons)

        print('Function value', sol.fun)
        print('Solution', sol.x)
        print('\n')
        print('-----------------------------')
        return sol 

    def make_bounds(self, edge_or_cell):
        """
        Define bounds. Possible to be based on the initial guess of tension or pressure
        Parameters
        ---------------------
        edge_or_cell - list of edges or cells (variables)
        """
        b = []
        tol_perc = 0.5

        for j, e_or_c in enumerate(edge_or_cell):
            if type(e_or_c) == edge:
                if not e_or_c.guess_tension:
                    b.append((0, np.inf))
                else:
                    # Possible to add tolerance to bounds based on previous initial guess
                    tolerance = e_or_c.guess_tension * tol_perc
                    b.append((0, np.inf))
            else:
                if not e_or_c.guess_pressure:
                    b.append((-np.inf, np.inf))
                else:
                    tolerance = e_or_c.guess_pressure * tol_perc
                    b.append((-np.inf, np.inf))
                    # b.append((e_or_c.guess_pressure - tolerance, e_or_c.guess_pressure + tolerance))   
        return tuple(b)

    def initial_conditions(self, edge_or_cell):
        """
        Define initial conditions based on previous solution for tension/pressure
        Parameters 
        ------------------
        edge_or_cell - list of edge or cells (variables)
        """

        initial_gues = []

        for j, e_or_c in enumerate(edge_or_cell):
            if type(e_or_c) == edge:
                if not e_or_c.guess_tension:
                    # If no guess, we assign guess based on surrounding edge tension guesses
                    node_a, node_b = e_or_c.node_a, e_or_c.node_b
                    tensions_a = [e.guess_tension for e in node_a.edges if e.guess_tension != []]
                    tensions_b = [e.guess_tension for e in node_b.edges if e.guess_tension != []]
                    guess_a = np.mean(tensions_a) if tensions_a != [] else 0
                    guess_b = np.mean(tensions_b) if tensions_b != [] else 0

                    guess = (guess_a + guess_b)/2 if guess_a != 0 and guess_b != 0 else\
                            guess_a if guess_b == 0 and guess_a != 0 else\
                            guess_b if guess_a == 0 and guess_b != 0 else 0.2
                    initial_gues.append(guess)
                    # Update the guess tension used
                    e_or_c.guess_tension = initial_gues[j]
                else:
                    initial_gues.append(e_or_c.guess_tension)
                
            else:
                if not e_or_c.guess_pressure:
                    adj_press = self.get_adjacent_pressures(e_or_c)
                    guess = np.mean(adj_press) if adj_press != [] else 0

                    edges_in_this_cell = [e for e in e_or_c.edges]
                    tension_of_edges = [e.tension for e in edges_in_this_cell]
                    radii_of_edges = [e.radius for e in edges_in_this_cell]
                    ratio_of_tension_to_radius = [x/y for x, y in zip(tension_of_edges, radii_of_edges)]
                    guess2 = np.mean(ratio_of_tension_to_radius)

                    [initial_gues.append(guess) if guess != 0 else initial_gues.append(guess2)]
                    e_or_c.guess_pressure = initial_gues[j]
                else:
                    initial_gues.append(e_or_c.guess_pressure)

        return initial_gues

    def get_adjacent_pressures(self, e_or_c):
        """
        Get a list of adjacent pressures of adjacent tensions if there is no initial guess for this edge tension
        or cell pressure
        """
        adj_press = []
        for ee in e_or_c.edges:
            adj_cell = [c for c in ee.cells if c != e_or_c]
            if adj_cell:
                if adj_cell[0].guess_pressure:
                    adj_press.append(adj_cell[0].guess_pressure)
        return adj_press

    def objective_function_tension(self, x):
        """
        Main objective function to be minimized in the tension calculation
        i.e sum(row^2) for every row in tension matrix A + a regularizer based on Tikhonov regularization
        for overdetermined problems
        see - https://epubs.siam.org/doi/pdf/10.1137/050624418
        """
        objective = 0

        for node in self.tot_nodes:
            if len(node.edges) > 2:
                indices = node.edge_indices

                starting_tensions = []
                previous_tensions = []
                resid_mag_old = 0

                node_vecs = node.tension_vectors

                for i in range(len(indices)):
                    starting_tensions.append([x[indices[i]]])

                # Use previous node labels if needed

                # if all(e.label == e.previous_label for e in node.edges):
                #     previous_tensions.append(e.guess_tension)
                #     previous_vecs.append(e.previous_tension_vectors)
                #
                # if len(previous_tensions) > 0:
                #     previous_tensions = np.array(previous_tensions)
                #     old_tension_vecs = np.multiply(previous_vecs, previous_tensions)
                #     previous_vec_mags = [np.hypot(*vec) for vec in old_tenson_vecs]
                #     resid_vec_old = np.sum(old_tension_vecs, 0)
                #     resid_mag_old = np.hypot(*resid_vec_old)

                starting_tensions = np.array(starting_tensions)
                tension_vecs = np.multiply(node_vecs, starting_tensions)

                tension_vec_mags = [np.hypot(*vec) for vec in tension_vecs]
                resid_vec = np.sum(tension_vecs, 0)
                resid_mag = np.hypot(*resid_vec)

                coeff, hyper, coeff2 = 1, 0, 0

                if len(previous_tensions) > 0:
                    if len(node.velocity_vector) == 0:

                        objective = objective + resid_mag + coeff * resid_mag/np.sum(tension_vec_mags) + \
                                    coeff2 * np.exp(hyper*(resid_mag - resid_mag_old))

                    else:
                        objective = objective + resid_mag + coeff * resid_mag/np.sum(tension_vec_mags) + \
                                    coeff2 * np.exp(hyper*(resid_mag - resid_mag_old))
                else:
                    if len(node.velocity_vector ) == 0:
                        objective = objective + resid_mag + coeff * resid_mag/np.sum(tension_vec_mags) 

                    else:
                        objective = objective + resid_mag + coeff * resid_mag/np.sum(tension_vec_mags)

        return objective 

    def objective_function_pressure(self, x):
        """
        Main objective function to be minimzed in the pressure calculation 
        i.e sum((row - rhs)^2). We need rhs here because in the pressure case
        rhs is not 0.
        """
        objective = 0
        for e in self.edges:
            if len(e.cells) == 2:
                cell_indices = e.cell_indices
                cell_values = e.cell_coefficients
                cell_rhs = e.cell_rhs
                edge_pressure_lhs = 0

                for i in range(len(cell_indices)):
                    edge_pressure_lhs = edge_pressure_lhs + cell_values[i]*x[cell_indices[i]]

                objective = objective + (edge_pressure_lhs - cell_rhs)**2

        return objective 

    def equality_constraint_tension(self, x):
        """
        Assigns equality constraint - i.e mean of tensions = 1
        """
        a = self.tension_matrix

        num_of_edges = len(a[0, :])
        constraint = 0
        for i in range(num_of_edges):
            constraint = constraint + x[i]
        return constraint - num_of_edges    

    def equality_constraint_pressure(self, x):
        """
        Assigns equality constraint - i.e mean of pressures = 0
        """
        a = self.pressure_matrix

        num_of_cells = len(a[0, :])
        constraint = 0
        for i in range(num_of_cells):
            constraint = constraint + x[i]
        return constraint

    def remove_outliers(self, bad_tensions, tensions):
        """
        Removing those edges that are giving tensions more than 3 std away from mean value
        -------
        Parameters
        -------
        bad_tensions - values of tensions that are more than 3 std away
        tensions - list of all edge tensions
        """
        nodes = self.tot_nodes
        edges = self.tot_edges
        # Find indices of bad tensions
        indices = [np.where(tensions - bad_tensions[i] == 0)[0][0] for i in range(len(bad_tensions))]

        bad_edges = [edges[i] for i in indices]
        for ed in bad_edges:
            ed.kill_edge(ed.node_a)
            ed.kill_edge(ed.node_b)
            if len(ed.node_a.edges) == 0:
                if ed.node_a in nodes:
                    nodes.remove(ed.node_a)
            if len(ed.node_b.edges) == 0:
                if ed.node_b in nodes:
                    nodes.remove(ed.node_b)
            edges.remove(ed)

        return nodes, edges 

    def make_pressure_matrix(self):
        """
        Make pressure matrix A and rhs matrix (AX = rhs)
        A is m * n matrix
        m is number of equations - equals number of edges that have 2 cells on either side
        n is number of cells 
        rhs is m * 1 matrix - each element is tension/radius
        """

        a_mat = np.zeros((len(self.cells), 1))
        list_of_edges = []

        # of the form tension/radius
        rhs = []

        for e in self.edges:
            if e not in list_of_edges:
                if len(e.cells) == 2:

                    cell1 = e.cells[0]
                    cell2 = e.cells[1]

                    c_edges = [e for e in set(cell1.edges).intersection(set(cell2.edges))]
                    indices = []
                    indices.append(self.cells.index(cell1))
                    indices.append(self.cells.index(cell2))

                    e.cell_indices = indices

                    e.cell_coefficients = np.array([1, -1])

                    temp = np.zeros((len(self.cells), 1))

                    for j, i in enumerate(indices):
                        # here we assign +1 to cell (cell1) and -1 to cell (cell2)
                        temp[i] = e.cell_coefficients[j]

                    a_mat = np.append(a_mat, temp, axis=1)

                    convex_cell = e.convex_concave(cell1, cell2)

                    if convex_cell == cell1:
                        if e.radius is not None:
                            if e.tension is not []:
                                rhs.append(e.tension/e.radius)
                                e.cell_rhs = e.tension/e.radius
                        else:
                            # Radius is None
                            rhs.append(0)
                            e.cell_rhs = 0

                    elif convex_cell == cell2:
                        if e.radius is not None:
                            rhs.append(np.negative(e.tension/e.radius))
                            e.cell_rhs = np.negative(e.tension/e.radius)
                        else:
                            # Radius is None
                            rhs.append(0)
                            e.cell_rhs = 0
                    else:
                        # Cells are equidistant
                        if e.radius is not None:
                            if e.tension is not []:
                                rhs.append(e.tension/e.radius)
                                e.cell_rhs = e.tension/e.radius
                        else: 
                            rhs.append(0)
                            e.cell_rhs = 0

                    list_of_edges.append(e)

        a_mat = a_mat.T
        a_mat = np.delete(a_mat, (0), axis=0)
        rhs = np.array(rhs)

        self.pressure_matrix = a_mat
        self.pressure_rhs = rhs

        return a_mat, rhs

    def calculate_pressure(self, solver=None, **kwargs):
        """
        Calculate pressure using calculated tensions and edge curvatures (radii). 
        Pressure is unique to every cell
        """
        a_mat, rhs = self.make_pressure_matrix()

        # Old solver
        if solver == 'KKT' or solver == 'CellFIT':
            pressures, p = self.solve_constrained_lsq(a_mat, 1, rhs)

        # New solver
        cells = self.cells

        if not solver or solver == 'DLITE':
            sol = self.scipy_opt_minimze(cells, **kwargs)
            pressures = sol.x
            p = []

        if solver == 'KKT' or solver == 'CellFIT':
            for j, c in enumerate(self.cells):
                c.pressure = pressures[j]
        else:
            mean_pres = np.mean(pressures)
            for j, c in enumerate(self.cells):
                c.pressure = pressures[j]/mean_pres - 1
            # cell.pressure = pressures[j]

        return pressures, p, a_mat

    def plot_tensions(self, ax, fig, tensions, min_x=None, max_x=None, min_y=None, max_y=None,
                      min_ten=None, max_ten=None, specify_color=None, type=None, cbar='yes', **kwargs):
        """
        Plot normalized tensions (min, width) with colorbar
        """

        edges = self.tot_edges

        ax.set(xlim=[min_x, max_x], ylim=[min_y, max_y], aspect=1)
        if type == 'surface_evolver_cellfit':
            ax.set(xlim=[200, 800], ylim=[200, 750], aspect=1)

        def norm(tensions, min_ten=None, max_ten=None):
            if not min_ten and not max_ten:
                return (tensions - min(tensions)) / float(max(tensions) - min(tensions))
            else:
                norm_tens = [(a - min_ten)/(max_ten - min_ten) for a in tensions]
                return norm_tens

        c1 = norm(tensions, min_ten, max_ten)
        # Plot tensions

        for j, an_edge in enumerate(edges):
            if specify_color is not None:
                an_edge.plot(ax, ec=cm.jet(c1[j]), **kwargs)
            else:
                an_edge.plot(ax, ec=cm.viridis(c1[j]), **kwargs)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min_ten, vmax=max_ten))
            # sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=min_ten, vmax=max_ten))
            # sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))

        # fake up the array of the scalar mappable. 
        sm._A = []
        if cbar == 'yes':
            cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
            cl = plt.colorbar(sm, cax=cbaxes)
            cl.set_label('Normalized tension', fontsize=13, labelpad=-60)

    def plot_pressures(self, ax, fig, pressures, min_pres=None, max_pres=None, specify_color=None, **kwargs):
        """
        Plot normalized pressures (mean, std) with colorbar 
        """
        ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)

        def norm2(pressures, min_pres=None, max_pres=None):
            if not min_pres and not max_pres:
                return (pressures - min(pressures)) / float(max(pressures) - min(pressures))
            else:
                return (pressures - min_pres) / float(max_pres - min_pres)

        c2 = norm2(pressures, min_pres, max_pres)

        # Plot pressures
        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            if specify_color is not None:
                plt.fill(x, y, c=cm.jet(c2[j]), alpha=0.2)
            else:
                plt.fill(x, y, c=cm.viridis(c2[j]), alpha=0.2)

            for e in c.edges:
                if specify_color is not None:
                    e.plot_fill(ax, color=cm.jet(c2[j]), alpha=0.2)
                else:
                    e.plot_fill(ax, color=cm.viridis(c2[j]), alpha=0.2)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-1, vmax=1))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax=cbaxes)
        cl.set_label('Normalized pressure', fontsize=13, labelpad=10)

    def plot(self, ax, fig, tensions, pressures, min_ten=None,
             max_ten=None, min_pres=None, max_pres=None,
             specify_color=None, **kwargs):
        """
        Plot both tensions and pressures on a single axes
        """
        edges = self.tot_edges
        nodes = self.nodes
        ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)

        def norm(tensions, min_ten=None, max_ten=None):
            if not min_ten and not max_ten:
                return (tensions - min(tensions)) / float(max(tensions) - min(tensions))
            else:
                return [(a - min_ten) / float(max_ten - min_ten) for a in tensions]

        def norm2(pressures, min_pres=None, max_pres=None):
            if not min_pres and not max_pres:
                return (pressures - min(pressures)) / float(max(pressures) - min(pressures))
            else:
                return [(p - min_pres) / float(max_pres - min_pres) for p in pressures]

        c1 = norm(tensions, min_ten, max_ten)
        c2 = norm2(pressures, min_pres, max_pres)
        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            if specify_color is not None:
                plt.fill(x, y, c=cm.jet(c2[j]), alpha=0.2)
            else:
                plt.fill(x, y, c=cm.viridis(c2[j]), alpha=0.2)
            for e in c.edges:
                # Plots a filled arc
                if specify_color is not None:
                    e.plot_fill(ax, color=cm.jet(c2[j]), alpha=0.2)
                else:
                    e.plot_fill(ax, color=cm.viridis(c2[j]), alpha=0.2)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-min_pres, vmax=max_pres))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-min_pres, vmax=max_pres))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax=cbaxes)
        cl.set_label('Normalized pressure', fontsize=13, labelpad=10)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            if specify_color is not None:
                an_edge.plot(ax, ec=cm.jet(c1[j]), **kwargs)
            else:
                an_edge.plot(ax, ec=cm.viridis(c1[j]), **kwargs)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=min_ten, vmax=max_ten))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=min_ten, vmax=max_ten))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax=cbaxes)
        cl.set_label('Normalized tension', fontsize=13, labelpad=-60)


if __name__ == '__main__':
    print('Running code!')
