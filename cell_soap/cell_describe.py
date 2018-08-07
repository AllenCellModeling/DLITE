import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from useful_functions import solution
from filled_arc import arc_patch
from scipy import ndimage, optimize
import numpy.linalg as la
import collections
import scipy.linalg as linalg 
from scipy.optimize import minimize
from matplotlib import cm
import os, sys
import matplotlib.patches as mpatches
import matplotlib.animation as manimation
from collections import defaultdict
import pylab
import scipy
#from Dave_cell_find import find_all_cells, cells_on_either_side, trace_cell_cycle

class node:
    def __init__(self, loc):
        """loc is the (x,y) location of the node"""
        self.loc = loc
        self._edges = []
        self._tension_vectors = []
        self._label = []
    
    # def __str__(self):
    #     return "x:%04i, y:%04i"%tuple(self.loc)

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]
    
    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edge):
        if edge not in self._edges:
            self._edges.append(edge)

    def remove_edge(self, edge):

        ind = self._edges.index(edge) 
        self._edges.pop(ind)
        self._tension_vectors.pop(ind)

    @property
    def tension_vectors(self):
        return self._tension_vectors

    @tension_vectors.setter
    def tension_vectors(self, vector):
        if vector not in self._tension_vectors:
            self._tension_vectors.append(vector)

    @property    
    def label(self):
        """
        Give a label to a node so we can track it over time
        """
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def plot(self, ax, **kwargs):
        ax.plot(self.loc[0], self.loc[1], ".", **kwargs)

class edge:
    def __init__(self, node_a, node_b, radius=None, xc = None, yc = None, x_co_ords = None, y_co_ords = None):
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
        self._guess_tension = []

    # def __str__(self):
    #     return "["+"   ->   ".join([str(n) for n in self.nodes])+"]"

    @property
    def co_ordinates(self):
        return self.x_co_ords, self.y_co_ords
    
    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cell):
        if cell not in self._cells:
            self._cells.append(cell)

    def kill_edge(self, node):
        """
        Kill edge for a specified node i.e kill the edge and tension vector 
        associated with the specified node 
        """

        if node == self.node_a:
            self.node_a.remove_edge(self)
        if node == self.node_b:
            self.node_b.remove_edge(self)

    @property
    def tension(self):
        return self._tension

    @tension.setter
    def tension(self, tension):
        self._tension = tension

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
        x0, y0 = 0.5*np.subtract(point2, point1)+point1 # midpoint
        a = 0.5*np.linalg.norm(np.subtract(point2, point1)) # dist to midpoint
        #assert a<radius, "Impossible arc asked for, radius too small"
        if a>radius:
            self.radius = radius + a - radius + 1
            radius = radius + a - radius + 1
        b = np.sqrt(radius**2-a**2) # midpoint to circle center
        xc = x0 + (b*(y0-y1))/a # location of circle center
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
        theta1 = np.rad2deg(np.arctan2(y2-yc, x2-xc)) # starting angle
        theta2 = np.rad2deg(np.arctan2(y1-yc, x1-xc)) # stopping angle
        return (xc, yc), theta1, theta2

    def plot(self, ax, **kwargs):
        a, b = self.node_a, self.node_b
        if self.radius is not None:

            center, th1, th2 = self.arc_translation(a.loc, b.loc, self.radius)
            patch = matplotlib.patches.Arc(center, 2*self.radius, 2*self.radius,
                                           0, th1, th2, **kwargs)
            ax.add_patch(patch)
            # if th1 - th2 > 180:
            #     th2 = th1 + abs(th1 - th2)

            # arc_patch(center, self.radius, th1, th2, ax=ax, fill=True, **kwargs)

        else:

            # ax.plot([a.x, b.x], [a.y, b.y], **kwargs)
             ax.plot([a.x, b.x], [a.y, b.y])

    def plot_fill(self, ax, resolution = 50, **kwargs):
        a, b = self.node_a, self.node_b
        if self.radius is not None:
            center, th1, th2 = self.arc_translation(a.loc, b.loc, self.radius)
            old_th1, old_th2 = th1, th2
            th1, th2 = (th1 + 720) % 360, (th2 + 720) % 360
            if abs(th2 - th1) > 180:
                th2, th1 = old_th2, old_th1
            #print(2, th1, th2)

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
        #edges_b = [[i, e] for i, e in enumerate(self.node_b.edges) if e is not self]       
        edges_b = [e for e in self.node_b.edges if e is not self]
        return edges_a, edges_b

    @property
    def nodes(self):
        return set((self.node_a, self.node_b))    

    def edge_angle(self, other_edge):
        """What is the angle between this edge and another edge connected
        at a node?
        """
        ## find the common node between the two edges
        try:
            common_node = self.nodes.intersection(other_edge.nodes).pop()
            this_other_node = self.nodes.difference([common_node]).pop()
            other_other_node = other_edge.nodes.difference([common_node]).pop()
            # find edge vectors from common node
            this_vec = np.subtract(this_other_node.loc, common_node.loc)
            other_vec = np.subtract(other_other_node.loc, common_node.loc)

            # My method
            # find angles of each 
            cosang = np.dot(this_vec, other_vec)
            #sinang = np.linalg.norm(np.cross(this_vec, other_vec))
            sinang = np.cross(this_vec, other_vec)
            return np.rad2deg(np.arctan2(sinang, cosang))  # maybe use %(2*np.pi)

            # Dave's method
            # this_ang = np.arctan2(this_vec[1], this_vec[0])
            # other_ang = np.arctan2(other_vec[1], other_vec[0]) 
            # angle = (2*np.pi + other_ang - this_ang)%(2*np.pi) # convert to 0->2pi
            # return angle         
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
            center = 0.5*np.subtract(b, a)+a # midpoint
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
        Cell1, Cell2

        Returns which of the 2 cells the edge (self) is curving into
        """

        perp_a, perp_b = self.unit_vectors()

        # lets only focus on node_a

        # find the other edges coming into node_a in cell1 and cell2
        edge1 = [e for e in cell1.edges if e.node_a == self.node_a or e.node_b == self.node_a if e!= self][0]
        edge2 = [e for e in cell2.edges if e.node_a == self.node_a or e.node_b == self.node_a if e!= self][0]

        # find the unit vectors associated with these edges 
        edge1_p_a, edge1_p_b = edge1.unit_vectors()
        edge2_p_a, edge2_p_b = edge2.unit_vectors()

        # choose the correct unit vector in edge1 coming into node_a
        if edge1.node_a == self.node_a:
            edge1_v = edge1_p_a
        else:
            edge1_v = edge1_p_b

        # choose the correct unit vector in edge2 coming into node_a
        if edge2.node_a == self.node_a:
            edge2_v = edge2_p_a
        else:
            edge2_v = edge2_p_b

        # Now we have 3 vectors - perp_a, edge1_v and edge2_v on 3 edges all coming into node_a 
        # to check convexivity, we get the angles between edge1_v and perp_a and between edge2_v and perp_a

        cosang = np.dot(perp_a, edge1_v)
        sinang = np.cross(perp_a, edge1_v)
        angle1 = np.rad2deg(np.arctan2(sinang, cosang)) 

        cosang = np.dot(perp_a, edge2_v)
        sinang = np.cross(perp_a, edge2_v)
        angle2 = np.rad2deg(np.arctan2(sinang, cosang))

        # the one with the larger angle difference should be the more convex cell

        if abs(angle1) > abs(angle2):
            return cell1
        else:
            return cell2 

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
        cell_nodes = []
        cell_edges = []

        # Find connected edges to current edge at node_a

        con_edges0 = self.connected_edges[0]

        # Find angles of these connected edges
        angles1 = [self.edge_angle(e2)  for e2 in con_edges0]

        # Find either a max negative angle or min positive angle
        if ty == 1:
            angle_node0 = max([n for n in angles1 if n<0], default = 9000)
        else:
            # min positive number from node 0
            angle_node0 = min([n for n in angles1 if n>0], default = 9000)    

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
                cells = self.recursive_cycle_finder([self], edge1, 1, cell_nodes, cell_edges, p, max_iter)
            
            else:
                # two edge cell
                cells = cell(list(self.nodes),[edge, edge1])

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
            return []
        else:
            # find the index of common and non common nodes between edge1 and edge2
            common_node = edge1[0].nodes.intersection(edge2[0].nodes).pop()
            other_node = edge2[0].nodes.difference([common_node]).pop()

            # = list(edge1[0].nodes).index(other_node) # has error because set reorders things
            # Check whether node_a or node_b in edge2 corresponds to other_node
            if edge2[0].node_a == other_node:
                i = 0
            else:
                i = 1

            # Find connected edges to edge2 at node_a
            con_edges0 = edge2[0].connected_edges[i]

            # Find angles of those edges
            angles1 = [edge2[0].edge_angle(e2)  for e2 in con_edges0]

            # Find max negative or min positive angle
            if ty == 1:
                angle_node0 = max([n for n in angles1 if n<0], default = 9000)
            else:
            # min positive number from node 0
                angle_node0 = min([n for n in angles1 if n>0], default = 9000)

            # CHeck that its not 9000, if it is, stop recursion - no cells
            if angle_node0 != 9000:
                # find edge corresponding to the angle 
                edge3 = [e for e in con_edges0 if edge2[0].edge_angle(e) == angle_node0]

                # Fine index of common and non-common node between the new edge (edge3) and 
                # its connected edge (edge2)
                common_node = edge3[0].nodes.intersection(edge2[0].nodes).pop()
                other_node = edge3[0].nodes.difference([common_node]).pop()

                # check if non-commomn node is final node
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
                    cells = self.recursive_cycle_finder(edge2, edge3, ty, cell_nodes, cell_edges, p , max_iter)

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

        for edge in edges:
            edge.cells = self

    # def __eq__(self, other):
    #     return set(self.edges) == set(other.edges)

    def __str__(self):
        return "{\n "+" ".join([str(e)+"\n" for e in self.edges])+"}"

    def plot(self, ax, **kwargs):

        """Plot the cell on a given axis"""
        [e.plot(ax, **kwargs) for e in self.edges]
        [n.plot(ax, **kwargs) for n in self.nodes]

    @property
    def colony_cell(self):
        return self._colony_cell

    @colony_cell.setter
    def colony_cell(self, colony):
        if colony not in self._colony_cell:
            self._colony_cell.append(colony)

    def perimeter(self):
        return sum([e.straight_length for e in self.edges ])  
    
    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        self._pressure = pres

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

    @property
    def guess_pressure(self):
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

        for cell in cells:
            cell.colony_cell = self

    def plot(self, ax):
        """
        plot the colony on a given axis
        """
        [e.plot(ax) for e in self.cells]

    @property
    def tension_matrix(self):
        return self._tension_matrix

    @tension_matrix.setter
    def tension_matrix(self, A):
        self._tension_matrix = A

    @property
    def pressure_matrix(self):
        return self._pressure_matrix

    @pressure_matrix.setter
    def pressure_matrix(self, B):
        self._pressure_matrix = B
    
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
        [edges.append(x) for cell in self.cells for x in cell.edges if x not in edges]
        return edges

    @property
    def nodes(self):
        """
        Nodes is not the total list of nodes - that is tot_nodes. This gives list of nodes that make up cells
        """
        nodes = []
        [nodes.append(x) for cell in self.cells for edge in cell.edges for x in edge.nodes if x not in nodes]
        return nodes

    @staticmethod
    def solve_constrained_lsq(A, type, B = None, Guess = None):
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
        N = np.shape(A)[1]

        # Define matrix of ones
        P = np.ones((N + 1, N + 1))

        # Get A^TA
        A1 = np.dot(A.T, A)

        # Plug this into appropriate place in P 
        P[0:N, 0:N] = A1

        # Set last element in N,N position to 0 
        P[N, N] = 0

        if type == 0:
            # Define Q
            Q = np.zeros((N+1, 1))

            # Set last element to N (number of edges) 
            # Effectively says average value is 1
            Q[N, 0] = N

        if type == 1:
            # Define Q
            Q = np.zeros((N+1, 1))
            # print(Q)
            # print(B)
            # print(1)
            # print(np.shape(Q))
            # print(np.shape(B))
            C = np.dot(A.T, B)
            C = C.reshape((len(C), ))
            # C = B.reshape((len(B), ))
            # print(np.shape(C))
            # print(np.shape(Q))
            if len(C) >= len(Q[0:N,0]):
                Q[0:N, 0] = C[0:N]
            else:
                Q[0:len(C), 0] = C[0:len(C)]
            # Effectively says average is 0 
            Q[N, 0] = 0


        # Solve PX = Q
        try:
            # By QR decomposition
            R1, R2 = linalg.qr(P) # QR decomposition with qr function
            y = np.dot(R1.T, Q) # Let y=R1'.Q using matrix multiplication
            x = linalg.solve(R2, y) # Solve Rx=y 

            # By least squares - gives same result
            #x = linalg.lstsq(R2, y)


            # By LU decomposition - Both give same results        
            # L, U = scipy.linalg.lu_factor(P)
            # x = scipy.linalg.lu_solve((L, U), Q)
        except np.linalg.LinAlgError as err:
            if 'Matrix is singular' in str(err):
                return None, P
            else:
                raise

        return x[0:N][:,0], P # use this if solved using linalg.solve
        #return x[0][0:N], P # use this if solved using linalg.lstsq



    def make_tension_matrix(self, nodes = None, edges = None):
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
        #nodes = self.nodes
        if nodes == None:
            nodes = self.tot_nodes
        #edges = self.edges
        if edges == None:
            edges = self.tot_edges
        # change above to self.nodes and self.edges to plot tensions only in cells found

        # We want to solve for AX = 0 where A is the coefficient matrix - 
        # A is m * n and X is n * 1 where n is the number of the edges
        # m can be more than n
        # we initialize A as n * m (n*1 plus appends as we loop) because I couldnt figure out how to append rows
        A = np.zeros((len(edges), 1))

        for node in nodes:
            # only add a force balance if more than 2 edge connected to a node
            if len(node.edges) > 2:
                # create a temporay list of zeros thats useful for stuff
                temp = np.zeros((len(edges),1))

                # node.edges should give the same edge as the edge corresponding to node.tension_vectors since they are added together
                # only want to add indices of edges that are part of colony edge list
                #indices = np.array([edges.index(x) for x in node.edges if x in edges if x.radius is not None])
                # Use this for the networkx plot
                indices = np.array([edges.index(x) for x in node.edges if x in edges])

                # similarly, only want to consider horizontal vectors that are a part of the colony edge list 
                # x[0]
                #horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges if node.edges[node.tension_vectors.index(x)].radius is not None])[np.newaxis]
                #Use this for networkx plot
                horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                
                # add the horizontal vectors to the corresponding indices in temp
                temp[indices] = horizontal_vectors.T

                # append this list to A. This column now has the horizontal force balance information for the node being looped
                A = np.append(A, temp, axis=1)

                # repeat the process for the vertical force balance
                temp = np.zeros((len(edges),1))
                vertical_vectors = np.array([x[1] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                temp[indices] = vertical_vectors.T
                A = np.append(A, temp, axis=1)

        # A is the coefficient matrix that contains all horizontal and vertical force balance information of all nodes.
        # its almost definitely overdetermined. Plus its homogenous. Headache to solve. So we use SVD
        # transpose the matrix because we want it of the form AX = 0 where A is m * n and X is n * 1 where n is number of edges 
        A = A.T
        A = np.delete(A, (0), axis=0)

        return A


    def calculate_tension(self, nodes = None, edges = None, **kwargs):
        """
        Calls a solver to calculate tension. Cellfit paper used 
        (1) self.solve_constrained_lsq
        We use 
        (2) self.scipy_opt_minimize
        This optimization is slower but allows us to set initial conditions, bounds, constraints
        """

        # Use Solve_constrained_lsq

        ## MAIN SOLVER
        # Used in cellfit paper
        A = self.make_tension_matrix(nodes, edges)
        #tensions, P = self.solve_constrained_lsq(A, 0, None, Guess)

        # New scipy minimze solver
        if nodes == None:
            nodes = self.tot_nodes
        #edges = self.edges
        if edges == None:
            edges = self.tot_edges

        # Try scipy minimize 
        sol = self.scipy_opt_minimze(edges, **kwargs)

        tensions = sol.x

        # Im setting P = [] because i dont get a P matrix when i do optimization. 
        # Remember to remove this if we switch back to constrained_lsq
        P = []

        # Add tensions to edge
        for j, edge in enumerate(edges):
            edge.tension = tensions[j]

        return tensions, P, A

    def scipy_opt_minimze(self, edges, **kwargs):
        """
        Calls minimize function from scipy optimize. 
        Parameters:
        ----------------------
        edges - really either edges (for tension calculation)
        or cells (for pressure calculation). Just give the solver a list of 
        variables and it will give a solution
        """

        bnds = self.make_bounds(edges)

        x0 = self.initial_conditions(edges)
        

        if type(edges[0]) == edge:
            cons = [{'type': 'eq', 'fun':self.equality_constraint_tension}]
            # x0 = np.ones(len(edges))*0.002
            if not edges[0].guess_tension:

                #sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', bounds = bnds, constraints = cons)

                # This is correct, use this
                #sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', constraints = cons)
                sol = minimize(self.objective_function_tension, x0, method = 'Nelder-Mead', options = {**kwargs})
            else:
                sol = minimize(self.objective_function_tension, x0, method = 'Nelder-Mead', options = {**kwargs})
        else:
            cons = [{'type': 'eq', 'fun':self.equality_constraint_pressure}]
            # x0 = np.zeros(len(edges))
            if not edges[0].guess_pressure:
                #sol = minimize(self.objective_function_pressure, x0, method = 'SLSQP', bounds = bnds, constraints = cons)

                # This is correct, use this
                # sol = minimize(self.objective_function_pressure, x0, method = 'SLSQP', constraints = cons)
                sol = minimize(self.objective_function_pressure, x0, method = 'Nelder-Mead', options = {**kwargs})
            else:
                sol = minimize(self.objective_function_pressure, x0, method = 'Nelder-Mead', options = {**kwargs})
        # print(sol)
        print('Success', sol.success)
        print('Function value', sol.fun)
        print('Function evaluations', sol.nfev)
        print('Number of iterations', sol.nit)
        print('Solution', sol.x)
        print('\n')
        print('-----------------------------')
        return sol

    def make_bounds(self, edge_or_cell):
        """
        Define bounds based on the initial guess of tension or pressure 
        Parameters
        ---------------------
        edge_or_cell - list of edges or cells (variables)
        """


        b = []
        tol_perc = 0.5

        for j, e_or_c in enumerate(edge_or_cell):
            if type(e_or_c) == edge:
                if not e_or_c.guess_tension:
                    b.append((-np.inf, np.inf))
                else:
                    tolerance = e_or_c.guess_tension * tol_perc
                    b.append((e_or_c.guess_tension - tolerance, e_or_c.guess_tension + tolerance))                    
            else:
                if not e_or_c.guess_pressure:
                    b.append((-np.inf, np.inf))
                else:
                    tolerance = e_or_c.guess_pressure * tol_perc
                    b.append((e_or_c.guess_pressure - tolerance, e_or_c.guess_pressure + tolerance))   
        return tuple(b)

    def initial_conditions(self, edge_or_cell):
        """
        Define initial conditions based on previous solution for tension/pressure
        Parameters 
        ------------------
        edge_or_cell - list of edge or cells (variables)
        """

        I = []

        for j, e_or_c in enumerate(edge_or_cell):
            if type(e_or_c) == edge:
                if not e_or_c.guess_tension:
                    # If no guess, we assign guess based on surrounding edge tension guesses
                    node_a, node_b = e_or_c.node_a, e_or_c.node_b
                    tensions_a = [e.guess_tension for e in node_a.edges if e.guess_tension != []]
                    tensions_b = [e.guess_tension for e in node_b.edges if e.guess_tension != []]
                    guess = np.mean(tensions_a) if tensions_a != [] else 0
                    guess = (guess + np.mean(tensions_b))/2 if tensions_b != [] else guess
                    [I.append(guess) if guess != 0 else I.append(0.002)]
                else:
                    I.append(e_or_c.guess_tension)
            else:
                if not e_or_c.guess_pressure:
                    adj_press = self.get_adjacent_pressures(e_or_c)
                    guess = np.mean(adj_press) if adj_press != [] else 0
                    [I.append(guess) if guess != 0 else I.append(0)]

                    #testing
                    if any(I) != 0:
                        mn = np.mean(I)
                        for j, a in enumerate(I):
                            if a == 0:
                                I[j] = mn
                else:
                    if e_or_c.guess_pressure != 0:
                        I.append(e_or_c.guess_pressure)
                    else:
                        adj_press = self.get_adjacent_pressures(e_or_c)
                        guess = np.mean(adj_press) if adj_press != [] else 0
                        [I.append(guess) if guess != 0 else I.append(1e-5)]

        return I

    def get_adjacent_pressures(self, e_or_c):
        adj_press = []
        for ee in e_or_c.edges:
            adj_cell = [c for c in ee.cells if c != e_or_c]
            if adj_cell != []:
                if adj_cell[0].guess_pressure != []:
                    adj_press.append(adj_cell[0].guess_pressure)
        return adj_press



    def objective_function_tension(self, x):
        """
        Main objective function to be minimzed in the tension calculation 
        i.e sum(row^2) for every row in tension matrix A
        """

        A = self.make_tension_matrix()

        num_of_eqns = len(A[:,0])
        objective = 0
        for j, row in enumerate(A[:,:]):
            row_obj = 0
            for k, element in enumerate(row):
                if element != 0:
                    row_obj = row_obj + element*x[k]
            objective = objective + (row_obj)**2 

        return objective 

    def objective_function_pressure(self, x):
        """
        Main objective function to be minimzed in the pressure calculation 
        i.e sum((row - rhs)^2). We need rhs here because in the pressure case
        rhs is not 0.
        """

        A, rhs = self.make_pressure_matrix()

        num_of_eqns = len(A[:,0])
        objective = 0
        for j, row in enumerate(A[:,:]):
            row_obj = 0
            for k, element in enumerate(row):
                if element != 0:
                    row_obj = row_obj + element*x[k]
            objective = objective + (row_obj - rhs[j])**2 

        return objective 

    def equality_constraint_tension(self, x):
        """
        Assigns equality constraint - i.e mean of tensions = 1
        """
        
        A = self.make_tension_matrix()

        num_of_edges = len(A[0,:])
        constraint = 0
        for i in range(num_of_edges):
            constraint = constraint + x[i]
        return constraint - num_of_edges    

    def equality_constraint_pressure(self, x):
        """
        Assigns equality constraint - i.e mean of pressures = 0
        """      
        A, _ = self.make_pressure_matrix()

        num_of_cells = len(A[0,:])
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

        A = np.zeros((len(self.cells), 1))

        # of the form tension/radius
        rhs = []
        #rhs = np.zeros((len(edges), 1))
        for c in self.cells:


            # find cells with a common edge to c
            common_edge_cells = [cell for cell in self.cells if set(c.edges).intersection(set(cell.edges)) != set() if cell != c]

            # If there are two cells that share an edge, can calculate pressure difference across it
            for cell in common_edge_cells:
                # find common edges between cell and c
                c_edges = [e for e in set(cell.edges).intersection(set(c.edges))]
                indices = []
                indices.append(self.cells.index(c))
                indices.append(self.cells.index(cell))


                for e in c_edges:

                    temp = np.zeros((len(self.cells),1))
                    # we are finding the pressure difference between 2 cells - (cell, c)
                    values = np.array([1,-1])
                    for j, i in enumerate(indices):
                        # here we assign +1 to cell (c) and -1 to cell (cell)
                        temp[i] = values[j]

                    A = np.append(A, temp, axis=1)

                    convex_cell = e.convex_concave(c, cell)
                    if convex_cell == c:
                        if e.radius is not None:
                            if e.tension is not []:
                                rhs.append(e.tension/ e.radius)
                        else: 
                            rhs.append(0)
                    else:
                        if e.radius is not None:
                            rhs.append(np.negative(e.tension/ e.radius))
                        else:
                            rhs.append(0)

                    
        A = A.T
        A = np.delete(A, (0), axis=0)
        rhs = np.array(rhs)

        # Check for all zero columns. If any column is all zero, that means the cell doesnt share a common edge with any other cell
        def delete_column(A, index):
            A = np.delete(A, np.s_[index], axis=1)
            new_index = np.where(~A.any(axis=0))[0]

            if len(new_index) > 0:
                A = delete_column(A, new_index[0])
            return A

        # Save indicies of cells that we cant calculate pressure for (that cell doesnt have a common edge with any other cell)
        zero_column_index = np.sort(np.where(~A.any(axis=0))[0])

        if len(zero_column_index) > 0:
            for i in zero_column_index:
                self.cells[i].pressure = None
            A = delete_column(A, zero_column_index[0])

        return A, rhs


    def calculate_pressure(self, **kwargs):
        """
        Calculate pressure using calculated tensions and edge curvatures (radii). 
        Pressure is unique to every cell
        """
        A, rhs = self.make_pressure_matrix()

        # Old solver
        #pressures, P  = self.solve_constrained_lsq(A, 1, rhs)

        # New solver 
        cells = self.cells
        sol = self.scipy_opt_minimze(cells, **kwargs)
        pressures = sol.x
        P = []

        for j, cell in enumerate(self.cells):
            cell.pressure = pressures[j]
        
        return pressures, P, A

    def plot_tensions(self, ax, fig, tensions, min_ten = None, max_ten = None, **kwargs):
        """
        Plot normalized tensions (min, width) with colorbar
        """

        edges = self.tot_edges

        ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        def norm(tensions, min_ten = None, max_ten = None):
            if min_ten == None and max_ten == None:
                return (tensions - min(tensions)) / float(max(tensions) - min(tensions))
            else:
                return (tensions - min_ten) / float(max_ten - min_ten)

        c1 = norm(tensions, min_ten, max_ten)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            an_edge.plot(ax, ec = cm.viridis(c1[j]), lw = 3)

        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)
        cl.set_label('Normalized tension', fontsize = 13, labelpad = -60)

    def plot_pressures(self, ax, fig, pressures, min_pres = None, max_pres = None, **kwargs):
        """
        Plot normalized pressures (mean, std) with colorbar 
        """
        ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        def norm2(pressures, min_pres = None, max_pres = None):
            if min_pres == None and max_pres == None:
                return (pressures - min(pressures)) / float(max(pressures) - min(pressures))
            else:
                return (pressures - min_pres) / float(max_pres - min_pres)

        # def norm2(pressures):
        #     return (pressures - np.mean(pressures))/float(np.std(pressures))

        c2 = norm2(pressures, min_pres, max_pres)

        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            plt.fill(x, y, c= cm.viridis(c2[j]), alpha = 0.2)
            for e in c.edges:
                e.plot_fill(ax, color = cm.viridis(c2[j]), alpha = 0.2)

        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)


    def plot(self, ax, fig, tensions, pressures, min_ten = None, max_ten = None, min_pres = None, max_pres = None, **kwargs):
        """
        Plot both tensions and pressures on a single axes
        """
        edges = self.tot_edges
        nodes = self.nodes
        ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        def norm(tensions, min_ten = None, max_ten = None):
            if min_ten == None and max_ten == None:
                return (tensions - min(tensions)) / float(max(tensions) - min(tensions))
            else:
                return (tensions - min_ten) / float(max_ten - min_ten)

        def norm2(pressures, min_pres = None, max_pres = None):
            if min_pres == None and max_pres == None:
                return (pressures - min(pressures)) / float(max(pressures) - min(pressures))
            else:
                return (pressures - min_pres) / float(max_pres - min_pres)

        c1 = norm(tensions, min_ten, max_ten)
        c2 = norm2(pressures, min_pres, max_pres)
        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            plt.fill(x, y, c= cm.viridis(c2[j]), alpha = 0.2)
            for e in c.edges:
                # Plots a filled arc
                e.plot_fill(ax, color = cm.viridis(c2[j]), alpha = 0.2)

        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            an_edge.plot(ax, ec = cm.viridis(c1[j]), lw = 3)

        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)
        cl.set_label('Normalized tension', fontsize = 13, labelpad = -60)

class data:
    def __init__(self, V, t):
        """
        Parameters
        ---------
        V is data structure obtained after loading the pickle file
        t is time step
        ---------
        """
        self.V = V
        self.t = t
        self.length = len(self.V[2][self.t])

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
            return self.V[1][self.t][self.V[2][self.t][index][0], 1]
        elif f_or_l == "last":
            loc = len(self.V[2][self.t][index][:]) - 1
            return self.V[1][self.t][self.V[2][self.t][index][loc], 1]
        else:
            return self.V[1][self.t][self.V[2][self.t][index][:], 1]     

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
            return self.V[1][self.t][self.V[2][self.t][index][0], 0]
        elif f_or_l == "last":
            loc = len(self.V[2][self.t][index][:]) - 1
            return self.V[1][self.t][self.V[2][self.t][index][loc], 0]
        else:
            return self.V[1][self.t][self.V[2][self.t][index][:], 0]    

    def add_node(self, index, f_or_l):
        """
        Define a node on branch "index" and location on branch "f_or_l" (str)
        """
        return node((self.x(index, f_or_l), self.y(index, f_or_l)))

    def add_edge(self, node_a, node_b, index = None, x = None, y = None):
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
        radius, xc, yc = self.fit(x,y)


        # Check the direction of the curve. Do this by performing cross product 
        x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]

        v1 = [x1 - xc, y1 - yc]
        v2 = [x2 - xc, y2 - yc]

        cr = np.cross(v1, v2)


        a = 0.5*np.linalg.norm(np.subtract([x2,y2], [x1,y1])) # dist to midpoint


        # Check if radius is 0
        if radius > 0:

            # Check for impossible arc
            if a < radius:
                # if cross product is negative, then we want to go from node_a to node_b
                # if positive, we want to go from node_b to node_a
                # All the cases where i specify a radius are unique cases that i have yet to figure out
                if cr > 0: # correct is cr > 0
                    if radius == 110.29365917569841:
                        ed = edge(node_a, node_b, radius, None, None, x, y)
                    else:

                        ed = edge(node_b, node_a, radius, None, None, x, y)
                    #ed = edge(node_b, node_a, radius, xc, yc)
                else:
                    if radius == 310.7056676687468 or radius == 302.67735946711764:
                        ed = edge(node_b, node_a, radius, None, None, x, y)
                    else:
                        ed = edge(node_a, node_b, radius, None, None, x, y)
                    #ed = edge(node_a, node_b, radius, xc, yc)
            else:

                rnd = a - radius + 5              
                if cr > 0:
                    if cr == 11076.485197677383 or cr == 202.12988846862288:
                        ed = edge(node_a, node_b, radius + rnd,  None, None, x, y)
                    else:
                    #ed = edge(node_b, node_a, None,  None, None, x, y)
                        ed = edge(node_b, node_a, radius + rnd,  None, None, x, y)
                else:
                    if radius == 37.262213713433155 or radius == 62.61598322629542 or radius == 76.8172271622748 or radius == 42.1132395657534:
                        ed = edge(node_b, node_a, radius + rnd,  None, None, x, y)
                    else:
                    #ed = edge(node_a, node_b, None,  None, None, x, y)
                        ed = edge(node_a, node_b, radius + rnd,  None, None, x, y)
        else:
            # if no radius, leave as None
            ed = edge(node_a, node_b, None, None, None, x, y)
        return ed


    def fit(self, x,y):
        """
        Fit a circular arc to a list of co-ordinates
        -----------
        Parameters
        -----------
        x, y
        """

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()


        x_m = np.mean(x)
        y_m = np.mean(y)

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2       = calc_R(*center_2)
        R_2        = Ri_2.mean()
        residu_2   = np.sum((Ri_2 - R_2)**2)

        theta1 = np.rad2deg(np.arctan2(y[np.argmax(x)]-yc_2, x[np.argmax(x)]-xc_2)) # starting angle
        theta2 = np.rad2deg(np.arctan2(y[np.argmin(x)]-yc_2, x[np.argmin(x)]-xc_2)) 

        return R_2, xc_2, yc_2

    def post_processing(self, cutoff, num = None):
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

        if num == None:
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

        # Remove dangling edges (single edges connected to an interface at nearly 90 deg angle)
        new_edges = []
        # Below are the next 3 post processing steps

        # Step 1 - remove small stray edges (nodes connected to 1 edge)
        nodes, edges = self.remove_dangling_edges(nodes, edges)

        # Step 2 - remove small cells
        nodes, edges, new_edges = self.remove_small_cells(nodes, edges)

        # Step 3 - remove nodes connected to 2 edges
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
                perps = [b for b in other_edges if 85 < abs(e.edge_angle(b)) < 95 ] # 85 - 95, 40 - 140
                # If there is such a perpendicular edge, we want to delete e
                if perps != []:
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
                    # Get non common node in egde 0
                    node_a = [a for a in n.edges[0].nodes if a != n ][0]
                    # Get non common node in edge 1
                    node_b = [a for a in n.edges[1].nodes if a != n ][0]
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
        # Get unique cells
        cells = self.find_cycles(edges)
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
                if len(n.edges)>0:
                    for ned in n.edges:
                        node_b = [a for a in ned.nodes if a != n ][0]

                        x1, y1 = ned.co_ordinates
                        new_x1, new_y1 =  np.append(x1, new_x), np.append(y1, new_y)
                        ned.kill_edge(n)
                        ned.kill_edge(node_b)

                        # Finish cleanup
                        # Delete memory of the old edge from the nodes and then remove it from the list of edges
                        if ned in edges:
                            edges.remove(ned)
                        # Add new edge
                        new_edge = self.add_edge(node_b, new_node, None, new_x1, new_y1)
                        #new_edge = self.add_edge(new_node, node_b, None, new_x1, new_y1)
                        new_edges.append(new_edge)
                        edges.append(new_edge)
                    if n in nodes:
                        nodes.remove(n)
            nodes.append(new_node)

        # Check for some weird things
        for n in nodes:
            if len(n.edges) == 0:        
                nodes.remove(n)

        return nodes, edges, new_edges

    @staticmethod
    def find_cycles(edges):

        # Set max iterations for cycle finding
        max_iter = 100
        # Set initial cells
        cells = []

        # My method
        for e in edges:
            cell = e.which_cell(edges, 1, max_iter)
            check = 0
            if cell != []:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    cells.append(cell)

            cell = e.which_cell(edges, 0, max_iter)
            check = 0
            if cell != []:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    cells.append(cell)
        return cells


    def compute(self, cutoff, nodes = None, edges = None):
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
        tensions, P_T, A = col1.calculate_tension()

        # Check for bad tension values
        # Find mean and std
        mean = np.mean(tensions)
        sd = np.std(tensions)

        # # Find tensions more than 3 standard deviations away
        bad_tensions = [x for x in tensions if (x < mean - 3 * sd) or (x > mean + 3 * sd)]

        # if len(bad_tensions) > 0:
        #     new_nodes, new_edges = col1.remove_outliers(bad_tensions, tensions)
        #     col1, tensions, _, P_T, _, A, _ =  self.compute(cutoff, new_nodes, new_edges)

        pressures, P_P, B = col1.calculate_pressure()
        

        return col1, tensions, pressures, P_T, P_P, A, B


    def plot(self, ax, type = None, num = None,  **kwargs):
        """
        Plot the data set 
        ----------------
        Parameters 
        ----------------
        ax - axes to be plotted on 
        type - "edge_and_node", "node", "edge", "image" - specifying what you want to plot
        num - number of branches to be plotted
        """
        if num == None:
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
            img = ndimage.rotate(self.V[0][self.t] == 2, 0)
            # plot the image with origin at lower left
            ax.imshow(img, origin = 'lower')

        ax.set(xlim = [0, 1000], ylim = [0, 1000], aspect = 1)

class manual_tracing(data):
    def __init__(self, X, Y):
        """
        Manual tracing that outputs an array of X and Y co-ordinates
        length(X) == number of edges 
        length(X[0]) == X co-ordinates on edge 0
        length(Y[0]) == Y co-ordinates on edge 0
        """
        self.X = X
        self.Y = Y
        self.length = len(self.X)

    def co_ordinates(self, edge_num):
        """
        Get X and Y co-ordinates for specified edge number
        """
        return X[edge_num], Y[edge_num]

    def fit_X_Y(self, edge_num):
        """
        Fit a circular arc to an edge.
        Call self.fit - .fit is a function in the data class
        """
        R_2, xc_2, yc_2 = self.fit(self.X[edge_num], self.Y[edge_num])
        return R_2, xc_2, yc_2

    def cleanup(self,cutoff):
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
            node_a = node((self.X[index][0], self.Y[index][0]))
            node_b = node((self.X[index][-1], self.Y[index][-1]))
        
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
                ed = self.add_edge(node_a, node_b, None, self.X[index], self.Y[index])
                edges.append(ed)

        # Remove dangling edges (single edges connected to an interface at nearly 90 deg angle)
        new_edges = []
        # Below are the next 3 post processing steps

        # # Step 1 - remove small stray edges (nodes connected to 1 edge)
        nodes, edges = self.remove_dangling_edges(nodes, edges)

        # # Step 2 - remove small cells
        # nodes, edges, new_edges = self.remove_small_cells(nodes, edges)

        # # Step 3 - remove nodes connected to 2 edges
        nodes, edges = self.remove_two_edge_connections(nodes, edges)

        return nodes, edges, new_edges

class manual_tracing_multiple:
    def __init__(self, numbers):
        """
        Class to handle colonies at mutliple time points that have been manually traced out
        using NeuronJ
        Numbers is a list containing start and stop index e.g [2,4]
        of the files labeled as -
        'MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt'
                                                ^ this number changes - 0,1,2,3,..30
        """
        self.name_first = 'MAX_20170123_I01_003-Scene-4-P4-split_T'
        self.name_last = '.ome.txt'
        names = [] 
        for i in range(numbers[0],numbers[-1],1):
            names.append(self.name_first+ str(i)+ self.name_last)
        names.append(self.name_first+ str(numbers[-1])+ self.name_last)
        self.names = names

    def get_X_Y_data(self, number):
        """
        Retrieve X and Y co-ordinates of a colony at a time point specified by number
        """

        file = self.name_first + str(number) + self.name_last

        with open(file,'r') as f:
            a = [l.split(',') for l in f]

        x,y, X, Y = [],[], [],[]

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
        """

        X, Y = self.get_X_Y_data(number)

        ex = manual_tracing(X, Y)

        cutoff= 14 if  0 <= number <= 3 \
                or 5 <= number <= 7 \
                or 10 <= number <= 11 \
                or number == 13 or number == 17 else \
                16 if 14 <= number <= 15 \
                or 18 <= number <= 30 else \
                17 if 8 <= number <= 9 or number == 16 \
                or number == 16 else \
                20 if number == 4 else 12
        print('File %d used a Cutoff value ------> %d' % (number, cutoff))

        nodes, edges, new = ex.cleanup(cutoff)

        cells = ex.find_cycles(edges)

        return nodes, edges, cells

    def initial_numbering(self, number0):
        """
        Assign random labels to nodes and cells in the colony specified by number0
        Returns labeled nodes and cells. 
        Also returns a dictionary defined as {node.label: edges connected to node label, vectors of edges connected to node label}
        Also returns the edge list (not labeled)
        """

        # Get the list of nodes for name0
        temp_nodes, edges, initial_cells = self.get_nodes_edges_cells(number0)

        def func(p, common_node):
            # This function outputs the absolute angle (0 to 360) that the edge makes with the horizontal
            if p.node_a == common_node:
                this_vec = np.subtract(p.node_b.loc, p.node_a.loc)
            else:
                this_vec = np.subtract(p.node_a.loc, p.node_b.loc)
            angle = np.arctan2(this_vec[1], this_vec[0])
            #angle = np.rad2deg((2*np.pi + angle)%(2*np.pi))
            return this_vec

        # Create an empty dictionary
        old_dictionary = defaultdict(list)
        for j, node in enumerate(temp_nodes):
            # Give every node a label -> in this case we're arbitrarily givig labels as we loop through
            node.label = j

            # We do 2 sorting steps for the edges in the node.edges list ->
            # (1) sort by length of the edge
            # (2) sort by the angle the edge makes with the horizontal
            sort_edges = node.edges
            #sort_edges = sorted(sort_edges, key = lambda p: func(p, node))
            this_vec = [func(p, node) for p in sort_edges]
            #sort_edges = sorted(sort_edges, key = lambda p: p.straight_length)
            
            # Add these sorted edges to a dictionary associated with the node label
            old_dictionary[node.label].append(sort_edges)
            old_dictionary[node.label].append(this_vec)

        for k, cell in enumerate(initial_cells):
            cell.label = k

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
            angle = np.arctan2(this_vec[1], this_vec[0])
            #return np.rad2deg((2*np.pi + angle)%(2*np.pi))
            return this_vec

        def py_ang(v1, v2):
            """ Returns the angle in degrees between vectors 'v1' and 'v2'    """
            cosang = np.dot(v1, v2)
            sinang = la.norm(np.cross(v1, v2))
            return np.rad2deg(np.arctan2(sinang, cosang))

        # def get_new_edges(old_dictionary, node)
        #     old_angles = old_dictionary[node.label][1]
        #     temp_edges = []
        #     new_vec = [func(p, node) for p in node.edges]             
        #     for old_e in old_angles:
        #         v1_v2_angs = [py_ang(old_e, nw) for nw in new_vec]
        #         min_ang = min(v1_v2_angs)
        #         if min_ang > 20:                    
        #             for ed in node.edges:
        #                 vec = func(ed, node)
        #                 if py_ang(old_e, vec) == min_ang:
        #                     new_edge = ed
        #                     return new_edge
        #         else:
        #             return []


        # Get list of nodes and edges for every time point
        old_nodes = old_colony.tot_nodes
        old_edges = old_colony.tot_edges
        old_cells = old_colony.cells

        # Get list of nodes and edges for names_now
        # No labelling
        now_nodes, now_edges, now_cells = self.get_nodes_edges_cells(number_now)

        # Find the node in now_nodes that is closest to a node in old_nodes and give same label
        for j, prev_node in enumerate(old_nodes):
            # Give the same label as the previous node 
            closest_new_node = min([node for node in now_nodes], key = lambda p: np.linalg.norm(np.subtract(prev_node.loc, p.loc)))

            # Check that the edge vectors on this node are similar to the edge vectors on the prev node
            if len(closest_new_node.edges) == 1:
                # Want to check that angles are similar
                if py_ang(closest_new_node.tension_vectors[0], prev_node.tension_vectors[0]) < 15:
                    closest_new_node.label = prev_node.label
            else:
                closest_new_node.label = prev_node.label

        # Check for any node labels that are empty and assign a 100+ number
        #count = 100
        #testing
        count = max([n.label for n in now_nodes])
        for node in now_nodes:
            if node.label == []:
                node.label = count 
                count += 1

        # Sort now_nodes by label
        now_nodes = sorted(now_nodes, key = lambda p: p.label)


        # Make a new dictionary for the now_nodes list
        new_dictionary = defaultdict(list)
        for node in now_nodes:
            if node.label < 100:
                # old_edges = old_dictionary[node.label][0]
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

                #temp_edges = sorted(temp_edges, key = lambda p: p.straight_length)
                new_vecs = [func(p, node) for p in temp_edges]
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

        now_cells = self.label_cells(now_nodes, old_cells, now_cells)


        # Define a colony 
        edges2 = [e for e in now_edges if e.radius is not None]
        now_nodes, now_cells, edges2 = self.assign_intial_guesses(now_nodes, combined_dict, now_cells, old_cells, edges2)
        new_colony = colony(now_cells, edges2, now_nodes)

        return new_colony, new_dictionary

    def label_cells(self, now_nodes, old_cells, now_cells):
        """
        Now_nodes is the list of nodes at the current time step
        These nodes have labels based on previous time steps
        old_cells - cells from prev time step
        now_cells - cells in current time step
        """
        for j, cell in enumerate(old_cells):
            nodes = cell.nodes
            now_cell_nodes = []
            for node in nodes:
                match_now_node = [n for n in now_nodes if n.label == node.label]                
                if match_now_node != []:
                    now_cell_nodes.append(match_now_node[0])

            if -4 < len(cell.nodes) - len(now_cell_nodes) < 4:
                # This means we found a matching label node for every node in cell
                labels = [n.label for n in nodes]
                match_cell = [c for c in now_cells if len(set([n.label for n in c.nodes]).intersection(set(labels))) > 3 ]
                if match_cell != []:
                    match_cell[0].label = cell.label
                    # if match_cell[0].perimeter() - cell.perimeter() < 20:                  
                        # match_cell[0].label = cell.label
        return now_cells

    def assign_intial_guesses(self, now_nodes, combined_dict, now_cells, old_cells, edges2):
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

        for cell in now_cells:
            if cell.label != []:
                match_old_cell = [old for old in old_cells if old.label == cell.label][0]
                cell.guess_pressure = match_old_cell.pressure
        

        for k,v in combined_dict.items():
            # v[0] is list of old edges and v[1] is list of matching new edges
            for old, new in zip(v[0], v[1]):
                match_edge = [e for e in edges2 if e == new][0]
                match_edge.guess_tension = old.tension

        # Note - right now edges2 not changing at all. Left it here so that if we want to add labels to edges2, can do it here

        return now_nodes, now_cells, edges2

    def first_computation(self, number_first, **kwargs):
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
        name = str(number_first)
        colonies[name] = colony(cells, edges2, nodes)

        tensions, P_T, A = colonies[name].calculate_tension(**kwargs)
        pressures, P_P, B = colonies[name].calculate_pressure(**kwargs)

        colonies[name].tension_matrix = A
        colonies[name].pressure_matrix = B

        return colonies, dictionary

    def computation_based_on_prev(self, numbers, colonies = None, index = None, old_dictionary = None, **kwargs):
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

        if colonies == None:
            colonies, old_dictionary = self.first_computation(numbers[0], **kwargs)
            colonies[str(numbers[0])].dictionary = old_dictionary
            index = 0

        colonies[str(numbers[index + 1])], new_dictionary = self.track_timestep(colonies[str(numbers[index])], old_dictionary, numbers[index + 1])
        colonies[str(numbers[index + 1])].dictionary = new_dictionary
        tensions, P_T, A = colonies[str(numbers[index+1])].calculate_tension(**kwargs)
        pressures, P_P, B = colonies[str(numbers[index+1])].calculate_pressure(**kwargs)

        # Save tension and pressure matrix
        colonies[str(numbers[index+1])].tension_matrix = A
        colonies[str(numbers[index+1])].pressure_matrix = B

        index = index + 1

        if index < len(numbers) - 1:
            colonies = self.computation_based_on_prev(numbers, colonies, index, new_dictionary, **kwargs)

        return colonies

    def check_repeat_labels(self, colonies, max_num):
        """
        Find node labels that are present in a specified number of colonies
        """
        testin = []
        for t, v in colonies.items():
            index = str(t)
            if int(t) < max_num:
                testin.append(colonies[index].dictionary)
        res = testin.pop()
        for d in testin:
            res = res & d.keys()
        return res 

    def all_tensions_and_radius_and_pressures(self, colonies):
        """
        Return all unique edge tensions, edge radii and cell pressures in all colonies
        """
        all_tensions = []
        all_radii = []
        all_pressures = []
        for t, v in colonies.items():
            index = str(t)
            cells = colonies[index].cells
            edges = colonies[index].tot_edges
            [all_tensions.append(e.tension) for e in edges if e.tension not in all_tensions]
            [all_radii.append(e.radius) for e in edges if e.radius not in all_radii]
            [all_pressures.append(c.pressure) for c in cells if c.pressure not in all_pressures]
        return all_tensions, all_radii, all_pressures

    def plot_single_nodes(self, fig, ax, label, colonies, max_num):
        """
        Plot the edges connected to a node specified by label
        Parameters
        ---------------
        label - label of node that is present in all colonies specified by colonies
        colonies - dictionary of colonies
        """
        ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        #min_t, max_t, _, _, _, _ = self.get_min_max(colonies, label)
        all_tensions, all_radii, _ = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_t, min_t = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)

        
        for cindex, v in colonies.items():
            # Get all nodes, all edges
            if int(cindex) < max_num:
                nodes = colonies[str(cindex)].tot_nodes
                all_edges = colonies[str(cindex)].tot_edges

                # Get tensions
                tensions = [e.tension for n in nodes for e in n.edges if n.label == label]
                all_tensions = [e.tension for e in all_edges]

                def norm(tensions, min_t = None, max_t = None):
                    return (tensions - min_t) / float(max_t - min_t)

                c1 = norm(tensions, min_t, max_t)

                # Get edges on node label
                edges = [e for n in nodes for e in n.edges if n.label == label]

                for j, an_edge in enumerate(edges):
                    an_edge.plot(ax, ec = cm.viridis(c1[j]), lw = 3)

                sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
                # fake up the array of the scalar mappable. 
                sm._A = []

                # Plot all edges 
                for edd in all_edges:
                    edd.plot(ax, lw = 0.2)
                cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
                cl = plt.colorbar(sm, cax = cbaxes)
                cl.set_label('Normalized tension', fontsize = 13, labelpad = -60)            
                pylab.savefig('_tmp%05d.png'%int(cindex), dpi=200)
                plt.cla()
                plt.clf()
                plt.close()
                fig, ax = plt.subplots(1,1, figsize = (8, 5))
                ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        fps = 1
        os.system("rm movie_single_node.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_single_node.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()


    def plot_tensions(self, fig, ax, colonies):
        """
        Make a tension movie over the colonies
        """
        max_num = len(colonies)

        all_tensions, _, _ = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)

        #min_ten, max_ten = None, None

        for t, v in colonies.items():
            index = str(t)
            t= int(t)
            nodes = colonies[index].tot_nodes
            edges = colonies[index].tot_edges
            tensions = [e.tension for e in edges]
            colonies[index].plot_tensions(ax, fig, tensions, min_ten, max_ten)
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            pylab.savefig('_tmp%05d.png'%t, dpi=200)
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1,1, figsize = (8, 5))

        fps = 1
        os.system("rm movie_tension.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_tension.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_single_cells(self, fig, ax, ax1, colonies, cell_label):
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)        
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')
        frames = [i for i in colonies.keys()]
        
        pressures = []
        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            pres = [c.pressure for c in cells if c.label == cell_label]
            if pres != []:
                pressures.append(pres[0])
            else:
                frames = frames[0:j]

        ax1.plot(frames, pressures, lw = 3, color = 'black')
        ax1.set_ylabel('Pressures', color='black')
        ax1.set_xlabel('Frames')

        for i in frames:
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
 #           [n.plot(ax, markersize = 10) for n in nodes if n.label == node_label]

            current_cell = [c for c in cells if c.label == cell_label][0]
            # ax.plot(current_cell, color = 'red')
            [current_cell.plot(ax, color = 'red', )]


            x = [n.loc[0] for n in current_cell.nodes]
            y = [n.loc[1] for n in current_cell.nodes]
            ax.fill(x, y, c= 'red', alpha = 0.2)
            for e in current_cell.edges:
                e.plot_fill(ax, color = 'red', alpha = 0.2)

            ax1.plot(i, current_cell.pressure, 'ok', color = 'red')

            fname = '_tmp%05d.png'%int(i)   
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1) = plt.subplots(1,2, figsize = (14,6)) 
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, pressures, lw = 3, color = 'black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')

        fps = 1
        os.system("rm movie_cell.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_cell.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()


    def plot_single_edges(self, fig, ax, ax1, colonies, node_label, edge_label):

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        frames = [i for i in colonies.keys()]
        # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])

        tensions = []
        radii = []

        for j, i in enumerate(frames):
            dictionary = colonies[str(i)].dictionary
            try:
                tensions.append(dictionary[node_label][0][edge_label].tension)
                radii.append(dictionary[node_label][0][edge_label].radius)
            except:
                frames = frames[0:j]

        ax1.plot(frames, tensions, lw = 3, color = 'black')
        ax1.set_ylabel('Tension', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, radii, 'blue')
        ax2.set_ylabel('Radius', color='blue')
        ax2.tick_params('y', colors='blue')

        for i in frames:
            edges = colonies[str(i)].tot_edges
            nodes = colonies[str(i)].tot_nodes
            dictionary = colonies[str(i)].dictionary

            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
          #  ax1.set(xlim = [0,31], ylim = [0,0.004])
            ax1.set(xlim = [0,31], ylim = [min_ten, max_ten])
            ax2.set(xlim = [0,31], ylim = [min_rad, max_rad])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
 #           [n.plot(ax, markersize = 10) for n in nodes if n.label == node_label]

            current_edge = dictionary[node_label][0][edge_label]
            [current_edge.plot(ax, lw=3, color = 'red')]

            fname = '_tmp%05d.png'%int(i)   
            ax1.plot(i, current_edge.tension, 'ok', color = 'red')
            ax2.plot(i, current_edge.radius, 'ok', color = 'red')
            plt.savefig(fname)

            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1) = plt.subplots(1,2, figsize = (14,6)) 
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, tensions, lw = 3, color = 'black')
            ax1.set_ylabel('Tension', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, radii, 'blue')
            ax2.set_ylabel('Radius', color='blue')
            ax2.tick_params('y', colors='blue')

        fps = 1
        os.system("rm movie_edge.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_edge.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()


    def get_min_max_by_outliers_iqr(self, ys, type = None):
        """
        Get the maximum and minimum of a data set by ignoring outliers
        uses interquartile method
        code from - http://colingorrie.github.io/outlier-detection.html
        """
        quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        # lower_bound, upper_bound = np.percentile(ys, (5,95))
        updated_list = np.where((ys > upper_bound) | (ys < lower_bound), np.inf, ys)
        max_t = max([e for e in updated_list if e != np.inf])
        min_t = min([e for e in updated_list if e != np.inf])
        if type == None:
            return updated_list, max_t, min_t
        else:
            #replace min_t with mean_t and max_t - min_t with standard deviation
            std_t = np.std([e for e in updated_list if e != np.inf])
            mean_t = np.mean([e for e in updated_list if e != np.inf])
            return updated_list, mean_t + std_t, mean_t


    def outliers_modified_z_score(self, ys, y):
        """
        Alternative outlier check. Not used currently
        """
        threshold = 3.5

        median_y = np.median(ys)
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
        modified_z_scores = 0.6745 * (y - median_y) / median_absolute_deviation_y
        return np.where(np.abs(modified_z_scores) > threshold)

    def plot_pressures(self, fig, ax, colonies):
        """
        Make a pressure movie over colonies
        """

        max_num = len(colonies)

        _, _, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        #min_pres, max_pres = None, None

        for t, v in colonies.items():
            index = str(t)
            t= int(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            colonies[index].plot_pressures(ax, fig, pressures, min_pres, max_pres)
            [e.plot(ax) for e in colonies[index].edges]
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            pylab.savefig('_tmp%05d.png'%t, dpi=200)
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1,1, figsize = (8, 5))

        fps = 1
        os.system("rm movie_pressure.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_pressure.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_both_tension_pressure(self, fig, ax, colonies):
        """
        Make a pressure movie over colonies
        """
        max_num = len(colonies)

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')
        #min_ten, max_ten, min_pres, max_pres = None, None, None, None


        for t, v in colonies.items():
            index = str(t)
            t=int(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            edges = colonies[index].tot_edges
            tensions = [e.tension for e in edges]
            colonies[index].plot(ax, fig, tensions, pressures, min_ten, max_ten, min_pres, max_pres)
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            pylab.savefig('_tmp%05d.png'%t, dpi=200)
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1,1, figsize = (8, 5))

        fps = 1
        os.system("rm movie_ten_pres.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_ten_pres.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()


class data_multiple:
    def __init__(self, pkl):
        """
        Define a class to store the pickle file including all the time points
        """
        self.pkl = pkl

    def compute(self, t, cutoff):
        """
        Perform the computation at a specified time and cutoff
        ---------
        Parameters 
        ---------
        t - time to calculate things 
        cutoff - minimum distance below which we merge nodes
        """
        V = data(self.pkl, t)
        col1, tensions, pressures, P_T, P_P, A, B = V.compute(cutoff)
        return col1, tensions, pressures, P_T, P_P, A, B

    def plot(self, ax, fig, t, cutoff, **kwargs):
        """
        Plot stuff at specified time and cutoff
        """
        col1, tensions, pressures, P_T, P_P, A, B = self.compute(t, cutoff)
        #col1.plot(ax, fig, tensions, pressures)
        col1.plot_tensions(ax, fig, tensions, **kwargs)

    def plot_cells(self, ax, t, cutoff, **kwargs):
        """
        Plot all cells found for specified time and cutoff
        """
        ax.set(xlim = [0,1000], ylim = [0,1000], aspect = 1)
        col1, tensions, pressures, P_T, P_P, A, B = self.compute(t, cutoff)
        cells = col1.cells
        [c.plot(ax) for c in cells]


    def save_pictures(self, ax, fig, max_t, cutoff, **kwargs):
        """
        Save a specified number of pictures 
        ---------
        Parameters 
        ---------
        ax, fig - fig stuff
        max_t - the maximum number of pictures to save
        cutoff - merge value 
        """
        for g, t in enumerate(range(max_t)):
            self.plot(ax, fig, t, cutoff)
            pylab.savefig('0000{0}.png'.format(g), dpi=200)
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1,1, figsize = (8, 5))

    def CreateMovie(self, ax, fig, number_of_times, cutoff, fps=10):
        """
        Create a movie for a specified number of frames
        ----------
        Parameters
        ----------
        ax, fig - fig stuff
        number_of_times - number of frames to save to the movie
        cutoff - merge value
        """
         
        for i in range(number_of_times):
            self.plot(ax, fig, i, cutoff)

            fname = '_tmp%05d.png'%i
     
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, ax = plt.subplots(1,1, figsize = (8,5))   
        os.system("rm movie.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie.mp4")
        os.system("rm _tmp*.png")





            













