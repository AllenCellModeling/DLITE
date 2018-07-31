import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from useful_functions import solution
from filled_arc import arc_patch
from scipy import ndimage, optimize
import numpy.linalg as la
import collections
import scipy.linalg as linalg 
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
        self._edge_label = []
    
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

    @property
    def edge_label(self):
        return self._edge_label
    
    @edge_label.setter
    def edge_label(self, edge_label):
        if edge_label not in self._edge_label:
            self._edge_label.append(edge_label)

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
        self._label = []

    # def __str__(self):
    #     return "["+"   ->   ".join([str(n) for n in self.nodes])+"]"

    @property
    def co_ordinates(self):
        return self.x_co_ords, self.y_co_ords

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label
    
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
       # assert a<radius, "Impossible arc asked for, radius too small"
        if a>radius:
            self.radius = radius + 60
            radius = radius + 60
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

        for edge in edges:
            edge.cells = self

    # def __eq__(self, other):
    #     return set(self.edges) == set(other.edges)

    def __str__(self):
        return "{\n "+" ".join([str(e)+"\n" for e in self.edges])+"}"

    def plot(self, ax):


        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        # color = [x for x in colors if colors.index(x) + 2 == len(self.nodes) ]
        # ''.join(color)
        # ax.facecolor = color

        """Plot the cell on a given axis"""
        [e.plot(ax) for e in self.edges]
        [n.plot(ax) for n in self.nodes]

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
        if pres not in self._pressure:
            self._pressure.append(pres)


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

        for cell in cells:
            cell.colony_cell = self

    def plot(self, ax):
        """
        plot the colony on a given axis
        """
        [e.plot(ax) for e in self.cells]

    @property
    def edges(self):
        edges = []
        [edges.append(x) for cell in self.cells for x in cell.edges if x not in edges]
        return edges

    @property
    def nodes(self):
        nodes = []
        [nodes.append(x) for cell in self.cells for edge in cell.edges for x in edge.nodes if x not in nodes]
        return nodes

    @staticmethod
    def solve_constrained_lsq(A, type, B = None):
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



    def calculate_tension(self, nodes = None, edges = None):
        """
        Calculate tension along every edge of the colony (including stray edges)
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

        # # OLD MAIN SOLVER
        # # decompose matrix A as A = U * diag(S) * V
        # # S is list of singular values in decreasing order of magnitude.
        # # If any diagonal element is 0, that means we found a singular value
        # # look up the column in V.T corresponding to that column in S. This is the solution for X.
        # # if no diagonal element is 0, that means there is no solution. BUT, its reasonable to assume
        # # that the value of S of smallest magnitude corresponds to the best fit. 
        # # Thus, for solution we always pick the last column of V.T

        # U, S, V = np.linalg.svd(A)
        # tensions = V.T[:,-1]

        # Use Solve_constrained_lsq
        
        ## MAIN SOLVER
        tensions, P = self.solve_constrained_lsq(A, 0, None)


        # Add tensions to edge
        for j, edge in enumerate(edges):
            edge.tension = tensions[j]

        return tensions, P, A

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



    def calculate_pressure(self, tensions):
        """
        Calculate pressure using calculated tensions and edge curvatures (radii). 
        Pressure is unique to every cell
        """
        # get the list of nodes and edges in the colony
        #nodes = self.nodes
        #edges = self.edges

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

        # print(A)
        
        # OLD SOLVER
        # U, S, V = np.linalg.svd(A)
        # pressures = V.T[:,-1]
        #x = la.lstsq(A,rhs, rcond = None)
        #pressures = x[0]

        # pressures = np.dot(np.linalg.pinv(A), rhs)

        pressures, P  = self.solve_constrained_lsq(A, 1, rhs)

        

        # Output None if it is a singular matrix
        if pressures is not None:
            j = 0
            for i in range(len(pressures)):
                if self.cells[j].pressure != [None]:
                    self.cells[j].pressure = pressures[i]
                    j += 1
                else:
                    j += 1
                    self.cells[j].pressure = pressures[i]
                    j += 1

        return pressures, P, A

    def plot_tensions(self, ax, fig, tensions, **kwargs):
        """
        Plot normalized tensions (min, width) with colorbar
        """

        edges = self.tot_edges

        ax.set(xlim = [0,1000], ylim = [0,1000], aspect = 1)

        def norm(tensions):
            return (tensions - min(tensions)) / float(max(tensions) - min(tensions))

        c1 = norm(tensions)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            an_edge.plot(ax, ec = cm.jet(c1[j]), lw = 3)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)
        cl.set_label('Normalized tension', fontsize = 13, labelpad = -60)

    def plot_pressures(self, ax, fig, pressures, **kwargs):
        """
        Plot normalized pressures (mean, std) with colorbar 
        """
        ax.set(xlim = [0,1000], ylim = [0,1000], aspect = 1)

        def norm2(pressures):
            return (pressures - np.mean(pressures))/float(np.std(pressures))

        c2 = norm2(pressures)

        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            plt.fill(x, y, c= cm.jet(c2[j]), alpha = 0.2)
            for e in c.edges:
                e.plot_fill(ax, color = cm.jet(c2[j]), alpha = 0.2)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)


    def plot(self, ax, fig, tensions, pressures, **kwargs):
        """
        Plot both tensions and pressures on a single axes
        """
        edges = self.tot_edges
        nodes = self.nodes
        ax.set(xlim = [0,1000], ylim = [0,1000], aspect = 1)

        def norm(tensions):
            return (tensions - min(tensions)) / float(max(tensions) - min(tensions))

        def norm2(pressures):
            return (pressures - np.mean(pressures))/float(np.std(pressures))

        c1 = norm(tensions)
        c2 = norm2(pressures)
        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            plt.fill(x, y, c= cm.jet(c2[j]), alpha = 0.2)
            for e in c.edges:
                # Plots a filled arc
                e.plot_fill(ax, color = cm.jet(c2[j]), alpha = 0.2)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            an_edge.plot(ax, ec = cm.jet(c1[j]), lw = 3)

        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
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
                if cr > 0:
                    if cr == 11076.485197677383:
                        ed = edge(node_a, node_b, radius + 30,  None, None, x, y)
                    else:
                    #ed = edge(node_b, node_a, None,  None, None, x, y)
                        ed = edge(node_b, node_a, radius + 30,  None, None, x, y)
                else:
                    if radius == 37.262213713433155 or radius == 62.61598322629542 or radius == 76.8172271622748 or radius == 42.1132395657534:
                        ed = edge(node_b, node_a, radius + 30,  None, None, x, y)
                    else:
                    #ed = edge(node_a, node_b, None,  None, None, x, y)
                        ed = edge(node_a, node_b, radius + 30,  None, None, x, y)
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

        pressures, P_P, B = col1.calculate_pressure(tensions)
        

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
        length(X[0]) == number of co-ordinates on edge 0
        """
        self.X = X
        self.Y = Y
        self.length = len(self.X)

    def co_ordinates(self, edge_num):
        return X[edge_num], Y[edge_num]

    def fit_X_Y(self, edge_num):
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
    def __init__(self, names):
        """
        Name is of the form 'MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt'
        names = [Name1, Name2...]
        """
        self.names = names

    def get_nodes_edges(self, name):

        with open(name,'r') as f:
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

        ex = manual_tracing(X, Y)
        nodes, edges, new = ex.cleanup(14)

        return nodes, edges

    def initial_numbering(self, name0):
        temp_nodes, _ = self.get_nodes_edges(name0)

        # temp_edges = []
        old_dictionary = {}
        for j, node in enumerate(temp_nodes):
            # number all nodes in list 0 from 0 to N
            node.label = j
            sort_edges = sorted(node.edges, key = lambda p: p.straight_length)
            old_dictionary[node.label] = sort_edges


        return temp_nodes, old_dictionary


    def track_based_on_prev_t(self, names_prev, names_now, old_nodes = None):

        # Get list of nodes and edges for every time point

        if old_nodes == None:
            old_nodes, old_dictionary = self.initial_numbering(names_prev)

        now_nodes, now_edges = self.get_nodes_edges(names_now)

        for j, prev_node in enumerate(old_nodes):
            # Give the same label as the previous node 
            closest_new_node = min([node for node in now_nodes], key = lambda p: np.linalg.norm(np.subtract(prev_node.loc, p.loc)))
            closest_new_node.label = j

        count = 100
        for node in now_nodes:
            if node.label == []:
                node.label = count 
                count += 1

        now_nodes = sorted(now_nodes, key = lambda p: p.label)


        repeat_now_edges = []
        count = 100
        new_dictionary = {}
        for node in now_nodes:
            if node.label < 100:
                temp_edges = sorted(node.edges, key = lambda p: p.straight_length)
                new_dictionary[node.label] = temp_edges

        set1 = set(old_dictionary)
        set2 = set(new_dictionary)

        combined_dict = defaultdict(list)

        for label in set1.intersection(set2):
            if old_dictionary[label] != [] and new_dictionary[label] != []:
                combined_dict[label].append(old_dictionary[label])
                combined_dict[label].append(new_dictionary[label])       

        return now_nodes, new_dictionary, old_nodes, old_dictionary, combined_dict





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





            













