import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from useful_functions import solution
from filled_arc import arc_patch
import numpy.linalg as la
import matplotlib.patches as mpatches

class node:
    def __init__(self, loc):
        """loc is the (x,y) location of the node"""
        self.loc = loc
        self._edges = []
        self._tension_vectors = []
    
    def __str__(self):
        return "x:%04i, y:%04i"%tuple(self.loc)

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

    # @edges.deleter
    # def edges(self, edge): 
    #     print('hi')
    #     if edge in self._edges:
    #         self._edges.remove(edge)

    @property
    def tension_vectors(self):
        return self._tension_vectors

    @tension_vectors.setter
    def tension_vectors(self, vector):
        if vector not in self._tension_vectors:
            self._tension_vectors.append(vector)

    def plot(self, ax, **kwargs):
        ax.plot(self.loc[0], self.loc[1], ".", **kwargs)

class edge:
    def __init__(self, node_a, node_b, radius=None, xc = None, yc = None):
        self.node_a = node_a
        self.node_b = node_b
        self.radius = radius
        self.xc = xc
        self.yc = yc 

        node_a.edges = self
        node_b.edges = self


        perp_a, perp_b = self.unit_vectors()
        perp_a = list(perp_a.reshape(1, -1)[0])
        perp_b = list(perp_b.reshape(1, -1)[0])


        node_a.tension_vectors = perp_a
        node_b.tension_vectors = perp_b


        self._cells = []
        self._tension = []
        # self._pressures = []

    def __str__(self):
        return "["+"   ->   ".join([str(n) for n in self.nodes])+"]"

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cell):
        if cell not in self._cells:
            self._cells.append(cell)

    def kill_edge(self, node):

        if node == self.node_a:
            self.node_a.remove_edge(self)
        if node == self.node_b:
            self.node_b.remove_edge(self)

    @property
    def tension(self):
        return self._tension

    @tension.setter
    def tension(self, tension):
        if tension not in self._tension:
            self._tension.append(tension)

    @property
    def straight_length(self):
        """The distance from node A to node B"""
        return np.linalg.norm(np.subtract(self.node_a.loc, self.node_b.loc))

    @staticmethod
    def _circle_arc_center(point1, point2, radius):
        """Get the center of a circle from arc endpoints and radius"""
        x1, y1 = point1
        x2, y2 = point2
        x0, y0 = 0.5*np.subtract(point2, point1)+point1 # midpoint
        a = 0.5*np.linalg.norm(np.subtract(point2, point1)) # dist to midpoint
        assert a<radius, "Impossible arc asked for, radius too small"
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
            ax.plot([a.x, b.x], [a.y, b.y], **kwargs)

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

    # @property
    # def pressures(self):
    #     return self._pressures

    # @pressures.setter
    # def pressures(self, two_pressures):
    #     if two_pressures not in self._pressures:
    #         self._pressures.append(two_pressures)
    

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
            # find angles of each 
            cosang = np.dot(this_vec, other_vec)
            #sinang = np.linalg.norm(np.cross(this_vec, other_vec))
            sinang = np.cross(this_vec, other_vec)
            return np.rad2deg(np.arctan2(sinang, cosang))  # maybe use %(2*np.pi)
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
        Is the edge convex with respect to cell1 or cell2
        Check the angle between unit vectors coming into a node
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

class cell:
    def __init__(self, nodes, edges):
        """
        Parameters
        ----------
        nodes: list of nodes
            Nodes that make up vertices of the cell
        edges: list of edges
            Directed edges that compose the cell
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

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, pres):
        if pres not in self._pressure:
            self._pressure.append(pres)


class colony:
    def __init__(self, cells, edges):
        """
        Parameters
        ________________
        cells: list of cells
        edges: total list of edges (including those not part of a cell)
        """
        self.cells = cells
        self.tot_edges = edges 

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


    def calculate_tension(self):
        """
        Calculate tension along every edge of the colony
        """
        # get the list of nodes and edges in the colony
        nodes = self.nodes
        edges = self.edges
        #edges = self.tot_edges

        # We want to solve for AX = 0 where A is the coefficient matrix - 
        # A is m * n and X is n * 1 where n is the number of the edges
        # m can be more than n
        # we initialize A as n * m (n*1 plus appends as we loop) because I couldnt figure out how to append rows
        A = np.zeros((len(edges), 1))
        y = np.zeros(len(edges))

        for node in nodes:
            # create a temporay list of zeros thats useful for stuff
            temp = np.zeros((len(edges),1))

            # node.edges should give the same edge as the edge corresponding to node.tension_vectors since they are added together
            # only want to add indices of edges that are part of colony edge list
            indices = np.array([edges.index(x) for x in node.edges if x in edges])

            # similarly, only want to consider horizontal vectors that are a part of the colony edge list 
            # x[0]
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

        # MAIN SOLVER
        # decompose matrix A as A = U * diag(S) * V
        # S is list of singular values in decreasing order of magnitude.
        # If any diagonal element is 0, that means we found a singular value
        # look up the column in V.T corresponding to that column in S. This is the solution for X.
        # if no diagonal element is 0, that means there is no solution. BUT, its reasonable to assume
        # that the value of S of smallest magnitude corresponds to the best fit. 
        # Thus, for solution we always pick the last column of V.T

        U, S, V = np.linalg.svd(A)
        tensions = V.T[:,-1]

        for j, edge in enumerate(edges):
            edge.tension = tensions[j]

        return tensions, A, U, S, V

    def calculate_pressure(self, tensions):
        """
        Calculate pressure using calculated tensions and edge curvatures
        """
        # get the list of nodes and edges in the colony
        nodes = self.nodes
        edges = self.edges

        A = np.zeros((len(self.cells), 1))

        # of the form tension/radius
        rhs = []

        for c in self.cells:

            # find cells with a common edge to c
            common_edge_cells = [cell for cell in self.cells if set(c.edges).intersection(set(cell.edges)) != set() if cell != c]
            for cell in common_edge_cells:
                # find common edges between cell and c
                edges = [e for e in set(cell.edges).intersection(set(c.edges))]
                indices = []
                indices.append(self.cells.index(c))
                indices.append(self.cells.index(cell))

                for e in edges:

                    temp = np.zeros((len(self.cells),1))
                    # we are finding the pressure difference between 2 cells - (cell, c)
                    values = np.array([1,-1])
                    for j, i in enumerate(indices):
                        # here we assign +1 to cell (c) and -1 to cell (cell)
                        temp[i] = values[j]

                    A = np.append(A, temp, axis=1)

                    convex_cell = e.convex_concave(c, cell)
                    if convex_cell == c:
                        rhs.append(e.tension/ e.radius)
                    else:
                        rhs.append(np.negative(e.tension/ e.radius))

                    

        A = A.T
        A = np.delete(A, (0), axis=0)
        rhs = np.array(rhs)
        rhs = rhs[:,0]

        # U, S, V = np.linalg.svd(A)
        # pressures = V.T[:,-1]
        #x = la.lstsq(A,rhs, rcond = None)
        #pressures = x[0]

        pressures = np.dot(np.linalg.pinv(A), rhs)


        for j, cell in enumerate(self.cells):
            cell.pressure = pressures[j]

        pressures = pressures.T

        return pressures, A, rhs








