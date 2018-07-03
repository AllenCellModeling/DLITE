import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class node:
    def __init__(self, loc):
        """loc is the (x,y) location of the node"""
        self.loc = loc
        self._edges = []
    
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

    def plot(self, ax):
        ax.plot(self.loc[0], self.loc[1], ".")

class edge:
    def __init__(self, node_a, node_b, radius=None):
        self.node_a = node_a
        self.node_b = node_b
        self.radius = radius
        node_a.edges = self
        node_b.edges = self
        self._cells = []

    def __str__(self):
        return "["+"   ->   ".join([str(n) for n in self.nodes])+"]"

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cell):
        if cell not in self._cells:
            self._cells.append(cell)

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
        xc, yc = self._circle_arc_center(point1, point2, radius)
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
        else:
            ax.plot([a.x, b.x], [a.y, b.y], **kwargs)

    @property
    def connected_edges(self):
        """The edges connected to nodes a and b"""
        edges_a = [e for e in self.node_a.edges if e is not self]
        edges_b = [e for e in self.node_b.edges if e is not self]
        return edges_a, edges_b

    @property
    def nodes(self):
        return set((self.node_a, self.node_b))

    def _edge_angle(self, other_edge):
        """Straight-line angle between this edge and other"""
        ## find the common node between the two edges
        common_node = self.nodes.intersection(other_edge.nodes).pop()
        this_other_node = self.nodes.difference([common_node]).pop()
        other_other_node = other_edge.nodes.difference([common_node]).pop()
        # find edge vectors from common node
        this_vec = np.subtract(this_other_node.loc, common_node.loc)
        other_vec = np.subtract(other_other_node.loc, common_node.loc)
        # find angles of each 
        this_ang = np.arctan2(this_vec[1], this_vec[0])
        other_ang = np.arctan2(other_vec[1], other_vec[0]) 
        #length = lambda v: np.sqrt(np.dot(v,v))
        #angle = lambda v,w: np.arccos(np.dot(v,w) / (length(v) * length(w)))
        return other_ang - this_ang

    def unit_vectors(self):
        """What are the unit vectors of the angles the edge makes as it goes 
        into nodes A and B? Unlike _edge_angle, this accounts for the curvature 
        of the edge.
        """
        a, b = self.node_a.loc, self.node_b.loc
        center = self._circle_arc_center(a, b, self.radius)
        perp_a = np.array((a[1]-center[1], -(a[0]-center[0])))
        perp_a = perp_a/np.linalg.norm(perp_a)
        perp_b = np.array((-(b[1]-center[1]), b[0]-center[0]))
        perp_b = perp_b/np.linalg.norm(perp_b)
        return perp_a, perp_b



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
        for edge in edges:
            edge.cells = self

    def __str__(self):
        return "{\n "+" ".join([str(e)+"\n" for e in self.edges])+"}"

    def plot(self, ax):
        """Plot the cell on a given axis"""
        [e.plot(ax) for e in self.edges]
        [n.plot(ax) for n in self.nodes]



