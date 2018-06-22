import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class node:
    def __init__(self, loc, cells=[], edges=[]):
        """loc is the (x,y) location of the node"""
        self.loc = loc
        self.cells = cells
        self.edges = edges

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    def add_cell(self, cell):
        if cell not in self.cells:
            self.cells.append(cell)

    def del_cell(self, cell):
        self.cells.remove(cell)

    def plot(self, ax):
        ax.plot(self.loc[0], self.loc[1], ".")

class edge:
    def __init__(self, node_a, node_b, radius=None):
        self.node_a = node_a
        node_a.edges.append(self)
        self.node_b = node_b
        node_b.edges.append(self)
        self.radius = radius

    @property
    def straight_length(self):
        """The distance from node A to node B"""
        return np.linalg.norm(np.subtract(self.node_a.loc, self.node_b.loc))

    @staticmethod
    def arc_translation(point1, point2, radius):
        """Get arc center and angles from endpoints and radius

        We want to be able to plot circular arcs on matplotlib axes.
        Matplotlib only supports plotting such by giving the center,
        starting angle, and stopping angle for such. But we want to
        plot using two points on the circle and the radius.

        For explanation, check out https://tinyurl.com/ya7wxoax
        """
        x1, y1 = point1
        x2, y2 = point2
        x0, y0 = 0.5*np.subtract(point2, point1)+point1 # midpoint
        a = 0.5*np.linalg.norm(np.subtract(point2, point1)) # dist to midpoint
        assert a<radius, "Impossible arc asked for, radius too small"
        b = np.sqrt(radius**2-a**2) # midpoint to circle center
        xc = x0 + (b*(y0-y1))/a # location of circle center
        yc = y0 - (b*(x0-x1))/a
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
        [node.add_cell(self) for node in nodes]
        self.nodes = nodes
        self.edges = edges

    def plot(self, ax):
        """Plot the cell on a given axis"""
        [e.plot(ax) for e in self.edges]
        [n.plot(ax) for n in self.nodes]
