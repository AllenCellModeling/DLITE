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
from scipy import stats
from matplotlib import cm
import itertools
import math
import os, sys
import matplotlib.patches as mpatches
import matplotlib.animation as manimation
from collections import defaultdict
import pylab
import scipy
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
import seaborn as sns
import pandas as pd
import random
#from Dave_cell_find import find_all_cells, cells_on_either_side, Cycle_tracer
# To go back to my cycle find, change 'find_all_cells' to 'cycle_finder' and also change the way the angle is calculated


class CycleTracer:
    """Trace around a network to discover a cell cycle
    
    Typical use goes like
    >>> cell_cycle = CycleTracer(edge, 1)
    >>> cycle_nodes, cycle_edges = cell_cycle.trace_cycle(10)
    """
    def __init__(self, edge, direction):
        """Remember the edge where you start and the direction
        """
        self.start_node = edge.node_a
        self.current_node = edge.node_b
        # Cycle is [(node, edge_list_under_consideration), ...]
        self.cycle = [(self.start_node, [edge])]
        # Direction choice
        assert direction in [-1,1]
        if direction==1:
            self.dir = 0
        elif direction==-1:
            self.dir = -1
    
    @staticmethod
    def _other_node(edge, node):
        """The other node on an edge"""
        other_node = [n for n in edge.nodes if n is not node][0]
        return other_node

    @staticmethod
    def _sort_edges(edge, node):
        """Return a list of edges on a node, sorted by angle from 
        smallest to largest, relative to a given edge
        """

        other_edges = [e for e in node.edges if e is not edge]
        edge_angles = [edge.edge_angle(e) for e in other_edges]
        sorted_edges = [(a, e) for a, e in sorted(zip(edge_angles, other_edges))]
        sorted_edges = [e for a,e in sorted_edges]
        return sorted_edges
    
    def _pop_edge(self):
        """We hit a dead end and need to remove an edge 
        from the candidacy for inclusion in the cycle
        """
        self.cycle[-1][1].pop(self.dir)
        # and if that means there are no more edges on this node, recurse
        if len(self.cycle[-1][1]) == 0:
            self.cycle.pop()
            if len(self.cycle)>=2:  # don't eat the first node
                self._pop_edge() 
        return
    
    def _follow_to_next_edge(self):
        """Trace along to the next possible edge, add it to the cycle
        """
        #[print(e) for e in self.cycle]
        #print("\n")
        prior_node = self.cycle[-1][0]
        next_edge = self.cycle[-1][1][self.dir]
        next_node = [n for n in next_edge.nodes if n is not prior_node][0]
        next_edges = self._sort_edges(next_edge, next_node)
        if next_edges == []: #hit dead end
            self._pop_edge()
            return
        self.cycle.append((next_node, next_edges))
        self.current_node = next_node
        return

    def trace_cycle(self, lim=10):
        """Trace the cycle, of length up to lim
        """
        # Loop

        while self.start_node is not self.current_node and 1<=len(self.cycle)<lim:
            print(len(self.cycle), self.start_node, self.current_node)
            self._follow_to_next_edge()
        # If you didn't find a cycle    
        if self.start_node is not self.current_node:
            self.nodes = None  # no cycle found
            self.edges = None  # no cycle found
        else:  # or if you did
            self.cycle.pop()  # last entry was dupe of first
            self.nodes = [node for node, edges in self.cycle]
            self.edges = [edges[self.dir] for node, edges in self.cycle]
        return self.nodes, self.edges


class node:
    def __init__(self, loc):
        """loc is the (x,y) location of the node"""
        self.loc = loc
        self._edges = []
        self._tension_vectors = []
        self._horizontal_vectors = []
        self._vertical_vectors = []
        self._label = []
        # edge indices in colony.tot_edges
        self._edge_indices = []
    
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
    def horizontal_vectors(self):
        return self._horizontal_vectors

    @horizontal_vectors.setter
    def horizontal_vectors(self, hor_vector):
        self._horizontal_vectors = hor_vector

    @property
    def vertical_vectors(self):
        return self._vertical_vectors

    @vertical_vectors.setter
    def vertical_vectors(self, ver_vector):
        self._vertical_vectors = ver_vector

    @property
    def edge_indices(self):
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
        self._label = []
        self._center_of_circle = []
        self._cell_indices = []
        self._cell_coefficients = []
        self._cell_rhs = []

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
        check = 0
        if cell not in self._cells:
            for c in self._cells:
                if len(set(c.edges).intersection(set(cell.edges))) == len(set(cell.edges)):
                    check = 1
            if check == 0:
                self._cells.append(cell)

    @property
    def cell_indices(self):
        return self._cell_indices

    @cell_indices.setter
    def cell_indices(self, indices):
        self._cell_indices = indices

    @property
    def cell_coefficients(self):
        return self._cell_coefficients

    @cell_coefficients.setter
    def cell_coefficients(self, coeff):
        self._cell_coefficients = coeff

    @property
    def cell_rhs(self):
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
    def center_of_circle(self):
        return self._center_of_circle

    @center_of_circle.setter
    def center_of_circle(self, center):
        self._center_of_circle = center

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

            #return np.rad2deg(np.arctan2(sinang, cosang))  # maybe use %(2*np.pi)
            return np.degrees(np.math.atan2(np.linalg.det([this_vec,other_vec]),np.dot(this_vec,other_vec)))
            #return np.arccos(np.clip(np.dot(this_vec/np.linalg.norm(this_vec), other_vec/np.linalg.norm(other_vec)), -1.0, 1.0))

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

        Returns which of the 2 cells the edge (self) is curving out of
        """

        def py_ang(v1, v2):
            """ Returns the angle in degrees between vectors 'v1' and 'v2'    """
            cosang = np.dot(v1, v2)
            sinang = la.norm(np.cross(v1, v2))
            return np.rad2deg(np.arctan2(sinang, cosang))

        perp_a, perp_b = self.unit_vectors()

        # lets only focus on node_a

        # find the other edges coming into node_a in cell1 and cell2
        edge1 = [e for e in cell1.edges if e.node_a == self.node_a or e.node_b == self.node_a if e!= self][0]
        edge2 = [e for e in cell2.edges if e.node_a == self.node_a or e.node_b == self.node_a if e!= self][0]

        # NEW method

        centroid_cell1 = list(cell1.centroid())
        centroid_cell2 = list(cell2.centroid())
        n_a = list(self.node_a.loc)
        n_b = list(self.node_b.loc)

        midpoint = [(x + y)/2 for x,y in zip(self.node_a.loc, self.node_b.loc)]
        actual_co_ord = [np.mean(self.x_co_ords), np.mean(self.y_co_ords)]


        distance1 = math.sqrt( ((centroid_cell1[0]-self.center_of_circle[0])**2)+((centroid_cell1[1]-self.center_of_circle[1])**2) )
        distance2 = math.sqrt( ((centroid_cell2[0]-self.center_of_circle[0])**2)+((centroid_cell2[1]-self.center_of_circle[1])**2) )
        if distance1 > distance2:
            return cell2
        elif distance2 > distance1:
            return cell1
        else:
            print('how')

        # v1 = [(x - y) for x, y in zip(midpoint, n_a)]
        # v2 = [(x - y) for x, y in zip(actual_co_ord, n_a)]

        # # # find the unit vectors associated with these edges 
        # edge1_p_a, edge1_p_b = edge1.unit_vectors()
        # edge2_p_a, edge2_p_b = edge2.unit_vectors()

        # # choose the correct unit vector in edge1 coming into node_a
        # if edge1.node_a == self.node_a:
        #     edge1_v = edge1_p_a
        # else:
        #     edge1_v = edge1_p_b

        # # choose the correct unit vector in edge2 coming into node_a
        # if edge2.node_a == self.node_a:
        #     edge2_v = edge2_p_a
        # else:
        #     edge2_v = edge2_p_b

        # angle1 = py_ang(edge1_v, v1)
        # angle2 = py_ang(edge1_v, v2)
        # if angle1 > angle2:
        #     return cell2
        # elif angle2 > angle1:
        #     return cell1
        # else:
        #     print(angle1, angle2, angle1 > angle2, angle2 > angle1)
        #     print('really how')


        # # print(len(self.x_co_ords))
        # # #actual_co_ord = [self.x_co_ords[2], self.y_co_ords[2]]

        #     distance_between_cell1_midpoint = math.sqrt( ((centroid_cell1[0]-midpoint[0])**2)+((centroid_cell1[1]-midpoint[1])**2) )
        #     distance_between_cell2_midpoint = math.sqrt( ((centroid_cell2[0]-midpoint[0])**2)+((centroid_cell2[1]-midpoint[1])**2) )
        #     distance_between_cell1_actual_co_ord = math.sqrt( ((centroid_cell1[0]-actual_co_ord[0])**2)+((centroid_cell1[1]-actual_co_ord[1])**2) )
        #     distance_between_cell2_actual_co_ord = math.sqrt( ((centroid_cell2[0]-actual_co_ord[0])**2)+((centroid_cell2[1]-actual_co_ord[1])**2) )
        #     print(distance_between_cell1_actual_co_ord, distance_between_cell1_midpoint, distance_between_cell2_actual_co_ord, distance_between_cell2_midpoint)
        #     if distance_between_cell1_actual_co_ord > distance_between_cell1_midpoint and distance_between_cell2_actual_co_ord < distance_between_cell2_midpoint:
        #         return cell1
        #     if distance_between_cell1_actual_co_ord < distance_between_cell1_midpoint and distance_between_cell2_actual_co_ord > distance_between_cell2_midpoint:
        #         return cell2
        #     else:
        #         print('how')

        # OLD METHOD

        # angle1 = self.edge_angle(edge1)
        # angle2 = self.edge_angle(edge2)

        # ODLER METHOD

        # # find the unit vectors associated with these edges 
        # edge1_p_a, edge1_p_b = edge1.unit_vectors()
        # edge2_p_a, edge2_p_b = edge2.unit_vectors()

        # # choose the correct unit vector in edge1 coming into node_a
        # if edge1.node_a == self.node_a:
        #     edge1_v = edge1_p_a
        # else:
        #     edge1_v = edge1_p_b

        # # choose the correct unit vector in edge2 coming into node_a
        # if edge2.node_a == self.node_a:
        #     edge2_v = edge2_p_a
        # else:
        #     edge2_v = edge2_p_b

        # Now we have 3 vectors - perp_a, edge1_v and edge2_v on 3 edges all coming into node_a 
        # to check convexivity, we get the angles between edge1_v and perp_a and between edge2_v and perp_a

        # cosang = np.dot(perp_a, edge1_v)
        # sinang = np.cross(perp_a, edge1_v)
        # angle1 = np.rad2deg(np.arctan2(sinang, cosang)) 

        # cosang = np.dot(perp_a, edge2_v)
        # sinang = np.cross(perp_a, edge2_v)
        # angle2 = np.rad2deg(np.arctan2(sinang, cosang))



        # the one with the larger angle difference should be the more convex cell

        # if abs(angle1) > abs(angle2):
        #     return cell1
        # else:
        #     return cell2 

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

        # print('First', angles1)

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
                # print('max', max_iter)
                cells = self.recursive_cycle_finder([self], edge1, ty, cell_nodes, cell_edges, p, max_iter)

                # print(cells)
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
        # print(p, 'start')

        if p > max_iter:
            print('how')
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

            # print('all_angles', angles1)

            # BEGIN HACKY BLOCK

            # if all(i < 0 for i in angles1) and ty == 0 and len(angles1) != 0:
            #     if angles1[0] == -177.92477691580484:
            #         angle_node0 = angles1[0]
            #     else:
            #         angle_node0 = angles1[1]


            # if all(i > 0 for i in angles1) and ty == 1 and len(angles1) != 0:
            #     angle_node0 = angles1[1]

            # END HACKY BLOCK, PLS REMOVE IF NOT FEEL LIKE HACKY

            # print('angle', angle_node0)
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

                    # HACKY STUFF< PLS REMOVE IF STATEMENT
                    if len(cell_edges) != 11 and len(cell_edges) != 12:
                        cells = cell(cell_nodes, cell_edges)
                        # print('found')

                    # cells = cell(cell_nodes, cell_edges)
                    # print('found')

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

        # for edge in edges:
        #     edge.cells = self

    # def __eq__(self, other):
    #     return set(self.edges) == set(other.edges)

    # def __str__(self):
    #     return "{\n "+" ".join([str(e)+"\n" for e in self.edges])+"}"

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

    def centroid(self):
        x = [n.loc[0] for n in self.nodes]
        y = [n.loc[1] for n in self.nodes]
        return (np.mean(x), np.mean(y))

    def area(self):
        vertices = [n.loc for n in self.nodes]
        n = len(vertices) # of corners
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += abs(vertices[i][0] * vertices[j][1]-vertices[j][0] * vertices[i][1])
        result = a / 2.0
        return result
    
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
        self._pressure_rhs = []

        for cell in cells:
            cell.colony_cell = self

    def plot(self, ax):
        """
        plot the colony on a given axis
        """
        [e.plot(ax) for e in self.cells]

    def add_cell(self, cell):
        self.cells.append(cell)

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
    def pressure_rhs(self):
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

                node.edge_indices = indices

                # similarly, only want to consider horizontal vectors that are a part of the colony edge list 
                # x[0]
                #horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges if node.edges[node.tension_vectors.index(x)].radius is not None])[np.newaxis]
                #Use this for networkx plot
                horizontal_vectors = np.array([x[0] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]
                
                node.horizontal_vectors = horizontal_vectors[0]
                # add the horizontal vectors to the corresponding indices in temp
                temp[indices] = horizontal_vectors.T

                # append this list to A. This column now has the horizontal force balance information for the node being looped
                A = np.append(A, temp, axis=1)

                # repeat the process for the vertical force balance
                temp = np.zeros((len(edges),1))
                vertical_vectors = np.array([x[1] for x in node.tension_vectors if node.edges[node.tension_vectors.index(x)] in edges])[np.newaxis]

                node.vertical_vectors = vertical_vectors[0]

                temp[indices] = vertical_vectors.T
                A = np.append(A, temp, axis=1)

        # A is the coefficient matrix that contains all horizontal and vertical force balance information of all nodes.
        # its almost definitely overdetermined. Plus its homogenous. Headache to solve. So we use SVD
        # transpose the matrix because we want it of the form AX = 0 where A is m * n and X is n * 1 where n is number of edges 
        A = A.T
        A = np.delete(A, (0), axis=0)

        self.tension_matrix = A

        return A


    def calculate_tension(self, nodes = None, edges = None, solver = None, **kwargs):
        """
        Calls a solver to calculate tension. Cellfit paper used 
        (1) self.solve_constrained_lsq
        We use 
        (2) self.scipy_opt_minimize
        This optimization is slower but allows us to set initial conditions, bounds, constraints
        """

        # Use Solve_constrained_lsq



        # New scipy minimze solver
        if nodes == None:
            nodes = self.tot_nodes
        #edges = self.edges
        if edges == None:
            edges = self.tot_edges

        ## MAIN SOLVER
        # Used in cellfit paper

        
        
        if solver == 'KKT':
            A = self.make_tension_matrix(nodes, edges)
            tensions, P = self.solve_constrained_lsq(A, 0, None)

        # Try scipy minimize 
        if solver == None:
            A = self.make_tension_matrix(nodes, edges)
            sol = self.scipy_opt_minimze(edges, **kwargs)

            tensions = sol.x

            # Im setting P = [] because i dont get a P matrix when i do optimization. 
            # Remember to remove this if we switch back to constrained_lsq
            P = []

        # Add tensions to edge
        for j, edge in enumerate(edges):
            edge.tension = tensions[j]

        return tensions, P, A

    def scipy_opt_minimze(self, edges, i =[0], **kwargs):
        """
        Calls minimize function from scipy optimize. 
        Parameters:
        ----------------------
        edges - really either edges (for tension calculation)
        or cells (for pressure calculation). Just give the solver a list of 
        variables and it will give a solution
        """

        i[0]+=1 # mutable variable get evaluated ONCE

        bnds = self.make_bounds(edges)

        x0 = self.initial_conditions(edges)
        

        if type(edges[0]) == edge:
            # Not using equality constraint, useful for redoing cellfit stuff
            cons = [{'type': 'eq', 'fun':self.equality_constraint_tension}]
            # x0 = np.ones(len(edges))*0.002

            # Check if the first element is empty (which it shouldnt be)
            if not edges[0].guess_tension:
                pass

                # Use this for cellfit stuff
                #sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', constraints = cons)

                #sol = minimize(self.objective_function_tension, x0, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs})
            else:
                # Assign constant initial values if you want
                # for k, xxx in enumerate(x0):
                #     x0[k] = 0.2

                # If all elements are the same, run basin hopping with random initial guess
                if x0.count(x0[0]) == len(x0):

                    # Assign random initial guesses in range 0-1 upto 2 digits
                    for k, xxx in enumerate(x0):
                        x0[k] = random.randint(0,101)/100

                    print('Initial Tension guess is', x0)
                    
                    print('Trying L-BFGS')

                    # Correct BLOCK

                    # #Run basin hopping
                    minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : bnds} # used only BFGS and no bounds before
                    sol = basinhopping(self.objective_function_tension, x0, T = 0.5, interval = 10,  minimizer_kwargs=minimizer_kwargs, niter=100,  stepsize = 0.05, disp = True)
                    zoom = sol['x']
                    print(zoom)
                    # Run normal L-BFGS
                    sol = minimize(self.objective_function_tension, zoom, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs}, tol = 1e-9)


                    # END CORRECT BLOCK
                    # sols = []
                    # for j, i in enumerate(range(10)):
                    #     lower = np.random.uniform(0, 0.5, len(edges))
                    #     upper = lower + 0.5
                    #     # print(len(lower), len(upper), len(x0))
                    #     bnds = [(l, u) for (l, u) in zip(lower, upper)]
                    #     # print(bnds)
                    #     # for k, xxx in enumerate(x0):
                    #     #     x0[k] = random.randint(0,101)/100
                    #     sols.append([minimize(self.objective_function_tension,  x0, bounds = bnds)])
                    #     print(sols[j][0].fun, sols[j][0].success)
                        

                    # print(sols)
                    # idx = np.argmin([sol[0].fun for sol in sols if sol[0].success == True])
                    # sol = sols[idx][0]

                    # print('Trying differential_evolution')
                    # sol = differential_evolution(self.objective_function_tension, bnds,  strategy = 'best1bin')



                    #sol = minimize(self.objective_function_tension, x0, method = 'Nelder-Mead', bounds = bnds, options = {**kwargs})

                    # Run Cellfit stuff
                    #sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', constraints = cons)

                    # testing

                    # from hyperopt import tpe, fmin, Trials, hp

                    # space = hp.uniform('x', 0, 100)
                    # tpe_algo = tpe.suggest
                    # tpe_trials = Trials()
                    # tpe_best = fmin(fn=self.objective_function_tension, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=2000)
                    # print(tpe_best)
                else:
                    # Run normal L-BFGS
                    sol = minimize(self.objective_function_tension, x0, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs})

                    # Run Cellfit stuff
                    #sol = minimize(self.objective_function_tension, x0, method = 'SLSQP', constraints = cons)
        else:
            # Constraint for cellfit paper
            cons = [{'type': 'eq', 'fun':self.equality_constraint_pressure}]
            # x0 = np.zeros(len(edges))

            # Check for empty element (which it shouldnt be)
            if not edges[0].guess_pressure and edges[0].guess_pressure != 0:
                pass
                # This is correct, use this
                # For cellfit stuff
                #sol = minimize(self.objective_function_pressure, x0, method = 'L-BFGS-B', constraints = cons)
                # Normal L-BFGS 
                #sol = minimize(self.objective_function_pressure, x0, method = 'L-BFGS-B', bounds = bnds,  options = {**kwargs})
            else:
                # Assign constant initial guesses if needed
                # for k, xxx in enumerate(x0):
                #     x0[k] = 0.001
                if x0.count(x0[0]) == len(x0):

                    # If all initial guesses are the same, use random initial guesses
                    # for k, xxx in enumerate(x0):
                    #     x0[k] = random.randint(0,101)/10000

                    # BEGIN CORRECT BLOCK

                    # print('Initial pressure guess is', x0)
                    # minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : bnds}
                    # print('Trying L-BFGS')
                    # #sol = basinhopping(self.objective_function_pressure, x0, minimizer_kwargs=minimizer_kwargs, niter=2, disp = True)

                    # # RUn normal
                    sol = minimize(self.objective_function_pressure, x0, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs})

                    # END CORRECT BLOCK

                    # Run Cellfit stuff
                    #sol = minimize(self.objective_function_pressure, x0, method = 'SLSQP', constraints = cons)

                else:
                    # i[0] should say how many times the function has been called. If its 2, this is the first pressure iteration. Use basin hopping
                    if i[0] ==2:
                        print('Pressure initial basin hopping')
                        print('Initial pressure guess is', x0)
                        minimizer_kwargs = {"method": "L-BFGS-B", "bounds" : bnds}
                        print('Trying L-BFGS')
                        # Try basin hoppinh
                        #sol = basinhopping(self.objective_function_pressure, x0, minimizer_kwargs=minimizer_kwargs, niter=2, disp = True)

                        # Run CellFit stuff
                        #sol = minimize(self.objective_function_pressure, x0, method = 'SLSQP', constraints = cons)

                        # L-BFGS
                        sol = minimize(self.objective_function_pressure, x0, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs})
                        #sol = minimize(self.objective_function_tension, x0, method = 'Nelder-Mead', bounds = bnds, options = {**kwargs})
                    else:
                        # RUn normal
                        sol = minimize(self.objective_function_pressure, x0, method = 'L-BFGS-B', bounds = bnds, options = {**kwargs})

                        # Run Cellfit stuff
                        #sol = minimize(self.objective_function_pressure, x0, method = 'SLSQP', constraints = cons)
        # print(sol)
        # print('Success', sol.success)
        print('Function value', sol.fun)
        # print('Function evaluations', sol.nfev)
        # print('Number of iterations', sol.nit)
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
                    #b.append((0, np.inf))
                    b.append((0.0001, 1))
                    # b.append((-np.inf, np.inf))
                else:
                    tolerance = e_or_c.guess_tension * tol_perc
                    #b.append((0, np.inf))
                    b.append((0.0001, 1))
                    # b.append((e_or_c.guess_tension - tolerance, e_or_c.guess_tension + tolerance))                    
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

        I = []

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
                            guess_b if guess_a == 0 and guess_b != 0 else\
                            0.2
                    I.append(guess)
                    # Update the guess tension used
                    e_or_c.guess_tension = I[j]
                else:
                    I.append(e_or_c.guess_tension)
                
            else:
                if not e_or_c.guess_pressure:
                    adj_press = self.get_adjacent_pressures(e_or_c)
                    guess = np.mean(adj_press) if adj_press != [] else 0

                    edges_in_this_cell = [e for e in e_or_c.edges]
                    tension_of_edges = [e.tension for e in edges_in_this_cell]
                    radii_of_edges = [e.radius for e in edges_in_this_cell]
                    ratio_of_tension_to_radius = [x/y for x, y in zip(tension_of_edges, radii_of_edges)]
                    guess2 = np.mean(ratio_of_tension_to_radius)

                    [I.append(guess) if guess != 0 else I.append(guess2)]
                    # I.append(0)
                    # Upodate guess pressure used
                    e_or_c.guess_pressure = I[j]

                    #testing
                    # if any(I) != 0:
                    #     mn = np.mean(I)
                    #     for j, a in enumerate(I):
                    #         if a == 0:
                    #             I[j] = mn
                else:
                    I.append(e_or_c.guess_pressure)
                    # if e_or_c.guess_pressure != 0:
                    #     I.append(e_or_c.guess_pressure)
                    # else:
                    #     adj_press = self.get_adjacent_pressures(e_or_c)
                    #     guess = np.mean(adj_press) if adj_press != [] else 0
                    #     [I.append(guess) if guess != 0 else I.append(1e-5)]

        #print(I)

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
        objective = 0

        # DAVE WAY
        for node in self.tot_nodes:
            if len(node.edges) > 2:
                indices = node.edge_indices

                starting_tensions = []

                node_vecs = node.tension_vectors

                for i in range(len(indices)):
                    starting_tensions.append([x[indices[i]]])

                starting_tensions = np.array(starting_tensions)

                tension_vecs = np.multiply(node_vecs, starting_tensions)
                # print(node_vecs, starting_tensions)

                # print(tension_vecs)

                tension_vec_mags = [np.hypot(*vec) for vec in tension_vecs]

                # print(tension_vec_mags)

                resid_vec = np.sum(tension_vecs, 0)

                # print(resid_vec)

                resid_mag = np.hypot(*resid_vec)

                # print(resid_mag)

                objective = objective + resid_mag + resid_mag/np.sum(tension_vec_mags)
                #objective = objective + resid_mag 


        # MY WAY - split horizontal and vertical force balance

        # for node in self.tot_nodes:
        #     # only add a force balance if more than 2 edge connected to a node
        #     if len(node.edges) > 2:
        #         indices = node.edge_indices
        #         hor_vectors = node.horizontal_vectors
        #         ver_vectors = node.vertical_vectors

        #         node_ver_balance, node_hor_balance, sum_of_vec_mags = 0, 0, 0
        #         for i in range(len(indices)):
        #             node_hor_balance = node_hor_balance + hor_vectors[i]*x[indices[i]]
        #             node_ver_balance = node_ver_balance + ver_vectors[i]*x[indices[i]]
        #             sum_of_vec_mags = sum_of_vec_mags + x[indices[i]]

        #         if sum_of_vec_mags != 0:
        #             # This is correct
        #             objective = objective + (node_hor_balance + node_hor_balance/sum_of_vec_mags)**2 
        #             objective = objective + (node_ver_balance + node_ver_balance/sum_of_vec_mags)**2
                    
        #             # objective = objective + (node_hor_balance)**2 + 0.1*(node_hor_balance/sum_of_vec_mags)**2
        #             # objective = objective + (node_ver_balance)**2 + 0.1*(node_ver_balance/sum_of_vec_mags)**2 
        #             # objective = objective + (node_hor_balance)**2 + 0.5*(node_hor_balance)**2
        #             # objective = objective + (node_ver_balance)**2 + 0.5*(node_ver_balance)**2 

        #             #objective = objective + (node_hor_balance )**2 
        #             #objective = objective + (node_ver_balance )**2
        #         else:
        #             objective = objective + (node_hor_balance )**2
        #             objective = objective + (node_ver_balance )**2
                    # objective = objective + (node_hor_balance - node_hor_balance/sum_of_vec_mags)**2
                    # objective = objective + (node_ver_balance - node_ver_balance/sum_of_vec_mags)**2

        # OLD WAY


        # #A = self.make_tension_matrix()
        # A = self.tension_matrix


        # num_of_eqns = len(A[:,0])
        # objective = 0

        # for j, row in enumerate(A[:,:]):
        #     row_obj = 0
        #     mag = 0
        #     regularizer = 0
        #     for k, element in enumerate(row):
        #         if element != 0:
        #             row_obj = row_obj + element*x[k]
        #             mag = mag + x[k]
        #             regularizer = regularizer + x[k]**2
        #         if mag != 0:
        #             rhs = row_obj/mag
        #         else:
        #             rhs = 0
        #         # temporary
        #         #rhs = 0
        #     objective = objective + (row_obj - rhs)**2 
        #     # This works good
        #     #objective = objective + (row_obj )**2  + rhs**2

        #     #objective = objective + (row_obj )**2 + rhs **2 + abs(rhs)
        #     #objective = objective + (row_obj )**2  + abs(rhs)

        #     # This works good for 0.01
        #     #objective = objective + (row_obj )**2  + 0.05 * regularizer**2

        # # num_of_edges = len(A[:,:])
        # # for i in range(num_of_edges-1):
        # #     objective = objective + 0.1*x[i]**2

        return objective 

    def objective_function_pressure(self, x):
        """
        Main objective function to be minimzed in the pressure calculation 
        i.e sum((row - rhs)^2). We need rhs here because in the pressure case
        rhs is not 0.
        """

        #A, rhs = self.make_pressure_matrix()

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


        # OLD WAY
        # A = self.pressure_matrix
        # rhs = self.pressure_rhs

        # num_of_eqns = len(A[:,0])
        # objective = 0
        # for j, row in enumerate(A[:,:]):
        #     row_obj = 0
        #     mag = 0
        #     for k, element in enumerate(row):
        #         if element != 0:
        #             mag = mag + x[k]
        #             row_obj = row_obj + element*x[k]
        #         if mag != 0:
        #             regularizer = row_obj/mag
        #         else:
        #             regularizer = 0
        #     objective = objective + (row_obj - rhs[j])**2 
            #objective = objective + (row_obj - rhs[j])**2 + regularizer
            #objective = objective + (row_obj - rhs[j])**2 + regularizer**2 + abs(regularizer)

        # num_of_cells = len(A[:,:])
        # for i in range(num_of_cells):
        #     if i < num_of_cells:
        #         objective = objective + 0.1*x[i]**2

        return objective 

    def equality_constraint_tension(self, x):
        """
        Assigns equality constraint - i.e mean of tensions = 1
        """
        
        #A = self.make_tension_matrix()
        A = self.tension_matrix

        num_of_edges = len(A[0,:])
        constraint = 0
        for i in range(num_of_edges):
            constraint = constraint + x[i]
        return constraint - num_of_edges    

    def equality_constraint_pressure(self, x):
        """
        Assigns equality constraint - i.e mean of pressures = 0
        """      
        #A, _ = self.make_pressure_matrix()
        A = self.pressure_matrix

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

                    e.cell_coefficients = np.array([1,-1])

                    temp = np.zeros((len(self.cells),1))

                    for j, i in enumerate(indices):
                        # here we assign +1 to cell (cell1) and -1 to cell (cell2)
                        temp[i] = e.cell_coefficients[j]

                    A = np.append(A, temp, axis=1)

                    convex_cell = e.convex_concave(cell1, cell2)

                    if convex_cell == cell1:
                        if e.radius is not None:
                            if e.tension is not []:
                                #rhs.append(np.negative(e.tension/ e.radius))
                                rhs.append(e.tension/ e.radius)
                                e.cell_rhs = e.tension/e.radius
                        else: 
                            print('radius is None cell1')
                            rhs.append(0)
                            e.cell_rhs = 0

                    elif convex_cell == cell2:
                        if e.radius is not None:
                            #rhs.append(e.tension/ e.radius)
                            rhs.append(np.negative(e.tension/ e.radius))
                            e.cell_rhs = np.negative(e.tension/ e.radius)
                        else:
                            print('radius is none cell2')
                            rhs.append(0)
                            e.cell_rhs = 0
                    else:
                        print('how did this happen, no cell1, no cell2')
                        if e.radius is not None:
                            if e.tension is not []:
                                #rhs.append(np.negative(e.tension/ e.radius))
                                rhs.append(e.tension/ e.radius)
                                e.cell_rhs = e.tension/ e.radius
                        else: 
                            rhs.append(0)
                            e.cell_rhs = 0

                    list_of_edges.append(e)

        A = A.T
        A = np.delete(A, (0), axis=0)
        rhs = np.array(rhs)

        self.pressure_matrix = A
        self.pressure_rhs = rhs



        #OLD WAY

        # #rhs = np.zeros((len(edges), 1))
        # for c in self.cells:


        #     # find cells with a common edge to c
        #     common_edge_cells = [cell for cell in self.cells if set(c.edges).intersection(set(cell.edges)) != set() if cell != c]


        #     # If there are two cells that share an edge, can calculate pressure difference across it
        #     for cell in common_edge_cells:
        #         # find common edges between cell and c
        #         c_edges = [e for e in set(cell.edges).intersection(set(c.edges))]
        #         indices = []
        #         indices.append(self.cells.index(c))
        #         indices.append(self.cells.index(cell))




        #         for e in c_edges:

        #             if e not in list_of_edges:

        #                 e.cell_indices = indices


        #                 temp = np.zeros((len(self.cells),1))
        #                 # we are finding the pressure difference between 2 cells - (cell, c)
        #                 values = np.array([1,-1])
        #                 for j, i in enumerate(indices):
        #                     # here we assign +1 to cell (c) and -1 to cell (cell)
        #                     temp[i] = values[j]

        #                 e.cell_coefficients = values

        #                 A = np.append(A, temp, axis=1)

        #                 convex_cell = e.convex_concave(c, cell)

        #                 if convex_cell == c:
        #                     if e.radius is not None:
        #                         if e.tension is not []:
        #                             #rhs.append(np.negative(e.tension/ e.radius))
        #                             rhs.append(e.tension/ e.radius)
        #                             e.cell_rhs = e.tension/e.radius
        #                     else: 
        #                         rhs.append(0)
        #                         e.cell_rhs = 0

        #                 elif convex_cell == cell:
        #                     if e.radius is not None:
        #                         #rhs.append(e.tension/ e.radius)
        #                         rhs.append(np.negative(e.tension/ e.radius))
        #                         e.cell_rhs = np.negative(e.tension/ e.radius)
        #                     else:
        #                         rhs.append(0)
        #                         e.cell_rhs = 0
        #                 else:
        #                     if e.radius is not None:
        #                         if e.tension is not []:
        #                             #rhs.append(np.negative(e.tension/ e.radius))
        #                             rhs.append(e.tension/ e.radius)
        #                             e.cell_rhs = e.tension/ e.radius
        #                     else: 
        #                         rhs.append(0)
        #                         e.cell_rhs = 0

        #                 list_of_edges.append(e)

                    
        # A = A.T
        # A = np.delete(A, (0), axis=0)
        # rhs = np.array(rhs)

        # self.pressure_matrix = A
        # self.pressure_rhs = rhs

        # Check for all zero columns. If any column is all zero, that means the cell doesnt share a common edge with any other cell
        # def delete_column(A, index):
        #     A = np.delete(A, np.s_[index], axis=1)
        #     new_index = np.where(~A.any(axis=0))[0]

        #     if len(new_index) > 0:
        #         A = delete_column(A, new_index[0])
        #     return A

        # # Save indicies of cells that we cant calculate pressure for (that cell doesnt have a common edge with any other cell)
        # zero_column_index = np.sort(np.where(~A.any(axis=0))[0])

        # if len(zero_column_index) > 0:
        #     for i in zero_column_index:
        #         self.cells[i].pressure = None
        #     A = delete_column(A, zero_column_index[0])

        return A, rhs


    def calculate_pressure(self, solver = None,  **kwargs):
        """
        Calculate pressure using calculated tensions and edge curvatures (radii). 
        Pressure is unique to every cell
        """
        A, rhs = self.make_pressure_matrix()

        # Old solver
        if solver == 'KKT':
            pressures, P  = self.solve_constrained_lsq(A, 1, rhs)

        # New solver 
        cells = self.cells

        if solver == None:
            sol = self.scipy_opt_minimze(cells, **kwargs)
            pressures = sol.x
            P = []

        for j, cell in enumerate(self.cells):
            cell.pressure = pressures[j]
        
        return pressures, P, A

    def plot_tensions(self, ax, fig, tensions, min_ten = None, max_ten = None, specify_color = None, **kwargs):
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
            if specify_color is not None:
                an_edge.plot(ax, ec = cm.jet(c1[j]), **kwargs)
            else:
                an_edge.plot(ax, ec = cm.viridis(c1[j]), **kwargs)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))

        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)
        cl.set_label('Normalized tension', fontsize = 13, labelpad = -60)

    def plot_pressures(self, ax, fig, pressures, min_pres = None, max_pres = None, specify_color = None, **kwargs):
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
            if specify_color is not None:
                plt.fill(x, y, c= cm.jet(c2[j]), alpha = 0.2)
            else:
                plt.fill(x, y, c= cm.viridis(c2[j]), alpha = 0.2)

            for e in c.edges:
                if specify_color is not None:
                    e.plot_fill(ax, color = cm.jet(c2[j]), alpha = 0.2)
                else:
                    e.plot_fill(ax, color = cm.viridis(c2[j]), alpha = 0.2)


        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-1, vmax=1))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)


    def plot(self, ax, fig, tensions, pressures, min_ten = None, max_ten = None, min_pres = None, max_pres = None, specify_color = None, **kwargs):
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
                #return (pressures - np.mean(pressures)) / np.std(pressures)
                return (pressures - min(pressures)) / float(max(pressures) - min(pressures))
            else:
                return (pressures - min_pres) / float(max_pres - min_pres)

        c1 = norm(tensions, min_ten, max_ten)
        c2 = norm2(pressures, min_pres, max_pres)
        # Plot pressures

        for j, c in enumerate(self.cells):
            x = [n.loc[0] for n in c.nodes]
            y = [n.loc[1] for n in c.nodes]
            if specify_color is not None:
                plt.fill(x, y, c= cm.jet(c2[j]), alpha = 0.2)
            else:
                plt.fill(x, y, c= cm.viridis(c2[j]), alpha = 0.2)
            for e in c.edges:
                # Plots a filled arc
                if specify_color is not None:
                    e.plot_fill(ax, color = cm.jet(c2[j]), alpha = 0.2)
                else:
                    e.plot_fill(ax, color = cm.viridis(c2[j]), alpha = 0.2)


        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=-1, vmax=1))
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=-1, vmax=1))
        # fake up the array of the scalar mappable. 
        sm._A = []

        cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
        cl = plt.colorbar(sm, cax = cbaxes)  
        cl.set_label('Normalized pressure', fontsize = 13, labelpad = 10)

        # # Plot tensions

        for j, an_edge in enumerate(edges):
            if specify_color is not None:
                an_edge.plot(ax, ec = cm.jet(c1[j]), **kwargs)
            else:
                an_edge.plot(ax, ec = cm.viridis(c1[j]), **kwargs)

        if specify_color is not None:
            sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
        else:
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
        # print(radius == 26.73131381210073, radius)
        # Check if radius is 0
        # print(x1, y1, x2, x2, radius)
        if radius > 0:

            # Check for impossible arc
            if a < radius:
                # if cross product is negative, then we want to go from node_a to node_b
                # if positive, we want to go from node_b to node_a
                # All the cases where i specify a radius are unique cases that i have yet to figure out
                if cr > 0: # correct is cr > 0
                    # print(radius == 44.869497086324635)
                    if radius == 110.29365917569841 or radius == 15.13898863318815:
                        ed = edge(node_a, node_b, radius, None, None, x, y)
                    else:

                        ed = edge(node_b, node_a, radius, None, None, x, y)
                    #ed = edge(node_b, node_a, radius, xc, yc)
                else:
                    if radius == 310.7056676687468 or radius == 302.67735946711764 or radius == 20.50806241927082 or radius == 26.73131381210073 or radius == 44.869497086324635:
                        ed = edge(node_b, node_a, radius, None, None, x, y)
                    else:
                        ed = edge(node_a, node_b, radius, None, None, x, y)
                    #ed = edge(node_a, node_b, radius, xc, yc)
            else:
                rnd = a - radius + 5
                print(cr, radius)              
                if cr > 0:
                    if cr == 11076.485197677383 or cr == 202.12988846862288 or cr == 145.1160155195729 or radius == 20.124618428290365 or radius == 21.71639076327512 or radius == 13.296557101922646 or radius == 16.102180466236412:
                        ed = edge(node_a, node_b, radius + rnd,  None, None, x, y)
                    else:
                    #ed = edge(node_b, node_a, None,  None, None, x, y)
                        ed = edge(node_b, node_a, radius + rnd,  None, None, x, y)
                else:
                    if radius == 37.262213713433155 or radius == 62.61598322629542 or radius == 76.8172271622748 or radius == 42.1132395657534 or radius == 14.59048212177372 or radius == 22.032023519154624 or radius == 24.943669050666582 or radius == 12.168152196856052 or radius == 11.461675085309 or radius == 13.14379201471666 or radius == 15.237588186248265:
                        ed = edge(node_b, node_a, radius + rnd,  None, None, x, y)
                    else:
                    #ed = edge(node_a, node_b, None,  None, None, x, y)
                        ed = edge(node_a, node_b, radius + rnd,  None, None, x, y)
        else:
            # if no radius, leave as None
            ed = edge(node_a, node_b, None, None, None, x, y)

        ed.center_of_circle = [xc, yc]

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
        #cells = self.find_all_cells(edges)
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

    def find_all_cells(self, edges):
        """Find all the cells in a list of edges
        Parameters
        ----------
            edges: list of edges
        Returns
        -------
            cells: list of cells
        """
        cells = []
        for edge in edges:
            new = self.cells_on_either_side(edge)
            print('Found a cell', new)
            for cell in new:
                if cell is not None and cell not in cells:
                    cells.append(cell)
        return cells

    def cells_on_either_side(self, edge):
        """Find the cells on either side of this edge, if they exist
        What we have to work with are the connected edges on each side 
        of the starting edge and the edge angles they make
        Parameters
        ----------
        edge: edge class
            a single seed edge
        Returns
        -------
        cells: list
            all (0-2) cells that bound this edge
        """
        cycles = [CycleTracer(edge,  1).trace_cycle(10), 
                  CycleTracer(edge, -1).trace_cycle(10)]
        cells = [cell(nodes, edges) for nodes, edges in cycles if nodes is not None]
        return cells

    @staticmethod
    def find_cycles(edges):

        # Set max iterations for cycle finding
        max_iter = 300
        # Set initial cells
        cells = []

        # My method
        for e in edges:
            cell = e.which_cell(edges, 0, max_iter)
            check = 0
            if cell != []:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    for edge in cell.edges:
                        edge.cells = cell
                    cells.append(cell)


            cell = e.which_cell(edges, 1, max_iter)
            check = 0
            if cell != []:
                for c in cells:
                    if set(cell.edges) == set(c.edges):
                        check = 1
                if check == 0:
                    for edge in cell.edges:
                        edge.cells = cell
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
        #cells = self.find_all_cells(edges)
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

class synthetic_data(data):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def compute(self, solver = None, **kwargs):

        cells = self.find_cycles(self.edges)


        colonies = {}

        edges2 = [e for e in self.edges if e.radius is not None]
        name = str(1)
        colonies[name] = colony(cells, edges2, self.nodes)

        tensions, P_T, A = colonies[name].calculate_tension(None, None, solver, **kwargs)
        pressures, P_P, B = colonies[name].calculate_pressure( solver, **kwargs)

        colonies[name].tension_matrix = A
        colonies[name].pressure_matrix = B

        return colonies


class manual_tracing_multiple:
    def __init__(self, numbers, type = None):
        """
        Class to handle colonies at mutliple time points that have been manually traced out
        using NeuronJ
        Numbers is a list containing start and stop index e.g [2,4]
        of the files labeled as -
        'MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt'
                                                ^ this number changes - 0,1,2,3,..30
        """

        if type == None:
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
        else:
            self.name_first = 'AVG_20170124_I07_001-Scene-4-P2-100'
            self.name_last = '.txt'

        names = [] 
        for i in range(numbers[0],numbers[-1],1):
            names.append(self.name_first+ str(i)+ self.name_last)
        names.append(self.name_first+ str(numbers[-1])+ self.name_last)

        self.names = names

    def get_X_Y_data(self, number):
        """
        Retrieve X and Y co-ordinates of a colony at a time point specified by number
        """

        if number <10 and self.name_first == 'AVG_20170124_I07_001-Scene-4-P2-100':
            file = self.name_first + str(0) +  str(number) + self.name_last
        elif number < 10 and self.name_first == '20170123_I01_003.czi - 20170123_I01_00300':
            file = self.name_first + str(0) +  str(number) + self.name_last
        elif number < 10 and self.name_first == '7_20170123_I01_003.czi - 20170123_I01_00300':
            file = self.name_first + str(0) +  str(number) + self.name_last
        elif number < 10 and self.name_first == '002_20170123_I01_002.czi - 20170123_I01_00200':
            file = self.name_first + str(0) +  str(number) + self.name_last
        else:
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

        if self.name_first ==  'MAX_20170123_I01_003-Scene-4-P4-split_T':
            cutoff= 14 if  0 <= number <= 3 \
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
        #cells = ex.find_all_cells(edges)

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
                # ind = p.node_a.edges.index(p)
                # this_vec = p.node_a.tension_vectors[ind]
                this_vec = np.subtract(p.node_b.loc, p.node_a.loc)
            else:
                # ind = p.node_b.edges.index(p)
                # this_vec = p.node_b.tension_vectors[ind]
                this_vec = np.subtract(p.node_a.loc, p.node_b.loc)
            angle = np.arctan2(this_vec[1], this_vec[0])
            #return np.rad2deg((2*np.pi + angle)%(2*np.pi))
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
                # If its connected to 3 edges, closest node is fine. only single edge nodes had problems 
                closest_new_node.label = prev_node.label


        #testing
        upper_limit = max([n.label for n in now_nodes if n.label != []])
        upper_edge_limit = max([e.label for e in old_edges if e.label != []])


        # NEW TRY

        # new_dictionary = []

        # for n in now_nodes:
        #     if n.label == []:
        #         n.label = upper_limit + 1
        #         upper_limit += 1

        # for e in old_edges:
        #     old_node_labels = [e.node_a.label, e.node_b.label]
        #     for new_edge in now_edges:
        #         new_node_labels = [new_edge.node_a.label, new_edge.node_b.label]
        #         if set(new_node_labels) == set(old_node_labels):
        #             new_edge.label = e.label

        # for ed in now_edges:
        #     if ed.label == []:
        #         for old_ed in old_edges:
        #             if py_ang(old_ed.unit_vectors()[0], ed.unit_vectors()[0]) < 15 or py_ang(old_ed.unit_vectors()[0], ed.unit_vectors()[1]) < 15 or py_ang(old_ed.unit_vectors()[1], ed.unit_vectors()[0]) < 15 or py_ang(old_ed.unit_vectors()[1], ed.unit_vectors()[1]) < 15:
        #                 labels = [e.label for e in now_edges]
        #                 if old_ed.label not in labels:

        #                     ed.label = old_ed.label

        # for ed in now_edges:
        #     if ed.label == []:
        #         ed.label = upper_edge_limit + 1
        #         upper_edge_limit += 1

        # END NEW TRY





        # Old stuff

        # count = upper_limit
        # for node in now_nodes:
        #     if node.label == []:
        #         node.label = count
        #         count += 1

        # # Sort now_nodes by label
        # now_nodes = sorted(now_nodes, key = lambda p: p.label)

        # end Old stuff


        # Make a new dictionary for the now_nodes list

        # CORRECT BLOCK

        new_dictionary = defaultdict(list)
        total_now_edges = []

        for node in now_nodes:
            if node.label != []:
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
                            print('breaking')
                            temp_edges = []

                        #temp_edges = sorted(temp_edges, key = lambda p: p.straight_length)
                        new_vecs = [func(p, node) for p in temp_edges]
                        # print(new_vecs)
                        for k, p in zip(old_edges_node, temp_edges):
                            #print(p.label, p, k)
                            if p.label == []:
                                labels = [e.label for e in total_now_edges]
                                if k.label not in labels:
                                    # print(k, p)
                                    p.label = k.label
                        new_dictionary[node.label].append(temp_edges)
                        new_dictionary[node.label].append(new_vecs)
                    except:
                        pass


        # New stuff
        if upper_limit < 1000:
            count = 1000
        else:
            count = upper_limit + 1

        if upper_edge_limit < 1000:
            count_edge = 1000
        else:
            count_edge = upper_edge_limit + 1

        # count_edge += count_edge + 1 


        for node in now_nodes:
            check = 0
            if node.label == []:
                node.label = count
                count += 1
                check = 1
                print('node',node.label)
            for e in node.edges:
                if e.label == []:
                    e.label = count_edge
                    count_edge += 1
                    check = 1
                    print('edge',e.label)
            if check == 1:
                temp_edges = node.edges
                new_vecs = [func(p, node) for p in temp_edges]
                print(temp_edges, new_vecs)
                new_dictionary[node.label].append(temp_edges)
                new_dictionary[node.label].append(new_vecs)
        # end of stuff


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

        # END CORRECT BLOCK



        now_cells = self.label_cells(now_nodes, old_cells, now_cells)

        # Define a colony 
        edges2 = [e for e in now_edges if e.radius is not None]
        now_nodes, now_cells, edges2 = self.assign_intial_guesses(now_nodes, combined_dict, now_cells, old_cells, edges2, old_edges)
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
            closest_new_cell = min([c for c in now_cells], key = lambda p: np.linalg.norm(np.subtract(cell.centroid(), p.centroid())))
            if closest_new_cell.label == []:
                if np.linalg.norm(np.subtract(cell.centroid(), closest_new_cell.centroid())) < 100: #was 60 before
                    closest_new_cell.label = cell.label

        max_label = max([c.label for c in now_cells if c.label != []])
        if max_label > 999:
            count = max_label + 1
        else:
            count = 1000

        for j, cc in enumerate(now_cells):
            if cc.label == []:
                print('New cell label is', count)
                now_cells[j].label = count 
                count += 1


        return now_cells


    def assign_intial_guesses(self, now_nodes, combined_dict, now_cells, old_cells, edges2, old_edges):
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
                if cell.label in [old.label for old in old_cells]:
                    match_old_cell = [old for old in old_cells if old.label == cell.label][0]
                    cell.guess_pressure = match_old_cell.pressure
        
        for ed in old_edges:
            label = ed.label
            for new_ed in edges2:
                if new_ed.label == label:
                    if new_ed.guess_tension == []:
                        new_ed.guess_tension = ed.tension

        # for k,v in combined_dict.items():
        #     # v[0] is list of old edges and v[1] is list of matching new edges
        #     for old, new in zip(v[0], v[1]):
        #         match_edge = [e for e in edges2 if e == new][0]
        #         if match_edge.guess_tension == []:
        #             match_edge.guess_tension = old.tension
        #         else:
        #             match_edge.guess_tension = (old.tension + match_edge.guess_tension)/2

        # Note - right now edges2 not changing at all. Left it here so that if we want to add labels to edges2, can do it here

        return now_nodes, now_cells, edges2

    def first_computation(self, number_first, solver = None, type = None, **kwargs):
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

        tensions, P_T, A = colonies[name].calculate_tension(solver = solver, **kwargs)
        pressures, P_P, B = colonies[name].calculate_pressure(solver = solver, **kwargs)

        colonies[name].tension_matrix = A
        colonies[name].pressure_matrix = B

        return colonies, dictionary

    def jiggling_computation_based_on_prev(self, numbers, colonies = None, index = None, old_dictionary = None, solver= None, **kwargs):
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
            colonies, old_dictionary = self.first_computation(numbers[0], solver, type = 'Jiggling', **kwargs)
            print(colonies)
            colonies[str(0)].dictionary = old_dictionary
            index = 0

        print(index, numbers[index], numbers[index + 1])
        colonies[str(index + 1)], new_dictionary = self.track_timestep(colonies[str(index)], old_dictionary, numbers[index + 1])
        colonies[str(index + 1)].dictionary = new_dictionary
        tensions, P_T, A = colonies[str(index+1)].calculate_tension(solver = solver, **kwargs)
        pressures, P_P, B = colonies[str(index+1)].calculate_pressure(solver = solver, **kwargs)

        # Save tension and pressure matrix
        colonies[str(index+1)].tension_matrix = A
        colonies[str(index+1)].pressure_matrix = B

        index = index + 1
        if index < len(numbers) - 1:
            print('ok')
            colonies = self.jiggling_computation_based_on_prev(numbers, colonies, index, new_dictionary, solver,  **kwargs)

        return colonies

    def computation_based_on_prev(self, numbers, colonies = None, index = None, old_dictionary = None, solver= None, **kwargs):
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
            colonies, old_dictionary = self.first_computation(numbers[0], solver, **kwargs)
            colonies[str(numbers[0])].dictionary = old_dictionary
            index = 0

        if numbers[index + 1] == numbers[index]:
            colonies[str(index + 1)], new_dictionary = self.track_timestep(colonies[str(numbers[index])], old_dictionary, numbers[index + 1])
            colonies[str(index + 1)].dictionary = new_dictionary
            tensions, P_T, A = colonies[str(index+1)].calculate_tension(**kwargs)
            pressures, P_P, B = colonies[str(index+1)].calculate_pressure(**kwargs)

            # Save tension and pressure matrix
            colonies[str(index+1)].tension_matrix = A
            colonies[str(index+1)].pressure_matrix = B
        else:

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

    def all_perims_areas_lengths(self, colonies):
        """
        Return all unique edge tensions, edge radii and cell pressures in all colonies
        """
        all_perims = []
        all_areas = []
        all_lengths = []
        for t, v in colonies.items():
            index = str(t)
            cells = colonies[index].cells
            edges = colonies[index].tot_edges
            [all_lengths.append(e.straight_length) for e in edges if e.straight_length not in all_lengths]
            [all_perims.append(c.perimeter()) for c in cells if c.perimeter() not in all_perims]
            [all_areas.append(c.area()) for c in cells if c.area() not in all_areas]
        return all_lengths, all_perims, all_areas

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
        count = 0
        
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
                pylab.savefig('_tmp%05d.png'%count, dpi=200)
                plt.cla()
                plt.clf()
                plt.close()
                fig, ax = plt.subplots(1,1, figsize = (8, 5))
                ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
                count += 1

        fps = 1
        os.system("rm movie_single_node.mp4")

        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_single_node.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()


    def plot_tensions(self, fig, ax, colonies, specify_aspect = None, specify_color = None, **kwargs):
        """
        Make a tension movie over the colonies
        """
        max_num = len(colonies)

        all_tensions, _, _ = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)

        #min_ten, max_ten = None, None

        counter = 0
        for t, v in colonies.items():
            index = str(t)
            t= int(t)
            nodes = colonies[index].tot_nodes
            edges = colonies[index].tot_edges
            tensions = [e.tension for e in edges]
            colonies[index].plot_tensions(ax, fig, tensions, min_ten, max_ten, specify_color, **kwargs)
            if specify_aspect is not None:
                ax.set(xlim = [0,600], ylim = [0,600], aspect = 1)
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            pylab.savefig('_tmp%05d.png'%counter, dpi=200)
            counter += 1
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

    def plot_single_cells(self, fig, ax, ax1, ax3, colonies, cell_label):
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies) 
        all_lengths, all_perims, all_areas = self.all_perims_areas_lengths(colonies)       
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')
        _, max_perim, min_perim = self.get_min_max_by_outliers_iqr(all_perims)
        _, max_area, min_area = self.get_min_max_by_outliers_iqr(all_areas)
        frames = [i for i in colonies.keys()]

        
        pressures, areas, perimeters, change_in_area = [], [], [], [0]
        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            pres = [c.pressure for c in cells if c.label == cell_label]
            ares = [c.area() for c in cells if c.label == cell_label]
            perims = [c.perimeter() for c in cells if c.label == cell_label]
            if pres != []:
                pressures.append(pres[0])
                areas.append(ares[0])
                perimeters.append(perims[0])
                if j > 0:
                    change_in_area.append(perims[0] - perimeters[j-1])
            else:
                frames = frames[0:j]


        ax1.plot(frames, pressures, lw = 3, color = 'black')
        ax1.set_ylabel('Pressures', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, perimeters, 'blue')
        ax2.set_ylabel('Perimeters', color='blue')
        ax2.tick_params('y', colors='blue')

        ax3.plot(frames, areas, lw = 3, color = 'black')
        ax3.set_ylabel('Areas', color='black')
        ax3.set_xlabel('Frames')
        ax4 = ax3.twinx()
        ax4.plot(frames, change_in_area, 'blue')
        ax4.set_ylabel('Change in Area', color='blue')
        ax4.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax3.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax1.set(xlim = [0,31])
            ax2.set(xlim = [0,31], ylim = [min_perim, max_perim])
            ax3.set(xlim = [0,31], ylim = [min_area, max_area])
            ax4.set(xlim = [0,31])

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
            ax2.plot(i, current_cell.perimeter(), 'ok', color = 'red')
            ax3.plot(i, current_cell.area(), 'ok', color = 'red')
            ax4.plot(i, change_in_area[j], 'ok', color = 'red')

            fname = '_tmp%05d.png'%int(j) 
            plt.tight_layout()  
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1, ax3) = plt.subplots(3,1, figsize = (5.5,15)) 
            #fig, (ax, ax1) = plt.subplots(1,2, figsize = (14,6)) 
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, pressures, lw = 3, color = 'black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, perimeters, 'blue')
            ax2.set_ylabel('Perimeters', color='blue')
            ax2.tick_params('y', colors='blue')

            ax3.plot(frames, areas, lw = 3, color = 'black')
            ax3.set_ylabel('Areas', color='black')
            ax3.set_xlabel('Frames')
            ax4 = ax3.twinx()
            ax4.plot(frames, change_in_area, 'blue')
            ax4.set_ylabel('Change in Area', color='blue')
            ax4.tick_params('y', colors='blue')

        fps = 1
        os.system("rm movie_cell.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_cell.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def single_edge_plotting(self, fig, ax, ax1, ax3, colonies, node_label, edge_label):
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        all_lengths, all_perims, all_areas = self.all_perims_areas_lengths(colonies)   
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_len, min_len = self.get_min_max_by_outliers_iqr(all_lengths)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)

        frames = [i for i in colonies.keys()]
        # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])

        tensions = []
        radii = []
        length = []
        change_in_length = [0]

        for j, i in enumerate(frames):
            dictionary = colonies[str(i)].dictionary
            try:
                # for node edge plotting
                #edd = dictionary[node_label][0][edge_label]
                # for edge plotting based on edge label
                edd = [e for e in colonies[str(i)].tot_edges if e.label == edge_label][0]
                tensions.append(edd.tension)
                radii.append(edd.radius)
                length.append(edd.straight_length)
                if j >0:
                    change_in_length.append(edd.straight_length - length[j - 1])                    
            except:
                frames = frames[0:j]

        ax1.plot(frames, tensions, lw = 3, color = 'black')
        ax1.set_ylabel('Tension', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, radii, 'blue')
        ax2.set_ylabel('Radius', color='blue')
        ax2.tick_params('y', colors='blue')

        ax3.plot(frames, length, lw = 3, color = 'black')
        ax3.set_ylabel('Length', color='black')
        ax3.set_xlabel('Frames')
        ax4 = ax3.twinx()
        ax4.plot(frames, change_in_length, 'blue')
        ax4.set_ylabel('Change in length', color='blue')
        ax4.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            edges = colonies[str(i)].tot_edges
            nodes = colonies[str(i)].tot_nodes
            dictionary = colonies[str(i)].dictionary

            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
          #  ax1.set(xlim = [0,31], ylim = [0,0.004])
            ax1.set(xlim = [0,31], ylim = [min_ten, max_ten])
            ax2.set(xlim = [0,31], ylim = [min_rad, max_rad])
            ax3.set(xlim = [0,31], ylim = [min_len, max_len])
            ax4.set(xlim = [0,31], ylim = [min(change_in_length), max(change_in_length)])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax3.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
 #           [n.plot(ax, markersize = 10) for n in nodes if n.label == node_label]

            #current_edge = dictionary[node_label][0][edge_label]
            current_edge = [e for e in colonies[str(i)].tot_edges if e.label == edge_label][0]
            [current_edge.plot(ax, lw=3, color = 'red')]

            fname = '_tmp%05d.png'%int(j)   
            ax1.plot(i, current_edge.tension, 'ok', color = 'red')
            ax2.plot(i, current_edge.radius, 'ok', color = 'red')
            ax3.plot(i, current_edge.straight_length, 'ok', color = 'red')
            ax4.plot(i, change_in_length[j], 'ok', color = 'red')
            plt.tight_layout()

            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1, ax3) = plt.subplots(3,1, figsize = (5.5,15))  # figsize (14,6) before
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, tensions, lw = 3, color = 'black')
            ax1.set_ylabel('Tension', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, radii, 'blue')
            ax2.set_ylabel('Radius', color='blue')
            ax2.tick_params('y', colors='blue')

            ax3.plot(frames, length, lw = 3, color = 'black')
            ax3.set_ylabel('Length', color='black')
            ax3.set_xlabel('Frames')
            ax4 = ax3.twinx()
            ax4.plot(frames, change_in_length, 'blue')
            ax4.set_ylabel('Change in length', color='blue')
            ax4.tick_params('y', colors='blue')


    def plot_single_edges(self, fig, ax, ax1, ax3, colonies, node_label, edge_label):

        self.single_edge_plotting(fig, ax, ax1, ax3, colonies, node_label, edge_label)

        fps = 1
        os.system("rm movie_edge.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_edge.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_guess_tension(self, fig, ax, ax1, colonies, node_label, edge_label):
        frames = [i for i in colonies.keys()]

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        all_lengths, all_perims, all_areas = self.all_perims_areas_lengths(colonies)   
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)

        tensions = []
        guesses = []

        for j, i in enumerate(frames):
            dictionary = colonies[str(i)].dictionary
            try:
                edd = dictionary[node_label][0][edge_label]
                tensions.append(edd.tension)
                guesses.append(edd.guess_tension)
                # if type(edd.guess_tension) == int:
                #     guesses.append(edd.guess_tension) 
                # else:
                #     guesses.append([])                
            except:
                frames = frames[0:j]

        for j, i in enumerate(guesses):
            if i == []:
                guesses[j] = 0.002

        ax1.plot(frames, tensions, lw = 3, color = 'black')
        ax1.set_ylabel('Tension', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, guesses, 'blue')
        ax2.set_ylabel('Guess Tension', color='blue')
        ax2.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            edges = colonies[str(i)].tot_edges
            nodes = colonies[str(i)].tot_nodes
            dictionary = colonies[str(i)].dictionary

            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
          #  ax1.set(xlim = [0,31], ylim = [0,0.004])
            ax1.set(xlim = [0,31], ylim = [min_ten, max_ten])
            ax2.set(xlim = [0,31], ylim = [min_ten, max_ten])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
            current_edge = dictionary[node_label][0][edge_label]
            [current_edge.plot(ax, lw=3, color = 'red')]

            fname = '_tmp%05d.png'%int(j)   
            ax1.plot(i, current_edge.tension, 'ok', color = 'red')
            ax2.plot(i, guesses[j], 'ok', color = 'red')
            plt.tight_layout()

            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1) = plt.subplots(2,1, figsize = (5.5,10))  # figsize (14,6) before
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, tensions, lw = 3, color = 'black')
            ax1.set_ylabel('Tension', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, guesses, 'blue')
            ax2.set_ylabel('Guess Tension', color='blue')
            ax2.tick_params('y', colors='blue')
        fps = 1
        os.system("rm movie_edge_guess.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_edge_guess.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_guess_pressures(self, fig, ax, ax1,colonies, cell_label):


        frames = [i for i in colonies.keys()]

        
        pressures, guesses = [], []
        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            pres = [c.pressure for c in cells if c.label == cell_label]
            gess = [c.guess_pressure for c in cells if c.label == cell_label]
            if pres != []:
                pressures.append(pres[0])
                guesses.append(gess[0])
            else:
                frames = frames[0:j]

        for j, i in enumerate(guesses):
            if i == []:
                guesses[j] = 0


        ax1.plot(frames, pressures, lw = 3, color = 'black')
        ax1.set_ylabel('Pressures', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, guesses, 'blue')
        ax2.set_ylabel('Guess Pressure', color='blue')
        ax2.tick_params('y', colors='blue')


        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax1.set(xlim = [0,31])


            [e.plot(ax) for e in edges]
            current_cell = [c for c in cells if c.label == cell_label][0]
            # ax.plot(current_cell, color = 'red')
            [current_cell.plot(ax, color = 'red', )]


            x = [n.loc[0] for n in current_cell.nodes]
            y = [n.loc[1] for n in current_cell.nodes]
            ax.fill(x, y, c= 'red', alpha = 0.2)
            for e in current_cell.edges:
                e.plot_fill(ax, color = 'red', alpha = 0.2)

            ax1.plot(i, current_cell.pressure, 'ok', color = 'red')
            ax2.plot(i, guesses[j], 'ok', color = 'red')

            fname = '_tmp%05d.png'%int(j) 
            plt.tight_layout()  
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1) = plt.subplots(2,1, figsize = (5.5,10)) 
            ax1.plot(frames, pressures, lw = 3, color = 'black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, guesses, 'blue')
            ax2.set_ylabel('Guess Pressure', color='blue')
            ax2.tick_params('y', colors='blue')

        fps = 1
        os.system("rm movie_cell_guess.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_cell_guess.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_histogram(self, fig, ax, ax1, ax2, colonies):

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        max_ten, min_ten, max_pres, min_pres = max(all_tensions), min(all_tensions), max(all_pressures), min(all_pressures)
        # _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        # _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        frames = [i for i in colonies.keys()]

        ensemble_tensions, ensemble_pressures = [], []

        for j, i in enumerate(frames):
            this_colony_dict = dict((k, v) for k, v in colonies.items() if int(i) + 1 >int(k) >= int(i)) 
            try:
                this_tensions, this_radii, this_pressures = self.all_tensions_and_radius_and_pressures(this_colony_dict)
                ensemble_tensions.append(this_tensions)
                ensemble_pressures.append(this_pressures)                                  
            except:
                frames = frames[0:j]

        for j, i in enumerate(frames):

            edges = colonies[str(i)].tot_edges
            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])

            [e.plot(ax) for e in edges]

            # the histogram of the data
            n, bins, patches = ax1.hist(ensemble_tensions[j], 25, range = (min_ten, max_ten))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # scale values to interval [0,1]
            col = bin_centers - min(bin_centers)
            col /= max(col)

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm.viridis(c))

            # the histogram of the data
            n, bins, patches = ax2.hist(ensemble_pressures[j], 25, range = (min_pres, max_pres))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # scale values to interval [0,1]
            col = bin_centers - min(bin_centers)
            col /= max(col)

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm.viridis(c))

            ax1.set_xlabel('Tension')
            ax2.set_xlabel('Pressure')
            ax1.set_ylabel('Frequency')
            ax2.set_ylabel('Frequency')

            fname = '_tmp%05d.png'%int(j) 
            plt.tight_layout()  
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1, ax2) = plt.subplots(3,1, figsize = (5.5,15)) 
        fps = 1
        os.system("rm movie_histograms.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_histograms.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def get_repeat_edge(self, colonies):
        labels = []
        for t, v in colonies.items():    
            labels.append([e.label for e in v.tot_edges if e.label != []])

        repeat_edge_labels = set(labels[0]).intersection(*labels)
        return list(repeat_edge_labels)

    def get_repeat_cell(self, colonies):
        labels = []
        for t, v in colonies.items():    
            labels.append([c.label for c in v.cells if c.label != []])

        repeat_cell_labels = set(labels[0]).intersection(*labels)
        return list(repeat_cell_labels)

    def get_repeat_nodes(self, colonies):
        labels = []
        for t, v in colonies.items():    
            labels.append([n.label for n in v.tot_nodes if n.label != []])

        repeat_node_labels = set(labels[0]).intersection(*labels)
        return list(repeat_node_labels)

    def simple_plot_all_edges(self, fig, ax, colonies):

        labels = self.get_repeat_edge(colonies)

        tensions, frames = [], []
        ax.set_xlabel('Frames')
        ax.set_ylabel('Tensions')
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.set(xlim = [0,31])

        for lab in labels:
            for t, v in colonies.items():
                tensions.append([e.tension for e in v.tot_edges if e.label == lab][0])
                frames.append(int(t))
            ax.plot(frames, tensions)
            # ax.plot(frames, tensions, lw = 3, color = cm.viridis(min(frames)/31))

        fname = '_tmp%05d.png'%int(min(frames) -1) 
        plt.tight_layout()  
        plt.savefig(fname)

        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > min(frames))
        # if new_colony_range != {}:
        #     self.simple_plot_all_edges(fig, ax, new_colony_range)




    def plot_all_edges(self, fig, ax, ax1, colonies, index = None, old_labels = None, counter = None, tensions = None, frames = None):

        labels = self.get_repeat_edge(colonies)

        if tensions == None and frames == None:
            tensions, frames = [], []
        ax1.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
        ax.set_xlabel('Frames')
        ax.set_ylabel('Tensions')
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.set(xlim = [0,31])

        if index == None:
            index = 0

        for lab in labels:
            if old_labels == None or lab not in old_labels:
                current_frames, current_tensions = [], []
                for t, v in colonies.items():
                    current_tensions.append([e.tension for e in v.tot_edges if e.label == lab][0])
                    current_frames.append(int(t))
                    tensions.append([e.tension for e in v.tot_edges if e.label == lab][0])
                    frames.append(int(t))
                #ax.plot(frames, tensions, lw = 1, color = 'black')
                ax.plot(frames, tensions, '.', color = 'black')
                ax.plot(current_frames, current_tensions, 'ok', color = 'yellow')
                #ax.plot(current_frames, current_tensions, 'ok', color = cm.viridis(lab/ max(labels)))
                count = 0
                for t,v in colonies.items():
                    ax1.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
                    [e.plot(ax1) for e in v.tot_edges]
                    [e.plot(ax1, color = 'red', lw = 3) for e in v.tot_edges if e.label == lab]
                    ax.plot(current_frames[count], current_tensions[count], 'ok', color = 'red')
                    count += 1

                    fname = '_tmp%05d.png'%int(index) 
                    plt.tight_layout()  
                    plt.savefig(fname)
                    index += 1
                    plt.clf() 
                    plt.cla() 
                    plt.close()
                    fig, (ax1, ax) = plt.subplots(2,1, figsize = (5.5,10))
                    ax.set_xlabel('Frames')
                    ax.set_ylabel('Tensions')
                    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
                    ax.set(xlim = [0,31])
                    #ax.plot(frames, tensions, lw = 1, color = 'black')
                    ax.plot(frames, tensions, '.', color = 'black')
                    ax.plot(current_frames, current_tensions, 'ok', color = 'yellow')
                    #ax.plot(current_frames, current_tensions, 'ok', color = cm.viridis(lab/ max(labels)))
            #[e.plot(ax1, color = cm.viridis(lab/max(labels)), lw = 3) for e in v.tot_edges if e.label == lab]

        if counter == None:
            counter = 1
        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > counter)
        if new_colony_range != {}:
            old_labels = labels
            self.plot_all_edges(fig, ax, ax1, new_colony_range, index, old_labels, counter + 1, tensions, frames)

    def make_all_edge_movie(self, fig, ax, ax1, colonies):
        ax1.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
        self.plot_all_edges(fig, ax, ax1, colonies)
        fps = 1
        os.system("rm movie_all_edges.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_all_edges.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_all_cells(self, ax, colonies):

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)      
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        frames = [i for i in colonies.keys()]

        
        pressures, areas, perimeters, change_in_area = [], [], [], [0]
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


        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))

            ax1.set(xlim = [0,31])

            [e.plot(ax) for e in edges]
            current_cell = [c for c in cells if c.label == cell_label][0]
            # ax.plot(current_cell, color = 'red')
            [current_cell.plot(ax, color = 'red' )]


            x = [n.loc[0] for n in current_cell.nodes]
            y = [n.loc[1] for n in current_cell.nodes]
            ax.fill(x, y, c= 'red', alpha = 0.2)
            for e in current_cell.edges:
                e.plot_fill(ax, color = 'red', alpha = 0.2)

            ax1.plot(i, current_cell.pressure, 'ok', color = 'red')
            ax2.plot(i, current_cell.perimeter(), 'ok', color = 'red')
            ax3.plot(i, current_cell.area(), 'ok', color = 'red')
            ax4.plot(i, change_in_area[j], 'ok', color = 'red')

            fname = '_tmp%05d.png'%int(j) 
            plt.tight_layout()  
            plt.savefig(fname)
            plt.clf() 
            plt.cla() 
            plt.close()
            fig, (ax, ax1) = plt.subplots(2,1, figsize = (5.5,15)) 
            #fig, (ax, ax1) = plt.subplots(1,2, figsize = (14,6)) 
            # ax.set(xlim = [0,1030], ylim = [0,1030], aspect = 1)
            # ax1.set(xlim = [frames[0], frames[-1]], ylim = [0, 0.004])
            ax1.plot(frames, pressures, lw = 3, color = 'black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, perimeters, 'blue')
            ax2.set_ylabel('Perimeters', color='blue')
            ax2.tick_params('y', colors='blue')

            ax3.plot(frames, areas, lw = 3, color = 'black')
            ax3.set_ylabel('Areas', color='black')
            ax3.set_xlabel('Frames')
            ax4 = ax3.twinx()
            ax4.plot(frames, change_in_area, 'blue')
            ax4.set_ylabel('Change in Area', color='blue')
            ax4.tick_params('y', colors='blue')

    def seaborn_cells_dataframe_tensor(self, colonies, jump_number = 1, data = None):

        initial_index = [int(k) for k,v in colonies.items()][0]
        # labels = [e.label for e in colonies[str(initial_index)].cells if e.label != []]
        # labels = sorted(labels)

        if data is None:
            data = {'Index_Time': [], 'Time': [], 'Velocity_x': [], 'Velocity_y': [],'x_pos': [], 'y_pos': [],'u_translational': [], \
             'v_translational': [], 'ux_translational': [], 'uy_translational': [], 'vx_translational': [],'vy_translational': [], \
              'Velocity_gradient_tensor': [], 'Rotation': [], 'Strain_rate': [], 'Rate_of_area_change': [], 'Eigenvalues_strain_1': [],\
               'Eigenvalues_strain_2': [], 'Eigenvectors_strain': [],'Eigenvectors_strain_1': [],'Eigenvectors_strain_2': [], \
               'Eigenvalues_rotation': [], 'Eigenvectors_rotation': [], 'First_invariant': [], 'Second_invariant': [], \
               'Mean_resid': [], 'Std_resid': [], 'Mean_tension': [], 'Mean_pressure': [], 'Area': [], 'Perimeter': [], \
               'Number_of_edges': [], 'Length_of_edges': [], 'Radius_of_edges': [], 'Std_tension': [], 'Change_in_mean_tension': [], 'Change_in_std_tension': [],\
               'Std_pressure': [], 'Area_std': [], 'Perimeter_std': [], 'Number_of_edges_std': [],'Count_topology': [],'Change_in_connectivity': [], 'Length_of_edges_std': [], 'Radius_of_edges_std': []}                  
            tensor_dataframe = pd.DataFrame(data)
            tensor_dataframe.set_index(['Index_Time'], inplace = True)


        for t, v in colonies.items():

            # Intial 0

            if int(t) == [int(k) for k,v in colonies.items()][-1]:
                pass
                # data['Index_Time'].append(int(t))
                # data['Time'].append(int(t))
                # data['Velocity_x'].append(0)
                # data['Velocity_y'].append(0)
                # data['x_pos'].append(0)
                # data['y_pos'].append(0)
                # data['Velocity_gradient_tensor'].append(0)
                # data['Rotation'].append(0)
                # data['Strain_rate'].append(0)
                # data['Rate_of_area_change'].append(0)
                # data['Eigenvalues_strain_1'].append(0)
                # data['Eigenvalues_strain_2'].append(0)
                # data['Eigenvectors_strain'].append(0)
                # data['Eigenvalues_rotation'].append(0)
                # data['Eigenvectors_rotation'].append(0)
                # data['First_invariant'].append(0)
                # data['Second_invariant'].append(0)
                # data['ux_translational'].append(0)
                # data['uy_translational'].append(0)
                # data['vx_translational'].append(0)
                # data['vy_translational'].append(0)
                # data['Eigenvectors_strain_1'].append(0)
                # data['Eigenvectors_strain_2'].append(0)
                # data['u_translational'].append(0)
                # data['v_translational'].append(0)
                # data['Mean_resid'].append(0)
                # data['Std_resid'].append(0)
                # data['Mean_tension'].append(0)
                # data['Std_tension'].append(0)
                # data['Mean_pressure'].append(0)
                # data['Std_pressure'].append(0)

                # data['Area'].append(0)
                # data['Area_std'].append(0)
                # data['Perimeter'].append(0)
                # data['Perimeter_std'].append(0)
                # data['Number_of_edges'].append(0)
                # data['Number_of_edges_std'].append(0)
                # data['Length_of_edges'].append(0)
                # data['Length_of_edges_std'].append(0)
                # data['Radius_of_edges'].append(0)
                # data['Radius_of_edges_std'].append(0)


            # Check that its not the last one

            if int(t) < [int(k) for k,v in colonies.items()][-1]:
                # define new colony as this time frame and the next
                new_colony_range = dict((k, v) for k, v in colonies.items() if int(t) <= int(k) <= int(t) + jump_number)

                labels = self.get_repeat_cell(new_colony_range)
                nodes_labels = self.get_repeat_nodes(new_colony_range)

                min_t = [int(t) for t, v in new_colony_range.items()][0]
                print(new_colony_range)
                x_pos1, y_pos1, x_pos2, y_pos2, total_con_labels_1, total_con_labels_2 = [], [], [], [], [], []
                tension_mean_1, tension_mean_2, tension_std_1, tension_std_2 = [], [], [], []
                for t, v in new_colony_range.items():

                    cells = sorted(v.cells, key = lambda x: x.label)
                    for c in cells:
                        if c.label in labels:

                            xc, yc = c.centroid()[0], c.centroid()[1]
                            if int(t) == min_t:
                                x_pos1.append(xc)
                                y_pos1.append(yc)
                            else:
                                x_pos2.append(xc)
                                y_pos2.append(yc)

                    sorted_nodes = sorted(v.tot_nodes, key = lambda x: x.label)

                    for n in sorted_nodes:
                        if n.label in nodes_labels:
                            con_labels = [e.label for e in n.edges]
                            if int(t) == min_t:
                                total_con_labels_1.append(con_labels)
                            else:
                                total_con_labels_2.append(con_labels)
                    if int(t) == min_t:
                        tension_mean_1 = np.mean(np.array([e.tension for e in v.tot_edges]))
                        tension_std_1 = np.std(np.array([e.tension for e in v.tot_edges]))
                    else:
                        tension_mean_2 = np.mean(np.array([e.tension for e in v.tot_edges]))
                        tension_std_2 = np.std(np.array([e.tension for e in v.tot_edges]))



                for t, v in new_colony_range.items():

                    if int(t) == min_t:

                        residuals = []
                        for n in v.tot_nodes:
                            x  = [e.tension for e in v.tot_edges]
                            if len(n.edges) > 2:
                                tensions  =[]
                                indices = n.edge_indices
                                for i in range(len(indices)):
                                    tensions.append([x[indices[i]]])
                                tensions = np.array(tensions)

                                node_vecs = n.tension_vectors
                                tension_vecs = np.multiply(node_vecs, tensions)
                                tension_vec_mags = [np.hypot(*vec) for vec in tension_vecs]
                                resid_vec = np.sum(tension_vecs, 0)  
                                resid_mag = np.hypot(*resid_vec)
                                residuals.append(resid_mag)
                        data['Mean_resid'].append(np.mean(np.array(residuals)))
                        data['Std_resid'].append(np.std(np.array(residuals)))

                        data['Mean_tension'].append(np.mean(np.array([e.tension for e in v.tot_edges])))
                        data['Std_tension'].append(np.std(np.array([e.tension for e in v.tot_edges])))
                        data['Mean_pressure'].append(np.mean(np.array([c.pressure for c in v.cells])))
                        data['Std_pressure'].append(np.std(np.array([c.pressure for c in v.cells])))

                        data['Change_in_mean_tension'].append(tension_mean_2 - tension_mean_1)
                        data['Change_in_std_tension'].append(tension_std_2 - tension_std_1)


                        data['Area'].append(np.mean(np.array([c.area() for c in v.cells])))
                        data['Area_std'].append(np.std(np.array([c.area() for c in v.cells])))
                        data['Perimeter'].append(np.mean(np.array([c.perimeter() for c in v.cells])))
                        data['Perimeter_std'].append(np.std(np.array([c.perimeter() for c in v.cells])))
                        data['Number_of_edges'].append(np.mean(np.array([len(n.edges) for n in v.tot_nodes])))
                        data['Number_of_edges_std'].append(np.std(np.array([len(n.edges) for n in v.tot_nodes])))
                        data['Length_of_edges'].append(np.mean(np.array([e.straight_length for e in v.tot_edges])))
                        data['Length_of_edges_std'].append(np.std(np.array([e.straight_length for e in v.tot_edges])))
                        data['Radius_of_edges'].append(np.mean(np.array([e.radius for e in v.tot_edges])))
                        data['Radius_of_edges_std'].append(np.std(np.array([e.radius for e in v.tot_edges])))

                        Number_of_topology_changes = 0
                        Change_in_number = 0

                        for k, v in zip(total_con_labels_1, total_con_labels_2):
                            try:
                                if set(k) != set(v):
                                    Number_of_topology_changes += 1
                                    print(int(t), k, v)
                            except:
                                pass
                            if len(k) != len(v):
                                Change_in_number += 1

                        data['Count_topology'].append(Number_of_topology_changes)
                        data['Change_in_connectivity'].append(Change_in_number)


                        u = np.array(x_pos2) - np.array(x_pos1)
                        v = np.array(y_pos2) - np.array(y_pos1)
                        data['Index_Time'].append(int(t))
                        data['Time'].append(int(t))
                        data['Velocity_x'].append(np.mean(u))
                        data['Velocity_y'].append(np.mean(v))
                        data['x_pos'].append(np.mean(x_pos1))
                        data['y_pos'].append(np.mean(y_pos1))
                        dudx, intercept_ux, r_value_ux, p_value_ux, std_err_ux = stats.linregress(x_pos1,u)
                        dudy, intercept_uy, r_value_uy, p_value_uy, std_err_uy = stats.linregress(y_pos1,u)
                        dvdx, intercept_vx, r_value_vx, p_value_vx, std_err_vx = stats.linregress(x_pos1,v)
                        dvdy, intercept_vy, r_value_vy, p_value_vy, std_err_vy = stats.linregress(y_pos1,v)

                        data['ux_translational'].append(intercept_ux)
                        data['uy_translational'].append(intercept_uy)
                        data['u_translational'].append(np.sqrt(intercept_uy**2 + intercept_ux**2))
                        data['v_translational'].append(np.sqrt(intercept_vy**2 + intercept_vx**2))
                        data['vx_translational'].append(intercept_vx)
                        data['vy_translational'].append(intercept_vy)

                        tensor = np.array([[dudx,dudy], [dvdx,dvdy]])
                        data['Velocity_gradient_tensor'].append(tensor)
                        omega = (tensor - tensor.T)/2
                        strain = (tensor + tensor.T)/2

                        data['Rotation'].append(omega)
                        data['Strain_rate'].append(strain)
                        trace = strain[0,0] + strain[1,1]
                        data['Rate_of_area_change'].append(trace)
                        w, v = np.linalg.eig(strain)

                        data['Eigenvalues_strain_1'].append(w[0])
                        data['Eigenvalues_strain_2'].append(w[1])
                        data['Eigenvectors_strain'].append(v)
                        v = v.T
                        angle1 = np.rad2deg(np.arctan2(v[0][1], v[0][0])) 
                        angle2 = np.rad2deg(np.arctan2(v[1][1], v[1][0])) 

                        data['Eigenvectors_strain_1'].append(angle1)
                        data['Eigenvectors_strain_2'].append(angle2)

                        w2, v2 = np.linalg.eig(omega)
                        print(np.imag(w2[1]))
                        # data['Eigenvalues_rotation'].append(np.imag(w2[1]))
                        data['Eigenvalues_rotation'].append(omega[0,1])
                        data['Eigenvectors_rotation'].append(v2)

                        P = -tensor[0,0] - tensor[1,1]
                        Q = 1/2*P**2 -1/2*tensor[1,0] * tensor[0,1] -1/2 *tensor[0,1]*tensor[1,0] - 1/2 *tensor[0,0] * tensor[0,0] -1/2 *tensor[1,1] * tensor[1,1]
                        data['First_invariant'].append(P)
                        data['Second_invariant'].append(Q)



        tensor_dataframe = pd.DataFrame(data)
        tensor_dataframe.set_index(['Index_Time'], inplace = True)
        return tensor_dataframe
                
    def plot_tensor_dataframe(self,ax,  colonies, tensor_dataframe):

        count = 0
        for t,v in colonies.items():
            if int(t) < [int(t) for t, v in colonies.items()][-1]:

                #tensor_first = tensor_dataframe.Rotation[[int(t) for t, v in colonies.items()][0]]
                tensor_first = tensor_dataframe.Strain_rate[[int(t) for t, v in colonies.items()][0]]

                #tensor_first = tensor_dataframe.Velocity_gradient_tensor[[int(t) for t, v in colonies.items()][0]]

                radius = max(abs(np.sqrt(tensor_first[0,0]**2 + tensor_first[0,1]**2)), abs(np.sqrt(tensor_first[1,1]**2 + tensor_first[1,0]**2)))
                circle1 = plt.Circle((0, 0), radius, fill = False)
                ax.add_artist(circle1)
                ax.set_aspect(1)
                ax.set(xlim = [-0.06, 0.06], ylim = [-0.06, 0.06])

                #tensor = tensor_dataframe.Rotation[int(t)]
                tensor = tensor_dataframe.Strain_rate[int(t)]
                tensor_rotation = tensor_dataframe.Rotation[int(t)]
                #tensor = tensor_dataframe.Velocity_gradient_tensor[int(t)]


                # pt1_u = [tensor[0,0], tensor[0,1]]
                # pt2_u = [-tensor[0,0], -tensor[0,1]]
                # pt1_u = [tensor[0,0], 0]
                # pt2_u = [-tensor[0,0], 0]
                w, v = np.linalg.eig(tensor)

                # f, m = np.linalg.eig(tensor_rotation)
                angle = tensor_rotation[0,1]

                # For rotation matrix
                # w = [np.imag(i) for i in w]

                v = v.T

                # pt1_u = [w[0], 0]
                # pt2_u = [-w[0], 0]
                pt1_u = -w[0] * v[0]
                pt2_u = +w[0] * v[0]


                pts_u = [pt1_u, pt2_u]
                x_u, y_u = zip(*pts_u)

                # pt1_v = [tensor[1,0], tensor[1,1]]
                # pt2_v = [-tensor[1,0], -tensor[1,1]]
                # pt1_v = [0, tensor[1,1]]
                # pt2_v = [0, -tensor[1,1]]
                # pt1_v = [0, w[0]]
                # pt2_v = [0, -w[0]]

                pt1_v = -w[1] * v[1]
                pt2_v = +w[1] * v[1]

                pts_v = [pt1_v, pt2_v]
                x_v, y_v = zip(*pts_v)


                # Old stuff

                # For rotation matrix
                # w = [np.imag(i) for i in w]

                # pt1_u = [w[0], 0]
                # pt2_u = [-w[0], 0]



                # pts_u = [pt1_u, pt2_u]
                # x_u, y_u = zip(*pts_u)

                # # pt1_v = [tensor[1,0], tensor[1,1]]
                # # pt2_v = [-tensor[1,0], -tensor[1,1]]
                # # pt1_v = [0, tensor[1,1]]
                # # pt2_v = [0, -tensor[1,1]]
                # pt1_v = [0, w[0]]
                # pt2_v = [0, -w[0]]

                # pts_v = [pt1_v, pt2_v]
                # x_v, y_v = zip(*pts_v)

                if tensor[0,0] > 0:
                    ax.plot(x_u, y_u, color = 'blue')
                else:
                    ax.plot(x_u, y_u, color = 'red')

                if tensor[1,1] > 0:
                    ax.plot(x_v, y_v, color = 'blue')
                else:
                    ax.plot(x_v, y_v, color = 'red')

                if angle > 0:
                    center, th1, th2 = (0,0), 0, 180
                    #ax.annotate("", xy=(-(radius + 0.01), 0), arrowprops=dict(arrowstyle="->"))
                    ax.arrow(-(radius + 0.01), 0.005, 0, -0.005, color = 'green')
                else:
                    center, th1, th2 = (0,0), 180, 0
                    ax.arrow((radius + 0.01), -0.005, 0, 0.005, color = 'green')

                print(th1, th2)

                patch = matplotlib.patches.Arc(center, 2*(radius + 0.01), 2*(radius + 0.01),
                                               0, th1, th2, color = 'green')
                ax.add_patch(patch)



                fname = '_tmp%05d.png'%int(count) 
                plt.tight_layout()  
                plt.savefig(fname)
                plt.clf() 
                plt.cla() 
                plt.close()
                fig, ax = plt.subplots(1,1, figsize = (8,5)) 
                count += 1

        fps = 1
        os.system("rm movie_strain_rate.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_strain_rate.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

        





        # w, v = np.linalg.eig(tensor)
        # vectors = w *v 

        # print(w)
        # print(v)
        # print(vectors)

        # for j, s in enumerate(w):
        #     if s > 0:
        #         ax.plot(vectors[j].real, vectors[j].imag, color = 'blue')
        #     else:
        #         ax.plot(vectors[j].real, vectors[j].imag, color = 'red')

        # strain_eigenvectors = tensor_dataframe.Eigenvectors_strain[t]
        # strain_eigenvalues = tensor_dataframe.Eigenvalues_strain[t]
        # strain_vectors = strain_eigenvalues * strain_eigenvectors

        # rotation_eigenvectors = tensor_dataframe.Eigenvectors_rotation[t]
        # rotation_eigenvalues = tensor_dataframe.Eigenvalues_rotation[t]
        # rotation_vectors = rotation_eigenvalues * rotation_eigenvectors

        # plt.Circle((1,1), radius = max(strain_eigenvalues), fill = 'False')
        # for j, s in enumerate(strain_eigenvalues):

        #     if s > 0:
        #         ax.plot(strain_vectors[j] + [1,1] + , color = 'red')
        #     else:
        #         ax.plot(strain_vectors[j] + [1,1], color = 'blue')

        # plt.Circle((-1,1), radius = max(rotation_eigenvalues), fill = 'False')
        # for j, s in enumerate(rotation_eigenvalues):

        #     if s > 0:
        #         ax.plot(np.imag(rotation_vectors[j]) + [-1,1], color = 'red')
        #     else:
        #         ax.plot(np.imag(rotation_vectors[j]) + [-1,1], color = 'blue')






    def seaborn_nodes_dataframe(self, colonies, data, old_labels = None, counter = None):

        initial_index = [int(k) for k,v in colonies.items()][0]
        labels = [e.label for e in colonies[str(initial_index)].tot_nodes if e.label != []]
        labels = sorted(labels)

        if data is None:
            data = {'Index_Node_Label': [],'Index_Time': [], 'Time': [], 'Number_of_connected_edges':[],'Connected_edge_labels': [],  'Residual': [], 'Change_in_num_con_edges': [], 'Length_of_connected_edges': [], 'Movement_from_prev_t': [], 'Mean_radius_of_connected_edges': [], 'Node_Label': [], 'Mean_Tension': [], 'Change_in_mean_tension': []}
            nodes_dataframe = pd.DataFrame(data)
            nodes_dataframe.set_index(['Index_Node_Label','Index_Time'], inplace = True)

        for lab in labels:
            if old_labels == None or lab not in old_labels:
                tensions, num_of_con_edges = [], []
                node_index = 0
                locs = []
                for t, v in colonies.items():
                    if [len(n.edges) for n in v.tot_nodes if n.label == lab] != []:
                        x  = [e.tension for e in v.tot_edges]
                        if [len(n.edges) for n in v.tot_nodes if n.label == lab][0] > 2:
                            start_tensions  =[]
                            node1 = [n for n in v.tot_nodes if n.label == lab][0]
                            indices = node1.edge_indices
                            for i in range(len(indices)):
                                start_tensions.append([x[indices[i]]])
                            start_tensions = np.array(start_tensions)

                            node_vecs = node1.tension_vectors
                            tension_vecs = np.multiply(node_vecs, start_tensions)
                            tension_vec_mags = [np.hypot(*vec) for vec in tension_vecs]
                            resid_vec = np.sum(tension_vecs, 0)  
                            resid_mag = np.hypot(*resid_vec)
                        else:
                            resid_mag = np.NaN

                        con_labels = [e.label for n in v.tot_nodes for e in n.edges if n.label == lab]           
                        data['Connected_edge_labels'].append(con_labels)
                        
                        data['Residual'].append(resid_mag)
                        data['Time'].append(int(t))
                        data['Index_Time'].append(int(t))
                        data['Node_Label'].append(lab)
                        data['Index_Node_Label'].append(lab)
                        data['Number_of_connected_edges'].append([len(n.edges) for n in v.tot_nodes if n.label == lab][0])
                        num_of_con_edges.append([len(n.edges) for n in v.tot_nodes if n.label == lab][0])
                        data['Length_of_connected_edges'].append(sum([e.straight_length for n in v.tot_nodes for e in n.edges if n.label == lab]))
                        data['Mean_radius_of_connected_edges'].append(sum([e.radius for n in v.tot_nodes for e in n.edges if n.label == lab]))
                        data['Mean_Tension'].append(np.mean([e.tension for n in v.tot_nodes for e in n.edges if n.label == lab]))
                        tensions.append(np.mean([e.tension for n in v.tot_nodes for e in n.edges if n.label == lab]))
                        locs.append([n.loc for n in v.tot_nodes if n.label == lab][0])
                        if node_index == 0:
                            data['Movement_from_prev_t'].append(0)
                            data['Change_in_mean_tension'].append(0)
                            data['Change_in_num_con_edges'].append(0)
                            node_index += 1
                        else:
                            dist = np.linalg.norm(np.subtract(locs[node_index], locs[node_index - 1]))
                            data['Movement_from_prev_t'].append(dist)
                            data['Change_in_mean_tension'].append(tensions[node_index] - tensions[node_index - 1])
                            data['Change_in_num_con_edges'].append(abs(num_of_con_edges[node_index] - num_of_con_edges[node_index - 1]))
                            node_index += 1

                        nodes_dataframe = pd.DataFrame(data)
                        nodes_dataframe.set_index(['Index_Node_Label','Index_Time'], inplace = True)


        if counter == None:
            counter = 1
            #counter = 30
        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > counter)
        #new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) < counter)
        if new_colony_range != {}:
            old_labels = labels
            counter = counter + 1
            #counter = counter - 1
            self.seaborn_nodes_dataframe(new_colony_range, data,  old_labels,  counter)
            nodes_dataframe = pd.DataFrame(data)
            nodes_dataframe.set_index(['Index_Node_Label','Index_Time'], inplace = True)
            return nodes_dataframe



    def seaborn_plot(self, ax, colonies, common_edge_labels, common_cell_labels, data = None, cell_data = None, old_labels = None, old_cell_labels = None, counter = None, min_ten = None, max_ten = None, min_pres = None, max_pres = None):

        #labels = self.get_repeat_edge(colonies)


        if min_ten == None and max_ten == None and min_pres == None and max_pres == None:
            all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)      
            _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')
            _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
            print(len(all_tensions))
            min_ten = min(all_tensions)
            max_ten = max(all_tensions)
            print(min_ten, max_ten)

        initial_index = [int(k) for k,v in colonies.items()][0]
        labels = [e.label for e in colonies[str(initial_index)].tot_edges if e.label != []]
        cell_labels = [c.label for c in colonies[str(initial_index)].cells if c.label != []]

        if data is None:
            data = {'Index_Edge_Labels': [], 'Index_Time':[], 'Edge_Labels': [], 'Strain_rate': [],'Normalized_Tensions': [],'Local_normalized_tensions': [], 'Deviation': [], 'Tensions': [], 'Repeat_Tensions': [], 'Change_in_tension': [], 'Time': [], 'Curvature': [], 'Radius': [], 'Straight_Length': [], 'Total_connected_edge_length':[], 'Change_in_length': [], 'Change_in_connected_edge_length': [],'Binary_length_change': [] , 'Binary_connected_length_change':[]}
            edges_dataframe = pd.DataFrame(data)
            edges_dataframe.set_index(['Index_Edge_Labels','Index_Time'])
            #edges_dataframe.set_index(['Index_Edge_Labels', 'Index_Time'], inplace = True)

        if cell_data is None:
            cell_data = {'Index_Cell_Labels': [], 'Index_Cell_Time':[], 'Cell_Labels': [], 'Centroid_movement': [], 'Rotation': [], 'Binary_rotation': [],  'Number_of_edges': [],  'Normalized_Pressures': [], 'Pressures': [], 'Mean_node_edge_tension': [], 'Sum_edge_tension': [], 'Repeat_Pressures': [], 'Change_in_pressure':[] , 'Cell_Time': [], 'Area': [], 'Perimeter': [], 'Change_in_area': [], 'Binary_area_change': [], 'Change_in_perimeter': [], 'Binary_perim_change': []}
            cells_dataframe = pd.DataFrame(cell_data)
            cells_dataframe.set_index(['Index_Cell_Labels', 'Index_Cell_Time'], inplace = True)


        for lab in labels:
            if old_labels == None or lab not in old_labels:
                edge_index = 0
                lengths, con_lengths, tensions = [], [], []
                for t, v in colonies.items():
                    mean_tens = np.mean([e.tension for e in v.tot_edges])
                    if [e.tension for e in v.tot_edges if e.label == lab] != []:
                        data['Edge_Labels'].append(lab)
                        data['Index_Edge_Labels'].append(lab)
                        data['Tensions'].append([e.tension for e in v.tot_edges if e.label == lab][0])
                        data['Normalized_Tensions'].append(([e.tension for e in v.tot_edges if e.label == lab][0] - min_ten)/float(max_ten - min_ten))
                        if lab in common_edge_labels:
                            data['Repeat_Tensions'].append([e.tension for e in v.tot_edges if e.label == lab][0])
                        else:
                            data['Repeat_Tensions'].append(np.NaN)

                        data['Local_normalized_tensions'].append([e.tension for e in v.tot_edges if e.label == lab][0]/mean_tens)

                        current_edge = [e for e in v.tot_edges if e.label == lab][0]
                        con_edges = [e for n in current_edge.nodes for e in n.edges if e != current_edge]
                        con_lengths.append(sum([e.straight_length for e in con_edges]))
                        data['Deviation'].append([e.tension for e in v.tot_edges if e.label == lab][0] - np.mean(np.array([e.tension for e in v.tot_edges ])))
                        data['Total_connected_edge_length'].append(sum([e.straight_length for e in con_edges]))
                        data['Time'].append(int(t))
                        data['Index_Time'].append(int(t))
                        data['Radius'].append([e.radius for e in v.tot_edges if e.label == lab][0])
                        data['Curvature'].append([1/e.radius for e in v.tot_edges if e.label == lab][0])
                        [tensions.append([e.tension for e in v.tot_edges if e.label == lab][0])]
                        [lengths.append([e.straight_length for e in v.tot_edges if e.label == lab][0])]
                        data['Straight_Length'].append([e.straight_length for e in v.tot_edges if e.label == lab][0])
                        if edge_index == 0:
                            data['Change_in_length'].append(0)
                            data['Change_in_tension'].append(0)
                            data['Strain_rate'].append(0)
                            data['Change_in_connected_edge_length'].append(0)
                            data['Binary_length_change'].append('Initial Length')
                            data['Binary_connected_length_change'].append('Initial Connected Edge Length')
                            edge_index += 1
                        else:
                            data['Strain_rate'].append((lengths[edge_index] - lengths[edge_index - 1])/lengths[edge_index - 1])
                            data['Change_in_length'].append(lengths[edge_index] - lengths[edge_index - 1])
                            data['Change_in_tension'].append(tensions[edge_index] - tensions[edge_index - 1])
                            data['Change_in_connected_edge_length'].append(con_lengths[edge_index] - con_lengths[edge_index - 1])
                            if lengths[edge_index] > lengths[edge_index - 1]:
                                data['Binary_length_change'].append('Increasing Length')
                            else:
                                data['Binary_length_change'].append('Decreasing Length')
                            if con_lengths[edge_index] > con_lengths[edge_index - 1]:
                                data['Binary_connected_length_change'].append('Increasing Connected Edge Length')
                            else:
                                data['Binary_connected_length_change'].append('Decreasing Connected Edge Length')

                            edge_index += 1


        for cell_lab in cell_labels:
            if old_cell_labels == None or cell_lab not in old_cell_labels:
                cell_index = 0
                areas, perims, pressures, centroids = [], [], [], []
                for t, v in colonies.items():
                    if [c.pressure for c in v.cells if c.label == cell_lab] != []:
                        cell_data['Cell_Labels'].append(cell_lab)
                        cell_data['Index_Cell_Labels'].append(cell_lab)
                        cell_data['Pressures'].append([c.pressure for c in v.cells if c.label == cell_lab][0])
                        cell_data['Normalized_Pressures'].append(([c.pressure for c in v.cells if c.label == cell_lab][0] - min_pres)/float(max_pres - min_pres))
                        if cell_lab in common_cell_labels:
                            cell_data['Repeat_Pressures'].append([c.pressure for c in v.cells if c.label == cell_lab][0])
                        else:
                            cell_data['Repeat_Pressures'].append(np.NaN)
                        cell_data['Cell_Time'].append(int(t))
                        cell_data['Index_Cell_Time'].append(int(t))
                        temp_node_tensions = [e.tension for c in v.cells for n in c.nodes for e in n.edges if c.label == cell_lab]
                        temp_edge_tensions = [e.tension for c in v.cells for e in c.edges if c.label == cell_lab]
                        cell_data['Number_of_edges'].append([len(c.edges) for c in v.cells if c.label == cell_lab][0])

                        cell_data['Mean_node_edge_tension'].append(np.mean(temp_node_tensions))
                        cell_data['Sum_edge_tension'].append(np.sum(temp_edge_tensions))
                        [areas.append([c.area() for c in v.cells if c.label == cell_lab][0])]
                        [centroids.append([c.centroid() for c in v.cells if c.label == cell_lab][0])]
                        [pressures.append([c.pressure for c in v.cells if c.label == cell_lab][0])]
                        [perims.append([c.perimeter() for c in v.cells if c.label == cell_lab][0])]
                        if cell_index == 0:
                            cell_data['Change_in_area'].append(0)
                            cell_data['Change_in_pressure'].append(0)
                            cell_data['Binary_area_change'].append('Initial Area')
                            cell_data['Change_in_perimeter'].append(0)
                            cell_data['Binary_perim_change'].append('Initial Perimeter')
                            cell_data['Centroid_movement'].append(0)
                            cell_data['Binary_rotation'].append(0)
                            cell_data['Rotation'].append(0)
                            cell_index += 1
                        else: 
                            x1, y1 = centroids[cell_index - 1][0], centroids[cell_index -1][1]
                            x2, y2 = centroids[cell_index][0], centroids[cell_index][1]
                            cosang = np.dot([x1, y1], [x2, y2])

                            sinang = np.cross([x1, y1], [x2, y2])
                            
                            # theta1 = np.rad2deg(np.arctan2(y2-y1, x2-x1)) 
                            theta1 = np.rad2deg(np.arctan2(sinang, cosang))
                            cell_data['Rotation'].append(theta1)
                            cell_data['Centroid_movement'].append(np.linalg.norm(np.subtract(centroids[cell_index], centroids[cell_index - 1])))
                            cell_data['Change_in_area'].append(areas[cell_index] - areas[cell_index - 1])
                            cell_data['Change_in_perimeter'].append(perims[cell_index] - perims[cell_index - 1])
                            cell_data['Change_in_pressure'].append(pressures[cell_index] - pressures[cell_index - 1])
                            if areas[cell_index] > areas[cell_index - 1]:
                                cell_data['Binary_area_change'].append('Increasing Area')
                            else:
                                cell_data['Binary_area_change'].append('Decreasing Area')

                            if perims[cell_index] > perims[cell_index - 1]:
                                cell_data['Binary_perim_change'].append('Increasing Perimeter')
                            else:
                                cell_data['Binary_perim_change'].append('Decreasing Perimeter')

                            if theta1 > 0:
                                cell_data['Binary_rotation'].append(1)
                            else:
                                cell_data['Binary_rotation'].append(-1)
                            cell_index += 1

                        cell_data['Area'].append([c.area() for c in v.cells if c.label == cell_lab][0])
                        cell_data['Perimeter'].append([c.perimeter() for c in v.cells if c.label == cell_lab][0])



        if counter == None:
            counter = 1
        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > counter)
        if new_colony_range != {}:
            old_labels = labels
            old_cell_labels = cell_labels
            self.seaborn_plot(ax,  new_colony_range, common_edge_labels, common_cell_labels,  data, cell_data, old_labels, old_cell_labels,  counter + 1, min_ten, max_ten, min_pres, max_pres)
            edges_dataframe = pd.DataFrame(data)
            edges_dataframe.set_index(['Index_Edge_Labels', 'Index_Time'], inplace = True)
            cells_dataframe = pd.DataFrame(cell_data)
            cells_dataframe.set_index(['Index_Cell_Labels', 'Index_Cell_Time'], inplace = True)
            return edges_dataframe, cells_dataframe

        

    def make_all_edge_movie_simple(self, fig, ax, colonies):
        self.simple_plot_all_edges(fig, ax, colonies)

        fps = 1
        os.system("rm movie_all_edges_simple.mp4")
        os.system("ffmpeg -r "+str(fps)+" -b 1800 -i _tmp%05d.png movie_all_edges_simple.mp4")
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

    def plot_pressures(self, fig, ax, colonies, specify_aspect = None, specify_color = None, **kwargs):
        """
        Make a pressure movie over colonies
        """

        max_num = len(colonies)

        _, _, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')

        #min_pres, max_pres = None, None

        counter = 0
        for t, v in colonies.items():
            index = str(t)
            t= int(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            colonies[index].plot_pressures(ax, fig, pressures, min_pres, max_pres, specify_color, **kwargs)
            [e.plot(ax) for e in colonies[index].edges]
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            if specify_aspect is not None:
                ax.set(xlim = [0,600], ylim = [0,600], aspect = 1)
            pylab.savefig('_tmp%05d.png'%counter, dpi=200)
            counter += 1
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

    def plot_both_tension_pressure(self, fig, ax, colonies, specify_aspect = None, specify_color = None, **kwargs):
        """
        Make a pressure movie over colonies
        """
        max_num = len(colonies)

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type = 'pressure')
        #min_ten, max_ten, min_pres, max_pres = None, None, None, None


        counter = 0
        for t, v in colonies.items():
            index = str(t)
            t=int(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            edges = colonies[index].tot_edges
            tensions = [e.tension for e in edges]
            colonies[index].plot(ax, fig, tensions, pressures, min_ten, max_ten, min_pres, max_pres, specify_color, **kwargs)
            if specify_aspect is not None:
                ax.set(xlim = [0,600], ylim = [0,600], aspect = 1)
            #pylab.savefig('_tmp0000{0}.png'.format(t), dpi=200)
            pylab.savefig('_tmp%05d.png'%counter, dpi=200)
            counter += 1
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





            













