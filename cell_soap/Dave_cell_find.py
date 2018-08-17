# encoding: utf-8
"""
Functions for finding cells in graphs of edges and nodes
"""

from cell_describe import cell
import numpy as np
import itertools

def find_all_cells(edges):
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
        new = cells_on_either_side(edge)
        for cell in new:
            if cell is not None and cell not in cells:
                cells.append(cell)
    return cells

def cells_on_either_side(edge):
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
        edge_angles = [edge._edge_angle(e) for e in other_edges]
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
    
