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
        for c in new:
            if c is not None and c not in cells:
                cells.append(c)
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
    cycles = [trace_cell_cycle(edge, sign) for sign in (1,-1)]
    cells = []
    for cycle in cycles:
        if cycle is not None:
            nodes = list(set(itertools.chain(*[edge.nodes for edge in cycle])))
            cells.append(cell(nodes, cycle))
    return cells
    
def trace_cell_cycle(edge, sign=1, cycle_len_lim=10):
    """Trace out a cell cycle, directionality depends on sign (-1 or 1)
    
    Parameters
    ----------
    edge: edge class
        the edge we start on
    sign: (-1,1)
        whether we take the largest- or smallest-angled next edge
    cycle_len_lim: int
        how many edges we trace along before concluding we won't complete 
        this cell
    Returns
    -------
    cell_cycle: list of edges
        a list of edges comprising a minimal cell cycle in the specified 
        direction
    """
    cell_cycle = [edge]
    start_node = edge.node_a
    next_node = edge.node_b
    current_cell_length = 1
    while start_node != next_node and current_cell_length <= cycle_len_lim:
        current_edge = cell_cycle[-1]
        # dead-end case
        if len(next_node.edges)==1:
            cell_cycle = None
            break
        # find the next edge
        next_edges = [edge for edge in next_node.edges if edge is not current_edge]
        next_edge_angles = [current_edge.edge_angle(ne) for ne in next_edges]
        if sign == 1:
            next_edge = next_edges[np.argmin(np.asarray(next_edge_angles))]
        else:
            next_edge = next_edges[np.argmin(np.asarray(next_edge_angles))]
        # update 
        cell_cycle.append(next_edge)
        next_node = [node for node in next_edge.nodes if node!=next_node][0]
        current_cell_length += 1
    if current_cell_length>cycle_len_lim:
        cell_cycle = None # no cycle found in length
    return cell_cycle