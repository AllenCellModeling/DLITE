from cell_describe import node, edge, cell
# Node xy locations taken from image, radii are # Node  
# pretty random guesses with some intentionally left out
nodes = [node((250, 290)), #n_a, loc0
         node((343, 424)), #n_b, loc1
         node((520, 249)), #n_c, loc2
         node((321, 108)), #n_d, loc3
         node(( 92,  34)), #n_e, loc4
         node(( 42, 246)), #n_f, loc5
         node(( 83, 446))] #n_g, loc6

edges = [edge(nodes[1], nodes[0],  200), #m_ba, loc0
         edge(nodes[2], nodes[1], None), #m_cb, loc1
         edge(nodes[2], nodes[3],  200), #m_cd, loc2
         edge(nodes[0], nodes[3],  300), #m_ad, loc3
         edge(nodes[5], nodes[0],  900), #m_fa, loc4
         edge(nodes[5], nodes[4], None), #m_fe, loc5
         edge(nodes[3], nodes[4],  300), #m_de, loc6
         edge(nodes[5], nodes[6], 1000), #m_fg, loc7
         edge(nodes[6], nodes[1], 1000)] #m_gb, loc8

## Create list of cell nodes and edges
# Normally I'd do this in the function call but will break out
# here for clarity

cell_a_nodes = [nodes[0], #n_a
                nodes[1], #n_b
                nodes[6], #n_g
                nodes[5]] #n_f
cell_a_edges = [edges[0], #m_ba
                edges[8], #m_gb
                edges[7], #m_fg
                edges[4]] #m_fa

cell_b_nodes = [nodes[0], #n_a
                nodes[3], #n_d
                nodes[2], #n_c
                nodes[1]] #n_b
cell_b_edges = [edges[3], #m_ad
                edges[2], #m_cd
                edges[1], #m_cb
                edges[0]] #m_ba
                
cell_c_nodes = [nodes[4], #n_e
                nodes[3], #n_d
                nodes[0], #n_a
                nodes[5]] #n_f
cell_c_edges = [edges[6], #m_de
                edges[3], #m_ad
                edges[4], #m_fa
                edges[5]] #m_fe

# Create cells
cells = [cell(cell_a_nodes, cell_a_edges),
         cell(cell_b_nodes, cell_b_edges), 
         cell(cell_c_nodes, cell_c_edges)]