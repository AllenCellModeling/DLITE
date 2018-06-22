import matplotlib.pyplot as plt
import matplotlib
from data import nodes, edges, cells


fig, axes = plt.subplots(1,2,figsize=(10,6))

## By itself
[cell.plot(axes[0]) for cell in cells];
axes[0].set_title("cell vertices")

## With original image
#img = matplotlib.image.imread('cell_vertex_graph.png')
# axes[1].imshow(img)
# [cell.plot(axes[1]) for cell in cells]
# axes[1].set_title("cell vertices and original")

# [a.set(xlim=(0,img.shape[1]), ylim=(img.shape[0], 0), aspect=1) for a in axes]
# plt.show()