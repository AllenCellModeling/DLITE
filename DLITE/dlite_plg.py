import numpy as np
import matplotlib.pyplot as plt

from imagepy.core.engine import Filter
from .EdgeExtractor import EdgeExtractor
from .ManualTracing import ManualTracing
from .cell_describe import node, edge, cell, colony

class Plugin(Filter):
    title = 'DLITE'
    note = ['all', 'auto_msk', 'auto_snap']

    def run(self, ips, snap, img, para = None):

        extractor = EdgeExtractor()
        img, edges = extractor.run(img, preprocess=True)

        X, Y = [], []
        for edge in edges:
            X.append([])
            Y.append([])
            for x, y in edge:
                X[-1].append(x)
                Y[-1].append(y)

        ex = ManualTracing(X, Y)
        cutoff = 5
        nodes, edges, new = ex.cleanup(cutoff)
        cells = ex.find_cycles(edges)

        col = colony(cells, edges, nodes)
        col.calculate_tension(solver="DLITE")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharey=True)

        tensions = [e.tension for e in col.tot_edges]
        mean_ten = np.mean(tensions)
        tensions = [e / mean_ten for e in tensions]

        col.plot_tensions(ax, fig, tensions, min_x=0, max_x=1024, min_y=0, max_y=1024,
                          min_ten=0, max_ten=3, specify_color='jet', cbar='no', lw=3)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set(xlim=[0, 1024], ylim=[0, 1024], aspect=1)
        plt.gca().invert_yaxis()

        plt.show()
        return img
