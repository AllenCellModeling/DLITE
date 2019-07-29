import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import pandas as pd
import pylab
from scipy import stats
import numpy.linalg as la
import numpy as np
import os

class PlottingFunctions:
    def __init__(self):
        """
        List of plotting functions given a colony time-series
        List of dataframe making functions
        """

    def check_repeat_labels(self, colonies, max_num):
        """
        Find node labels that are present in a specified number of colonies
        Used in plotting functions
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
        Used in plotting functions
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
        Used in plotting functions
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
        PLOTTING FUNCTION
        Plot the edges connected to a node specified by label
        Parameters
        ---------------
        label - label of node that is present in all colonies specified by colonies
        colonies - dictionary of colonies
        """
        ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)

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

                def norm(tensions, min_t=None, max_t=None):
                    return (tensions - min_t) / float(max_t - min_t)

                c1 = norm(tensions, min_t, max_t)

                # Get edges on node label
                edges = [e for n in nodes for e in n.edges if n.label == label]

                for j, an_edge in enumerate(edges):
                    an_edge.plot(ax, ec=cm.viridis(c1[j]), lw=3)

                sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
                # fake up the array of the scalar mappable.
                sm._A = []

                # Plot all edges
                for edd in all_edges:
                    edd.plot(ax, lw=0.2)

                cbaxes = fig.add_axes([0.13, 0.1, 0.03, 0.8])
                cl = plt.colorbar(sm, cax=cbaxes)
                cl.set_label('Normalized tension', fontsize=13, labelpad=-60)
                pylab.savefig('_tmp%05d.png' % count, dpi=200)
                plt.cla()
                plt.clf()
                plt.close()
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)
                count += 1

        fps = 1
        os.system("rm movie_single_node.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_single_node.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_tensions(self, fig, ax, colonies, min_x=None, max_x=None, min_y=None, max_y=None, min_ten=None,
                      max_ten=None, specify_aspect=None, specify_color=None, type=None, **kwargs):
        """
        PLOTTING FUNCTION
        Make a tension movie (colormap) for all timepoints of the colony
        """
        all_tensions, _, _ = self.all_tensions_and_radius_and_pressures(colonies)
        if not min_ten and not max_ten:
            _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)

        counter = 0
        for t, v in colonies.items():
            index = str(t)
            edges = colonies[index].tot_edges
            if type == 'Ground_truth':
                tensions = [e.ground_truth for e in edges]
            else:

                tensions = [e.tension for e in edges]
                mean_ten = np.mean(tensions)
                tensions = [i / mean_ten for i in tensions]
            colonies[index].plot_tensions(ax, fig, tensions, min_x, max_x, min_y, max_y,
                                          min_ten, max_ten, specify_color, **kwargs)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            if specify_aspect is not None:
                ax.set(xlim=[0, 600], ylim=[0, 600], aspect=1)

            pylab.savefig('_tmp%05d.png' % counter, dpi=200, bbox_inches='tight')
            counter += 1
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        fps = 1.5
        os.system("rm movie_tension.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_tension.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_single_cells(self, fig, ax, ax1, ax3, colonies, cell_label):
        """
        PLOTTING FUNCTION
        Make a movie tracking showing the evolution of a single cell over time, specified by cell_label
        Also plots
        Pressure, Perimeter, Area and Change in area of that cell over time
        """
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        all_lengths, all_perims, all_areas = self.all_perims_areas_lengths(colonies)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type='pressure')
        _, max_perim, min_perim = self.get_min_max_by_outliers_iqr(all_perims)
        _, max_area, min_area = self.get_min_max_by_outliers_iqr(all_areas)

        frames = [i for i in colonies.keys()]
        pressures, areas, perimeters, change_in_area = [], [], [], [0]

        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            pres = [c.pressure for c in cells if c.label == cell_label]
            ares = [c.area() for c in cells if c.label == cell_label]
            perims = [c.perimeter() for c in cells if c.label == cell_label]
            if pres:
                pressures.append(pres[0])
                areas.append(ares[0])
                perimeters.append(perims[0])
                if j > 0:
                    change_in_area.append(perims[0] - perimeters[j - 1])
            else:
                frames = frames[0:j]

        ax1.plot(frames, pressures, lw=3, color='black')
        ax1.set_ylabel('Pressures', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, perimeters, 'blue')
        ax2.set_ylabel('Perimeters', color='blue')
        ax2.tick_params('y', colors='blue')

        ax3.plot(frames, areas, lw=3, color='black')
        ax3.set_ylabel('Areas', color='black')
        ax3.set_xlabel('Frames')
        ax4 = ax3.twinx()
        ax4.plot(frames, change_in_area, 'blue')
        ax4.set_ylabel('Change in Area', color='blue')
        ax4.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax3.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax1.set(xlim=[0, 31])
            ax2.set(xlim=[0, 31], ylim=[min_perim, max_perim])
            ax3.set(xlim=[0, 31], ylim=[min_area, max_area])
            ax4.set(xlim=[0, 31])

            [e.plot(ax) for e in edges]
            current_cell = [c for c in cells if c.label == cell_label][0]
            [current_cell.plot(ax, color='red', )]
            x = [n.loc[0] for n in current_cell.nodes]
            y = [n.loc[1] for n in current_cell.nodes]
            ax.fill(x, y, c='red', alpha=0.2)
            for e in current_cell.edges:
                e.plot_fill(ax, color='red', alpha=0.2)

            ax1.plot(i, current_cell.pressure, 'ok', color='red')
            ax2.plot(i, current_cell.perimeter(), 'ok', color='red')
            ax3.plot(i, current_cell.area(), 'ok', color='red')
            ax4.plot(i, change_in_area[j], 'ok', color='red')

            fname = '_tmp%05d.png' % int(j)
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1, ax3) = plt.subplots(3, 1, figsize=(5.5, 15))
            ax1.plot(frames, pressures, lw=3, color='black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, perimeters, 'blue')
            ax2.set_ylabel('Perimeters', color='blue')
            ax2.tick_params('y', colors='blue')

            ax3.plot(frames, areas, lw=3, color='black')
            ax3.set_ylabel('Areas', color='black')
            ax3.set_xlabel('Frames')
            ax4 = ax3.twinx()
            ax4.plot(frames, change_in_area, 'blue')
            ax4.set_ylabel('Change in Area', color='blue')
            ax4.tick_params('y', colors='blue')

        fps = 1
        os.system("rm movie_cell.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_cell.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def single_edge_plotting(self, fig, ax, ax1, ax3, colonies, node_label, edge_label):
        """
        PLOTTING FUNCTION
        This is a function that is called by plot_single_edges (the next function)
        """
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        all_lengths, all_perims, all_areas = self.all_perims_areas_lengths(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_len, min_len = self.get_min_max_by_outliers_iqr(all_lengths)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type='pressure')

        frames = [i for i in colonies.keys()]
        tensions = []
        radii = []
        length = []
        change_in_length = [0]

        for j, i in enumerate(frames):
            try:
                edd = [e for e in colonies[str(i)].tot_edges if e.label == edge_label][0]
                all_tensions_frame = [e.tension for e in colonies[str(i)].tot_edges]
                mean_tens = np.mean(all_tensions_frame)
                tensions.append(edd.tension/mean_tens)
                radii.append(edd.radius)
                length.append(edd.straight_length)
                if j > 0:
                    change_in_length.append(edd.straight_length - length[j - 1])
            except:
                frames = frames[0:j]

        ax1.plot(frames, tensions, lw=3, color='black')
        ax1.set_ylabel('Tension', color='black', fontsize=18)
        ax1.set_xlabel('Frames', fontsize=18)
        ax2 = ax1.twinx()
        ax2.plot(frames, radii, 'blue')
        ax2.set_ylabel('Radius', color='blue', fontsize=18)
        ax2.tick_params('y', colors='blue')

        ax3.plot(frames, length, lw=3, color='black')
        ax3.set_ylabel('Length', color='black', fontsize=18)
        ax3.set_xlabel('Frames', fontsize=18)
        ax4 = ax3.twinx()
        ax4.plot(frames, change_in_length, 'blue')
        ax4.set_ylabel('Change in length', color='blue', fontsize=18)
        ax4.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            edges = colonies[str(i)].tot_edges
            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)
            ax1.set(xlim=[0, 31], ylim=[min_ten, max_ten])
            ax2.set(xlim=[0, 31], ylim=[min_rad, max_rad])
            ax3.set(xlim=[0, 31], ylim=[min_len, max_len])
            ax4.set(xlim=[0, 31], ylim=[min(change_in_length), max(change_in_length)])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax3.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
            current_edge = [e for e in colonies[str(i)].tot_edges if e.label == edge_label][0]
            [current_edge.plot(ax, lw=3, color='red')]

            fname = '_tmp%05d.png' % int(j)
            mean_tens = np.mean([e.tension for e in edges])
            this_tension = current_edge.tension/mean_tens
            ax1.plot(i, this_tension, 'ok', color='red')
            ax2.plot(i, current_edge.radius, 'ok', color='red')
            ax3.plot(i, current_edge.straight_length, 'ok', color='red')
            ax4.plot(i, change_in_length[j], 'ok', color='red')
            plt.tight_layout()

            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1, ax3) = plt.subplots(3, 1, figsize=(5.5, 15))
            ax1.plot(frames, tensions, lw=3, color='black')
            ax1.set_ylabel('Tension', color='black', fontsize=18)
            ax1.set_xlabel('Frames', fontsize=18)
            ax2 = ax1.twinx()
            ax2.plot(frames, radii, 'blue')
            ax2.set_ylabel('Radius', color='blue', fontsize=18)
            ax2.tick_params('y', colors='blue',)

            ax3.plot(frames, length, lw=3, color='black')
            ax3.set_ylabel('Length', color='black', fontsize=18)
            ax3.set_xlabel('Frames', fontsize=18)
            ax4 = ax3.twinx()
            ax4.plot(frames, change_in_length, 'blue')
            ax4.set_ylabel('Change in length', color='blue', fontsize=18)
            ax4.tick_params('y', colors='blue')

    def plot_single_edges(self, fig, ax, ax1, ax3, colonies, node_label, edge_label):
        """
        PLOTTING FUNCTION
        Plots a single edge over time specified by edge_label (dont really use node_label, used it before when i wasnt tracking edge labels explicitly)
        Also plots
        tension, straight_length, radius and change in straight_length of that edge
        """
        self.single_edge_plotting(fig, ax, ax1, ax3, colonies, node_label, edge_label)

        fps = 1
        os.system("rm movie_edge.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_edge.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_compare_single_edge_tension(self, fig, ax, ax1,
                                         colonies_1, colonies_2, edge_label, type=None,
                                         ground_truth=None, xlim_end=None):
        """
        PLOTTING FUNCTION
        Plot single edge over time specified by edge label (dont really use node_label)
        Also plots
        Tension of that edge store2 in colonies_1
        Tension of that edge stored in colonies_2
        Meant to be used as a comparison of 2 methods - CELLFIT, unconstrained
        """

        all_tensions_1, _, _ = self.all_tensions_and_radius_and_pressures(colonies_1)
        _, max_ten_1, min_ten_1 = self.get_min_max_by_outliers_iqr(all_tensions_1)

        all_tensions_2, _, _ = self.all_tensions_and_radius_and_pressures(colonies_2)
        _, max_ten_2, min_ten_2 = self.get_min_max_by_outliers_iqr(all_tensions_2)

        frames = [i for i in colonies_1.keys()]
        tensions_1, tensions_2 = [], []
        g_t = []

        for j, i in enumerate(frames):
            try:
                mean_t = np.mean([e.tension for e in colonies_1[str(i)].tot_edges])
                edd = [e for e in colonies_1[str(i)].tot_edges if e.label == edge_label][0]
                tensions_1.append(edd.tension / mean_t)

                if ground_truth is not None:
                    mean_gt = np.mean([e.ground_truth for e in colonies_1[str(i)].tot_edges])
                    g_t.append(edd.ground_truth/mean_gt)
            except:
                frames = frames[0:j]

        for j, i in enumerate(frames):
            try:
                mean_t = np.mean([e.tension for e in colonies_2[str(i)].tot_edges])
                edd = [e for e in colonies_2[str(i)].tot_edges if e.label == edge_label][0]
                tensions_2.append(edd.tension / mean_t)
            except:
                frames = frames[0:j]

        ax1.plot(frames, tensions_1, lw=3, color='black')
        ax1.set_ylabel('Tension_CELLFIT', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, tensions_2, 'blue')
        ax2.set_ylabel('Tension_Unconstrained', color='blue')
        ax2.tick_params('y', colors='blue')

        if ground_truth is not None:
            ax1.plot(frames, g_t, lw=3, color='green')

        for j, i in enumerate(frames):
            edges_1 = colonies_1[str(i)].tot_edges

            if type == 'surface_evolver':
                ax.set(xlim=[-2, 2], ylim=[-2, 2], aspect=1)
                ax1.set(xlim=[0, 55], ylim=[0, 2.2])
                ax2.set(xlim=[0, 55], ylim=[0, 2.2])
            elif type == 'surface_evolver_cellfit':
                if xlim_end is None:
                    ax.set(xlim=[200, 800], ylim=[200, 800], aspect=1)
                    ax1.set(xlim=[0, 55], ylim=[0, 2.2])
                    ax2.set(xlim=[0, 55], ylim=[0, 2.2])
                else:
                    ax.set(xlim=[400, 650], ylim=[400, 650], aspect=1)
                    ax1.set(xlim=[0, xlim_end], ylim=[0, 2.2])
                    ax2.set(xlim=[0, xlim_end], ylim=[0, 2.2])

            else:
                ax.set(xlim=[400, 650], ylim=[400, 650], aspect=1)
                ax1.set(xlim=[0, 31], ylim=[0, 2.2])
                ax2.set(xlim=[0, 31], ylim=[0, 2.2])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            [e.plot(ax) for e in edges_1]

            current_edge = [e for e in colonies_1[str(i)].tot_edges if e.label == edge_label][0]
            [current_edge.plot(ax, lw=3, color='red')]

            fname = '_tmp%05d.png' % int(j)
            mean_t_1 = np.mean([e.tension for e in colonies_1[str(i)].tot_edges])
            mean_gt_1 = np.mean([e.ground_truth for e in colonies_1[str(i)].tot_edges])
            ax1.plot(i, current_edge.tension / mean_t_1, 'ok', color='red')
            ax1.plot(i, current_edge.ground_truth/mean_gt_1, 'ok', color='red')

            current_new_edge = [e for e in colonies_2[str(i)].tot_edges if e.label == edge_label][0]
            mean_t_2 = np.mean([e.tension for e in colonies_2[str(i)].tot_edges])
            ax2.plot(i, current_new_edge.tension / mean_t_2, 'ok', color='red')

            plt.tight_layout()

            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(5.5, 10))
            ax1.plot(frames, tensions_1, lw=3, color='black')
            ax1.set_ylabel('Tension_CELLFIT', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, tensions_2, 'blue')
            ax2.set_ylabel('Tension_Unconstrained', color='blue')
            ax2.tick_params('y', colors='blue')
            ax1.plot(frames, g_t, lw=3, color='green')

        fps = 5
        os.system("rm movie_compare_edge.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_compare_edge.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_abnormal_edges(self, fig, ax, colonies_1, abnormal):
        """
        PLOTTING FUNCTION
        Abnormal edges defined as edges with large stochasticity
        Parameters
        -----------------
        colonies_1 - colony class with calculated tensions and pressures
        abnormal - of the form [[edge_label, time]]
        """
        # abnormal is of the form [[label, time]]

        frames = [i for i in colonies_1.keys()]
        temp_mean, temp_std = [], []

        for j, i in enumerate(frames):
            temp_mean.append(np.mean([e.tension for e in colonies_1[str(i)].tot_edges]))
            temp_std.append(np.std([e.tension for e in colonies_1[str(i)].tot_edges]))

        for j, i in enumerate(frames):
            edges_1 = colonies_1[str(i)].tot_edges
            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)

            [e.plot(ax) for e in edges_1]
            times = [a[1] for a in abnormal]

            if int(i) in times:
                labels = [a[0] for a in abnormal if a[1] == int(i)]
                for labe in labels:
                    [e.plot(ax, lw=3, color='red') for e in edges_1 if e.label == labe]

            fname = '_tmp%05d.png' % int(j)

            plt.tight_layout()

            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))

        fps = 5
        os.system("rm movie_abnormal_edge.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_abnormal_edge.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_guess_tension(self, fig, ax, ax1, colonies, node_label, edge_label):
        """
        PLOTTING FUNCTION
        Plot edge over time specified by edge_label
        Also plots
        Tension of that edge, guess tension of that edge
        Should be offset
        """
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
            except:
                frames = frames[0:j]

        for j, i in enumerate(guesses):
            if not i:
                guesses[j] = 0.002

        ax1.plot(frames, tensions, lw=3, color='black')
        ax1.set_ylabel('Tension', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, guesses, 'blue')
        ax2.set_ylabel('Guess Tension', color='blue')
        ax2.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            edges = colonies[str(i)].tot_edges
            dictionary = colonies[str(i)].dictionary

            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)
            ax1.set(xlim=[0, 31], ylim=[min_ten, max_ten])
            ax2.set(xlim=[0, 31], ylim=[min_ten, max_ten])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))

            [e.plot(ax) for e in edges]
            current_edge = dictionary[node_label][0][edge_label]
            [current_edge.plot(ax, lw=3, color='red')]

            fname = '_tmp%05d.png' % int(j)
            ax1.plot(i, current_edge.tension, 'ok', color='red')
            ax2.plot(i, guesses[j], 'ok', color='red')
            plt.tight_layout()

            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(5.5, 10))
            ax1.plot(frames, tensions, lw=3, color='black')
            ax1.set_ylabel('Tension', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, guesses, 'blue')
            ax2.set_ylabel('Guess Tension', color='blue')
            ax2.tick_params('y', colors='blue')
        fps = 1
        os.system("rm movie_edge_guess.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_edge_guess.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_guess_pressures(self, fig, ax, ax1, colonies, cell_label):
        """
        Plot single cell over time specified by cell_label
        ALso plots
        Pressure of that cell, guess pressure of that cell
        """
        frames = [i for i in colonies.keys()]
        pressures, guesses = [], []
        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            pres = [c.pressure for c in cells if c.label == cell_label]
            gess = [c.guess_pressure for c in cells if c.label == cell_label]
            if pres:
                pressures.append(pres[0])
                guesses.append(gess[0])
            else:
                frames = frames[0:j]

        for j, i in enumerate(guesses):
            if not i:
                guesses[j] = 0

        ax1.plot(frames, pressures, lw=3, color='black')
        ax1.set_ylabel('Pressures', color='black')
        ax1.set_xlabel('Frames')
        ax2 = ax1.twinx()
        ax2.plot(frames, guesses, 'blue')
        ax2.set_ylabel('Guess Pressure', color='blue')
        ax2.tick_params('y', colors='blue')

        for j, i in enumerate(frames):
            cells = colonies[str(i)].cells
            edges = colonies[str(i)].tot_edges
            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)
            # ax1.set(xlim = [0,31], ylim = [min_pres, max_pres])
            ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax1.set(xlim=[0, 31])

            [e.plot(ax) for e in edges]
            current_cell = [c for c in cells if c.label == cell_label][0]
            [current_cell.plot(ax, color='red', )]

            x = [n.loc[0] for n in current_cell.nodes]
            y = [n.loc[1] for n in current_cell.nodes]
            ax.fill(x, y, c='red', alpha=0.2)
            for e in current_cell.edges:
                e.plot_fill(ax, color='red', alpha=0.2)

            ax1.plot(i, current_cell.pressure, 'ok', color='red')
            ax2.plot(i, guesses[j], 'ok', color='red')

            fname = '_tmp%05d.png' % int(j)
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1) = plt.subplots(2, 1, figsize=(5.5, 10))
            ax1.plot(frames, pressures, lw=3, color='black')
            ax1.set_ylabel('Pressures', color='black')
            ax1.set_xlabel('Frames')
            ax2 = ax1.twinx()
            ax2.plot(frames, guesses, 'blue')
            ax2.set_ylabel('Guess Pressure', color='blue')
            ax2.tick_params('y', colors='blue')

        fps = 1
        os.system("rm movie_cell_guess.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_cell_guess.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_histogram(self, fig, ax, ax1, ax2, colonies):
        """
        PLOTTING FUNCTION
        Plots a histogram of tensions and pressures over time
        """

        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        max_ten, min_ten, max_pres, min_pres = max(all_tensions), min(all_tensions), \
                                               max(all_pressures), min(all_pressures)

        frames = [i for i in colonies.keys()]

        ensemble_tensions, ensemble_pressures = [], []

        for j, i in enumerate(frames):
            this_colony_dict = dict((k, v) for k, v in colonies.items() if int(i) + 1 > int(k) >= int(i))
            try:
                this_tensions, this_radii, this_pressures = self.all_tensions_and_radius_and_pressures(this_colony_dict)
                ensemble_tensions.append(this_tensions)
                ensemble_pressures.append(this_pressures)
            except:
                frames = frames[0:j]

        for j, i in enumerate(frames):

            edges = colonies[str(i)].tot_edges
            ax.set(xlim=[0, 1030], ylim=[0, 1030], aspect=1)

            [e.plot(ax) for e in edges]

            # the histogram of the data
            n, bins, patches = ax1.hist(ensemble_tensions[j], 25, range=(min_ten, max_ten))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # scale values to interval [0,1]
            col = bin_centers - min(bin_centers)
            col /= max(col)

            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm.viridis(c))

            # the histogram of the data
            n, bins, patches = ax2.hist(ensemble_pressures[j], 25, range=(min_pres, max_pres))
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

            fname = '_tmp%05d.png' % int(j)
            plt.tight_layout()
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
            fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=(5.5, 15))
        fps = 1
        os.system("rm movie_histograms.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_histograms.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def get_repeat_edge(self, colonies):
        """
        Get a list of edge_labels that are present in all colonies provided (repeat labels)
        used in plotting functions
        """
        labels = []
        for t, v in colonies.items():
            labels.append([e.label for e in v.tot_edges if e.label != []])

        repeat_edge_labels = set(labels[0]).intersection(*labels)
        return list(repeat_edge_labels)

    def get_repeat_cell(self, colonies):
        """
        Get a list of cell_labels that are present in all colonies provided (repeat labels)
        used in plotting functions
        """
        labels = []
        for t, v in colonies.items():
            labels.append([c.label for c in v.cells if c.label != []])

        repeat_cell_labels = set(labels[0]).intersection(*labels)
        return list(repeat_cell_labels)

    def get_repeat_nodes(self, colonies):
        """
        Get a list of node_labels that are present in all colonies provided (repeat labels)
        used in plotting functions
        """
        labels = []
        for t, v in colonies.items():
            labels.append([n.label for n in v.tot_nodes if n.label != []])

        repeat_node_labels = set(labels[0]).intersection(*labels)
        return list(repeat_node_labels)

    def seaborn_cells_dataframe_tensor(self, colonies, jump_number=1, data=None):
        """
        DATAFRAME FUNCTION
        Make a tensor dataframe, tracks movement over time
        calculates strain, rotation, velocities etc.
        """

        initial_index = [int(k) for k, v in colonies.items()][0]
        # labels = [e.label for e in colonies[str(initial_index)].cells if e.label != []]
        # labels = sorted(labels)

        if data is None:
            data = {'Index_Time': [], 'Time': [], 'Velocity_x': [], 'Velocity_y': [],
                    'x_pos': [], 'y_pos': [], 'u_translational': [],
                    'v_translational': [], 'ux_translational': [], 'uy_translational': [],
                    'vx_translational': [], 'vy_translational': [],
                    'Velocity_gradient_tensor': [], 'Rotation': [], 'Strain_rate': [],
                    'Rate_of_area_change': [], 'Eigenvalues_strain_1': [],
                    'Eigenvalues_strain_2': [], 'Eigenvectors_strain': [],
                    'Eigenvectors_strain_1': [], 'Eigenvectors_strain_2': [],
                    'Eigenvalues_rotation': [], 'Eigenvectors_rotation': [],
                    'First_invariant': [], 'Second_invariant': [],
                    'Mean_resid': [], 'Std_resid': [], 'Mean_tension': [], 'Mean_pressure': [],
                    'Area': [], 'Perimeter': [],
                    'Number_of_edges': [], 'Length_of_edges': [], 'Radius_of_edges': [], 'Std_tension': [],
                    'Change_in_mean_tension': [], 'Change_in_std_tension': [],
                    'Std_pressure': [], 'Area_std': [], 'Perimeter_std': [], 'Number_of_edges_std': [],
                    'Count_topology': [], 'Change_in_connectivity': [], 'Length_of_edges_std': [],
                    'Radius_of_edges_std': []}
            tensor_dataframe = pd.DataFrame(data)
            tensor_dataframe.set_index(['Index_Time'], inplace=True)

        for t, v in colonies.items():

            # Initial 0
            if int(t) == [int(k) for k, v in colonies.items()][-1]:
                pass

            # Check that its not the last one
            if int(t) < [int(k) for k, v in colonies.items()][-1]:
                # define new colony as this time frame and the next
                new_colony_range = dict((k, v) for k, v in colonies.items() if int(t) <= int(k) <= int(t) + jump_number)

                labels = self.get_repeat_cell(new_colony_range)
                nodes_labels = self.get_repeat_nodes(new_colony_range)

                min_t = [int(t) for t, v in new_colony_range.items()][0]
                x_pos1, y_pos1, x_pos2, y_pos2, total_con_labels_1, total_con_labels_2 = [], [], [], [], [], []
                tension_mean_1, tension_mean_2, tension_std_1, tension_std_2 = [], [], [], []
                for tt, vv in new_colony_range.items():

                    cells = sorted(vv.cells, key=lambda x: x.label)
                    for c in cells:
                        if c.label in labels:

                            xc, yc = c.centroid()[0], c.centroid()[1]
                            if int(tt) == min_t:
                                x_pos1.append(xc)
                                y_pos1.append(yc)
                            else:
                                x_pos2.append(xc)
                                y_pos2.append(yc)

                    sorted_nodes = sorted(vv.tot_nodes, key=lambda x: x.label)

                    for n in sorted_nodes:
                        if n.label in nodes_labels:
                            con_labels = [e.label for e in n.edges]
                            if int(tt) == min_t:
                                total_con_labels_1.append(con_labels)
                            else:
                                total_con_labels_2.append(con_labels)
                    if int(tt) == min_t:
                        tension_mean_1 = np.mean(np.array([e.tension for e in vv.tot_edges]))
                        tension_std_1 = np.std(np.array([e.tension for e in vv.tot_edges]))
                    else:
                        tension_mean_2 = np.mean(np.array([e.tension for e in vv.tot_edges]))
                        tension_std_2 = np.std(np.array([e.tension for e in vv.tot_edges]))

                for ttt, vvv in new_colony_range.items():

                    if int(ttt) == min_t:

                        residuals = []
                        for n in vvv.tot_nodes:
                            x = [e.tension for e in vvv.tot_edges]
                            if len(n.edges) > 2:
                                tensions = []
                                indices = n.edge_indices
                                for i in range(len(indices)):
                                    tensions.append([x[indices[i]]])
                                tensions = np.array(tensions)

                                node_vecs = n.tension_vectors
                                tension_vecs = np.multiply(node_vecs, tensions)
                                resid_vec = np.sum(tension_vecs, 0)
                                resid_mag = np.hypot(*resid_vec)
                                residuals.append(resid_mag)
                        data['Mean_resid'].append(np.mean(np.array(residuals)))
                        data['Std_resid'].append(np.std(np.array(residuals)))

                        data['Mean_tension'].append(np.mean(np.array([e.tension for e in vvv.tot_edges])))
                        data['Std_tension'].append(np.std(np.array([e.tension for e in vvv.tot_edges])))
                        data['Mean_pressure'].append(np.mean(np.array([c.pressure for c in vvv.cells])))
                        data['Std_pressure'].append(np.std(np.array([c.pressure for c in vvv.cells])))

                        data['Change_in_mean_tension'].append(tension_mean_2 - tension_mean_1)
                        data['Change_in_std_tension'].append(tension_std_2 - tension_std_1)

                        data['Area'].append(np.mean(np.array([c.area() for c in vvv.cells])))
                        data['Area_std'].append(np.std(np.array([c.area() for c in vvv.cells])))
                        data['Perimeter'].append(np.mean(np.array([c.perimeter() for c in vvv.cells])))
                        data['Perimeter_std'].append(np.std(np.array([c.perimeter() for c in vvv.cells])))
                        data['Number_of_edges'].append(np.mean(np.array([len(n.edges) for n in vvv.tot_nodes])))
                        data['Number_of_edges_std'].append(np.std(np.array([len(n.edges) for n in vvv.tot_nodes])))
                        data['Length_of_edges'].append(np.mean(np.array([e.straight_length for e in vvv.tot_edges])))
                        data['Length_of_edges_std'].append(np.std(np.array([e.straight_length for e in vvv.tot_edges])))
                        data['Radius_of_edges'].append(np.mean(np.array([e.radius for e in vvv.tot_edges])))
                        data['Radius_of_edges_std'].append(np.std(np.array([e.radius for e in vvv.tot_edges])))

                        Number_of_topology_changes = 0
                        Change_in_number = 0

                        for kkkk, vvvv in zip(total_con_labels_1, total_con_labels_2):
                            try:
                                if set(kkkk) != set(vvvv):
                                    Number_of_topology_changes += 1
                            except:
                                pass
                            if len(kkkk) != len(vvvv):
                                Change_in_number += 1

                        data['Count_topology'].append(Number_of_topology_changes)
                        data['Change_in_connectivity'].append(Change_in_number)

                        u_vel = np.array(x_pos2) - np.array(x_pos1)
                        v_vel = np.array(y_pos2) - np.array(y_pos1)
                        data['Index_Time'].append(int(ttt))
                        data['Time'].append(int(ttt))
                        data['Velocity_x'].append(np.mean(u_vel))
                        data['Velocity_y'].append(np.mean(v_vel))
                        data['x_pos'].append(np.mean(x_pos1))
                        data['y_pos'].append(np.mean(y_pos1))
                        dudx, intercept_ux, r_value_ux, p_value_ux, std_err_ux = stats.linregress(x_pos1, u_vel)
                        dudy, intercept_uy, r_value_uy, p_value_uy, std_err_uy = stats.linregress(y_pos1, u_vel)
                        dvdx, intercept_vx, r_value_vx, p_value_vx, std_err_vx = stats.linregress(x_pos1, v_vel)
                        dvdy, intercept_vy, r_value_vy, p_value_vy, std_err_vy = stats.linregress(y_pos1, v_vel)

                        data['ux_translational'].append(intercept_ux)
                        data['uy_translational'].append(intercept_uy)
                        data['u_translational'].append(np.sqrt(intercept_uy ** 2 + intercept_ux ** 2))
                        data['v_translational'].append(np.sqrt(intercept_vy ** 2 + intercept_vx ** 2))
                        data['vx_translational'].append(intercept_vx)
                        data['vy_translational'].append(intercept_vy)

                        tensor = np.array([[dudx, dudy], [dvdx, dvdy]])
                        data['Velocity_gradient_tensor'].append(tensor)
                        omega = (tensor - tensor.T) / 2
                        strain = (tensor + tensor.T) / 2

                        data['Rotation'].append(omega)
                        data['Strain_rate'].append(strain)
                        trace = strain[0, 0] + strain[1, 1]
                        data['Rate_of_area_change'].append(trace)
                        w_strain, v_strain = np.linalg.eig(strain)

                        data['Eigenvalues_strain_1'].append(w_strain[0])
                        data['Eigenvalues_strain_2'].append(w_strain[1])
                        data['Eigenvectors_strain'].append(v_strain)
                        v_strain = v_strain.T
                        angle1 = np.rad2deg(np.arctan2(v_strain[0][1], v_strain[0][0]))
                        angle2 = np.rad2deg(np.arctan2(v_strain[1][1], v_strain[1][0]))

                        data['Eigenvectors_strain_1'].append(angle1)
                        data['Eigenvectors_strain_2'].append(angle2)

                        w_strain_2, v_strain_2 = np.linalg.eig(omega)
                        data['Eigenvalues_rotation'].append(omega[0, 1])
                        data['Eigenvectors_rotation'].append(v_strain_2)

                        p_mat = -tensor[0, 0] - tensor[1, 1]
                        q_mat = 1 / 2 * p_mat ** 2 - 1 / 2 * tensor[1, 0] * tensor[0, 1] - \
                                1 / 2 * tensor[0, 1] * tensor[1, 0] - 1 / 2 * tensor[0, 0] * tensor[0, 0] - \
                                1 / 2 * tensor[1, 1] * tensor[1, 1]
                        data['First_invariant'].append(p_mat)
                        data['Second_invariant'].append(q_mat)

        tensor_dataframe = pd.DataFrame(data)
        tensor_dataframe.set_index(['Index_Time'], inplace=True)
        return tensor_dataframe

    def plot_tensor_dataframe(self, ax, colonies, tensor_dataframe):
        """
        PLOTTING FUNCTION
        Make strain rate movie
        """

        count = 0
        for t, v in colonies.items():
            if int(t) < [int(t) for t, v in colonies.items()][-1]:

                tensor_first = tensor_dataframe.Strain_rate[[int(t) for t, v in colonies.items()][0]]
                radius = max(abs(np.sqrt(tensor_first[0, 0] ** 2 +
                                         tensor_first[0, 1] ** 2)),
                             abs(np.sqrt(tensor_first[1, 1] ** 2 +
                                         tensor_first[1, 0] ** 2)))
                circle1 = plt.Circle((0, 0), radius, fill=False)
                ax.add_artist(circle1)
                ax.set_aspect(1)
                ax.set(xlim=[-0.02, 0.02], ylim=[-0.02, 0.02])
                tensor = tensor_dataframe.Strain_rate[int(t)]
                tensor_rotation = tensor_dataframe.Rotation[int(t)]

                w_tensor, v_tensor = np.linalg.eig(tensor)

                angle = tensor_rotation[0, 1]
                v_tensor = v_tensor.T

                pt1_u = - w_tensor[0] * v_tensor[0]
                pt2_u = + w_tensor[0] * v_tensor[0]

                pts_u = [pt1_u, pt2_u]
                x_u, y_u = zip(*pts_u)

                pt1_v = -w_tensor[1] * v_tensor[1]
                pt2_v = +w_tensor[1] * v_tensor[1]

                pts_v = [pt1_v, pt2_v]
                x_v, y_v = zip(*pts_v)

                if tensor[0, 0] > 0:
                    ax.plot(x_u, y_u, color='blue')
                else:
                    ax.plot(x_u, y_u, color='red')

                if tensor[1, 1] > 0:
                    ax.plot(x_v, y_v, color='blue')
                else:
                    ax.plot(x_v, y_v, color='red')

                if angle > 0:
                    center, th1, th2 = (0, 0), 0, 180
                    ax.arrow(-(radius + 0.01), 0.005, 0, -0.005, color='green')
                else:
                    center, th1, th2 = (0, 0), 180, 0
                    ax.arrow((radius + 0.01), -0.005, 0, 0.005, color='green')

                patch = matplotlib.patches.Arc(center, 2 * (radius + 0.01), 2 * (radius + 0.01),
                                               0, th1, th2, color='green')
                ax.add_patch(patch)
                fname = '_tmp%05d.png' % int(count)
                plt.tight_layout()
                plt.savefig(fname)
                plt.clf()
                plt.cla()
                plt.close()
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                count += 1

        fps = 5
        os.system("rm movie_strain_rate.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_strain_rate.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def seaborn_nodes_dataframe(self, colonies, data, old_labels=None, counter=None):
        """
        DATAFRAME FUNCTION
        Make nodes_dataframe which containts node related information like
        number of connected edges, tension residual, average curvature of conn edges etc.
        """

        initial_index = [int(k) for k, v in colonies.items()][0]
        labels = [e.label for e in colonies[str(initial_index)].tot_nodes if e.label != []]
        labels = sorted(labels)

        if data is None:
            data = {'Index_Node_Label': [], 'Index_Time': [], 'Time': [],
                    'Number_of_connected_edges': [], 'Average_curvature_of_connected_edges': [],
                    'Connected_edge_labels': [], 'Residual': [], 'Change_in_num_con_edges': [],
                    'Length_of_connected_edges': [], 'Movement_from_prev_t': [],
                    'Mean_radius_of_connected_edges': [], 'Node_Label': [], 'Mean_Tension': [],
                    'Std_Tension': [], 'Change_in_mean_tension': [], 'Net_residual': []}
            nodes_dataframe = pd.DataFrame(data)
            nodes_dataframe.set_index(['Index_Node_Label', 'Index_Time'], inplace=True)

        for lab in labels:
            if not old_labels or lab not in old_labels:
                tensions, num_of_con_edges = [], []
                node_index = 0
                locs = []
                for t, v in colonies.items():
                    if [len(n.edges) for n in v.tot_nodes if n.label == lab] != []:
                        x = [e.tension for e in v.tot_edges]
                        if [len(n.edges) for n in v.tot_nodes if n.label == lab][0] > 2:
                            start_tensions = []
                            node1 = [n for n in v.tot_nodes if n.label == lab][0]
                            indices = node1.edge_indices
                            for i in range(len(indices)):
                                start_tensions.append([x[indices[i]]])
                            start_tensions = np.array(start_tensions)

                            node_vecs = node1.tension_vectors
                            tension_vecs = np.multiply(node_vecs, start_tensions)
                            resid_vec = np.sum(tension_vecs, 0)
                            resid_mag = np.hypot(*resid_vec)
                        else:
                            resid_mag = np.NaN

                        con_labels = [e.label for n in v.tot_nodes for e in n.edges if n.label == lab]
                        data['Connected_edge_labels'].append(con_labels)

                        data['Net_residual'].append(
                            [np.linalg.norm(n.residual_vector) for n in v.tot_nodes if n.label == lab][0])
                        data['Residual'].append(resid_mag)
                        data['Time'].append(int(t))
                        data['Index_Time'].append(int(t))
                        data['Node_Label'].append(lab)
                        data['Index_Node_Label'].append(lab)
                        data['Number_of_connected_edges'].append([len(n.edges) for n in v.tot_nodes
                                                                  if n.label == lab][0])
                        num_of_con_edges.append([len(n.edges) for n in v.tot_nodes
                                                 if n.label == lab][0])
                        avg_radius_con_edges = sum([1 / e.radius for n in v.tot_nodes
                                                    for e in n.edges if n.label == lab])
                        data['Average_curvature_of_connected_edges'].append(avg_radius_con_edges)
                        data['Length_of_connected_edges'].append(sum([e.straight_length for n in v.tot_nodes
                                                                      for e in n.edges if n.label == lab]))
                        data['Mean_radius_of_connected_edges'].append(sum([e.radius for n in v.tot_nodes
                                                                           for e in n.edges if n.label == lab]))
                        data['Mean_Tension'].append(np.mean([e.tension for n in v.tot_nodes
                                                             for e in n.edges if n.label == lab]))
                        data['Std_Tension'].append(np.std([e.tension for n in v.tot_nodes
                                                           for e in n.edges if n.label == lab]))
                        tensions.append(np.mean([e.tension for n in v.tot_nodes
                                                 for e in n.edges if n.label == lab]))
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
                            data['Change_in_num_con_edges'].append(abs(num_of_con_edges[node_index] -
                                                                       num_of_con_edges[node_index - 1]))
                            node_index += 1

                        nodes_dataframe = pd.DataFrame(data)
                        nodes_dataframe.set_index(['Index_Node_Label', 'Index_Time'], inplace=True)
        if not counter:
            counter = 1
        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > counter)

        if new_colony_range != {}:
            old_labels = labels
            counter = counter + 1
            self.seaborn_nodes_dataframe(new_colony_range, data, old_labels, counter)
            nodes_dataframe = pd.DataFrame(data)
            nodes_dataframe.set_index(['Index_Node_Label', 'Index_Time'], inplace=True)
            return nodes_dataframe

    def seaborn_plot(self, ax, colonies, common_edge_labels, common_cell_labels,
                     data=None, cell_data=None, old_labels=None, old_cell_labels=None,
                     counter=None, min_ten=None, max_ten=None, min_pres=None, max_pres=None,
                     ground_truth=None):
        """
        DATAFRAME FUNCTION
        Make an edges_dataframe and cells_dataframe
        Edges dataframe has information about tension, stochasticity in tension etc.
        Cells dataframe has information about pressure, stochasticity in pressure etc.
        """

        if not min_ten and not max_ten and not min_pres and not max_pres:
            all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
            _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type='pressure')
            _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
            min_ten = min(all_tensions)
            max_ten = max(all_tensions)

        initial_index = [int(k) for k, v in colonies.items()][0]
        labels = [e.label for e in colonies[str(initial_index)].tot_edges if e.label != []]
        cell_labels = [c.label for c in colonies[str(initial_index)].cells if c.label != []]

        if data is None:
            if ground_truth is not None:
                data = {'Index_Edge_Labels': [], 'Index_Time': [], 'Edge_Labels': [],
                        'Strain_rate': [], 'Topological_changes': [], 'Normalized_Tensions': [],
                        'Ground_truth_stochasticity_in_tension': [], 'Stochasticity_in_tension': [],
                        'Local_normalized_tensions': [], 'Deviation': [], 'Tensions': [],
                        'Repeat_Tensions': [], 'Change_in_tension': [], 'Time': [],
                        'Curvature': [], 'Radius': [], 'Straight_Length': [], 'Curve_fit_residual': [],
                        'Total_connected_edge_length': [], 'Change_in_length': [],
                        'Change_in_connected_edge_length': [], 'Binary_length_change': [],
                        'Binary_connected_length_change': [], 'Ground_truth': [], 'Ground_truth_error': []}
            else:
                data = {'Index_Edge_Labels': [], 'Index_Time': [], 'Edge_Labels': [], 'Strain_rate': [],
                        'Topological_changes': [], 'Normalized_Tensions': [],
                        'Stochasticity_in_tension': [], 'Local_normalized_tensions': [],
                        'Deviation': [], 'Tensions': [], 'Repeat_Tensions': [], 'Change_in_tension': [],
                        'Time': [], 'Curvature': [], 'Radius': [], 'Straight_Length': [], 'Curve_fit_residual': [],
                        'Total_connected_edge_length': [], 'Change_in_length': [],
                        'Change_in_connected_edge_length': [], 'Binary_length_change': [],
                        'Binary_connected_length_change': []}
            edges_dataframe = pd.DataFrame(data)
            edges_dataframe.set_index(['Index_Edge_Labels', 'Index_Time'])

        if cell_data is None:
            cell_data = {'Index_Cell_Labels': [], 'Index_Cell_Time': [], 'Cell_Labels': [],
                         'Ground_truth_pressure': [], 'Centroid_movement': [], 'Rotation': [],
                         'Binary_rotation': [], 'Number_of_edges': [], 'Normalized_Pressures': [],
                         'Pressures': [], 'Mean_node_edge_tension': [], 'Sum_edge_tension': [],
                         'Repeat_Pressures': [], 'Change_in_pressure': [], 'Cell_Time': [],
                         'Area': [], 'Perimeter': [], 'Change_in_area': [], 'Binary_area_change': [],
                         'Change_in_perimeter': [], 'Binary_perim_change': [], 'Energy': []}
            cells_dataframe = pd.DataFrame(cell_data)
            cells_dataframe.set_index(['Index_Cell_Labels', 'Index_Cell_Time'], inplace=True)

        for lab in labels:
            if not old_labels or lab not in old_labels:
                edge_index = 0
                lengths, con_lengths, tensions, norm_tensions, norm_ground_truths, all_con_labels = [], [], \
                                                                                                    [], [], [], []
                for t, v in colonies.items():
                    mean_tens = np.mean([e.tension for e in v.tot_edges])
                    if ground_truth is not None:
                        mean_ground_truths = np.mean([e.ground_truth for e in v.tot_edges])
                    if [e.tension for e in v.tot_edges if e.label == lab] != []:
                        data['Edge_Labels'].append(lab)
                        data['Index_Edge_Labels'].append(lab)
                        data['Tensions'].append([e.tension for e in v.tot_edges
                                                 if e.label == lab][0])
                        data['Normalized_Tensions'].append(([e.tension for e in v.tot_edges
                                                             if e.label == lab][0] - min_ten) / float(
                            max_ten - min_ten))
                        if lab in common_edge_labels:
                            data['Repeat_Tensions'].append([e.tension for e in v.tot_edges
                                                            if e.label == lab][0])
                        else:
                            data['Repeat_Tensions'].append(np.NaN)

                        if ground_truth is not None:
                            data['Ground_truth'].append([e.ground_truth for e in v.tot_edges
                                                         if e.label == lab][0] / mean_ground_truths)
                            data['Ground_truth_error'].append(np.abs([e.ground_truth for e in v.tot_edges
                                                                      if e.label == lab][0] -
                                                                     [e.tension for e in v.tot_edges if e.label == lab][
                                                                         0] / mean_tens))

                        data['Local_normalized_tensions'].append([e.tension for e in v.tot_edges
                                                                  if e.label == lab][0] / mean_tens)
                        [norm_tensions.append([e.tension for e in v.tot_edges if e.label == lab][0] / mean_tens)]
                        if ground_truth is not None:
                            [norm_ground_truths.append(
                                [e.ground_truth for e in v.tot_edges if e.label == lab][0] / mean_ground_truths)]
                        current_edge = [e for e in v.tot_edges if e.label == lab][0]
                        con_edges = [e for n in current_edge.nodes for e in n.edges if e != current_edge]
                        con_labels = [e.label for e in con_edges]
                        con_lengths.append(sum([e.straight_length for e in con_edges]))
                        data['Deviation'].append([e.tension for e in v.tot_edges if e.label == lab][0]
                                                 - np.mean(np.array([e.tension for e in v.tot_edges])))
                        data['Total_connected_edge_length'].append(sum([e.straight_length for e in con_edges]))
                        data['Time'].append(int(t))
                        data['Index_Time'].append(int(t))
                        data['Radius'].append([e.radius for e in v.tot_edges if e.label == lab][0])
                        data['Curvature'].append([1 / e.radius for e in v.tot_edges if e.label == lab][0])
                        for e in v.tot_edges:
                            if len(e.cells) == 2:
                                ress = np.abs(e.tension - e.cells[0].pressure)
                        data['Curve_fit_residual'].append([e.curve_fit_residual for e in v.tot_edges if e.label == lab][0])
                        all_con_labels.append(con_labels)
                        [tensions.append([e.tension for e in v.tot_edges if e.label == lab][0])]
                        [lengths.append([e.straight_length for e in v.tot_edges if e.label == lab][0])]
                        data['Straight_Length'].append([e.straight_length for e in v.tot_edges if e.label == lab][0])
                        if edge_index == 0:
                            data['Change_in_length'].append(0)
                            data['Change_in_tension'].append(0)
                            data['Topological_changes'].append(0)
                            data['Stochasticity_in_tension'].append(0)
                            if ground_truth is not None:
                                data['Ground_truth_stochasticity_in_tension'].append(0)
                            data['Strain_rate'].append(0)
                            data['Change_in_connected_edge_length'].append(0)
                            data['Binary_length_change'].append('Initial Length')
                            data['Binary_connected_length_change'].append('Initial Connected Edge Length')
                            edge_index += 1
                        else:
                            # For topological changes
                            cur_con_lab = all_con_labels[edge_index]
                            old_con_lab = all_con_labels[edge_index - 1]

                            try:
                                if set(cur_con_lab) != set(old_con_lab):
                                    data['Topological_changes'].append(1)
                                else:
                                    data['Topological_changes'].append(0)
                            except:
                                data['Topological_changes'].append(None)

                            data['Strain_rate'].append((lengths[edge_index] -
                                                        lengths[edge_index - 1]) / lengths[edge_index - 1])
                            data['Change_in_length'].append(lengths[edge_index] - lengths[edge_index - 1])
                            data['Change_in_tension'].append(tensions[edge_index] - tensions[edge_index - 1])
                            data['Stochasticity_in_tension'].append(norm_tensions[edge_index] -
                                                                    norm_tensions[edge_index - 1])
                            if ground_truth is not None:
                                data['Ground_truth_stochasticity_in_tension'].append(norm_ground_truths[edge_index]
                                                                                     - norm_ground_truths[
                                                                                         edge_index - 1])
                            data['Change_in_connected_edge_length'].append(con_lengths[edge_index]
                                                                           - con_lengths[edge_index - 1])
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
            if not old_cell_labels or cell_lab not in old_cell_labels:
                cell_index = 0
                areas, perims, pressures, centroids = [], [], [], []
                for t, v in colonies.items():
                    min_pres, max_pres = np.min([c.pressure for c in v.cells]), np.max([c.pressure for c in v.cells]) 

                    if ground_truth is not None:
                        min_pres_gt, max_pres_gt = np.min([c.ground_truth_pressure for c in v.cells]), np.max([c.ground_truth_pressure for c in v.cells]) 
                    if [c.pressure for c in v.cells if c.label == cell_lab] != []:
                        cell_data['Cell_Labels'].append(cell_lab)
                        cell_data['Index_Cell_Labels'].append(cell_lab)
                        cell_data['Pressures'].append([2*((c.pressure - min_pres) / float(max_pres - min_pres)) - 1 for c in v.cells if c.label == cell_lab][0])
                        if ground_truth is not None:
                            cell_data['Ground_truth_pressure'].append([2*((c.ground_truth_pressure - min_pres_gt) / float(max_pres_gt - min_pres_gt)) - 1 for c in v.cells 
                                                                        if c.label == cell_lab][0])
                        else:
                            cell_data['Ground_truth_pressure'].append(0)
                        cell_data['Normalized_Pressures'].append(
                            ([c.pressure for c in v.cells if c.label == cell_lab][0]
                             - min_pres) / float(max_pres - min_pres))
                        if cell_lab in common_cell_labels:
                            cell_data['Repeat_Pressures'].append([c.pressure for c in v.cells
                                                                  if c.label == cell_lab][0])
                        else:
                            cell_data['Repeat_Pressures'].append(np.NaN)
                        cell_data['Cell_Time'].append(int(t))
                        cell_data['Index_Cell_Time'].append(int(t))
                        temp_node_tensions = [e.tension for c in v.cells
                                              for n in c.nodes for e in n.edges if c.label == cell_lab]
                        temp_edge_tensions = [e.tension for c in v.cells
                                              for e in c.edges if c.label == cell_lab]
                        cell_data['Number_of_edges'].append([len(c.edges) for c in v.cells
                                                             if c.label == cell_lab][0])
                        cell_data['Mean_node_edge_tension'].append(np.mean(temp_node_tensions))
                        cell_data['Sum_edge_tension'].append(np.sum(temp_edge_tensions)**2)
                        mean_t = np.mean([e.tension for e in v.tot_edges])
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
                            x1, y1 = centroids[cell_index - 1][0], centroids[cell_index - 1][1]
                            x2, y2 = centroids[cell_index][0], centroids[cell_index][1]
                            cosang = np.dot([x1, y1], [x2, y2])
                            sinang = np.cross([x1, y1], [x2, y2])
                            theta1 = np.rad2deg(np.arctan2(sinang, cosang))
                            cell_data['Rotation'].append(theta1)
                            cell_data['Centroid_movement'].append(np.linalg.norm(
                                np.subtract(centroids[cell_index], centroids[cell_index - 1])))
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
                        ar, per = [c.area() for c in v.cells if c.label == cell_lab][0], \
                                  [c.perimeter() for c in v.cells if c.label == cell_lab][0]
                        cell_data['Energy'].append(ar ** 2 + per ** 2)
        if not counter:
            counter = 1
        new_colony_range = dict((k, v) for k, v in colonies.items() if int(k) > counter)
        if new_colony_range != {}:
            old_labels = labels
            old_cell_labels = cell_labels
            length_dict = {key: len(value) for key, value in data.items()}
            self.seaborn_plot(ax, new_colony_range, common_edge_labels, common_cell_labels,
                              data, cell_data, old_labels, old_cell_labels, counter + 1,
                              min_ten, max_ten, min_pres, max_pres, ground_truth)

            length_dict = {key: len(value) for key, value in data.items()}
            edges_dataframe = pd.DataFrame(data)
            edges_dataframe.set_index(['Index_Edge_Labels', 'Index_Time'], inplace=True)
            cells_dataframe = pd.DataFrame(cell_data)
            cells_dataframe.set_index(['Index_Cell_Labels', 'Index_Cell_Time'], inplace=True)
            return edges_dataframe, cells_dataframe

    def get_min_max_by_outliers_iqr(self, ys, type=None):
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
            # replace min_t with mean_t and max_t - min_t with standard deviation
            std_t = np.std([e for e in updated_list if e != np.inf])
            mean_t = np.mean([e for e in updated_list if e != np.inf])
            return updated_list, mean_t + std_t, mean_t

    def outliers_modified_z_score(self, ys, y):
        """
        Alternative outlier check. Not used currently
        useful for plotting
        """
        threshold = 3.5

        median_y = np.median(ys)
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
        modified_z_scores = 0.6745 * (y - median_y) / median_absolute_deviation_y
        return np.where(np.abs(modified_z_scores) > threshold)

    def plot_pressures(self, fig, ax, colonies, specify_aspect=None, specify_color=None, **kwargs):
        """
        PLOTTING FUNCTION
        Make a pressure movie over colonies
        """
        _, _, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type='pressure')

        counter = 0
        for t, v in colonies.items():
            index = str(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            colonies[index].plot_pressures(ax, fig, pressures, min_pres, max_pres, specify_color, **kwargs)
            [e.plot(ax) for e in colonies[index].edges]
            if specify_aspect is not None:
                ax.set(xlim=[0, 600], ylim=[0, 600], aspect=1)
            pylab.savefig('_tmp%05d.png' % counter, dpi=200)
            counter += 1
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        fps = 1
        os.system("rm movie_pressure.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_pressure.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()

    def plot_both_tension_pressure(self, fig, ax, colonies, specify_aspect=None, specify_color=None, **kwargs):
        """
        PLOTTING FUNCTION
        Make a combined tension + pressure movie over colonies
        """
        all_tensions, all_radii, all_pressures = self.all_tensions_and_radius_and_pressures(colonies)
        _, max_ten, min_ten = self.get_min_max_by_outliers_iqr(all_tensions)
        _, max_rad, min_rad = self.get_min_max_by_outliers_iqr(all_radii)
        _, max_pres, min_pres = self.get_min_max_by_outliers_iqr(all_pressures, type='pressure')

        counter = 0
        for t, v in colonies.items():
            index = str(t)
            cells = colonies[index].cells
            pressures = [e.pressure for e in cells]
            edges = colonies[index].tot_edges
            tensions = [e.tension for e in edges]
            colonies[index].plot(ax, fig, tensions, pressures, min_ten, max_ten,
                                 min_pres, max_pres, specify_color, **kwargs)
            if specify_aspect is not None:
                ax.set(xlim=[0, 600], ylim=[0, 600], aspect=1)

            pylab.savefig('_tmp%05d.png' % counter, dpi=200)
            counter += 1
            plt.cla()
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        fps = 5
        os.system("rm movie_ten_pres.mp4")
        os.system("ffmpeg -r " + str(fps) + " -b 1800 -i _tmp%05d.png movie_ten_pres.mp4")
        os.system("rm _tmp*.png")

        plt.cla()
        plt.clf()
        plt.close()