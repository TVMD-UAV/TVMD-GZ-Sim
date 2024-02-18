from Visualizer import FlightDataVisualizer

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import os
import numpy as np

def figure_save_helper(fig, title):
    fig.align_ylabels()
    fname = os.path.join(FlightDataVisualizer.outfolder, title)
    plt.savefig("{}.eps".format(fname), format="eps", bbox_inches="tight")
    plt.savefig("{}.svg".format(fname), format="svg", bbox_inches="tight")
    plt.savefig("{}.pdf".format(fname), format="pdf", bbox_inches="tight")
    # plt.show()
    plt.pause(0.1)


def plot_add_cut_lines(ax, cut_time):
    for i in range(len(cut_time)):
        for j in range(len(ax)):
            ax[j].axvline(x=cut_time[i], ymin=-1.2 if j != len(ax)-1 else 0, ymax=1, color="red", 
                          linewidth=1.5, linestyle="--", zorder=0, clip_on=False)

def plot_vector3f(t, data, keys, title, colors=["#0072BD", "#D95319", "#EDB120"], limits=None):
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    for i in range(3):
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        ax[i].plot(t, data[:, i], color=colors[i], linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])

    ax[i].set_xlabel(r'$t$', fontsize=16)
    plot_add_cut_lines(ax, FlightDataVisualizer.x_cut_lines)
    figure_save_helper(fig, title)

"""
Data has the shape of M x [N x 3], where M is the number of data points, N is the number of data sets, and 3 is the dimension of the data
"""
def plot_vector3f_batch(t, data, keys, title, limits=None, marker_style=None, legend=None):
    colors = ["#0072BD", "#D95319", "#EDB120"]
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    num_instance = len(data)
    num_entry = data[0].shape[1]   # x, y, z

    for i in range(num_entry):    # x, y, z
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        for j in range(num_instance):
            ax[i].plot(t, data[j][:, i], color=colors[i], linestyle="-" if marker_style is None else marker_style[j], linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if legend is not None:
            ax[i].legend(legend, loc="upper right", ncol=num_entry)


    ax[i].set_xlabel(r'$t$', fontsize=16)
    plot_add_cut_lines(ax, FlightDataVisualizer.x_cut_lines)
    figure_save_helper(fig, title)

def plot_vector3f_batch_separate(t, data, keys, title, limits=None, legend=None, colors=["#0072BD", "#D95319", "#EDB120"], legend_loc="upper right"):
    
    num_instance = len(data)

    gs = gridspec.GridSpec(num_instance,1)
    fig = plt.figure(figsize=(5, 4.0/3.0*num_instance))
    ax = [ 0 for i in range(num_instance)]


    for i in range(num_instance):    # x, y, z
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        num_entry = data[i].shape[1]
        # if colors is None and num_entry == 3:
        if num_entry == 3:
            colors=["#0072BD", "#D95319", "#EDB120"]
        else:
            colormap = cm.get_cmap('jet', num_entry+1)
            colors = colormap(np.arange(num_entry))

        for j in range(num_entry):
            ax[i].plot(t, data[i][:, j], color=colors[j], linestyle="-", linewidth=2)
        print("keys: ", keys)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != num_instance-1:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if legend is not None:
            if num_entry == 3:
                ax[i].legend(legend, loc=legend_loc, ncol=num_entry)
            elif num_entry == 4:
                ax[i].legend(legend, loc=legend_loc, ncol=num_entry)
                # ax[i].legend(["x", "y", "z", "w"], loc=legend_loc, ncol=num_entry)


    ax[i].set_xlabel(r'$t$', fontsize=16)
    plot_add_cut_lines(ax, FlightDataVisualizer.x_cut_lines)
    figure_save_helper(fig, title)

def plot_vector3f_with_desire(t, data, t_d, data_d, keys, title, limits=None):
    colors = ["#0072BD", "#D95319", "#EDB120"]
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure(figsize=(5, 4))
    ax = [0, 0, 0]

    for i in range(3):
        if i == 0:
            ax[i] = fig.add_subplot(gs[i])
        else:
            ax[i] = fig.add_subplot(gs[i], sharex=ax[0])

        ax[i].plot(t, data[:, i], color=colors[i], linestyle="-", linewidth=2)
        ax[i].plot(t_d, data_d[:, i], color=colors[i], linestyle="--", linewidth=2)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if i != 2:
            plt.setp(ax[i].get_xticklabels(), visible=False)

    ax[2].set_xlabel(r'$t$', fontsize=16)
    plot_add_cut_lines(ax, FlightDataVisualizer.x_cut_lines)
    figure_save_helper(fig, title)

def plot_vector3f_batch_separate_with_desire(t, data, data_d, keys, title, limits=None, legend=None, colors=["#0072BD", "#D95319", "#EDB120"], legend_loc="upper right"):
    
    num_instance = len(data)

    gs = gridspec.GridSpec(num_instance,1)
    fig = plt.figure(figsize=(5, 4.0/3.0*num_instance))
    ax = [ 0 for i in range(num_instance)]


    for i in range(num_instance):    # x, y, z
        ax[i] = fig.add_subplot(gs[i], sharex= ax[0] if i != 0 else None)
        num_entry = data[i].shape[1]
        # if colors is None and num_entry == 3:
        if num_entry == 3:
            colors=["#0072BD", "#D95319", "#EDB120"]
        else:
            colormap = cm.get_cmap('jet', num_entry+1)
            colors = colormap(np.arange(num_entry))

        for j in range(num_entry):
            ax[i].plot(t, data_d[i][:, j], color=colors[j], linestyle="--", linewidth=2)

        for j in range(num_entry):
            ax[i].plot(t, data[i][:, j], color=colors[j], linestyle="-", linewidth=2)
        print("keys: ", keys)
        ax[i].set_ylabel(r'${}$'.format(keys[i]), fontsize=16)
        if i != num_instance-1:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        if limits is not None:
            ax[i].set_ylim(limits[i])
        if legend is not None:
            if num_entry == 3:
                ax[i].legend(legend, loc=legend_loc, ncol=num_entry)
            elif num_entry == 4:
                ax[i].legend([r"$x$", r"$y$", r"$z$", r"$w$"], loc=legend_loc, ncol=num_entry)


    ax[i].set_xlabel(r'$t$', fontsize=16)
    plot_add_cut_lines(ax, FlightDataVisualizer.x_cut_lines)
    figure_save_helper(fig, title)
