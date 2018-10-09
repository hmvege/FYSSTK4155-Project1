#!/usr/bin/env python2
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter




from matplotlib import rc, rcParams
rc("text", usetex=True)
rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
rcParams["font.family"] += ["serif"]


def FrankeFunction(x, y):
    """As retrieved from project description."""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def plot_simple_surface(x, y, z, filename="../../fig/simple_surface"):
    """Surface plotter from project notes."""

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plotting the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(-01.0, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # print(os.getcwd())
    figpath = os.path.abspath(filename + ".pdf")
    # print(figpath)
    # plt.savefig(figpath, dpi=350)
    plt.show()


def franke_function_example():
    """Plots a simple examply surface."""
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    plot_simple_surface(x, y, z)

def view_data():
    folder_path = "../../../MachineLearning/doc/Projects/2018/Project1/DataFiles/"
    abs_folder_path = os.path.abspath(folder_path)
    data_path = os.path.join(abs_folder_path, os.listdir(abs_folder_path)[0])

    print("Loading data from:", data_path)
    terrain = imageio.imread(data_path)

    # plt.figure()
    # plt.imshow(terrain, cmap="gray")
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")
    # plt.savefig("../../fig/terrain_data.png", dpi=350)
    # plt.show()

    # Extract a smaller patch of the terrain
    row_start = 1550#1950
    row_end = 1650#2050

    col_start = 600#1200
    col_end = 850#1450

    terrain_patch = terrain[row_start:row_end, col_start:col_end]


    # plt.figure()    
    # plt.imshow(terrain_patch, cmap="gray")
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")
    # plt.savefig("../../fig/terrain_data_zoomed.png", dpi=350)
    # plt.show()

    terrain_patch = terrain_patch / float(np.max(terrain))


    M, N = terrain_patch.shape

    # print (row_end-row_start, M)
    # print (col_end-col_start, N)

    # ax_rows = np.arange(row_start, row_start+M, 1)
    # ax_cols = np.arange(col_start, col_start+N, 1)

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [x, y] = np.meshgrid(ax_cols, ax_rows)
    z = terrain_patch

    plot_simple_surface(x,y,z,"../../fig/3d_terrain")





def main():
    # franke_function_example()
    view_data()


if __name__ == '__main__':
    # plot_terrain_data("data.npy")
    main()
