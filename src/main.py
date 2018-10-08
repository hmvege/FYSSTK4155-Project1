#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import os

import lib.regression as libreg
import lib.metrics as metrics

# Task a)-c), Franke function testing
import tasks.regression_generator as reggen
import lib.franke_function as ff_tools

# Task d)-e), real data
import imageio


def main():
    franke_func_tasks()
    # real_data()


def surface_plot(surface, title):
    M, N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X, Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, surface, cmap=cm.viridis, linewidth=0)
    plt.title(title)


def real_data():
    """Part d)-e)."""

    # Loads data
    folder_path = "../../MachineLearning/doc/Projects/2018/Project1/DataFiles/"
    abs_folder_path = os.path.abspath(folder_path)
    data_path = os.path.join(abs_folder_path, os.listdir(abs_folder_path)[0])

    print("Loading data from:", data_path)
    terrain = imageio.imread(data_path)

    # plt.figure()
    # plt.imshow(data)
    # plt.show()

    # Extract a smaller patch of the terrain
    row_start = 1950
    row_end = 2050

    col_start = 1200
    col_end = 1450

    terrain_patch = terrain[row_start:row_end, col_start:col_end]

    # Normalizes
    terrain_patch = terrain_patch/np.amax(terrain_patch)

    # Sets up X,Y,Z data
    M, N = terrain_patch.shape

    # print (row_end-row_start, M)
    # print (col_end-col_start, N)

    # ax_rows = np.arange(row_start, row_start+M, 1)
    # ax_cols = np.arange(col_start, col_start+N, 1)

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [x, y] = np.meshgrid(ax_cols, ax_rows)
    z = terrain_patch

    # surface_plot(terrain1, "Surface plot over Norway")
    # plt.show()

    # Analysis constants
    N_bs_resampling = 1000
    N_cv_bs = 100
    k_splits = 4
    test_percent = 0.4

    noise_sigma = 0.1
    polynom_degrees = [5]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4]

    np.random.seed(1234)

    regression_methods = []
    # regression_methods += ["ols"]
    # regression_methods += ["ridge"]
    regression_methods += ["lasso"]

    regression_implementation = []
    regression_implementation += ["sklearn"]
    # regression_implementation += ["manual"]

    run_regrssion_methods(regression_methods, polynom_degrees,
                          regression_implementation,
                          x, y, z, N_bs_resampling, N_cv_bs, test_percent,
                          alpha_values)

    # if "ols" in regression_methods:
    #     print("\nOrdinarty Linear Regression")
    #     for degree in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(degree))

    #         if "manual" in regression_implementation:
    #             ols = reggen.ManualOLS(x, y, z, deg=degree,
    #                                    N_bs=N_bs_resampling,
    #                                    N_cv_bs=N_cv_bs,
    #                                    test_percent=test_percent)

    #         if "sklearn" in regression_implementation:
    #             sk_ols = reggen.SKLearnOLS(x, y, z, deg=degree,
    #                                        N_bs=N_bs_resampling,
    #                                        N_cv_bs=N_cv_bs,
    #                                        test_percent=test_percent)

    # if "ridge" in regression_methods:
    #     print("\nRidge Regression")
    #     for deg in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(deg))
    #         for alpha in alpha_values:
    #             print("\n**** Ridge Lambda: {:-e} ****".format(alpha))

    #             if "manual" in regression_implementation:
    #                 ridge = reggen.ManualRidge(x, y, z, alpha, deg=deg,
    #                                            test_percent=test_percent)

    #             if "sklearn" in regression_implementation:
    #                 sk_ridge = reggen.SKLearnRidge(x, y, z, alpha, deg=deg,
    #                                                test_percent=test_percent)

    # if "lasso" in regression_methods:
    #     print("\nLasso Regression")
    #     for deg in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(deg))
    #         for alpha in alpha_values:
    #             print("\n**** Lasso Lambda: {:-e} ****".format(alpha))

    #             # if "manual" in regression_implementation:
    #             #     reggen.task_c_manual(x, y, z, alpha, deg=deg,
    #             #                   test_percent=test_percent)

    #             if "sklearn" in regression_implementation:
    #                 sk_lasso = reggen.SKLearnLasso(x, y, z, alpha, deg=deg,
    #                                                test_percent=test_percent)


def franke_func_tasks():
    """Part a)-c)."""

    # Generates data
    N_data_points = 150 # 150
    x = np.sort(np.random.uniform(0, 1, N_data_points))
    y = np.sort(np.random.uniform(0, 1, N_data_points))

    # Analysis constants
    N_bs_resampling = 1000 # 1000
    N_cv_bs = 100
    k_splits = 4
    test_percent = 0.4
    print_results = True

    noise_sigma = 0.1
    noise_mu = 0
    polynom_degrees = [5]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4]

    np.random.seed(1234)

    regression_methods = []
    regression_methods += ["ols"]
    regression_methods += ["ridge"]
    regression_methods += ["lasso"]

    regression_implementation = []
    regression_implementation += ["sklearn"]
    regression_implementation += ["manual"]

    # x += np.random.uniform(noise_mu, noise_sigma, N)
    # y += np.random.uniform(noise_mu, noise_sigma, N)

    x, y = np.meshgrid(x, y)
    z = ff_tools.FrankeFunction(x, y)

    np.save("surface_plotter/data", np.c_[x, y, z])
    run_regrssion_methods(regression_methods, polynom_degrees,
                          regression_implementation,
                          x, y, z, N_bs_resampling, N_cv_bs, test_percent,
                          alpha_values, print_results)

    # if "ols" in regression_methods:
    #     print("\nOrdinarty Linear Regression")
    #     for degree in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(degree))

    #         if "manual" in regression_implementation:
    #             ols = reggen.ManualOLS(x, y, z, deg=degree,
    #                                    N_bs=N_bs_resampling,
    #                                    N_cv_bs=N_cv_bs,
    #                                    test_percent=test_percent)

    #         if "sklearn" in regression_implementation:
    #             sk_ols = reggen.SKLearnOLS(x, y, z, deg=degree,
    #                                        N_bs=N_bs_resampling,
    #                                        N_cv_bs=N_cv_bs,
    #                                        test_percent=test_percent)

    # if "ridge" in regression_methods:
    #     print("\nRidge Regression")
    #     for deg in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(deg))
    #         for alpha in alpha_values:
    #             print("\n**** Ridge Lambda: {:-e} ****".format(alpha))

    #             if "manual" in regression_implementation:
    #                 ridge = reggen.ManualRidge(x, y, z, alpha, deg=deg,
    #                                            test_percent=test_percent)

    #             if "sklearn" in regression_implementation:
    #                 sk_ridge = reggen.SKLearnRidge(x, y, z, alpha, deg=deg,
    #                                                test_percent=test_percent)

    # if "lasso" in regression_methods:
    #     print("\nLasso Regression")
    #     for deg in polynom_degrees:
    #         print("\n**** Polynom degree: {} ****".format(deg))
    #         for alpha in alpha_values:
    #             print("\n**** Lasso Lambda: {:-e} ****".format(alpha))

    #             # if "manual" in regression_implementation:
    #             #     reggen.task_c_manual(x, y, z, alpha, deg=deg,
    #             #                   test_percent=test_percent)

    #             if "sklearn" in regression_implementation:
    #                 sk_lasso = reggen.SKLearnLasso(x, y, z, alpha, deg=deg,
    #                                                test_percent=test_percent)


def run_regrssion_methods(regression_methods, polynom_degrees,
                          regression_implementation, x, y, z, N_bs_resampling,
                          N_cv_bs, test_percent, alpha_values, print_results):

    if "ols" in regression_methods:
        print("\nOrdinarty Linear Regression")
        for degree in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(degree))

            if "manual" in regression_implementation:
                ols = reggen.ManualOLS(x, y, z, deg=degree,
                                       N_bs=N_bs_resampling,
                                       N_cv_bs=N_cv_bs,
                                       test_percent=test_percent,
                                       print_results=print_results)

            if "sklearn" in regression_implementation:
                sk_ols = reggen.SKLearnOLS(x, y, z, deg=degree,
                                           N_bs=N_bs_resampling,
                                           N_cv_bs=N_cv_bs,
                                           test_percent=test_percent,
                                           print_results=print_results)

    if "ridge" in regression_methods:
        print("\nRidge Regression")
        for deg in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(deg))
            for alpha in alpha_values:
                print("\n**** Ridge Lambda: {:-e} ****".format(alpha))

                if "manual" in regression_implementation:
                    ridge = reggen.ManualRidge(x, y, z, alpha, deg=deg,
                                               test_percent=test_percent,
                                               print_results=print_results)

                if "sklearn" in regression_implementation:
                    sk_ridge = reggen.SKLearnRidge(x, y, z, alpha, deg=deg,
                                                   test_percent=test_percent,
                                                   print_results=print_results)

    if "lasso" in regression_methods:
        print("\nLasso Regression")
        for deg in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(deg))
            for alpha in alpha_values:
                print("\n**** Lasso Lambda: {:-e} ****".format(alpha))

                # if "manual" in regression_implementation:
                #     reggen.task_c_manual(x, y, z, alpha, deg=deg,
                #                   test_percent=test_percent)

                if "sklearn" in regression_implementation:
                    sk_lasso = reggen.SKLearnLasso(x, y, z, alpha, deg=deg,
                                                   test_percent=test_percent,
                                                   print_results=print_results)


if __name__ == '__main__':
    main()
