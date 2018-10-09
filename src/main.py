#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
import collections
import os
import pickle
import time
import copy as cp

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

    t3 = time.clock()

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
    row_start = 1550  # 1950
    row_end = 1650  # 2050

    col_start = 600  # 1200
    col_end = 850  # 1450

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
    print_results = False

    noise_sigma_values = np.linspace(0, 0.5, 15)
    polynom_degrees = [1, 2, 3, 4, 5]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]

    np.random.seed(1234)

    regression_methods = []
    regression_methods += ["ols"]
    regression_methods += ["ridge"]
    regression_methods += ["lasso"]

    regression_implementation = []
    regression_implementation += ["sklearn"]
    regression_implementation += ["manual"]

    data = run_regrssion_methods(regression_methods, polynom_degrees,
                                 regression_implementation,
                                 x, y, z, N_bs_resampling, N_cv_bs,
                                 test_percent, alpha_values, print_results,
                                 noise_sigma_values)

    pickle_fname = "real_data.pickle"
    with open(pickle_fname, "wb") as f:
        pickle.dump(data, f)
        print("Data pickled and dumped to: {:s}".format(pickle_fname))

    t4 = time.clock()

    time_used = t4-t3
    print("\n")
    print("*"*100)
    print("Time used on Terrain data regression: {:.10f} secs/ "
          "{:.10f} minutes".format(time_used, time_used/60.))
    print("*"*100)


def franke_func_tasks():
    """Part a)-c)."""

    t1 = time.clock()

    # Generates data
    N_data_points = 10  # 100
    x = np.sort(np.random.uniform(0, 1, N_data_points))
    y = np.sort(np.random.uniform(0, 1, N_data_points))

    # Analysis constants
    N_bs_resampling = 20  # 1000
    N_cv_bs = 10 # 100
    k_splits = 4
    test_percent = 0.4
    print_results = True

    noise_sigma_values = np.linspace(0, 2.0, 10)
    noise_mu = 0
    polynom_degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    # alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    alpha_values = [1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1,
                    1e-1, 0.5, 1, 0.5e1, 1e1, 0.5e2, 1e2, 0.5e3, 1e3]

    # print(alpha_values)
    # print(np.logspace(-4, 4, 20))
    # exit(1)


    np.random.seed(1234)

    regression_methods = []
    regression_methods += ["ols"]
    regression_methods += ["ridge"]
    regression_methods += ["lasso"]

    regression_implementation = []
    regression_implementation += ["sklearn"]
    regression_implementation += ["manual"]

    x, y = np.meshgrid(x, y)
    z = ff_tools.FrankeFunction(x, y)

    # Stores data for plotting
    np.save("_various_scripts/data", np.c_[x, y, z])

    data = run_regrssion_methods(regression_methods, polynom_degrees,
                                 regression_implementation,
                                 x, y, z, N_bs_resampling, N_cv_bs,
                                 test_percent, alpha_values, print_results,
                                 noise_sigma_values)

    pickle_fname = "franke_func_ols_final3.pickle"
    with open(pickle_fname, "wb") as f:
        pickle.dump(data, f)
        print("Data pickled and dumped to: {:s}".format(pickle_fname))

    t2 = time.clock()

    time_used = t2-t1
    print("\n")
    print("*"*100)
    print("Time used on Franke function regression: {:.10f} secs/ "
          "{:.10f} minutes".format(time_used, time_used/60.))
    print("*"*100)

    t3 = time.clock()


def run_regrssion_methods(regression_methods, polynom_degrees,
                          regression_implementation, x, y, z, N_bs_resampling,
                          N_cv_bs, test_percent, alpha_values, print_results,
                          noise_sigma_values):

    data = []

    if "ols" in regression_methods:
        print("\nOrdinarty Linear Regression")

        for deg in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(deg))

            for noise in noise_sigma_values:
                z += np.random.normal(0, noise, size=z.shape)
                if "manual" in regression_implementation:
                    ols = reggen.ManualOLS(x, y, z, deg=deg,
                                           N_bs=N_bs_resampling,
                                           N_cv_bs=N_cv_bs,
                                           test_percent=test_percent,
                                           print_results=print_results)

                    data.append({
                        "reg_type": "ols",
                        "degree": deg,
                        "noise": noise,
                        "method": "manual",
                        "data": cp.deepcopy(ols.data),
                    })

                # if "sklearn" in regression_implementation:
                #     sk_ols = reggen.SKLearnOLS(x, y, z, deg=degree,
                #                                N_bs=N_bs_resampling,
                #                                N_cv_bs=N_cv_bs,
                #                                test_percent=test_percent,
                #                                print_results=print_results)

                #     data.append({
                #         "reg_type": "ols",
                #         "degree": degree,
                #         "noise": noise,
                #         "method": "sklearn",
                #         "data": cp.deepcopy(sk_ols.data),
                #     })

    if "ridge" in regression_methods:
        print("\nRidge Regression. N ridge regressions: ", 
            len(noise_sigma_values)*len(polynom_degrees)*len(alpha_values))

        for noise in noise_sigma_values:
            z += np.random.normal(0, noise, size=z.shape)
            for deg in polynom_degrees:
                print("\n**** Polynom degree: {} ****".format(deg))
                for alpha in alpha_values:
                    print("\n**** Ridge Lambda: {:-e} ****".format(alpha))

                    if "manual" in regression_implementation:
                        ridge = \
                            reggen.ManualRidge(x, y, z, alpha, deg=deg,
                                               test_percent=test_percent,
                                               print_results=print_results)

                        data.append({
                            "reg_type": "ridge",
                            "degree": deg,
                            "noise": noise,
                            "alpha": alpha,
                            "method": "manual",
                            "data": cp.deepcopy(ridge.data),
                        })

                    # if "sklearn" in regression_implementation:
                    #     sk_ridge = \
                    #         reggen.SKLearnRidge(x, y, z, alpha, deg=deg,
                    #                             test_percent=test_percent,
                    #                             print_results=print_results)

                    #     data.append({
                    #         "reg_type": "ridge",
                    #         "degree": degree,
                    #         "noise": noise,
                    #         "alpha": alpha,
                    #         "method": "sklearn",
                    #         "data": cp.deepcopy(sk_ridge.data),
                    #     })

    if "lasso" in regression_methods:
        print("\nLasso Regression. N lasso regressions: ", 
            len(noise_sigma_values)*len(polynom_degrees)*len(alpha_values))

        for noise in noise_sigma_values:
            z += np.random.normal(0, noise, size=z.shape)
            for deg in polynom_degrees:
                print("\n**** Polynom degree: {} ****".format(deg))
                for alpha in alpha_values:
                    print("\n**** Lasso Lambda: {:-e} ****".format(alpha))

                    # if "manual" in regression_implementation:
                    #     reggen.task_c_manual(x, y, z, alpha, deg=deg,
                    #                   test_percent=test_percent)

                    if "sklearn" in regression_implementation:
                        sk_lasso = \
                            reggen.SKLearnLasso(x, y, z, alpha, deg=deg,
                                                test_percent=test_percent,
                                                print_results=print_results)

                        data.append({
                            "reg_type": "lasso",
                            "degree": deg,
                            "noise": noise,
                            "alpha": alpha,
                            "method": "sklearn",
                            "data": cp.deepcopy(sk_lasso.data),
                        })

    return data


if __name__ == '__main__':
    main()
