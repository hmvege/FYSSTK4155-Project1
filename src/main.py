#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

import lib.regression as libreg
import lib.metrics as metrics

# Task a)-b), Franke function testing
import tasks.franke_func_reg as ff_tasks

# Sklearn modules
# import sklearn.linear_model as sk_model
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score

# def poly5setup(x, y, N):
#     """Polynom of degree 5 of x and y."""
#     # Degree 0
#     assert len(x) == len(y)
#     deg0 = np.ones((N, 1))
#     # deg0 = np.ones((N**2, 1))

#     # Degree 1
#     deg1 = x + y

#     # Degree 2
#     _x2 = x*x
#     _y2 = y*y
#     deg2 = _x2 + 2*x*y + _y2

#     # Degree 3
#     _x3 = _x2*x
#     _y3 = _y2*y
#     _y2x = _y2*x
#     _x2y = _x2*y
#     deg3 = _x3 + 3*_x2y + 3*_y2x + _y3

#     # Degree 4
#     _x4 = _x3*x
#     _y4 = _y3*y
#     _y3x = _y3*x
#     _x3y = _x3*y
#     _x2y2 = _x2*_y2
#     deg4 = _x4 + 4*_x3y + 6*_x2y2 + 4*_y3x + _y4

#     # Degree 5
#     _x5 = _x4*x
#     _y5 = _y4*y
#     _x4y = _x4*y
#     _y4x = _y4*x
#     deg5 = _x5 + 5*_x4y + 10*_x3*_y2 + 10*_x2*_y3 + 5*_y4x + _y5

#     return np.hstack([deg0, deg1, deg2, deg3, deg4, deg5])
#     # return np.hstack([deg0.reshape((N**2, 1)), deg1.reshape((N**2, 1)),
#     #                   deg2.reshape((N**2, 1)), deg3.reshape((N**2, 1)),
#     #                   deg4.reshape((N**2, 1)), deg5.reshape((N**2, 1))])


# def plot_simple_surface(x, y, z):
#     """Surface plotter from project notes."""
#     from mpl_toolkits.mplot3d import Axes3D
#     from matplotlib import cm
#     from matplotlib.ticker import LinearLocator, FormatStrFormatter

#     fig = plt.figure()
#     ax = fig.gca(projection="3d")

#     # Plotting the surface
#     surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)

#     # Customize the z axis
#     ax.set_zlim(-01.0, 1.40)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     plt.show()


# def franke_function_example():
#     """Code as given in project description."""

#     # Make data
#     x = np.arange(0, 1, 0.05)
#     y = np.arange(0, 1, 0.05)
#     x, y = np.meshgrid(x, y)
#     z = FrankeFunction(x, y)

#     _plot_simple_surface(x, y, z)


# def part1():
#     # Generate data
#     N = 20
#     noise_sigma = 0.1
#     noise_mu = 0
#     np.random.seed(1)
#     x = np.sort(np.random.uniform(0, 1, N))
#     y = np.sort(np.random.uniform(0, 1, N))
#     # x += np.random.uniform(noise_mu, noise_sigma, N)
#     # y += np.random.uniform(noise_mu, noise_sigma, N)
#     x, y = np.meshgrid(x, y)
#     z = FrankeFunction(x, y)

#     # X = poly5setup(np.ravel(x), np.ravel(y), N)
#     # X = poly5setup(x, y, N)

    # X = poly5setup(x.reshape(-1, 1), y.reshape(-1, 1), N*N)

#     # Setting up new vectors
#     Nnew = 20
#     x_new = np.linspace(0, 1, Nnew)
#     y_new = np.linspace(0, 1, Nnew)
#     x_new, y_new = np.meshgrid(x_new, y_new)
#     x_new = x_new.reshape(-1, 1)
#     y_new = y_new.reshape(-1, 1)

#     X_new = poly5setup(x_new.reshape(-1, 1),
#                        y_new.reshape(-1, 1), Nnew*Nnew)


#     # SCIKIT-LEARN
#     linfit5 = sk_model.LinearRegression(
#         copy_X=True, fit_intercept=False, n_jobs=None, normalize=False)
#     linfit5.fit(X, z.reshape((N*N, 1)))

#     # x_sk_new = poly5.fit_transform([x_new, y_new])
#     z_sk_new = linfit5.predict(X_new)
#     R2_sk = linfit5.score(X_new, z.reshape(-1, 1))
#     MSE_sk = mean_squared_error(z.reshape(-1, 1), z_sk_new)
#     print(linfit5.coef_)
#     print("MSE = ", MSE_sk)
#     print("R2 = ", R2_sk)

#     # MANUAL
#     Nnew = 20
#     linreg = libreg.LinearRegression()
#     linreg.fit(X, z.reshape(-1, 1))
#     z_approx, beta, _, beta_variance, eps = linreg.get_results()
#     x_new = np.linspace(0, 1, Nnew)
#     y_new = np.linspace(0, 1, Nnew)
#     x_new, y_new = np.meshgrid(x_new, y_new)
#     x_new = x_new.reshape(-1, 1)
#     y_new = y_new.reshape(-1, 1)
#     X_new = poly5setup(x_new.reshape(-1, 1),
#                        y_new.reshape(-1, 1), Nnew*Nnew)
#     z_new_predict = X_new.dot(beta)

#     mse_manual = metrics.mse(z.reshape(-1, 1), z_new_predict)[0]
#     r2_manual = metrics.R2(z.reshape(-1, 1), z_new_predict)[0]
#     print("MSE: {0:f}".format(mse_manual))
#     print("R2: {0:f}".format(r2_manual))
#     for i, b in enumerate(zip(np.ravel(beta),
#                               np.sqrt(np.ravel(beta_variance)))):
#         print("Beta_{2} = {0:.6f} +/- {1:.6f}".format(*b, i))

#     # _plot_simple_surface(x, y, z)
#     _plot_simple_surface(x_new.reshape(N, N), y_new.reshape(
#         N, N), z_new_predict.reshape(N, N))


def main():
    folder_path = "../../MachineLearning/doc/Projects/2018/Project1/DataFiles/"
    abs_folder_path = os.path.abspath(folder_path)

    # print(os.listdir(abs_folder_path))
    # franke_function_example()
    ff_tasks.task_a()
    ff_tasks.task_b()
    ff_tasks.task_c()
    # plt.show()

if __name__ == '__main__':
    main()
