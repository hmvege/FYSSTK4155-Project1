#!/usr/bin/env python3
"""
Program for testing regression and resampling on the Franke function.

Runs examples on implemented methods of bootstrapping, regression ect. and
SciKit learn.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Sometimes you might be too laze to change folders, and this is the
# solution...

import lib.metrics as metrics
import lib.bootstrap as bs
import lib.regression as reg
import lib.cross_validation as cv

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
# import sklearn.svm as

# from matplotlib import rc, rcParams
# rc("text", usetex=True)
# rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
# rcParams["font.family"] += ["serif"]


def poly5setup(x, y, N):
    """Polynom of degree 5 of x and y."""
    # Degree 0
    assert len(x) == len(y)
    deg0 = np.ones((N, 1))
    # deg0 = np.ones((N**2, 1))

    # Degree 1
    deg1 = x + y

    # Degree 2
    _x2 = x*x
    _y2 = y*y
    deg2 = _x2 + 2*x*y + _y2
    # deg2 = _x2 + x*y + _y2

    # Degree 3
    _x3 = _x2*x
    _y3 = _y2*y
    _y2x = _y2*x
    _x2y = _x2*y
    deg3 = _x3 + 3*_x2y + 3*_y2x + _y3
    # deg3 = _x3 + _x2y + _y2x + _y3

    # Degree 4
    _x4 = _x3*x
    _y4 = _y3*y
    _y3x = _y3*x
    _x3y = _x3*y
    deg4 = _x4 + 4*_x3y + 6*_x2*_y2 + 4*_y3x + _y4
    # deg4 = _x4 + _x3y + _x2y2 + _y3x + _y4

    # Degree 5
    _x5 = _x4*x
    _y5 = _y4*y
    _x4y = _x4*y
    _y4x = _y4*x
    deg5 = _x5 + 5*_x4y + 10*_x3*_y2 + 10*_x2*_y3 + 5*_y4x + _y5
    # deg5 = _x5 + _x4y + _x3*_y2 + _x2*_y3 + _y4x + _y5

    # return np.hstack([deg0, x, y])
    # return np.hstack([deg0, x, y, _x2, x*y, _y2])
    return np.hstack([deg0, x, y, _x2, x*y, _y2, _x3, _x2y,
                      _y2x, _y3, _x4, _x3y, _x2*_y2, _y3x, _y4, _x5, _x4y,
                      _x3*_y2, _x2*_y3, _y4x, _y5])
    # return np.hstack([deg0, deg1, deg2, deg3, deg4, deg5])


def plot_simple_surface(x, y, z, filename="../../fig/simple_surface"):
    """Surface plotter from project notes."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plotting the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
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
    print(figpath)
    plt.savefig(figpath)
    plt.show()


def franke_function_example():
    """Plots a simple examply surface."""
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    _plot_simple_surface(x, y, z)


def FrankeFunction(x, y):
    """As retrieved from project description."""
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def task_a():
    # Generate data
    N = 10
    N_bs_resampling = 100
    N_cv_bs = 100
    test_percent = 0.4
    k_fold_size = 0.25
    holdout_percent = 0.2

    noise_sigma = 0.1
    noise_mu = 0
    polynom_degrees = [5]

    np.random.seed(1234)

    # Sets up training data
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    # x += np.random.uniform(noise_mu, noise_sigma, N)
    # y += np.random.uniform(noise_mu, noise_sigma, N)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)


    np.save("surface_plotter/data", np.c_[x,y,z])

    for degree in polynom_degrees:
        task_a_manual(x, y, z, deg=degree, N_bs=N_bs_resampling,
                      N_cv_bs=N_cv_bs, test_percent=test_percent)
    # X = poly5setup(np.ravel(x), np.ravel(y), N)
    # X = poly5setup(x, y, N)

    # X = poly5setup(x.reshape(-1, 1), y.reshape(-1, 1), N*N)


def task_a_manual(x, y, z, deg=1, N_bs=100, N_cv_bs=100, test_percent=0.4,
                  k_fold_size=0.2, holdout_percent=0.4):
    """Manual implementation of the OLS."""

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

    linreg = reg.LinearRegression()
    linreg.fit(X, z.ravel())
    z_predict = linreg.predict(X).ravel()
    print("Regular linear regression")
    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(metrics.mse(z.ravel(), z_predict)))
    print("Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef))
    print("Beta coefs variances: {}".format(linreg.coef_var))

    # Resampling with k-fold cross validation
    print("k-fold Cross Validation")
    kfcv = cv.kFoldCrossValidation(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        reg.LinearRegression, poly.transform)
    kfcv.cross_validate(k_percent=k_fold_size, holdout_percent=holdout_percent)
    print("R2:    {:-20.16f}".format(kfcv.R2))
    print("MSE:   {:-20.16f}".format(kfcv.MSE))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.MSE, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.MSE)))


    # SK Learn
    print("SciKit-Learn k-fold Cross Validation")
    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=holdout_percent, random_state=0)
    kf = sk_modsel.KFold(n_splits=4)

    y_pred_list = []

    for train_index, test_index in kf.split(X_train):
        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]
        kf_linreg = reg.LinearRegression()
        kf_linreg.fit(kX_train, kY_train)
        y_pred_list.append(kf_linreg.predict(X_test))

    y_pred_list = np.asarray(y_pred_list)

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = (y_test - y_pred_list)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred_list, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    R2 = np.mean(metrics.R2(y_pred_list, y_test, axis=1))

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred_list, axis=0, keepdims=True))

    print("R2:    {:-20.16f}".format(R2))
    print("MSE:   {:-20.16f}".format(MSE))
    print("Bias^2:{:-20.16f}".format(bias))
    print("Var(y):{:-20.16f}".format(var))
    print(abs(MSE - bias - var))



    exit(1)
    # Resampling with kk-fold cross validation
    print("kk Cross Validation")
    kkcv = cv.kkFoldCrossValidation(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        reg.LinearRegression, poly.transform)
    kkcv.cross_validate(k_percent=k_fold_size,
                        holdout_percent=holdout_percent)
    print("R2:    {:-20.16f}".format(kkcv.R2))
    print("MSE:   {:-20.16f}".format(kkcv.MSE))
    print("Bias^2:{:-20.16f}".format(kkcv.bias))
    print("Var(y):{:-20.16f}".format(kkcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kkcv.MSE, kkcv.bias, kkcv.var,
                                     kkcv.bias + kkcv.var))
    print("Diff: {}".format(abs(kkcv.bias + kkcv.var - kkcv.MSE)))

    # Resampling with mc cross validation
    print("Monte Carlo Cross Validation")
    mccv = cv.MCCrossValidation(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        reg.LinearRegression, poly.transform)
    mccv.cross_validate(N_cv_bs, k_percent=k_fold_size,
                        holdout_percent=holdout_percent)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    # Resampling with bootstrapping
    print("\nBootstrapping")

    bs_reg = bs.BootstrapRegression(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        reg.LinearRegression, poly.transform)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))

    # plot_simple_surface(x, y, z, filename="../../fig/frankie_surface")


def task_a_sk_learn(x, y, z, deg=5):
    """SK-Learn implementation of OLS."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

    linreg = sk_model.LinearRegression()
    linreg.fit(X, z.ravel())
    z_predict = linreg.predict(X)

    mse_error = metrics.mse(z.ravel(), z_predict)
    beta_error = np.diag(np.linalg.inv(X.T @ X))*mse_error

    print("Scikit-Learn linear regression")
    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(mse_error))
    print("Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(beta_error))


def task_b():
    pass


def task_b_manual(deg=5):
    """Manual implementation of Ridge Regression."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform([x, y])


def task_b_sk_learn(deg=5):
    pass


def task_c():
    pass


def task_c_sk_learn(deg=5):
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    _X_transformed = poly.fit_transform(x)
    pass


if __name__ == '__main__':
    task_a()
