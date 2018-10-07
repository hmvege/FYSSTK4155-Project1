#!/usr/bin/env python3

"""
Program for testing regression and resampling on the Franke function.

Runs examples on implemented methods of bootstrapping, regression ect. and
SciKit learn.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import copy as cp

# Sometimes you might be too laze to change folders, and this is the
# solution...

import lib.metrics as metrics
import lib.bootstrap as bs
import lib.regression as reg
import lib.cross_validation as cv

import lib.scikit_resampling as sk_resampling

import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils

from matplotlib import rc, rcParams
rc("text", usetex=True)
rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
rcParams["font.family"] += ["serif"]


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


def task_a_manual(x, y, z, deg=1, N_bs=100, N_cv_bs=100, k_splits=4,
                  test_percent=0.4):
    """Manual implementation of the OLS."""

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]),
                           cp.deepcopy(z.ravel()))

    linreg = reg.OLSRegression()
    linreg.fit(X, cp.deepcopy(z.ravel()))
    z_predict = linreg.predict(X).ravel()
    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(metrics.mse(z.ravel(), z_predict)))
    print("Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(linreg.coef_var))

    # Resampling with k-fold cross validation
    print("k-fold Cross Validation")
    kfcv = cv.kFoldCrossValidation(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.OLSRegression(), poly.transform)
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.R2))
    print("MSE:   {:-20.16f}".format(kfcv.MSE))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("Beta coefs: {}".format(kfcv.coef_))
    print("Beta coefs variances: {}".format(kfcv.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.MSE, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.MSE)))

    # Resampling with mc cross validation
    print("Monte Carlo Cross Validation")
    mccv = cv.MCCrossValidation(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.OLSRegression(), poly.transform)
    mccv.cross_validate(N_cv_bs, k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("Beta coefs: {}".format(mccv.coef_))
    print("Beta coefs variances: {}".format(mccv.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    # Resampling with bootstrapping
    print("Bootstrapping")

    bs_reg = bs.BootstrapRegression(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.OLSRegression(), poly.transform)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta coefs: {}".format(bs_reg.coef_))
    print("Beta coefs variances: {}".format(bs_reg.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))

    # plot_simple_surface(x, y, z, filename="../../fig/frankie_surface")


def task_a_sk_learn(x, y, z, deg=1, N_bs=100, N_cv_bs=100,
                    k_splits=4, test_percent=0.4):
    """SK-Learn implementation of OLS."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(
        np.c_[cp.deepcopy(x).reshape(-1, 1), cp.deepcopy(y).reshape(-1, 1)])

    linreg = sk_model.LinearRegression(fit_intercept=False)
    linreg.fit(X, z.ravel())
    z_predict = linreg.predict(X)

    mse_error = metrics.mse(z.ravel(), z_predict)
    beta_error = np.diag(np.linalg.inv(X.T @ X))*mse_error

    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(mse_error))
    print(
        "Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(beta_error))

    # sk_resampling.sk_learn_bootstrap(cp.deepcopy(x), cp.deepcopy(y),
    #                                  cp.deepcopy(z), poly.transform,
    #                                  sk_model.LinearRegression,
    #                                  reg_kwargs={"fit_intercept": False},
    #                                  N_bs=N_bs, test_percent=test_percent)

    sk_resampling.sk_learn_k_fold_cv(cp.deepcopy(x), cp.deepcopy(y),
                                     cp.deepcopy(z),
                                     sk_model.LinearRegression(fit_intercept=False), 
                                     poly.transform,
                                     test_percent=test_percent,
                                     k_splits=k_splits)

    bs_reg = bs.BootstrapRegression(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        sk_model.LinearRegression(fit_intercept=False), poly.transform)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta coefs: {}".format(bs_reg.coef_))
    print("Beta coefs variances: {}".format(bs_reg.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


def task_b_manual(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                  k_splits=4, test_percent=0.4):
    """Manual implementation of Ridge Regression."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]),
                           cp.deepcopy(z.ravel()))

    linreg = reg.RidgeRegression(alpha)
    linreg.fit(X, cp.deepcopy(z.ravel()))
    z_predict = linreg.predict(X).ravel()
    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(metrics.mse(z.ravel(), z_predict)))
    print("Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(linreg.coef_var))

    # Resampling with k-fold cross validation
    print("k-fold Cross Validation")
    kfcv = cv.kFoldCrossValidation(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.RidgeRegression(alpha=alpha), poly.transform)
    kfcv.cross_validate(k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.R2))
    print("MSE:   {:-20.16f}".format(kfcv.MSE))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("Beta coefs: {}".format(kfcv.coef_))
    print("Beta coefs variances: {}".format(kfcv.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.MSE, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.MSE)))

    # Resampling with mc cross validation
    print("Monte Carlo Cross Validation")
    mccv = cv.MCCrossValidation(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.RidgeRegression(alpha=alpha), poly.transform)
    mccv.cross_validate(N_cv_bs, k_splits=k_splits,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("Beta coefs: {}".format(mccv.coef_))
    print("Beta coefs variances: {}".format(mccv.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    # Resampling with bootstrapping
    print("Bootstrapping")

    bs_reg = bs.BootstrapRegression(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        reg.RidgeRegression(alpha=alpha), poly.transform)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta coefs: {}".format(bs_reg.coef_))
    print("Beta coefs variances: {}".format(bs_reg.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


def task_b_sk_learn(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                    k_splits=4, test_percent=0.4):
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(
        cp.deepcopy(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]))

    ridge = sk_model.Ridge(alpha=alpha, solver="lsqr", fit_intercept=False)
    ridge.fit(X, z.ravel())

    # Gets the predicted y values
    z_predict = ridge.predict(X)

    # Ridge training score
    R2 = ridge.score(X, z.ravel())

    # Mean Square Error
    mse = metrics.mse(z.ravel(), z_predict)

    # Gets the beta coefs
    beta = ridge.coef_

    # Gets the beta variance
    beta_variance = metrics.ridge_regression_variance(
        X, mse, alpha)

    print("Lambda: {:-e}".format(alpha))
    print("R2:     {:-20.16f}".format(R2))
    print("MSE:    {:-20.16f}".format(mse))
    print("Bias:   {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(beta))
    print("Beta coefs variances: {}".format(beta_variance))

    reg_kwargs = {"alpha": alpha, "fit_intercept": False, "solver": "lsqr"}
    sk_resampling.sk_learn_k_fold_cv(cp.deepcopy(x), cp.deepcopy(y),
                                     cp.deepcopy(z),
                                     sk_model.Ridge(**reg_kwargs), poly.transform,
                                     test_percent=test_percent,
                                     k_splits=k_splits)

    # Resampling with bootstrapping
    print("Bootstrapping")
    bs_reg = bs.BootstrapRegression(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        sk_model.Ridge(**reg_kwargs), poly.transform)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta coefs: {}".format(bs_reg.coef_))
    print("Beta coefs variances: {}".format(bs_reg.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))



def task_c_sk_learn(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                    k_splits=4, test_percent=0.4):
    """Lasso method for scikit learn."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(
        cp.deepcopy(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]))

    ridge = sk_model.Lasso(alpha=alpha, fit_intercept=False)
    ridge.fit(X, z.ravel())

    # Gets the predicted y values
    z_predict = ridge.predict(X)

    # Ridge training score
    R2 = ridge.score(X, z.ravel())

    # Mean Square Error
    mse = metrics.mse(z.ravel(), z_predict)

    # Gets the beta coefs
    beta = ridge.coef_

    print("Lambda: {:-e}".format(alpha))
    print("R2:     {:-20.16f}".format(R2))
    print("MSE:    {:-20.16f}".format(mse))
    print("Bias:   {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(beta))

    reg_kwargs = {"alpha": alpha, "fit_intercept": False}
    sk_resampling.sk_learn_k_fold_cv(cp.deepcopy(x), cp.deepcopy(y),
                                     cp.deepcopy(z),
                                     sk_model.Lasso(**reg_kwargs), 
                                     poly.transform,
                                     test_percent=test_percent,
                                     k_splits=k_splits)

    bs_reg = bs.BootstrapRegression(
        cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
        sk_model.Lasso(**reg_kwargs), poly.transform)
    bs_reg.reg = sk_model.Lasso(alpha=alpha, fit_intercept=False)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta coefs: {}".format(bs_reg.coef_))
    print("Beta coefs variances: {}".format(bs_reg.coef_var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


if __name__ == '__main__':
    exit("Run from main.py")
