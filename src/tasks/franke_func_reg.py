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
import sklearn.utils as sk_utils
# import sklearn.svm as

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


def franke_func_tasks():
    # Generate data
    N = 100
    N_bs_resampling = 1000
    N_cv_bs = 100
    k_fold_size = 0.25
    test_percent = 0.4

    noise_sigma = 0.1
    noise_mu = 0
    polynom_degrees = [5]
    alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3, 1e4]

    np.random.seed(1234)

    regression_methods = []
    regression_methods += ["ols"]
    # regression_methods += ["ridge"]
    # regression_methods += ["lasso"]

    regression_implementation = []
    regression_implementation += ["sklearn"]
    # regression_implementation += ["manual"]

    # Sets up training data
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    # x += np.random.uniform(noise_mu, noise_sigma, N)
    # y += np.random.uniform(noise_mu, noise_sigma, N)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    np.save("surface_plotter/data", np.c_[x, y, z])

    if "ols" in regression_methods:
        print("\nOrdinarty Linear Regression")
        for degree in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(degree))

            if "manual" in regression_implementation:
                task_a_manual(x, y, z, deg=degree, N_bs=N_bs_resampling,
                              N_cv_bs=N_cv_bs,
                              test_percent=test_percent)

            if "sklearn" in regression_implementation:
                task_a_sk_learn(x, y, z, deg=degree, N_bs=N_bs_resampling,
                                N_cv_bs=N_cv_bs,
                                test_percent=test_percent)

    if "ridge" in regression_methods:
        print("\nRidge Regression")
        for deg in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(deg))
            for alpha in alpha_values:
                print("\n**** Ridge Lambda: {:-e} ****".format(alpha))

                if "manual" in regression_implementation:
                    task_b_manual(x, y, z, alpha, deg=deg,
                                  test_percent=test_percent)

                if "sklearn" in regression_implementation:
                    task_b_sk_learn(x, y, z, alpha, deg=deg,
                                    test_percent=test_percent)

    if "lasso" in regression_methods:
        print("\nLasso Regression")
        for deg in polynom_degrees:
            print("\n**** Polynom degree: {} ****".format(deg))
            for alpha in alpha_values:
                print("\n**** Lasso Lambda: {:-e} ****".format(alpha))

                # if "manual" in regression_implementation:
                #     task_c_manual(x, y, z, alpha, deg=deg,
                #                   test_percent=test_percent)

                if "sklearn" in regression_implementation:
                    task_c_sk_learn(x, y, z, alpha, deg=deg,
                                    test_percent=test_percent)


def task_a_manual(x, y, z, deg=1, N_bs=100, N_cv_bs=100, k_fold_size=0.2,
                  test_percent=0.4):
    """Manual implementation of the OLS."""

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

    linreg = reg.LinearRegression()
    linreg.fit(X, z.ravel())
    z_predict = linreg.predict(X).ravel()
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
    kfcv.cross_validate(k_percent=k_fold_size,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.R2))
    print("MSE:   {:-20.16f}".format(kfcv.MSE))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.MSE, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.MSE)))

    # Resampling with mc cross validation
    print("Monte Carlo Cross Validation")
    mccv = cv.MCCrossValidation(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        reg.LinearRegression, poly.transform)
    mccv.cross_validate(N_cv_bs, k_percent=k_fold_size,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    # Resampling with bootstrapping
    print("Bootstrapping")

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


def sk_learn_k_fold_cv(x, y, z, reg_method, test_percent=0.4,
                       reg_kwargs={"fit_intercept": False}):
    """Scikit Learn method for cross validation."""
    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=test_percent)
    kf = sk_modsel.KFold(n_splits=4)

    y_pred_list = []

    for train_index, test_index in kf.split(X_train):
        kX_train, kX_test = X_train[train_index], X_train[test_index]
        kY_train, kY_test = y_train[train_index], y_train[test_index]

        kf_reg = reg_method(**reg_kwargs)
        # linreg.fit(X, z.ravel())
        # z_predict = linreg.predict(X)

        # kf_reg = reg.LinearRegression()
        kf_reg.fit(kX_train, kY_train)
        y_pred_list.append(kf_reg.predict(X_test))

    y_pred_list = np.asarray(y_pred_list)

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = (y_test - y_pred_list)**2
    MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred_list, axis=0, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
    R2 = np.mean(metrics.R2(y_test, y_pred_list, axis=0))

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred_list, axis=0, keepdims=True))

    print("SciKit-Learn k-fold Cross Validation")
    print("R2:    {:-20.16f}".format(R2))
    print("MSE:   {:-20.16f}".format(MSE))
    print("Bias^2:{:-20.16f}".format(bias))
    print("Var(y):{:-20.16f}".format(var))
    print("abs(MSE - bias - var(y_approx)) = ", abs(MSE - bias - var))


def sk_learn_bootstrap(x, y, z, reg_method, N_bs=100,
                       test_percent=0.4,
                       reg_kwargs={"fit_intercept": False}):
    """Sci-kit learn bootstrap method."""

    X_train, X_test, y_train, y_test = sk_modsel.train_test_split(
        np.c_[x.ravel(), y.ravel()], z.ravel(),
        test_size=test_percent)

    # Ensures we are on axis shape (N_observations, N_predictors)
    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    y_pred = np.empty((y_test.shape[0], N_bs))

    # Initializes regression method
    kf_reg = reg_method(**reg_kwargs)

    R2_ = np.empty(N_bs)
    mse_ = np.empty(N_bs)
    bias2_ = np.empty(N_bs)

    for i_bs in range(N_bs):
        X_boot, y_boot = sk_utils.resample(X_train, y_train)

        kf_reg.fit(X_boot, y_boot)
        y_pred[:, i_bs] = kf_reg.predict(X_test).ravel()

        # print(sk_metrics.r2_score(y_test.flatten(), y_pred[:,i_bs].flatten()))
 
        R2_[i_bs] = metrics.R2(y_test.flatten(), y_pred[:,i_bs].flatten())
        mse_[i_bs] = metrics.mse(y_test.flatten(), y_pred[:,i_bs].flatten())
        bias2_[i_bs] = metrics.bias2(y_test.flatten(), y_pred[:,i_bs].flatten())


    # R2 = R2_.mean()
    # MSE = mse_.mean()
    # bias = bias2_.mean()

    R2 = (1 - np.sum((y_test - y_pred)**2, axis=1, keepdims=True) / np.sum((y_test - y_test.mean())**2, axis=0)).mean()

    # Mean Square Error, mean((y - y_approx)**2)
    _mse = ((y_test - y_pred))**2
    MSE = np.mean(np.mean(_mse, axis=1, keepdims=True))

    # Bias, (y - mean(y_approx))^2
    _mean_pred = np.mean(y_pred, axis=1, keepdims=True)
    bias = np.mean((y_test - _mean_pred)**2)

    # Variance, var(y_predictions)
    var = np.mean(np.var(y_pred, axis=1, keepdims=True))


    # # R^2 score, 1 - sum((y-y_approx)**2)/sum((y-mean(y))**2)
    # y_pred_mean = np.mean(y_pred, axis=1)
    # _y_test = y_test.reshape(-1)
    # print ("R2:", metrics.R2(_y_test, y_pred_mean))

    # _s1 = np.sum(((y_test - y_pred))**2, axis=1, keepdims=True)
    # _s2 = np.sum((y_test - np.mean(y_test))**2)
    # print (_s1.mean(), _s2)

    # R2 = 1 - _s1.mean()/_s2
    # print(np.array([sk_metrics.r2_score(y_test, y_pred[:,i]) for i in range(N_bs)]).mean())
    # R2 = metrics.R2(y_test, y_pred, axis=1)
    # R2 = np.mean(metrics.R2(y_test, y_pred, axis=1))
    # print(np.mean(metrics.R2(y_test, y_pred, axis=1)))
    # R2 = R2.mean()
    # print(R2.mean())

    print("SciKit-Learn bootstrap")
    print("R2:    {:-20.16f}".format(R2))
    print("MSE:   {:-20.16f}".format(MSE))
    print("Bias^2:{:-20.16f}".format(bias))
    print("Var(y):{:-20.16f}".format(var))
    print("abs(MSE - bias - var(y_approx)) = ", abs(MSE - bias - var))


def task_a_sk_learn(x, y, z, deg=1, N_bs=100, N_cv_bs=100,
                    k_fold_size=0.2, test_percent=0.4):
    """SK-Learn implementation of OLS."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

    linreg = sk_model.LinearRegression(fit_intercept=False)
    linreg.fit(X, z.ravel())
    z_predict = linreg.predict(X)

    mse_error = metrics.mse(z.ravel(), z_predict)
    beta_error = np.diag(np.linalg.inv(X.T @ X))*mse_error

    print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict)))
    print("MSE: {:-20.16f}".format(mse_error))
    print("Bias: {:-20.16f}".format(metrics.bias2(z.ravel(), z_predict)))
    print("Beta coefs: {}".format(linreg.coef_))
    print("Beta coefs variances: {}".format(beta_error))

    sk_learn_k_fold_cv(x, y, z, sk_model.LinearRegression,
                       reg_kwargs={"fit_intercept": False},
                       test_percent=test_percent)

    sk_learn_bootstrap(x, y, z, sk_model.LinearRegression,
                       reg_kwargs={"fit_intercept": False}, N_bs=N_bs,
                       test_percent=test_percent)


def task_b_manual(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                  k_fold_size=0.2, test_percent=0.4):
    """Manual implementation of Ridge Regression."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

    print("Manual k-fold Cross Validation")


def task_b_sk_learn(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                    k_fold_size=0.2, test_percent=0.4):
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

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

    sk_learn_k_fold_cv(x, y, z, sk_model.LinearRegression,
                       reg_kwargs={"fit_intercept": False},
                       test_percent=test_percent)


def task_c_sk_learn(x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                    k_fold_size=0.2, test_percent=0.4):
    """Lasso method for scikit learn."""
    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)])

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

    sk_learn_k_fold_cv(x, y, z, sk_model.LinearRegression,
                       reg_kwargs={"fit_intercept": False},
                       test_percent=test_percent)


if __name__ == '__main__':
    exit("Run from main.py")
