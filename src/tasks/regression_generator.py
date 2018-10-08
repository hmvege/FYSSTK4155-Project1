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


class _dataStorer:
    data = {}

    def _fill_data_linreg(self, y_pred, r2, mse, bias, beta,
                          beta_var=None):
        self.data["regression"] = {
            "y_pred": y_pred,
            "r2": r2,
            "mse": mse,
            "bias": bias,
            "beta_coefs": beta,
            "beta_coefs_var": beta_var,
            "beta_95c": np.sqrt(coef_var)*2,
        }

    def _fill_data(self, reg, method):
        self.data[method] = {
            "y_pred": reg.y_pred,
            "y_pred_var": reg.y_pred_var,
            "mse": reg.MSE,
            "r2": reg.R2,
            "var": reg.var,
            "bias": reg.bias,
            "beta_coefs": reg.coef_,
            "beta_coefs_var": reg.coef_var,
            "beta_95c": np.sqrt(reg.coef_var)*2,
            "diff": abs(reg.bias + reg.var - reg.MSE),
        }

    def get_data(self):
        return self.data

class ManualOLS(_dataStorer):
    def __init__(self, x, y, z, deg=1, N_bs=100, N_cv_bs=100, k_splits=4,
                      test_percent=0.4, print_results=False):
        """Manual implementation of the OLS."""

        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]),
                               cp.deepcopy(z.ravel()))

        linreg = reg.OLSRegression()
        linreg.fit(X, cp.deepcopy(z.ravel()))
        z_predict_ = linreg.predict(X).ravel()
        if print_results:
            print("R2:  {:-20.16f}".format(metrics.R2(z.ravel(), z_predict_)))
            print("MSE: {:-20.16f}".format(metrics.mse(z.ravel(), z_predict_)))
            print(
                "Bias: {:-20.16f}".format(
                    metrics.bias2(z.ravel(), z_predict_)))
            print("Beta coefs: {}".format(linreg.coef_))
            print("Beta coefs variances: {}".format(linreg.coef_var))

        self.data["regression"] = {
            "y_pred": z_predict_,
            "r2": metrics.R2(z.ravel(), z_predict_),
            "mse": metrics.mse(z.ravel(), z_predict_),
            "bias": metrics.bias2(z.ravel(), z_predict_),
            "beta_coefs": linreg.coef_,
            "beta_coefs_var": linreg.coef_var,
            "beta_95c": np.sqrt(linreg.coef_var)*2,
        }

        # Resampling with k-fold cross validation
        kfcv = cv.kFoldCrossValidation(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.OLSRegression(), poly.transform)
        kfcv.cross_validate(k_splits=k_splits,
                            test_percent=test_percent)

        if print_results:
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

        self._fill_data(kfcv, "kfoldcv")

        # Resampling with mc cross validation
        mccv = cv.MCCrossValidation(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.OLSRegression(), poly.transform)
        mccv.cross_validate(N_cv_bs, k_splits=k_splits,
                            test_percent=test_percent)
        if print_results:
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

        self._fill_data(kfcv, "mccv")

        # Resampling with bootstrapping
        bs_reg = bs.BootstrapRegression(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.OLSRegression(), poly.transform)
        bs_reg.bootstrap(N_bs, test_percent=test_percent)

        if print_results:
            print("R2:    {:-20.16f}".format(bs_reg.R2))
            print("MSE:   {:-20.16f}".format(bs_reg.MSE))
            print("Bias^2:{:-20.16f}".format(bs_reg.bias))
            print("Var(y):{:-20.16f}".format(bs_reg.var))
            print("Beta coefs: {}".format(bs_reg.coef_))
            print("Beta coefs variances: {}".format(bs_reg.coef_var))
            print("MSE = Bias^2 + Var(y) = ")
            print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias,
                                             bs_reg.var,
                                             bs_reg.bias + bs_reg.var))
            print("Diff: {}".format(
                abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))

        self._fill_data(bs_reg, "bootstrap")

        # plot_simple_surface(x, y, z, filename="../../fig/frankie_surface")


# TODO: implement a loop for different noise levels
# TODO: generate plots(make them simple) of FrankeFunction fits
# TODO: generate plots of the relevant terrain data
# TODO: plot different beta values for OLS, Ridge and Lasso
# TODO: plot bias + var + mse(see piazza)
# TODO: make a list over the different fits
# TODO: complete theory


class SKLearnOLS(_dataStorer):
    def __init__(self, x, y, z, deg=1, N_bs=100, N_cv_bs=100,
                 k_splits=4, test_percent=0.4, print_results=False):
        """SK-Learn implementation of OLS."""
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(
            np.c_[cp.deepcopy(x).reshape(-1, 1),
                  cp.deepcopy(y).reshape(-1, 1)])

        linreg = sk_model.LinearRegression(fit_intercept=False)
        linreg.fit(X, z.ravel())
        z_predict_ = linreg.predict(X)
        r2 = metrics.R2(z.ravel(), z_predict_)
        bias = metrics.bias2(z.ravel(), z_predict_)
        mse_error = metrics.mse(z.ravel(), z_predict_)

        N, P = X.shape
        z_variance = np.sum((z.ravel() - z_predict_)**2) / (N - P - 1)

        linreg_coef_var = np.diag(np.linalg.inv(X.T @ X))*z_variance
        self.data["regression"] = {
            "y_pred": z_predict_,
            "r2": r2,
            "mse": mse_error,
            "bias": bias,
            "beta_coefs": linreg.coef_,
            "beta_coefs_var": linreg_coef_var,
            "beta_95c": np.sqrt(linreg_coef_var)*2,
        }

        # Resampling coefs
        if print_results:
            print("R2:  {:-20.16f}".format(r2))
            print("MSE: {:-20.16f}".format(mse_error))
            print(
                "Bias: {:-20.16f}".format(bias))
            print("Beta coefs: {}".format(linreg.coef_))
            print("Beta coefs variances: {}".format(linreg_coef_var))

        sk_kfold_res = sk_resampling.sk_learn_k_fold_cv(
            cp.deepcopy(x), cp.deepcopy(y),
            cp.deepcopy(z),
            sk_model.LinearRegression(
                fit_intercept=False),
            poly.transform,
            test_percent=test_percent,
            k_splits=k_splits,
            print_results=print_results)

        self.data["kfoldcv"] = sk_kfold_res

        bs_reg = bs.BootstrapRegression(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            sk_model.LinearRegression(fit_intercept=False), poly.transform)
        bs_reg.bootstrap(N_bs, test_percent=test_percent)

        self._fill_data(bs_reg, "bootstrap")

        if print_results:
            print("R2:    {:-20.16f}".format(bs_reg.R2))
            print("MSE:   {:-20.16f}".format(bs_reg.MSE))
            print("Bias^2:{:-20.16f}".format(bs_reg.bias))
            print("Var(y):{:-20.16f}".format(bs_reg.var))
            print("Beta coefs: {}".format(bs_reg.coef_))
            print("Beta coefs variances: {}".format(bs_reg.coef_var))
            print("MSE = Bias^2 + Var(y) = ")
            print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias,
                                             bs_reg.var,
                                             bs_reg.bias + bs_reg.var))
            print("Diff: {}".format(
                abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


class ManualRidge(_dataStorer):
    def __init__(self, x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                 k_splits=4, test_percent=0.4, print_results=False):
        """Manual implementation of Ridge Regression."""
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]),
                               cp.deepcopy(z.ravel()))

        self.data["alpha"] = alpha

        linreg = reg.RidgeRegression(alpha)
        linreg.fit(X, cp.deepcopy(z.ravel()))
        z_predict_ = linreg.predict(X).ravel()
        mse_error = metrics.mse(z.ravel(), z_predict_)
        linreg_coef_var = np.diag(np.linalg.inv(X.T @ X))*mse_error
        r2 = metrics.R2(z.ravel(), z_predict_)
        bias = metrics.bias2(z.ravel(), z_predict_)
        self.data["regression"] = {
            "y_pred": z_predict_,
            "r2": r2,
            "mse": mse_error,
            "bias": bias,
            "beta_coefs": linreg.coef_,
            "beta_coefs_var": linreg_coef_var,
            "beta_95c": np.sqrt(linreg_coef_var)*2,
        }
        if print_results:
            print("R2:  {:-20.16f}".format(r2))
            print("MSE: {:-20.16f}".format(mse_error))
            print("Bias: {:-20.16f}".format(bias))
            print("Beta coefs: {}".format(linreg.coef_))
            print("Beta coefs variances: {}".format(linreg.coef_var))

        # Resampling with k-fold cross validation
        kfcv = cv.kFoldCrossValidation(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.RidgeRegression(alpha=alpha), poly.transform)
        kfcv.cross_validate(k_splits=k_splits,
                            test_percent=test_percent)
        self._fill_data(kfcv, "kfoldcv")

        if print_results:
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
        mccv = cv.MCCrossValidation(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.RidgeRegression(alpha=alpha), poly.transform)
        mccv.cross_validate(N_cv_bs, k_splits=k_splits,
                            test_percent=test_percent)
        self._fill_data(mccv, "mccv")

        if print_results:
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
        bs_reg = bs.BootstrapRegression(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            reg.RidgeRegression(alpha=alpha), poly.transform)
        bs_reg.bootstrap(N_bs, test_percent=test_percent)
        self._fill_data(bs_reg, "bootstrap")

        if print_results:
            print("R2:    {:-20.16f}".format(bs_reg.R2))
            print("MSE:   {:-20.16f}".format(bs_reg.MSE))
            print("Bias^2:{:-20.16f}".format(bs_reg.bias))
            print("Var(y):{:-20.16f}".format(bs_reg.var))
            print("Beta coefs: {}".format(bs_reg.coef_))
            print("Beta coefs variances: {}".format(bs_reg.coef_var))
            print("MSE = Bias^2 + Var(y) = ")
            print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias,
                                             bs_reg.var,
                                             bs_reg.bias + bs_reg.var))
            print("Diff: {}".format(
                abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


class SKLearnRidge(_dataStorer):
    def __init__(self, x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                 k_splits=4, test_percent=0.4, print_results=False):
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(
            cp.deepcopy(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]))

        ridge = sk_model.Ridge(alpha=alpha, solver="lsqr", fit_intercept=False)
        ridge.fit(X, z.ravel())

        # Gets the predicted y values
        z_predict = ridge.predict(X)

        R2 = ridge.score(X, z.ravel())
        mse = metrics.mse(z.ravel(), z_predict)
        bias = metrics.bias2(z.ravel(), z_predict)

        N, P = X.shape
        z_variance = np.sum((z.ravel() - z_predict)**2) / (N - P - 1)

        # Gets the beta variance
        beta_variance = metrics.ridge_regression_variance(
            X, z_variance, alpha)

        self.data["alpha"] = alpha
        self.data["regression"] = {
            "y_pred": z_predict,
            "r2": R2,
            "mse": mse,
            "bias": bias,
            "beta_coefs": ridge.coef_,
            "beta_coefs_var": beta_variance,
            "beta_95c": np.sqrt(beta_variance)*2,
        }

        if print_results:
            print("Lambda: {:-e}".format(alpha))
            print("R2:     {:-20.16f}".format(R2))
            print("MSE:    {:-20.16f}".format(mse))
            print("Bias:   {:-20.16f}".format(bias))
            print("Beta coefs: {}".format(ridge.coef_))
            print("Beta coefs variances: {}".format(beta_variance))

        reg_kwargs = {"alpha": alpha, "fit_intercept": False, "solver": "lsqr"}
        kfcf_results = sk_resampling.sk_learn_k_fold_cv(
            cp.deepcopy(x),
            cp.deepcopy(y),
            cp.deepcopy(z),
            sk_model.Ridge(
                **reg_kwargs),
            poly.transform,
            test_percent=test_percent,
            k_splits=k_splits,
            print_results=print_results)
        self.data["kfoldcv"] = kfcf_results

        # Resampling with bootstrapping
        bs_reg = bs.BootstrapRegression(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]),
            cp.deepcopy(z.ravel()),
            sk_model.Ridge(**reg_kwargs), poly.transform)
        bs_reg.bootstrap(N_bs, test_percent=test_percent)

        self._fill_data(bs_reg, "bootstrap")

        if print_results:
            print("R2:    {:-20.16f}".format(bs_reg.R2))
            print("MSE:   {:-20.16f}".format(bs_reg.MSE))
            print("Bias^2:{:-20.16f}".format(bs_reg.bias))
            print("Var(y):{:-20.16f}".format(bs_reg.var))
            print("Beta coefs: {}".format(bs_reg.coef_))
            print("Beta coefs variances: {}".format(bs_reg.coef_var))
            print("MSE = Bias^2 + Var(y) = ")
            print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias,
                                             bs_reg.var,
                                             bs_reg.bias + bs_reg.var))
            print("Diff: {}".format(
                abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


class SKLearnLasso(_dataStorer):
    def __init__(self, x, y, z, alpha, deg=5, N_bs=100, N_cv_bs=100,
                 k_splits=4, test_percent=0.4, print_results=False):
        """Lasso method for scikit learn."""
        poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
        X = poly.fit_transform(
            cp.deepcopy(np.c_[x.reshape(-1, 1), y.reshape(-1, 1)]))

        ridge = sk_model.Lasso(alpha=alpha, fit_intercept=False)
        ridge.fit(X, z.ravel())

        # Gets the predicted y values
        z_predict = ridge.predict(X)

        bias = metrics.bias2(z.ravel(), z_predict)
        R2 = ridge.score(X, z.ravel())
        mse = metrics.mse(z.ravel(), z_predict)

        # Gets the beta coefs
        beta = ridge.coef_

        self.data["alpha"] = alpha
        self.data["regression"] = {
            "y_pred": z_predict,
            "r2": R2,
            "mse": mse,
            "bias": bias,
            "beta_coefs": ridge.coef_,
            "beta_coefs_var": None,
        }

        if print_results:
            print("Lambda: {:-e}".format(alpha))
            print("R2:     {:-20.16f}".format(R2))
            print("MSE:    {:-20.16f}".format(mse))
            print("Bias:   {:-20.16f}".format(bias))
            print("Beta coefs: {}".format(beta))

        reg_kwargs = {"alpha": alpha, "fit_intercept": False}
        sk_results = sk_resampling.sk_learn_k_fold_cv(
            cp.deepcopy(x),
            cp.deepcopy(y),
            cp.deepcopy(z),
            sk_model.Lasso(
                **reg_kwargs),
            poly.transform,
            test_percent=test_percent,
            k_splits=k_splits,
            print_results=print_results)
        self.data["kfoldcv"] = sk_results

        bs_reg = bs.BootstrapRegression(
            cp.deepcopy(np.c_[x.ravel(), y.ravel()]), cp.deepcopy(z.ravel()),
            sk_model.Lasso(**reg_kwargs), poly.transform)
        bs_reg.reg = sk_model.Lasso(alpha=alpha, fit_intercept=False)
        bs_reg.bootstrap(N_bs, test_percent=test_percent)

        self._fill_data(bs_reg, "bootstrap")

        if print_results:
            print("R2:    {:-20.16f}".format(bs_reg.R2))
            print("MSE:   {:-20.16f}".format(bs_reg.MSE))
            print("Bias^2:{:-20.16f}".format(bs_reg.bias))
            print("Var(y):{:-20.16f}".format(bs_reg.var))
            print("Beta coefs: {}".format(bs_reg.coef_))
            print("Beta coefs variances: {}".format(bs_reg.coef_var))
            print("MSE = Bias^2 + Var(y) = ")
            print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias,
                                             bs_reg.var,
                                             bs_reg.bias + bs_reg.var))
            print("Diff: {}".format(
                abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))


if __name__ == '__main__':
    exit("Run from main.py")
