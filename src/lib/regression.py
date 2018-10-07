#!/usr/bin/env python3
import numpy as np
import scipy.linalg
try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics

__all__ = ["OLSRegression", "RidgeRegression", "LassoRegression"]


class __RegBackend:
    """Backend class in case we want to run with either scipy, numpy 
    (or something else)."""
    _fit_performed = False
    __possible_backends = ["numpy", "scipy"]
    __possible_inverse_methods = ["inv", "svd"]

    def __init__(self, linalg_backend="scipy", inverse_method="svd"):
        """Sets up the linalg backend."""
        assert linalg_backend in self.__possible_backends, \
            "{:s} backend not recognized".format(str(linalg_backend))
        self.linalg_backend = linalg_backend

        assert inverse_method in self.__possible_inverse_methods, \
            "{:s} inverse method not recognized".format(str(inverse_method))
        self.inverse_method = inverse_method

    def fit(self, X_train, y_train):
        raise NotImplementedError("Derived class missing fit()")

    def _inv(self, M):
        """Method for taking derivatives with either numpy or scipy."""

        if self.linalg_backend == "numpy":

            if self.inverse_method == "inv":
                return np.linalg.inv(M)

            elif self.inverse_method == "svd":
                U, S, VH = np.linalg.svd(M)
                S = np.diag(1.0/S)
                return U @ S @ VH

        elif self.linalg_backend == "scipy":

            if self.inverse_method == "inv":
                return scipy.linalg.inv(M)

            elif self.inverse_method == "svd":
                U, S, VH = scipy.linalg.svd(M)
                S = np.diag(1.0/S)
                return U @ S @ VH

    def _check_if_fitted(self):
        """Small check if fit has been performed."""
        assert self._fit_performed, "Fit not performed"

    def score(self, y_true, y_test):
        """Returns the R^2 score.

        Args:
            y_test (ndarray): X array of shape (N, p - 1) to test for
            y_true (ndarray): true values for X

        Returns:
            float: R2 score for X_test values.
        """
        return metrics.R2(y_true, y_test)

    def beta_variance(self):
        """Returns the variance of beta."""
        self._check_if_fitted()
        return self.coef_var

    def get_y_variance(self):
        if hasattr(self, "y_variance"):
            return self.y_variance
        else:
            raise AttributeError(
                ("Class {:s} does not contain "
                    "y_variance.".format(self.__class__)))

    def predict(self, X_test):
        """Performs a prediction for given beta coefs.

        Args:
            X_test (ndarray): test samples, size (N, p - 1)

        Returns:
            ndarray: test values for X_test
        """
        self._check_if_fitted()
        return X_test @ self.coef

    def get_results(self):
        """Method for retrieving results from fit.

        Returns:
            y_approx (ndarray): y approximated on training data x.
            beta (ndarray):  the beta fit paramters.
            beta_cov (ndarray): covariance matrix of the beta values.
            beta_var (ndarray): variance of the beta values.
            eps (ndarray): the residues of y_train and y_approx.

        """
        return self.y_approx, self.coef, self.coef_cov, self.coef_var, self.eps

    @property
    def coef_(self):
        return self.coef

    @coef_.getter
    def coef_(self):
        return self.coef

    @coef_.setter
    def coef_(self, value):
        self.coef = value

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.setter
    def coef_var(self, value):
        self.beta_coefs_var = value


class OLSRegression(__RegBackend):
    """
    An implementation of linear regression.

    Performs a fit on p features and s samples.
    """

    def __init__(self, **kwargs):
        """Initilizer for Linear Regression

        Args:
            linalg_backend (str): optional, default is "numpy". Choices: 
                numpy, scipy.
            inverse_method (str): optional, default is "svd". Choices:
                svd, inv.
        """
        super().__init__(**kwargs)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits/trains y_train with X_train using Linear Regression.

        X_train given as [1, x, x*2, ...]

        Args:
            X_train (ndarray): design matrix, (N, p - 1), 
            y_train (ndarray): (N),
        """
        self.X_train = X_train
        self.y_train = y_train

        # N samples, P features
        self.N, self.P = X_train.shape

        # X^T * X
        self.XTX = self.X_train.T @ self.X_train

        # (X^T * X)^{-1}
        self.XTX_inv = self._inv(self.XTX)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_inv @ self.X_train.T @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.coef

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = np.sum(self.eps**2) / float(self.N)

        # Beta fit covariance/variance. (X^T * X)^{-1} * sigma^2
        self.coef_cov = self.XTX_inv * self.y_variance
        self.coef_var = np.diag(self.coef_cov)

        self._fit_performed = True


class RidgeRegression(__RegBackend):
    """
    An implementation of ridge regression.
    """

    def __init__(self, alpha=1.0, **kwargs):
        """A method for Ridge Regression.

        Args:
            alpha (float): alpha/lambda to use in Ridge Regression.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits/trains y_train with X_train using Ridge Regression.

        X_train given as [1, x, x*2, ...]

        Args:
            X_train (ndarray): design matrix, (N, p - 1), 
            y_train (ndarray): (N, 1),
        """
        self.X_train = X_train
        self.y_train = y_train

        # N samples, P features
        self.N, self.P = X_train.shape

        # X^T * X
        self.XTX = self.X_train.T @ self.X_train

        # (X^T * X)^{-1}
        self.XTX_aI = self.XTX + self.alpha*np.eye(self.XTX.shape[0])
        self.XTX_aI_inv = self._inv(self.XTX_aI)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_aI_inv @ self.X_train.T @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.coef

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = metrics.mse(self.y_train, self.y_approx)

        # Beta fit covariance/variance.
        # See page 10 section 1.4 in https://arxiv.org/pdf/1509.09169.pdf
        # **REMEMBER TO CITE THIS/DERIVE THIS YOURSELF!**
        self.coef_cov = self.XTX_aI_inv @ self.XTX @ self.XTX_aI_inv.T
        self.coef_cov *= self.y_variance
        self.coef_var = np.diag(self.coef_cov)

        self._fit_performed = True


class LassoRegression(__RegBackend):
    """
    An implementation of lasso regression.
    """

    def __init__(self, alpha=1.0, **kwargs):
        """A method for Lasso Regression.

        Args:
            alpha (float): alpha/lambda to use in Lasso Regression.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

        raise NotImplementedError

    def fit(self, X_train, y_train):
        raise NotImplementedError


def __test_ols_regression(x, y, deg):
    print("Testing OLS for  degree={}".format(deg))
    import sklearn.preprocessing as sk_preproc

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x, y)

    reg = OLSRegression()
    reg.fit(X, y)
    print("R^2: {}".format(reg.score(y, reg.predict(X))))


def __test_ridge_regression(x, y, deg, alpha=1.0):
    print("Testing Ridge for degree={} for alpha={}".format(deg, alpha))
    import sklearn.preprocessing as sk_preproc

    poly = sk_preproc.PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(x, y)

    reg = RidgeRegression(alpha=alpha)
    reg.fit(X, y)
    print("R^2: {}".format(reg.score(y, reg.predict(X))))


def __test_regresssions():
    n = 100  # n cases, i = 0,1,2,...n-1
    deg = 5
    noise_strength = 0.1
    np.random.seed(1)
    x = np.random.rand(n, 1)
    y = 5.0*x*x + np.exp(-x*x) + noise_strength*np.random.randn(n, 1)

    __test_ols_regression(x, y, deg)

    for alpha_ in [1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2]:
        __test_ridge_regression(x, y, deg, alpha_)


if __name__ == '__main__':
    __test_regresssions()
