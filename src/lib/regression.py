import numpy as np
import scipy.linalg

class __reg_backend:
    possible_backends = ["numpy", "scipy"]
    linalg_backend = "numpy"

    def _set_backend(self, linalg_backend):
        """Sets up the linalg backend."""
        assert linalg_backend in possible_backends, \
            "{} backend not recognized".format(linalg_backend)
        self.linalg_backend = linalg_backend

    def _get_inv(self, M):
        if self.linalg_backend == "numpy":
            return np.linalg.inv(M)
        elif self.linalg_backend == "scipy":
            return scipy.linalg.inv(M)

    def _get_sum(self, V):
        if self.linalg_backend == "numpy":
            return np.sum(V)
        elif self.linalg_backend == "scipy":
            return scipy.sum(V)

    def _get_diag(self, M):
        if self.linalg_backend == "numpy":
            return np.diag(V)
        elif self.linalg_backend == "scipy":
            return scipy.diag(V)


class LinearRegression(__reg_backend):
    """
    An implementation of linear regression.

    Performs a fit on p features and s samples.
    """

    def __init__(self, linalg_backend="numpy"):
        self.X_train = None
        self.y_train = None

        # Sets up the linalg backend
        assert linalg_backend in possible_backends, \
            "{} backend not recognized".format(linalg_backend)
        self.linalg_backend = linalg_backend

    def fit(self, X_train, y_train):
        """

        Args:
            X_train (ndarray): (N x (p - 1)),
            y_train (ndarray): (N),
        """
        self.X_train = X_train
        self.y_train = y_train

        # N samples, P features
        self.N, self.P = X_train.shape

        # X^T * X
        self.XTX = self.X_train.T @ self.X_train

        # (X^T * X)^{-1}
        self.XTX_inv = self._get_inv(self.XTX)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_inv @ self.X_train.T @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.y_train

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = self._get_sum(eps**2) / float(self.N)

        # Beta fit covariance/variance. (X^T * X)^{-1} * sigma^2
        self.coef_cov = self.XTX_inv * self.y_variance
        self.coef_var = self._get_diag(self.coef_cov)

        raise NotImplementedError

    def predict(self, X_test):
        return X_test @ self.coef


class RidgeRegression(__reg_backend):
    """
    An implementation of ridge regression.
    """

    def __init__(self, X_train, y_train):
        raise NotImplementedError

    def fit():
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


class LassoRegression(__reg_backend):
    """
    An implementation of lasso regression.
    """

    def __init__(self, X_train, y_train):
        raise NotImplementedError

    def fit():
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


if __name__ == '__main__':
    exit("regression.py not intended as a standalone module.")
