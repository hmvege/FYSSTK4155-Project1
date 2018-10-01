import numpy as np
import scipy.linalg

# help(super)
class __reg_backend:
    __fit_performed = False
    __possible_backends = ["numpy", "scipy"]

    def __init__(self, linalg_backend):
        """Sets up the linalg backend."""
        assert linalg_backend in self.__possible_backends, \
            "{} backend not recognized".format(linalg_backend)
        self.linalg_backend = linalg_backend

    def __get_inv(self, M):
        if self.linalg_backend == "numpy":
            return np.linalg.inv(M)
        elif self.linalg_backend == "scipy":
            return scipy.linalg.inv(M)

    def __get_sum(self, V):
        if self.linalg_backend == "numpy":
            return np.sum(V)
        elif self.linalg_backend == "scipy":
            return scipy.sum(V)

    def __get_diag(self, M):
        if self.linalg_backend == "numpy":
            return np.diag(V)
        elif self.linalg_backend == "scipy":
            return scipy.diag(V)

    def __check_if_fitted(self):
        """Small check if fit has been performed."""
        assert self.__fit_performed, "Fit not performed"


class LinearRegression(__reg_backend):
    """
    An implementation of linear regression.

    Performs a fit on p features and s samples.
    """

    def __init__(self, linalg_backend="numpy"):
        super().__init__(linalg_backend)
        self.X_train = None
        self.y_train = None

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
        self.XTX_inv = self.__get_inv(self.XTX)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_inv @ self.X_train.T @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.y_train

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = self.__get_sum(eps**2) / float(self.N)

        # Beta fit covariance/variance. (X^T * X)^{-1} * sigma^2
        self.coef_cov = self.XTX_inv * self.y_variance
        self.coef_var = self.__get_diag(self.coef_cov)

        self.__fit_performed = True

    def predict(self, X_test):
        """Performs a prediction for given beta coefs.

        Args:
            X_test (ndarray): 

        Returns:
            y (ndarray):
        """
        self.__check_if_fitted()
        return X_test @ self.coef


class RidgeRegression(__reg_backend):
    """
    An implementation of ridge regression.
    """

    def __init__(self):
        super().__init__(linalg_backend)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, X_train):
        self.__fit_performed = True
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


class LassoRegression(__reg_backend):
    """
    An implementation of lasso regression.
    """

    def __init__(self):
        super().__init__(linalg_backend)
        self.X_train = None
        self.y_train = None
        raise NotImplementedError

    def fit(self, X_train, X_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError


if __name__ == '__main__':
    Test = LinearRegression()
    exit("regression.py not intended as a standalone module.")
