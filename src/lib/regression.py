import numpy as np
import scipy.linalg
import metrics

# help(super)


class __RegBackend:
    """Backend class in case we want to run with either scipy, numpy 
    (or something else)."""
    __fit_performed = False
    __possible_backends = ["numpy", "scipy"]
    __possible_inverse_methods = ["inv", "svd"]

    def __init__(self, linalg_backend="numpy", inverse_method="svd"):
        """Sets up the linalg backend."""
        assert linalg_backend in self.__possible_backends, \
            "{:s} backend not recognized".format(str(linalg_backend))
        self.linalg_backend = linalg_backend

        assert inverse_method in self.__possible_inverse_methods, \
            "{:s} inverse method not recognized".format(str(inverse_method))
        self.inverse_method = inverse_method

    def __inv(self, M):
        """Method for taking derivatives with either numpy or scipy."""

        if self.linalg_backend == "numpy":

            if self.inverse_method == "inv":
                return np.linalg.inv(M)

            elif self.inverse_method == "svd":
                S, V, D = np.linalg.svd(M)
                D = 1.0/np.diag(D)
                return V @ (D * U.T)

        elif self.linalg_backend == "scipy":

            if self.inverse_method == "inv":
                return scipy.linalg.inv(M)

            elif self.inverse_method == "svd":
                S, V, D = scipy.linalg.svd(M)
                D = 1.0/np.diag(D)
                return V @ (D * U.T)

    def __check_if_fitted(self, f):
        """Small check if fit has been performed."""
        assert self.__fit_performed, "Fit not performed"

    def score(self, X_test, y_true):
        """Returns the R^2 score.

        Args:
            X_test (ndarray): X array of shape (N, p - 1) to test for
            y_true (ndarray): true values for X

        Returns:
            float: R2 score for X_test values.
        """
        y_test = X_test @ self.coef
        return metrics.R2(y_test, y_true)

    def beta_variance(self):
        """Returns the variance of beta."""
        self.__check_if_fitted()
        return self.coef_var

    def predict(self, X_test):
        """Performs a prediction for given beta coefs.

        Args:
            X_test (ndarray): test samples, size (N, p - 1)

        Returns:
            ndarray: test values for X_test
        """
        self.__check_if_fitted()
        return X_test @ self.coef


class LinearRegression(__RegBackend):
    """
    An implementation of linear regression.

    Performs a fit on p features and s samples.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits/trains y_train with X_train using Linear Regression.

        X_train given as [1, x, x*2, ...]

        Args:
            X_train (ndarray): (N, p - 1), 
            y_train (ndarray): (N),
        """
        self.X_train = X_train
        self.y_train = y_train

        # N samples, P features
        self.N, self.P = X_train.shape

        # X^T * X
        self.XTX = self.X_train.T @ self.X_train

        # (X^T * X)^{-1}
        self.XTX_inv = self.__inv(self.XTX)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_inv @ self.X_train.T @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.coef

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = np.sum(eps**2) / float(self.N)

        # Beta fit covariance/variance. (X^T * X)^{-1} * sigma^2
        self.coef_cov = self.XTX_inv * self.y_variance
        self.coef_var = np.diag(self.coef_cov)

        self.__fit_performed = True


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
            X_train (ndarray): (N, p - 1), 
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
        self.XTX_aI_inv = self.__inv(self.XTX_aI)

        # Beta fit values: beta = (X^T * X)^{-1} @ X^T @ y
        self.coef = self.XTX_aI_inv @ self.X_train @ self.y_train

        # y approximate. X @ beta
        self.y_approx = self.X_train @ self.coef

        # Residues.
        self.eps = self.y_train - self.y_approx

        # Variance of y approximate values. sigma^2
        self.y_variance = metrics.mse(self.y, self.y_approx)

        # Beta fit covariance/variance.
        # See page 10 section 1.4 in https://arxiv.org/pdf/1509.09169.pdf
        # **REMEMBER TO CITE THIS/DERIVE THIS YOURSELF!**
        self.coef_cov = self.y_variance
        self.coef_cov *= self.XTX_aI_inv @ self.XTX @ self.XTX_aI_inv.T
        self.coef_var = self.__diag(self.coef_cov)

        self.__fit_performed = True


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

    def fit(self, X_train, y_train):
        raise NotImplementedError


if __name__ == '__main__':
    TestLinear = LinearRegression()
    TestRidge = RidgeRegression()
    TestLasso = LassoRegression()

    exit("regression.py not intended as a standalone module.")
