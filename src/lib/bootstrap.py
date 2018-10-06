#!/usr/bin/env python3
import numpy as np
try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics
from tqdm import tqdm


def boot(*data):
    """Strip-down version of the bootstrap method.

    Args:
        *data (ndarray): list of data arrays to resample.

    Return:
        *bs_data (ndarray): list of bootstrapped data arrays."""

    N_data = len(data)
    N = data[0].shape[0]
    assert np.all(np.array([len(d) for d in data]) == N), \
        "unequal lengths of data passed."

    index_lists = np.random.randint(N, size=N)

    return [d[index_lists] for d in data]


def bootstrap(*data, N_bs):
    bs_data = []
    for i_bs in range(N_bs):
        bs.data.append(boot(*data))
    return bs_data


def bootstrap_regression(x, y, N_bs, test_percent=0.2):
    """
    Method that splits dataset into test data and train data, 
    and performs a bootstrap on them.
    """
    assert test_percent < 1.0, "test_percent must be less than one."

    N = len(self.data)

    # Splits into k intervals
    test_size = np.floor(N * test_percent)
    k = int(N / test_size)

    x_test, x_train = np.split(x, [test_size])
    y_test, y_train = np.split(y, [test_size])

    x_boot_samples = []
    y_boot_samples = []

    for i_bs in range(N_bs):
        x_boot, y_boot = boot(x_test, y_test)

        x_boot_samples.append(x_boot)
        y_boot_samples.append(y_boot)

    return x_boot_samples, y_boot_samples


class BootstrapRegression:
    """Bootstrap class intended for use together with regression."""
    _reg = None
    _design_matrix = None

    def __init__(self, x_data, y_data, reg, design_matrix_func):
        """
        Initialises an bootstrap regression object.
        Args:
        """
        assert len(x_data) == len(y_data), "x and y data not of equal lengths"
        self.x_data = x_data
        self.y_data = y_data
        self._reg = reg()
        self._design_matrix = design_matrix_func

    @property
    def design_matrix(self):
        return self._design_matrix

    @design_matrix.setter
    def design_matrix(self, f):
        self._design_matrix = f

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, reg, **kwargs):
        self._reg = reg(**kwargs)

    @property
    def coef_(self):
        return self.coef_coefs

    @coef_.getter
    def coef_(self):
        return self.beta_coefs

    @property
    def coef_var(self):
        return self.beta_coefs_var

    @coef_var.getter
    def coef_var(self):
        return self.beta_coefs_var

    @metrics.timing_function
    def bootstrap(self, N_bs, test_percent=0.25):
        """
        Performs a bootstrap for a given regression type, design matrix 
        function and excact function.

        Args:
            N_bs (int): number of bootstraps to perform
            test_percent (float): what percentage of data to reserve for 
                testing.
        """

        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._design_matrix, type(None))

        assert test_percent < 1.0, "test_percent must be less than one."

        N = len(self.x_data)

        # Splits into k intervals
        test_size = np.floor(N * test_percent)
        k = int(N / test_size)
        test_size = int(test_size)

        x = self.x_data
        y = self.y_data

        # Splits into training and test set.
        x_test, x_train = np.split(x, [test_size], axis=0)
        y_test, y_train = np.split(y, [test_size], axis=0)

        # Sets up emtpy lists for gathering the relevant scores in
        R2_list = np.empty(N_bs)
        # MSE_list = np.empty(N_bs)
        # bias_list = np.empty(N_bs)
        # var_list = np.empty(N_bs)
        beta_coefs = []

        X_test = self._design_matrix(x_test)

        self.y_pred_list = np.empty((N_bs, test_size))

        # Bootstraps
        for i_bs in tqdm(range(N_bs), desc="Bootstrapping"):
            # Bootstraps test data
            x_boot, y_boot = boot(x_test, y_test)

            # Sets up design matrix
            X_boot = self._design_matrix(x_boot)

            # Fits the bootstrapped values
            self.reg.fit(X_boot, y_boot)

            # Tries to predict the y_test values the bootstrapped model
            y_predict = self.reg.predict(X_test)

            # Calculates R2
            R2_list[i_bs] = metrics.R2(y_predict, y_test)
            # MSE_list[i_bs] = metrics.mse(y_predict, y_test)
            # bias_list[i_bs] = metrics.bias2(y_predict, y_test)
            # var_list[i_bs] = np.var(y_predict)

            # Stores the prediction and beta coefs.
            self.y_pred_list[i_bs] = y_predict.ravel()
            beta_coefs.append(self.reg.coef_)

        # pred_list_bs = np.mean(self.y_pred_list, axis=0)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        self.R2 = np.mean(R2_list)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = np.mean((y_test.ravel() - self.y_pred_list)**2,
                       axis=0, keepdims=True)
        self.MSE = np.mean(_mse)

        # Bias, (y - mean(y_approx))^2
        _y_pred_mean = np.mean(self.y_pred_list, axis=0, keepdims=True)
        self.bias = np.mean((y_test.ravel() - _y_pred_mean)**2)

        # Variance, var(y_approx)
        self.var = np.mean(np.var(self.y_pred_list,
                                  axis=0, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)

        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        # print("R2:    ", R2_list.mean())
        # print("MSE:   ", MSE_list.mean())
        # print("bias2: ", bias_list.mean())
        # print("var:   ", var_list.mean())


def __test_bootstrap_fit():
        # A small implementation of a test case
    from regression import LinearRegression

    N_bs = 1000

    # Initial values
    n = 200
    noise = 0.2
    np.random.seed(1234)
    test_percent = 0.35

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    def design_matrix(_x):
        return np.c_[np.ones(_x.shape), _x, _x*_x]

    # Sets up design matrix
    X = design_matrix(x)

    # Performs regression
    reg = LinearRegression()
    reg.fit(X, y)
    y = y.ravel()
    y_predict = reg.predict(X).ravel()
    print("Regular linear regression")
    print("R2:  {:-20.16f}".format(reg.score(y_predict, y)))
    print("MSE: {:-20.16f}".format(metrics.mse(y, y_predict)))
    print("Beta:      ", reg.coef_.ravel())
    print("var(Beta): ", reg.coef_var.ravel())
    print("")

    # Performs a bootstrap
    print("Bootstrapping")
    bs_reg = BootstrapRegression(x, y, LinearRegression, design_matrix)
    bs_reg.bootstrap(N_bs, test_percent=test_percent)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("Beta:      ", bs_reg.coef_.ravel())
    print("var(Beta): ", bs_reg.coef_var.ravel())
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var,
                                     bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))

    # TODO recreate plot as shown on piazza


def __test_bootstrap():
    import matplotlib.pyplot as plt
    # Data to load and analyse
    data = np.random.normal(0, 2, 100)

    bs_data = np.empty((500, 100))
    # Histogram bins
    N_bins = 20

    # Bootstrapping
    N_bootstraps = int(500)
    for iboot in range(N_bootstraps):
        bs_data[iboot] = np.asarray(boot(data))

    print(data.mean(), data.std())
    bs_data = bs_data.mean(axis=0)
    print(bs_data.mean(), bs_data.std())

    plt.hist(data, label="Data")
    plt.hist(bs_data, label="Bootstrap")
    plt.show()


if __name__ == '__main__':
    __test_bootstrap_fit()
    # __test_bootstrap()
