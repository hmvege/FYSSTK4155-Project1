import numpy as np
import metrics


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
    _func_excact = None

    # def __init__(self, x_data, y_data, reg_method=None, 
    #     design_matrix_function=None):
    #     """
    #     Initialises an bootstrap regression object.
    #     Args:
    #     """
    #     self.x_data = x_data
    #     self.y_data = y_data

    def __init__(self, data, y=None):
        """
        Initialises an bootstrap regression object.
        Args:
        """
        self.data = data
        self.y = y

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
    def func_excact(self):
        return self._func_excact

    @func_excact.setter
    def func_excact(self, f):
        self._func_excact = f

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
        if isinstance(self._func_excact, type(None)) and \
           isinstance(self.y, type(None)):
           raise AssertionError("either provide a function "
                "to run data through, or provide the resulting data")

        assert test_percent < 1.0, "test_percent must be less than one."

        N = len(self.data)

        # Splits into k intervals
        test_size = np.floor(N * test_percent)
        k = int(N / test_size)
        test_size = int(test_size)

        x = self.data
        if not isinstance(self._func_excact, type(None)):
            y = self._func_excact(x)
        else:
            y = self.y

        # Splits into training and test set.
        x_test, x_train = np.split(x, [test_size], axis=0)
        y_test, y_train = np.split(y, [test_size], axis=0)

        # Sets up emtpy lists for gathering the relevant scores in
        R2_list = np.empty(N_bs)
        # MSE_list = np.empty(N_bs)
        # bias_list = np.empty((N_bs, test_size))
        # var_list = np.empty((N_bs, test_size))

        X_test = self._design_matrix(x_test)

        self.y_pred_list = np.empty((N_bs, test_size))
        self.y_test_list = np.empty((N_bs, test_size))

        # Bootstraps
        for i_bs in range(N_bs):
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

            # Stores the predicted variables for post calculation
            self.y_pred_list[i_bs] = y_predict.ravel()
            # self.y_test_list[i_bs] = y_test.ravel()

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        self.R2 = np.mean(R2_list)

        # Mean Square Error, mean((y - y_approx)**2)
        mse_temp = np.mean((y_test.ravel() - self.y_pred_list)**2, 
            axis=0, keepdims=True)
        self.MSE = np.mean(mse_temp)
        
        # Bias, (y - mean(y_approx))^2
        self.bias = np.mean((y_test.ravel() - np.mean(self.y_pred_list, 
            axis=0, keepdims=True))**2)

        # Variance, var(y_approx)
        self.var = np.mean(np.var(self.y_pred_list, 
            axis=0, keepdims=True))


def __test_bootstrap():
        # A small implementation of a test case
    from regression import LinearRegression

    N_bs = 200

    # Initial values
    n = 200
    noise = 0.2
    np.random.seed(1234)

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

    # Performs a bootstrap
    print("Bootstrapping")
    bs_reg = BootstrapRegression(x)
    bs_reg.reg = LinearRegression
    bs_reg.func_excact = func_excact
    bs_reg.design_matrix = design_matrix
    bs_reg.bootstrap(N_bs, test_percent=0.4)

    print("R2:    {:-20.16f}".format(bs_reg.R2))
    print("MSE:   {:-20.16f}".format(bs_reg.MSE))
    print("Bias^2:{:-20.16f}".format(bs_reg.bias))
    print("Var(y):{:-20.16f}".format(bs_reg.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(bs_reg.MSE, bs_reg.bias, bs_reg.var, 
        bs_reg.bias + bs_reg.var))
    print("Diff: {}".format(abs(bs_reg.bias + bs_reg.var - bs_reg.MSE)))

    # TODO recreate plot as shown on piazza


if __name__ == '__main__':
    __test_bootstrap()
