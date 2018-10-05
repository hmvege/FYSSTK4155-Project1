import numpy as np
import metrics

__all__ = ["kFoldCrossValidation", "MCCrossValidation"]


class __CV_core:
    """Core class for performing k-fold cross validation."""
    _reg = None
    _design_matrix = None
    _func_excact = None

    def __init__(self, data, y=None):
        self.data = data
        self.y = y

    def _get_split_percent(self, split_percentage, N,
        enforce_equal_intervals=True):
        """Parent method for getting interval split size.

        Args:
            split_percentage (float): percentage we split interval into. Goes
                from 0 to 1.
            N (int): total dataset size.
            enforce_equal_intervals (bool): if true, will add remainder 
                of N % test_size to test_size. Default is True.

        Returns:
            k_splits (int): number of split intervals
            test_size (int): size of test size(Size of 1 interval)
        """

        assert split_percentage < 1.0, "k_percent must be less than one."

        # Splits into k intervals
        test_size = np.floor(N * split_percentage)
        k_splits = int(N / test_size)
        test_size = int(test_size)

        if enforce_equal_intervals:
            if N % test_size != 0:
                raise ValueError("bad k_percent: N % k = {} != 0".format(
                    N % test_size))
        else:
            test_size += N % test_size

        return k_splits, test_size

    def _check_has_y_data_or_function(self):
        """Test for checking if y data is provided or not."""
        if isinstance(self._func_excact, type(None)) and \
           isinstance(self.y, type(None)):
           raise AssertionError("either provide a function "
                "to run data through, or provide the resulting data")

    def _check_provided_elements(self):
        """Checks we have a regression method, design matrix function and 
        either y data or y function data."""
        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._design_matrix, type(None))
        self._check_has_y_data_or_function()

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
        """Args:
            rmethod (regression class): regression class to use
        """
        self._reg = reg(**kwargs)

    @property
    def func_excact(self):
        return self._func_excact

    @func_excact.setter
    def func_excact(self, f):
        self._func_excact = f


class kFoldCrossValidation(__CV_core):
    """Class for performing k-fold cross validation."""

    def cross_validate(self, k_percent=0.2, holdout_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        self._check_has_y_data_or_function()

        # assert k_percent < 1.0, "k_percent must be less than one."

        N_total_size = len(self.data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        k_holdout, holdout_test_size = self._get_split_percent(
            holdout_percent, N_total_size, enforce_equal_intervals=False)
        
        # Splits X data and design matrix data
        x_holdout_test, x_kfold_train = np.split(self.data, 
            [holdout_test_size], axis=0)
        y_holdout_test, y_kfold_train = np.split(self._func_excact(self.data),
            [holdout_test_size], axis=0)
        
        N_kfold_data = len(x_kfold_train)

        X_holdout_test = self._design_matrix(x_holdout_test)



        print("len(X_holdout_test): {} len(x_kfold_train): {}".format(x_holdout_test.shape, x_kfold_train.shape))

        # Splits dataset into managable k fold tests
        k_splits, test_size = self._get_split_percent(k_percent, N_kfold_data)
        
        x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        print (self._func_excact(x_subdata)[0])
        print (y_subdata[0])

        print("len(subdata): {} subdata shape: {}".format(len(x_subdata), x_subdata[0].shape))

        # TODO: set off a part of the data to perform CV on

        # Stores the test values from each k trained data set in an array
        # R2_list = np.empty(k_splits)
        # MSE_list = np.empty(k_splits)
        # bias_list = np.empty((k_splits, holdout_test_size))
        # var_list = np.empty(k_splits)

        self.y_pred_list = np.empty((k_splits, holdout_test_size))
        self.y_test_list = np.empty((k_splits, holdout_test_size))
        # self.coef_list = []
        # coef_var_list = []

        for ik in range(k_splits):
            # Sets 
            k_x_test = x_subdata[ik]
            k_y_test = y_subdata[ik]

            k_y_test1 = self._func_excact(k_x_test)
            print(k_y_test[:5])
            print(k_y_test1[:5])
            assert k_y_test[0] == k_y_test1[0], "data not equal"
            exit(1)
            X_test = self._design_matrix(k_data)

            # Sets up indexes
            set_list = list(range(k_splits))
            set_list.pop(ik)

            # Sets up new data set
            data_train = np.concatenate([subdata[d] for d in set_list])

            # Gets the training data
            y_train = self._func_excact(data_train)

            # Sets up function to predict
            X_train = self._design_matrix(data_train)

            # Trains method bu fitting data
            self.reg.fit(X_train, y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_holdout_test).ravel()

            # # Appends R2, MSE, coef scores to list
            # R2_list[ik] = metrics.R2(y_predict, y_test)
            # MSE_list[ik] = metrics.mse(y_predict, y_test)
            # # MSE_list[ik] = np.mean((y_test - y_predict)**2, keepdims=True)
            # bias_list[ik] = metrics.bias2(y_predict, y_test)
            # # bias_list[ik] = (y_test - np.mean(y_predict**2, keepdims=True, axis=1))
            # var_list[ik] = np.var(y_predict)

            self.coef_list.append(self.reg.coef)
            self.y_pred_list[ik] = y_predict
            self.y_test_list[ik] = y_test

            # print(
            #     "R^2 = {:-20.16f}".format(R2_list[ik]),
            #     " MSE = {:-20.16f}".format(MSE_list[ik]),
            #     " Bias^2 = {:-20.16f}".format(bias_list[ik]),
            #     " Var^2 = {:-20.16f}".format(var_list[ik]))

        print("y_pred shape: ", self.y_pred_list.shape)
        print("y_test shape: ", self.y_test_list.shape)

        error_temp = (self.y_test_list - self.y_pred_list)**2
        error = np.mean(np.mean(error_temp, axis=0, keepdims=True))

        bias_temp = self.y_test_list 
        bias_temp -= np.mean(self.y_pred_list, axis=0, keepdims=True)
        bias = np.mean(bias_temp**2)

        variance = np.mean(np.var(self.y_pred_list, axis=0, keepdims=True))

        print(error, bias, variance, bias+variance, abs(bias+variance-error))

        # self.mean_bias = metrics.bias2(axis=0)
        # exit("ok")

        self.mean_MSE = np.mean(
            np.mean((self.y_pred_list - self.y_test_list)**2, keepdims=True, axis=1))
        self.mean_R2 = np.mean(metrics.R2(
            self.y_test_list, self.y_pred_list, axis=1))
        self.mean_bias = np.mean(np.mean(
            (self.y_pred_list - np.mean(self.y_test_list, axis=1, keepdims=True))**2))


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

        # self.mean_MSE = np.mean(MSE_list)
        # self.mean_bias = np.mean(bias_list)
        self.mean_var = np.mean(var_list)


class MCCrossValidation(__CV_core):
    pass


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""
    pass


def __test_k_fold_cross_validation():
    # A small implementation of a test case
    from regression import LinearRegression
    import matplotlib.pyplot as plt

    # Initial values
    n = 1000
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return -2*_x*_x + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    def design_matrix(_x):
        return np.c_[np.ones(_x.shape), _x, _x*_x, _x*_x*_x]

    # Sets up design matrix
    X = design_matrix(x)

    # Performs regression
    reg = LinearRegression()
    reg.fit(X, y)
    y = y.ravel()
    y_predict = reg.predict(X).ravel()
    print("Regular linear regression")
    print("R2:    {:-20.16f}".format(reg.score(y_predict, y)))
    print("MSE:   {:-20.16f}".format(metrics.mse(y, y_predict)))
    # print (metrics.bias(y, y_predict))
    print("Bias^2:{:-20.16f}".format(metrics.bias2(y, y_predict)))

    print("Cross Validation")
    cv = kFoldCrossValidation(x)
    cv.reg = LinearRegression
    cv.func_excact = func_excact
    cv.design_matrix = design_matrix
    cv.cross_validate(k_percent=0.25)
    print("R2:    {:-20.16f}".format(cv.mean_R2))
    print("MSE:   {:-20.16f}".format(cv.mean_MSE))
    print("Bias^2:{:-20.16f}".format(cv.mean_bias))
    print("Var(y):{:-20.16f}".format(cv.mean_var))


def __test_mc_cross_validation():
    pass


if __name__ == '__main__':
    __test_k_fold_cross_validation()
    __test_mc_cross_validation()
