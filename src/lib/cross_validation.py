import numpy as np
import metrics
from tqdm import tqdm

__all__ = ["kFoldCrossValidation", "MCCrossValidation"]


class __CV_core:
    """Core class for performing k-fold cross validation."""
    _reg = None
    _design_matrix = None

    def __init__(self, x_data, y_data, reg, design_matrix_func):
        """Initializer for Cross Validation.

        Args:
            x_data (ndarray): x data on the shape (N, 1)
            y_data (ndarray): y data on the shape (N, 1). Data to be 
                approximated.
            reg (Regression Instance):
            design_matrix_func (function): function that sets up the design
                matrix.
        """
        assert len(x_data) == len(y_data), "x and y data not of equal lengths"
        self.x_data = x_data
        self.y_data = y_data
        self._reg = reg()
        self._design_matrix = design_matrix_func

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

        assert split_percentage < 1.0, "percent must be less than one."

        # Splits into k intervals
        test_size = np.floor(N * split_percentage)
        k_splits = int(N / test_size)
        test_size = int(test_size)

        if enforce_equal_intervals:
            if N % test_size != 0:
                raise ValueError("bad percent: N % k = {} != 0".format(
                    N % test_size))
        else:
            test_size += N % test_size

        return k_splits, test_size

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


class kFoldCrossValidation(__CV_core):
    """Class for performing k-fold cross validation."""

    def cross_validate(self, k_percent=0.2, holdout_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        k_holdout, holdout_test_size = self._get_split_percent(
            holdout_percent, N_total_size, enforce_equal_intervals=False)

        # Splits X data and design matrix data
        x_holdout_test, x_kfold_train = np.split(self.x_data,
                                                 [holdout_test_size], axis=0)
        y_holdout_test, y_kfold_train = np.split(self.y_data,
                                                 [holdout_test_size], axis=0)

        N_kfold_data = len(x_kfold_train)

        # Sets up the holdout design matrix
        X_holdout_test = self._design_matrix(x_holdout_test)

        # Splits dataset into managable k fold tests
        k_splits, test_size = self._get_split_percent(k_percent, N_kfold_data)

        # Splits kfold train data into k actual folds
        x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        # Stores the test values from each k trained data set in an array
        R2_list = np.empty(k_splits)

        self.y_pred_list = np.empty((k_splits, holdout_test_size))

        for ik in tqdm(range(k_splits), desc="k-fold Cross Validation"):
            # Gets the testing data
            k_x_test = x_subdata[ik]
            k_y_test = y_subdata[ik]

            X_test = self._design_matrix(k_x_test)

            # Sets up indexes
            set_list = list(range(k_splits))
            set_list.pop(ik)

            # Sets up new data set
            k_x_train = np.concatenate([x_subdata[d] for d in set_list])
            k_y_train = np.concatenate([y_subdata[d] for d in set_list])

            # Sets up function to predict
            X_train = self._design_matrix(k_x_train)

            # Trains method bu fitting data
            self.reg.fit(X_train, k_y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_holdout_test).ravel()

            # # Appends R2, MSE, coef scores to list
            self.y_pred_list[ik] = y_predict

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_holdout_test - self.y_pred_list)**2
        self.MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
        _bias = y_holdout_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _R2 = metrics.R2(self.y_pred_list, y_holdout_test, axis=1)
        self.R2 = np.mean(_R2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=0, keepdims=True))


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""

    def cross_validate(self, k_percent=0.2, holdout_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted kk fold CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        k_holdout, holdout_test_size = self._get_split_percent(
            holdout_percent, N_total_size, enforce_equal_intervals=False)

        # Splits X data and design matrix data
        x_holdout_data = np.split(self.x_data, k_holdout, axis=0)
        y_holdout_data = np.split(self.y_data, k_holdout, axis=0)

        # Sets up some arrays for storing the different MSE, bias, var, R^2
        # scores.
        MSE_arr = np.empty(k_holdout)
        R2_arr = np.empty(k_holdout)
        var_arr = np.empty(k_holdout)
        bias_arr = np.empty(k_holdout)

        for i_holdout in tqdm(range(k_holdout),
                              desc="Nested k fold Cross Validation"):

            # Gets the testing holdout data to be used. Makes sure to use
            # every holdout test data once.
            x_holdout_test = x_holdout_data[i_holdout]
            y_holdout_test = y_holdout_data[i_holdout]

            # Sets up indexes
            holdout_set_list = list(range(k_holdout))
            holdout_set_list.pop(i_holdout)

            # Sets up new holdout data sets
            x_holdout_train = np.concatenate(
                [x_holdout_data[d] for d in holdout_set_list])
            y_holdout_train = np.concatenate(
                [y_holdout_data[d] for d in holdout_set_list])

            # Sets up the holdout design matrix
            X_holdout_test = self._design_matrix(x_holdout_test)

            # Splits dataset into managable k fold tests
            N_holdout_data = len(x_holdout_train)
            k_splits, test_size = self._get_split_percent(
                k_percent, N_holdout_data)

            # Splits kfold train data into k actual folds
            x_subdata = np.array_split(x_holdout_train, k_splits, axis=0)
            y_subdata = np.array_split(y_holdout_train, k_splits, axis=0)

            # Stores the test values from each k trained data set in an array
            R2_list = np.empty(k_splits)

            self.y_pred_list = np.empty((k_splits, holdout_test_size))
            # self.y_test_list = np.empty((k_splits, holdout_test_size))

            for ik in range(k_splits):
                # Gets the testing data
                k_x_test = x_subdata[ik]
                k_y_test = y_subdata[ik]

                X_test = self._design_matrix(k_x_test)

                # Sets up indexes
                set_list = list(range(k_splits))
                set_list.pop(ik)

                # Sets up new data set
                k_x_train = np.concatenate([x_subdata[d] for d in set_list])
                k_y_train = np.concatenate([y_subdata[d] for d in set_list])

                # Sets up function to predict
                X_train = self._design_matrix(k_x_train)

                # Trains method bu fitting data
                self.reg.fit(X_train, k_y_train)

                # Getting a prediction given the test data
                y_predict = self.reg.predict(X_holdout_test).ravel()

                # # Appends R2, MSE, coef scores to list
                self.y_pred_list[ik] = y_predict

            # Mean Square Error, mean((y - y_approx)**2)
            _mse = (y_holdout_test - self.y_pred_list)**2
            MSE_arr[i_holdout] = np.mean(np.mean(_mse, axis=0, keepdims=True))

            # Bias, (y - mean(y_approx))^2
            _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
            _bias = y_holdout_test - _mean_pred
            bias_arr[i_holdout] = np.mean(_bias**2)

            # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
            _R2 = metrics.R2(self.y_pred_list, y_holdout_test, axis=1)
            R2_arr[i_holdout] = np.mean(_R2)

            # Variance, var(y_predictions)
            _var = np.var(self.y_pred_list, axis=0, keepdims=True)
            var_arr[i_holdout] = np.mean(_var)

        self.var = np.mean(var_arr)
        self.bias = np.mean(bias_arr)
        self.R2 = np.mean(R2_arr)
        self.MSE = np.mean(MSE_arr)


class MCCrossValidation(__CV_core):
    """
    https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation
    """

    def cross_validate(self, N_mc_crossvalidations, k_percent=0.2,
                       holdout_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted MC CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        k_holdout, holdout_test_size = self._get_split_percent(
            holdout_percent, N_total_size, enforce_equal_intervals=False)

        # Splits X data and design matrix data
        x_holdout_test, x_mc_train = np.split(self.x_data,
                                              [holdout_test_size], axis=0)
        y_holdout_test, y_mc_train = np.split(self.y_data,
                                              [holdout_test_size], axis=0)

        N_mc_data = len(x_mc_train)

        # Sets up the holdout design matrix
        X_holdout_test = self._design_matrix(x_holdout_test)

        # Splits dataset into managable k fold tests
        _, mc_test_size = self._get_split_percent(
            k_percent, N_mc_data)

        # Splits kfold train data into k actual folds
        # x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        # y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        # All possible indices available
        mc_indices = list(range(N_mc_data))

        # Stores the test values from each k trained data set in an array
        R2_list = np.empty(N_mc_crossvalidations)

        self.y_pred_list = np.empty((N_mc_crossvalidations, holdout_test_size))

        for i_mc in tqdm(range(N_mc_crossvalidations),
                         desc="Monte Carlo Cross Validation"):

            # Gets retrieves indexes for MC-CV. No replacement.
            mccv_test_indexes = np.random.choice(mc_indices, mc_test_size)
            mccv_train_indices = np.array(
                list(set(mc_indices) - set(mccv_test_indexes)))

            # Gets the testing data
            k_x_test = x_mc_train[mccv_test_indexes]
            k_y_test = x_mc_train[mccv_test_indexes]

            X_test = self._design_matrix(k_x_test)

            # # Sets up indexes
            # set_list = list(range(k_splits))
            # set_list.pop(ik)

            # Sets up new data set
            k_x_train = x_mc_train[mccv_train_indices]
            k_y_train = y_mc_train[mccv_train_indices]

            # Sets up function to predict
            X_train = self._design_matrix(k_x_train)

            # Trains method bu fitting data
            self.reg.fit(X_train, k_y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_holdout_test).ravel()

            # # Appends R2, MSE, coef scores to list
            self.y_pred_list[i_mc] = y_predict

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_holdout_test - self.y_pred_list)**2
        self.MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
        _bias = y_holdout_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _R2 = metrics.R2(self.y_pred_list, y_holdout_test, axis=1)
        self.R2 = np.mean(_R2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=0, keepdims=True))


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

    print("k-fold Cross Validation")
    cv = kFoldCrossValidation(x, y, LinearRegression, design_matrix)
    cv.cross_validate(k_percent=0.25)
    print("R2:    {:-20.16f}".format(cv.R2))
    print("MSE:   {:-20.16f}".format(cv.MSE))
    print("Bias^2:{:-20.16f}".format(cv.bias))
    print("Var(y):{:-20.16f}".format(cv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(cv.MSE, cv.bias, cv.var,
                                     cv.bias + cv.var))
    print("Diff: {}".format(abs(cv.bias + cv.var - cv.MSE)))

    print("kk Cross Validation")
    kkcv = kkFoldCrossValidation(x, y, LinearRegression, design_matrix)
    kkcv.cross_validate(k_percent=0.25)
    print("R2:    {:-20.16f}".format(kkcv.R2))
    print("MSE:   {:-20.16f}".format(kkcv.MSE))
    print("Bias^2:{:-20.16f}".format(kkcv.bias))
    print("Var(y):{:-20.16f}".format(kkcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kkcv.MSE, kkcv.bias, kkcv.var,
                                     kkcv.bias + kkcv.var))
    print("Diff: {}".format(abs(kkcv.bias + kkcv.var - kkcv.MSE)))

    print("Monte Carlo Cross Validation")
    mccv = MCCrossValidation(x, y, LinearRegression, design_matrix)
    mccv.cross_validate(10000, k_percent=0.25)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    print("\nCross Validation methods tested.")

if __name__ == '__main__':
    __test_cross_validation_methods()
