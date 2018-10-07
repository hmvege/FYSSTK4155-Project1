#!/usr/bin/env python3
import numpy as np
try:
    import lib.metrics as metrics
except ModuleNotFoundError:
    import metrics
from tqdm import tqdm
import sklearn.model_selection as sk_modsel

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


class kFoldCrossValidation(__CV_core):
    """Class for performing k-fold cross validation."""

    def cross_validate(self, k_splits=5, test_percent=0.2):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """

        N_total_size = self.x_data.shape[0]

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        holdout_test_size = int(np.floor(N_total_size * test_percent))
        
        # Shuffles
        np.random.shuffle(self.x_data)
        np.random.shuffle(self.y_data)

        # Manual splitting
        x_holdout_test = self.x_data[:holdout_test_size,:]
        x_kfold_train = self.x_data[holdout_test_size:,:]
        y_holdout_test = self.y_data[:holdout_test_size]
        y_kfold_train = self.y_data[holdout_test_size:]

        np.random.shuffle(x_holdout_test)
        np.random.shuffle(y_holdout_test)
        np.random.shuffle(x_kfold_train)
        np.random.shuffle(y_kfold_train)

        # # print (x_kfold_train[:5])
        # x_kfold_train, x_holdout_test, y_kfold_train, y_holdout_test = \
        #     sk_modsel.train_test_split(self.x_data, self.y_data,
        #                                test_size=test_percent)
        # holdout_test_size = y_holdout_test.shape[0]

        N_kfold_data = len(y_kfold_train)

        # Sets up the holdout design matrix
        X_holdout_test = self._design_matrix(x_holdout_test)

        # Splits dataset into managable k fold tests
        test_size = int(np.floor(N_kfold_data / k_splits))

        # Splits kfold train data into k actual folds
        x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        # Stores the test values from each k trained data set in an array
        R2_list = np.empty(k_splits)
        beta_coefs = []
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

            # Trains method bu fitting data
            self.reg.fit(self._design_matrix(k_x_train), k_y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_holdout_test).ravel()

            # Appends prediction and beta coefs
            self.y_pred_list[ik] = y_predict
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_holdout_test - self.y_pred_list)**2
        self.MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
        _bias = y_holdout_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _R2 = metrics.R2(y_holdout_test, self.y_pred_list, axis=0)
        self.R2 = np.mean(_R2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=0, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = x_holdout_test
        self.y_pred = np.mean(self.y_pred_list, axis=0)
        self.y_pred_var = np.var(self.y_pred_list, axis=0)


class kkFoldCrossValidation(__CV_core):
    """A nested k fold CV for getting bias."""

    def cross_validate(self, k_splits=4, kk_splits=4, test_percent=0.2):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted kk fold CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        holdout_test_size = int(np.floor(N_total_size/k_splits))

        x_holdout_data = np.split(self.x_data, k_splits, axis=0)
        y_holdout_data = np.split(self.y_data, k_splits, axis=0)

        # Sets up some arrays for storing the different MSE, bias, var, R^2
        # scores.
        MSE_arr = np.empty(k_splits)
        R2_arr = np.empty(k_splits)
        var_arr = np.empty(k_splits)
        bias_arr = np.empty(k_splits)

        beta_coefs = []
        x_pred_test = []
        y_pred_mean_list = []
        y_pred_var_list = []

        for i_holdout in tqdm(range(k_splits),
                              desc="Nested k fold Cross Validation"):

            # Gets the testing holdout data to be used. Makes sure to use
            # every holdout test data once.
            x_holdout_test = x_holdout_data[i_holdout]
            y_holdout_test = y_holdout_data[i_holdout]

            # Sets up indexes
            holdout_set_list = list(range(k_splits))
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
            test_size = int(np.floor(N_holdout_data/kk_splits))

            # Splits kfold train data into k actual folds
            x_subdata = np.array_split(x_holdout_train, kk_splits, axis=0)
            y_subdata = np.array_split(y_holdout_train, kk_splits, axis=0)

            # Stores the test values from each k trained data set in an array
            R2_list = np.empty(kk_splits)

            self.y_pred_list = np.empty((kk_splits, holdout_test_size))
            # self.y_test_list = np.empty((kk_splits, holdout_test_size))

            for ik in range(kk_splits):
                # Gets the testing data
                k_x_test = x_subdata[ik]
                k_y_test = y_subdata[ik]

                X_test = self._design_matrix(k_x_test)

                # Sets up indexes
                set_list = list(range(kk_splits))
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

                # Appends prediction and beta coefs
                self.y_pred_list[ik] = y_predict
                beta_coefs.append(self.reg.coef_)

            # Mean Square Error, mean((y - y_approx)**2)
            _mse = (y_holdout_test - self.y_pred_list)**2
            MSE_arr[i_holdout] = np.mean(np.mean(_mse, axis=0, keepdims=True))

            # Bias, (y - mean(y_approx))^2
            _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
            _bias = y_holdout_test - _mean_pred
            bias_arr[i_holdout] = np.mean(_bias**2)

            # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
            _R2 = metrics.R2(y_holdout_test, self.y_pred_list, axis=1)
            R2_arr[i_holdout] = np.mean(_R2)

            # Variance, var(y_predictions)
            _var = np.var(self.y_pred_list, axis=0, keepdims=True)
            var_arr[i_holdout] = np.mean(_var)

            x_pred_test.append(x_holdout_test)
            y_pred_mean_list.append(np.mean(self.y_pred_list, axis=0))
            y_pred_var_list.append(np.var(self.y_pred_list, axis=0))

        self.var = np.mean(var_arr)
        self.bias = np.mean(bias_arr)
        self.R2 = np.mean(R2_arr)
        self.MSE = np.mean(MSE_arr)
        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = np.array(x_pred_test)
        self.y_pred = np.array(y_pred_mean_list)
        self.y_pred_var = np.array(y_pred_var_list)


class MCCrossValidation(__CV_core):
    """
    https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation
    """

    def cross_validate(self, N_mc_crossvalidations, k_splits=4,
                       test_percent=0.2):
        """
        Args:
            k_splits (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        # raise NotImplementedError("Not implemnted MC CV")

        N_total_size = len(self.x_data)

        # Splits dataset into a holdout test chuck to find bias, variance ect
        # on and one to perform k-fold CV on.
        # k_holdout, holdout_test_size = self._get_split_percent(
        #     test_percent, N_total_size, enforce_equal_intervals=False)

        # # Splits X data and design matrix data
        # x_holdout_test, x_mc_train = np.split(self.x_data,
        #                                       [holdout_test_size], axis=0)
        # y_holdout_test, y_mc_train = np.split(self.y_data,
        #                                       [holdout_test_size], axis=0)

        # Splits X data and design matrix data
        x_mc_train, x_holdout_test, y_mc_train, y_holdout_test = \
            sk_modsel.train_test_split(self.x_data, self.y_data,
                                       test_size=test_percent)
        holdout_test_size = y_holdout_test.shape[0]


        N_mc_data = len(x_mc_train)

        # Sets up the holdout design matrix
        X_holdout_test = self._design_matrix(x_holdout_test)

        # Splits dataset into managable k fold tests
        mc_test_size = int(np.floor(N_mc_data / k_splits))

        # Splits kfold train data into k actual folds
        # x_subdata = np.array_split(x_kfold_train, k_splits, axis=0)
        # y_subdata = np.array_split(y_kfold_train, k_splits, axis=0)

        # All possible indices available
        mc_indices = list(range(N_mc_data))

        # Stores the test values from each k trained data set in an array
        R2_list = np.empty(N_mc_crossvalidations)
        beta_coefs = []
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

            # Appends prediction and beta coefs
            self.y_pred_list[i_mc] = y_predict
            beta_coefs.append(self.reg.coef_)

        # Mean Square Error, mean((y - y_approx)**2)
        _mse = (y_holdout_test - self.y_pred_list)**2
        self.MSE = np.mean(np.mean(_mse, axis=0, keepdims=True))

        # Bias, (y - mean(y_approx))^2
        _mean_pred = np.mean(self.y_pred_list, axis=0, keepdims=True)
        _bias = y_holdout_test - _mean_pred
        self.bias = np.mean(_bias**2)

        # R^2 score, 1 - sum(y-y_approx)/sum(y-mean(y))
        _R2 = metrics.R2(y_holdout_test, self.y_pred_list, axis=1)
        self.R2 = np.mean(_R2)

        # Variance, var(y_predictions)
        self.var = np.mean(np.var(self.y_pred_list, axis=0, keepdims=True))

        beta_coefs = np.asarray(beta_coefs)
        self.beta_coefs_var = np.asarray(beta_coefs).var(axis=0)
        self.beta_coefs = np.asarray(beta_coefs).mean(axis=0)

        self.x_pred_test = x_holdout_test
        self.y_pred = np.mean(self.y_pred_list, axis=0)
        self.y_pred_var = np.var(self.y_pred_list, axis=0)


def __test_cross_validation_methods():
    # A small implementation of a test case
    from regression import LinearRegression
    import matplotlib.pyplot as plt

    # Initial values
    n = 100
    N_bs = 1000
    k_splits = 4
    test_percent = 0.2
    noise = 0.3
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + np.exp(-2*_x) + noise * \
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
    print("R2:    {:-20.16f}".format(reg.score(y, y_predict)))
    print("MSE:   {:-20.16f}".format(metrics.mse(y, y_predict)))
    # print (metrics.bias(y, y_predict))
    print("Bias^2:{:-20.16f}".format(metrics.bias2(y, y_predict)))

    # Small plotter
    import matplotlib.pyplot as plt
    plt.plot(x, y, "o", label="data")
    plt.plot(x, y_predict, "o",
             label=r"Pred, $R^2={:.4f}$".format(reg.score(y, y_predict)))

    print("k-fold Cross Validation")
    kfcv = kFoldCrossValidation(x, y, LinearRegression, design_matrix)
    kfcv.cross_validate(k_splits=k_fold_size,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kfcv.R2))
    print("MSE:   {:-20.16f}".format(kfcv.MSE))
    print("Bias^2:{:-20.16f}".format(kfcv.bias))
    print("Var(y):{:-20.16f}".format(kfcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kfcv.MSE, kfcv.bias, kfcv.var,
                                     kfcv.bias + kfcv.var))
    print("Diff: {}".format(abs(kfcv.bias + kfcv.var - kfcv.MSE)))

    plt.errorbar(kfcv.x_pred_test, kfcv.y_pred,
                 yerr=np.sqrt(kfcv.y_pred_var), fmt="o",
                 label=r"k-fold CV, $R^2={:.4f}$".format(kfcv.R2))

    print("kk Cross Validation")
    kkcv = kkFoldCrossValidation(x, y, LinearRegression, design_matrix)
    kkcv.cross_validate(k_splits=k_fold_size,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(kkcv.R2))
    print("MSE:   {:-20.16f}".format(kkcv.MSE))
    print("Bias^2:{:-20.16f}".format(kkcv.bias))
    print("Var(y):{:-20.16f}".format(kkcv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(kkcv.MSE, kkcv.bias, kkcv.var,
                                     kkcv.bias + kkcv.var))
    print("Diff: {}".format(abs(kkcv.bias + kkcv.var - kkcv.MSE)))

    plt.errorbar(kkcv.x_pred_test.ravel(), kkcv.y_pred.ravel(),
                 yerr=np.sqrt(kkcv.y_pred_var.ravel()), fmt="o",
                 label=r"kk-fold CV, $R^2={:.4f}$".format(kkcv.R2))

    print("Monte Carlo Cross Validation")
    mccv = MCCrossValidation(x, y, LinearRegression, design_matrix)
    mccv.cross_validate(N_bs, k_splits=k_fold_size,
                        test_percent=test_percent)
    print("R2:    {:-20.16f}".format(mccv.R2))
    print("MSE:   {:-20.16f}".format(mccv.MSE))
    print("Bias^2:{:-20.16f}".format(mccv.bias))
    print("Var(y):{:-20.16f}".format(mccv.var))
    print("MSE = Bias^2 + Var(y) = ")
    print("{} = {} + {} = {}".format(mccv.MSE, mccv.bias, mccv.var,
                                     mccv.bias + mccv.var))
    print("Diff: {}".format(abs(mccv.bias + mccv.var - mccv.MSE)))

    print("\nCross Validation methods tested.")

    plt.errorbar(mccv.x_pred_test, mccv.y_pred,
                 yerr=np.sqrt(mccv.y_pred_var), fmt="o",
                 label=r"MC CV, $R^2={:.4f}$".format(mccv.R2))

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"$y=2x^2$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    __test_cross_validation_methods()
