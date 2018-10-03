import numpy as np
from metrics import R2, mse

__all__ = ["CrossValidation"]


class CrossValidation:
    """Class for performing k-fold cross validation."""
    _reg = None
    _func_fit = None
    _func_excact = None

    def __init__(self, data):
        self.data = data

    @property
    def func_fit(self):
        return self._func_fit

    @func_fit.setter
    def func_fit(self, f):
        self._func_fit = f

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

    def cross_validate(self, k_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._func_fit, type(None))
        assert not isinstance(self._func_excact, type(None))

        assert k_percent < 1.0, "k_percent must be less than one."

        N = len(self.data)

        # Splits into k intervals
        test_size = np.floor(N * k_percent)
        k = int(N / test_size)

        if N % k != 0:
            raise ValueError("bad k_percent: N % k = {} != 0".format(N % k))

        subdata = np.split(self.data, k, axis=0)

        R2_list = np.empty(k)
        MSE_list = np.empty(k)
        coef_list = []
        # coef_var_list = []

        for ik in range(k):
            data_test = subdata[ik]
            y_test = self._func_excact(data_test)
            X_test = self._func_fit(data_test)

            # Sets up indexes
            set_list = list(range(k))
            set_list.pop(ik)

            # Sets up new data set
            data_train = np.concatenate([subdata[d] for d in set_list])

            # Gets the training data
            y_train = self._func_excact(data_train)

            # Sets up function to predict
            X_train = self._func_fit(data_train)

            # Trains method bu fitting data
            self.reg.fit(X_train, y_train)

            # Getting a prediction given the test data
            y_predict = self.reg.predict(X_test)

            # Appends R2, MSE, coef scores to list
            R2_list[ik] = R2(y_predict, y_test)
            MSE_list[ik] = mse(y_predict, y_test)
            coef_list.append(self.reg.coef)

            print(
                "R = {:-20.16f}".format(R2_list[ik]),
                " MSE = {:-20.16f}".format(MSE_list[ik]))

        print("Average R:   {:-20.16f}\nAverage MSE: {:-20.16f}".format(
            np.mean(R2_list), np.mean(MSE_list)))


def __test_cross_validation():
    # A small implementation of a test case
    from regression import LinearRegression

    # Initial values
    n = 200
    noise = 0.2
    np.random.seed(1234)

    # Sets up random matrices
    x = np.random.rand(n, 1)

    def func_excact(_x): return 2*_x*_x + noise * \
        np.random.randn(_x.shape[0], _x.shape[1])

    y = func_excact(x)

    def func_fit(_x):
        return np.c_[np.ones(_x.shape), _x, _x*_x]

    # Sets up design matrix
    X = func_fit(x)

    # Performs regression
    reg = LinearRegression()
    reg.fit(X, y)
    y_predict = reg.predict(X)
    print("Regular linear regression")
    print("R2:  ", reg.score(y_predict, y))
    print("MSE: ", mse(y, y_predict))

    print("Cross Validation")
    cv = CrossValidation(x)
    cv.reg = LinearRegression
    cv.func_excact = func_excact
    cv.func_fit = func_fit
    cv.cross_validate(k_percent=0.25)


if __name__ == '__main__':
    __test_cross_validation()
