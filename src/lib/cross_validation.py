import numpy as np

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
    def func_fit(self, value):
        self._func_fit = value

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, value, **kwargs):
        """Args:
            rmethod (regression class): regression class to use
        """
        self._reg = value(**kwargs)

    @property
    def func_excact(self):
        return self._func_excact

    @func_excact.setter
    def func_excact(self, value):
        self._func_excact = value

    def cross_validate(self, k=10):
        """
        Args:
            func (function): function that we are fitting.
        """
        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._func_fit, type(None))
        assert not isinstance(self._func_excact, type(None))

        N = len(self.data)

        subdata = np.split(self.data, k, axis=0)

        R2_list = []
        MSE_list = []
        coef_list = []
        coef_var_list = []

        for ik in range(k):
            data_test = subdata[ik]

            # Sets up indexes
            set_list = list(range(k))
            set_list.pop(ik)

            # Sets up new data set
            data_train = np.concatenate([subdata[d] for d in set_list])

            # Gets the training data
            y_train = self._func_excact(data_train)

            # Sets up function to predict
            X = self._func_fit(data_train)

            self.reg.fit(data_train, y_train)

            R2_list.append()

            




def __test():
    # A small implementation of a test case
    from regression import LinearRegression
    from metrics import R2, mse

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
    print("R2:  ", reg.score(X, y))
    print("MSE: ", mse(y, y_predict))

    print("Cross Validation")
    cv = CrossValidation(x)
    cv.reg = LinearRegression
    cv.func_excact = func_excact
    cv.func_fit = func_fit
    cv.cross_validate(k=10)


if __name__ == '__main__':
    __test()
