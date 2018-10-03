import numpy as np
import metrics

__all__ = ["CrossValidation"]


class CrossValidation:
    """Class for performing k-fold cross validation."""
    _reg = None
    _design_matrix = None
    _func_excact = None

    def __init__(self, data):
        self.data = data

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

    def cross_validate(self, k_percent=0.2):
        """
        Args:
            k_percent (float): percentage of the data which is to be used
                for cross validation. Default is 0.2
        """
        assert not isinstance(self._reg, type(None))
        assert not isinstance(self._design_matrix, type(None))
        assert not isinstance(self._func_excact, type(None))

        assert k_percent < 1.0, "k_percent must be less than one."

        N = len(self.data)

        # Splits into k intervals
        test_size = np.floor(N * k_percent)
        k = int(N / test_size)
        test_size = int(test_size)

        if N % test_size != 0:
            raise ValueError("bad k_percent: N % k = {} != 0".format(
                N % test_size))

        subdata = np.array_split(self.data, k, axis=0)

        R2_list = np.empty(k)
        MSE_list = np.empty(k)
        bias_list = np.empty((k, test_size))
        var_list = np.empty(k)
        self.y_pred_list = np.empty((k, test_size))
        self.y_test_list = np.empty((k, test_size))
        self.coef_list = []
        # coef_var_list = []

        for ik in range(k):
            data_test = subdata[ik]
            y_test = self._func_excact(data_test).ravel()
            X_test = self._design_matrix(data_test)

            # Sets up indexes
            set_list = list(range(k))
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
            y_predict = self.reg.predict(X_test).ravel()

            # Appends R2, MSE, coef scores to list
            R2_list[ik] = metrics.R2(y_predict, y_test)
            MSE_list[ik] = metrics.mse(y_predict, y_test)
            # MSE_list[ik] = np.mean((y_test - y_predict)**2, keepdims=True)
            bias_list[ik] = metrics.bias2(y_predict, y_test)
            # bias_list[ik] = (y_test - np.mean(y_predict**2, keepdims=True, axis=1))
            var_list[ik] = np.var(y_predict)

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

        error = np.mean( np.mean((self.y_test_list - self.y_pred_list)**2, axis=0, keepdims=True) )
        bias = np.mean( (self.y_test_list - np.mean(self.y_pred_list, axis=0, keepdims=True))**2 )
        variance = np.mean( np.var(self.y_pred_list, axis=0, keepdims=True) )

        print(error, bias, variance)

        # self.mean_bias = metrics.bias2(axis=0)
        exit(1)

        self.mean_MSE = np.mean( np.mean((self.y_pred_list - self.y_test_list)**2, keepdims=True, axis=1) )
        self.mean_R2 = np.mean(metrics.R2(self.y_test_list, self.y_pred_list, axis=1))
        self.mean_bias = np.mean( np.mean((self.y_pred_list - np.mean(self.y_test_list, axis=1, keepdims=True))**2))
        print (self.mean_MSE)
        print (self.mean_bias)
        # self.mean_MSE = np.mean(MSE_list)
        # self.mean_bias = np.mean(bias_list)
        # self.mean_var = np.mean(var_list)



def __test_cross_validation():
    # A small implementation of a test case
    from regression import LinearRegression
    import matplotlib.pyplot as plt

    # Initial values
    n = 100
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
    cv = CrossValidation(x)
    cv.reg = LinearRegression
    cv.func_excact = func_excact
    cv.design_matrix = design_matrix
    cv.cross_validate(k_percent=0.25)
    print("R2:    {:-20.16f}".format(cv.mean_R2))
    print("MSE:   {:-20.16f}".format(cv.mean_MSE))
    print("Bias^2:{:-20.16f}".format(cv.mean_bias))
    print("Var(y):{:-20.16f}".format(cv.mean_var))

if __name__ == '__main__':
    __test_cross_validation()
