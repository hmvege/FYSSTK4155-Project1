import numpy as np
from metrics import R2, mse

def boot(*data):
    """Strip-down version of the bootstrap method.

    Args:
        *data (ndarray): list of data arrays to resample.

    Return:
        *bs_data (ndarray): list of bootstrapped data arrays."""
    

    N_data = len(data)
    assert np.all(np.array([len(d) for d in data]) == N_data), \
        "unequal lengths of data passed."

    N = data[0].shape[0]

    index_lists = np.random.randint(N, size=N)

    return [d[index_lists] for d in data]

def bootstrap(*data, N_bs):
    bs_data = []
    for i in range(N_bs):
        bs.data.append(boot(*data))
    return bs_data

def bootstrap_regression(x, y, N_bs, test_data=0.2):
    """
    Method that splits dataset into test data and train data.
    """

    pass

def __test_bootstrap():
        # A small implementation of a test case
    from regression import LinearRegression

    N_bs = 100

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

    print("Bootstrapping")

    X_train


    for i_bs in range(N_bs):

    bs = Bootstrap(x, y, N_bs=10)
    cv.reg = LinearRegression
    cv.func_excact = func_excact
    cv.func_fit = func_fit
    cv.cross_validate(k_percent=0.25)


if __name__ == '__main__':
    __test_bootstrap()