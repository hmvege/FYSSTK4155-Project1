import numpy as np


def mse(y_excact, y_test):
    """Mean Square Error

    Uses numpy to calculate the mean square error.

    MSE = (1/n) sum^(N-1)_i=0 (y_i - y_test_i)^2

    Args:
        y_excact (ndarray): response/outcome/dependent variable, 
            size (N_samples, 1)
        y_test (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: mean square error
    """

    assert y_excact.shape == y_test.shape
    assert y_excact.shape[1] == y_test.shape[1] == 1

    return np.mean((y_excact-y_test)**2)


def R2(y_excact, y_test):
    """R^2 score

    Uses numpy to calculate the R^2 score.

    R^2 = 1 - sum(y - y_test)/sum(y - mean(y_test))

    Args:
        y_excact (ndarray): response/outcome/dependent variable, 
            size (N_samples, 1)
        y_test (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: R^2 score
    """

    assert y_excact.shape == y_test.shape
    return 1.0 - np.sum((y_excact-y_test)**2) / \
        np.sum((y_excact-np.mean(y_test))**2)


# TODO: add method for finding bias