import numpy as np


__all__ = ["mse", "R2", "bias"]


def mse(y_excact, y_predict, axis=0):
    """Mean Square Error

    Uses numpy to calculate the mean square error.

    MSE = (1/n) sum^(N-1)_i=0 (y_i - y_test_i)^2

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: mean square error
    """

    assert y_excact.shape == y_predict.shape

    return np.mean((y_excact - y_predict)**2, axis=axis)


def R2(y_excact, y_predict, axis=0):
    """R^2 score

    Uses numpy to calculate the R^2 score.

    R^2 = 1 - sum(y - y_test)/sum(y - mean(y_test))

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: R^2 score
    """

    assert y_excact.shape == y_excact.shape
    return 1.0 - np.sum((y_excact - y_predict)**2, axis=axis) / \
        np.sum((y_excact - np.mean(y_excact, axis=axis, keepdims=True))**2, axis=axis)


# TODO: add method for finding bias

def bias2(y_excact, y_predict, axis=0):
    """Bias^2 of a excact y and a predicted y

    Args:
        y_excact (ndarray): response/outcome/dependent variable,
            size (N_samples, 1)
        y_predict (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: Bias^2
    """
    return np.mean((y_predict - np.mean(y_excact, keepdims=True, axis=axis))**2)


def variance(y):
    """Variance between excact and predicted values.

    # TODO: remove this, as it is a completely, utterly redundant function.

    Args:
        y (ndarray): response/outcome/dependent variable, size (N_samples, 1)

    Returns
        float: Var(y)
    """
    return np.mean((y - np.mean(y))**2)