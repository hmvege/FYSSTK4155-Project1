import numpy as np


def mse(y, y_tilde):
    """Mean Square Error

    Uses numpy to calculate the mean square error.

    MSE = (1/n) sum^(N-1)_i=0 (y_i - y_tilde_i)^2

    Args:
        y (ndarray): response/outcome/dependent variable, size (N_samples, 1)
        y_tilde (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: mean square error
    """

    assert y.shape == y_tilde.shape
    assert y.shape[1] == y_tilde.shape[1] == 1

    return np.mean((y-y_tilde)**2)


def R2(y, y_tilde):
    """R^2 score

    Uses numpy to calculate the R^2 score.

    R^2 = 1 - sum(y - y_tilde)/sum(y - mean(y_tilde))

    Args:
        y (ndarray): response/outcome/dependent variable, size (N_samples, 1)
        y_tilde (ndarray): fitted response variable, size (N_samples, 1)

    Returns
        float: R^2 score
    """

    assert y.shape == y_tilde.shape
    assert y.shape[1] == y_tilde.shape[1] == 1

    return 1.0 - np.sum((y-y_tilde)**2)/np.sum((y-np.mean(y_tilde))**2)
