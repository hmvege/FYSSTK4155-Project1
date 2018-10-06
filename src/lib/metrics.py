#!/usr/bin/env python
import numpy as np


__all__ = ["mse", "R2", "bias", "timing_function"]


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


def timing_function(func):
    """Time function decorator."""
    import time
    def wrapper(*args, **kwargs):
        t1 = time.clock()
        
        val = func(*args, **kwargs)

        t2 = time.clock()

        time_used = t2-t1
        
        print ("Time used with function {:s}: {:.10f} secs/ "
            "{:.10f} minutes".format(func.__name__, time_used, time_used/60.))
        
        return val

    return wrapper
