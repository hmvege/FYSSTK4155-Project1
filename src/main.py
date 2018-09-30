import numpy as np
import matplotlib.pyplot as plt
import os
import numba as nb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def load_data(file_path):
    pass


def linreg(X, y):
    """Simple, manual, linear regression function."""
    # X = np.hstack([np.ones((n, 1)), x, x*x])  # X matrix
    XTX_inv = np.linalg.inv(X.T.dot(X))
    variances = np.diag(XTX_inv)
    beta = XTX_inv.dot(X.T).dot(y)  # Beta matrixx
    y_approx = X.dot(beta)  # Approximation to y
    eps = y - y_approx  # Residues
    return y_approx, beta, eps, variances


def mse(y, y_tilde):
    # Mean Square Error
    n = len(y)
    assert n == len(y_tilde), \
        "mismatch between y and y_tilde size %d %d" % (n, len(y_tilde))
    mse_sum = 0
    for i in range(0, n):
        mse_sum += (y[i] - y_tilde[i])**2
    return mse_sum/float(n)


def RSquared(y, y_tilde):
    n = len(y)
    assert n == len(y_tilde), \
        "mismatch between y and y_tilde size %d %d" % (n, len(y_tilde))
    y_mean = y.mean(axis=0)
    temp_sum1, temp_sum2 = 0, 0
    for i in range(0, n):
        temp_sum1 += (y[i] - y_tilde[i])**2
        temp_sum2 += (y[i] - y_mean)**2
    return 1 - temp_sum1/temp_sum2


def poly5setup(x, y, N):
    """Polynom of degree 5 of x and y."""
    # Degree 0
    assert len(x) == len(y)
    deg0 = np.ones((N, 1))
    # deg0 = np.ones((N**2, 1))

    # Degree 1
    deg1 = x + y

    # Degree 2
    _x2 = x*x
    _y2 = y*y
    deg2 = _x2 + 2*x*y + _y2

    # Degree 3
    _x3 = _x2*x
    _y3 = _y2*y
    _y2x = _y2*x
    _x2y = _x2*y
    deg3 = _x3 + 3*_x2y + 3*_y2x + _y3

    # Degree 4
    _x4 = _x3*x
    _y4 = _y3*y
    _y3x = _y3*x
    _x3y = _x3*y
    _x2y2 = _x2*_y2
    deg4 = _x4 + 4*_x3y + 6*_x2y2 + 4*_y3x + _y4

    # Degree 5
    _x5 = _x4*x
    _y5 = _y4*y
    _x4y = _x4*y
    _y4x = _y4*x
    deg5 = _x5 + 5*_x4y + 10*_x3*_y2 + 10*_x2*_y3 + 5*_y4x + _y5

    return np.hstack([deg0, deg1, deg2, deg3, deg4, deg5])
    # return np.hstack([deg0.reshape((N**2, 1)), deg1.reshape((N**2, 1)),
    #                   deg2.reshape((N**2, 1)), deg3.reshape((N**2, 1)),
    #                   deg4.reshape((N**2, 1)), deg5.reshape((N**2, 1))])


def _plot_simple_surface(x, y, z):
    """Surface plotter from project notes."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plotting the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(-01.0, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def franke_function_example():
    """Code as given in project description."""

    # Make data
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    _plot_simple_surface(x, y, z)


def part1():
    # Generate data
    N = 20
    noise_sigma = 0.1
    noise_mu = 0
    np.random.seed(1)
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    # x += np.random.uniform(noise_mu, noise_sigma, N)
    # y += np.random.uniform(noise_mu, noise_sigma, N)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # X = poly5setup(np.ravel(x), np.ravel(y), N)
    # X = poly5setup(x, y, N)
    X = poly5setup(x.reshape((N*N, 1)), y.reshape((N*N, 1)), N*N)

    # Setting up new vectors
    x_new = np.linspace(0, 1, N)
    y_new = np.linspace(0, 1, N)
    x_new, y_new = np.meshgrid(x_new, y_new)

    # # SCIKIT-LEARN
    # poly5 = PolynomialFeatures(degree=5)
    # # help(poly5.fit_transform)
    # print ("Hi")
    # _X_transformed = poly5.fit_transform(np.hstack([x.reshape(N*N,1),y.reshape(N*N,1)]))
    # print (_X_transformed.shape)
    # exit(1)

    # linfit5 = LinearRegression()
    # linfit5.fit(_X_transformed, z)

    # x_sk_new = poly5.fit_transform([x_new, y_new])
    # z_sk_new = linfit3.predict(x_sk_new)
    # # z_approx, beta, eps, variances = linreg(X, z)
    # # mse_val = mse(z, z_approx)
    # # Rval = RSquared(z, z_approx)

    # MANUAL
    Nnew = 20
    z_approx, beta, eps, beta_variance = linreg(X, z.reshape((N**2, 1)))
    x_new = np.linspace(0, 1, Nnew)
    y_new = np.linspace(0, 1, Nnew)
    x_new, y_new = np.meshgrid(x_new, y_new)
    x_new = x_new.reshape((Nnew**2, 1))
    y_new = y_new.reshape((Nnew**2, 1))
    _xy_new = poly5setup(x_new.reshape((Nnew*Nnew, 1)),
                         y_new.reshape((Nnew*Nnew, 1)), Nnew*Nnew)
    z_new_predict = _xy_new.dot(beta)

    mse_manual = mse(z.reshape((N**2, 1)), z_new_predict)[0]
    r2_manual = RSquared(z.reshape((N**2, 1)), z_new_predict)[0]
    print("MSE: {0:f}".format(mse_manual))
    print("R2: {0:f}".format(r2_manual))
    for i, b in enumerate(zip(np.ravel(beta),
                              np.sqrt(np.ravel(beta_variance)))):
        print("Beta_{2} = {0:.6f} +/- {1:.6f}".format(*b, i))

    # _plot_simple_surface(x, y, z)
    # _plot_simple_surface(x_new.reshape((N, N)), y_new.reshape(
    #     (N, N)), z_new_predict.reshape((N, N)))


def main():
    folder_path = "../../MachineLearning/doc/Projects/2018/Project1/DataFiles/"
    abs_folder_path = os.path.abspath(folder_path)

    # print(os.listdir(abs_folder_path))
    # franke_function_example()
    part1()


if __name__ == '__main__':
    main()
