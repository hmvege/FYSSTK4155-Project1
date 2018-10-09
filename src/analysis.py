import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import copy as cp

from lib.table_printer import TablePrinter
from lib.sciprint import sciprint

##Proper LaTeX font
import matplotlib as mpl
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
mpl.rcParams["font.family"] += ["serif"]


def load_pickle(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)
    return data


def print_elem(elem):
    try:
        print("implementation:", elem["method"], "deg:", elem["degree"],
              "noise:", elem["noise"],
              "reg:", elem["reg_type"],
              "r2:", elem["data"]["regression"]["r2"],
              "mse:", elem["data"]["regression"]["mse"],
              "bias:", elem["data"]["regression"]["bias"],
              "beta95c:", elem["data"]["regression"]["beta_95c"])

    except KeyError:
        print("implementation:", elem["method"], "deg:", elem["degree"],
              "noise:", elem["noise"],
              "reg:", elem["reg_type"],
              "r2:", elem["data"]["regression"]["r2"],
              "mse:", elem["data"]["regression"]["mse"],
              "bias:", elem["data"]["regression"]["bias"])


def franke_analysis(*data):
    data = data[0]

    ols_data = []
    ridge_data = []
    lasso_data = []

    print(data[0].keys(), data[0]["data"].keys(),
          data[0]["data"]["regression"].keys())

    for elem in data:
        # print_elem(elem)
        if elem["reg_type"] == "ols":
            ols_data.append(elem)

        if elem["reg_type"] == "ridge":
            ridge_data.append(elem)
            # print_elem(elem)

        if elem["reg_type"] == "lasso":
            lasso_data.append(elem)
            # print_elem(elem)

    print(lasso_data[0].keys(), lasso_data[0]["data"].keys(),
          lasso_data[0]["data"]["regression"].keys())

    ols_beta = []
    ols_beta_variance = []
    ols_noise = []
    ols_r2 = []

    # print(ols_data[0].keys())
    # print(ols_data[0]["data"].keys())

    for val in sorted(ols_data, key=lambda s: s["noise"]):
        # print(val["data"].keys())
        # ols_r2.append(val["data"]["regression"]["r2"])
        ols_noise.append(val["noise"])

    # create_beta_table(ols_data)

    plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="ols")
    plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="ridge")
    plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="lasso")

    plot_argx_argy(cp.deepcopy(data), "noise", "r2",
                   x_arg_latex=r"Noise($\mathcal{N}(',\infty)$)",
                   y_arg_latex=r"$R^2$", deg=5, reg_type="lasso")

    plot_bias_variance_all(cp.deepcopy(data), "mccv",
                           data_type_header=r"MC-CV")

    plot_bias_variance(cp.deepcopy(data), "ols", "mccv",
                       data_type_header=r"MC-CV")
    plot_bias_variance(cp.deepcopy(data), "ridge", "kfoldcv",
                       data_type_header=r"$k$-fold CV")
    plot_bias_variance(cp.deepcopy(data), "lasso", "bootstrap",
                       data_type_header=r"Bootstrap")

    heat_map(cp.deepcopy(ridge_data), "ridge",
             5, stat="r2", stat_latex=r"$R^2$")
    heat_map(cp.deepcopy(ridge_data), "ridge",
             5, stat="mse", stat_latex=r"MSE")
    heat_map(cp.deepcopy(ridge_data), "ridge",
             5, stat="bias", stat_latex=r"Bias$^2$")
    heat_map(cp.deepcopy(ridge_data), "ridge",
             5, stat="var", stat_latex=r"Var")

    heat_map(cp.deepcopy(lasso_data), "lasso",
             5, stat="r2", stat_latex=r"$R^2$")
    heat_map(cp.deepcopy(lasso_data), "lasso",
             5, stat="mse", stat_latex=r"MSE")
    heat_map(cp.deepcopy(lasso_data), "lasso",
             5, stat="bias", stat_latex=r"Bias$^2$")
    heat_map(cp.deepcopy(lasso_data), "lasso",
             5, stat="var", stat_latex=r"Var")

    # TODO: make a find_optimal_alpha()


def create_beta_table(data):
    header = [r"$R^2$", "MSE", r"Bias$^2$", '$\beta_0$',
              '$\beta_1$', '$\beta_2$', '$\beta_3$', '$\beta_4$']
    print(header)
    table = []
    print(data[0].keys())
    ptab = TablePrinter(header, table)
    print(ptab)
    ptab.print_table()


def filter_data2(data, noise_values=[], alpha_values=[], degree_values=[],
                 data_type="regression", data_prop_dict={}):

    if len(noise_values) == 0:
        noise_values, _ = select_data(data, "noise")
        noise_values = list(set(noise_values))
    if len(alpha_values) == 0:
        alpha_values, _ = select_data(data, "alpha")
        alpha_values = list(set(alpha_values))
    if len(degree_values) == 0:
        degree_values, _ = select_data(data, "degree")
        degree_values = list(set(degree_values))

    print(noise_values, alpha_values, degree_values)

    return_data = {}

    # for noise in noise_values:
    #     noise_dict = {}
    #     for alpha in alpha_values:
    #         alpha_dict = {}
    #         for degree in degree_values:
    #             degree_dict = {}


def filter_data(data, sort_by="", data_type="", property_dict={}):
    """Only selects items with what provided as arguments."""
    new_data = []
    for d in sorted(data, key=lambda s: s[sort_by]):
        for key, val in property_dict.items():

            if val == "manual" and d["reg_type"] == "lasso":
                continue

            if "reg_type" in property_dict:
                if property_dict["reg_type"] != d["reg_type"]:
                    break

            if isinstance(val, int):
                if val != d[key]:
                    break
            elif isinstance(val, float):
                # print(key, val, d.keys(), property_dict["reg_type"], d["reg_type"])
                if val != d[key]:
                    break
            elif type(d[key]) == np.float_:
                if val != d[key]:
                    break
            else:
                if not val in d[key]:
                    break
        else:
            new_data.append(d)

    print("Filtered {} items from {} to {} items.".format(
        len(data)-len(new_data), len(data), len(new_data)))
    return new_data


def select_data(data, sort_by="", data_type="regression", stats_to_select=[],
                suppress_warnings=True):
    """Selects data provided in dict and sorts by given item type. 
    Returns in matrix on order of select data, together with a list of what
    type we are looking at."""

    # The data we are sorting by
    sorted_data = []

    # Populating the new data into a dict of lists
    new_data = {stat_: [] for stat_ in stats_to_select}

    # Reg type order
    reg_type_order = []

    for d in sorted(data, key=lambda s: s[sort_by]):
        sorted_data.append(d[sort_by])

        for stat in stats_to_select:

            if stat in d["data"][data_type].keys():
                # print(d["data"][data_type].keys())
                new_data[stat].append(d["data"][data_type][stat])
            # else:
            #     print("discarding:", stat)

    return np.asarray(sorted_data), \
        {k: np.asarray(v) for k, v in new_data.items()}


def plot_beta_values(data, noise=0.0, deg=5, reg_type=""):
    pass


def plot_argx_argy(data, x_arg, y_arg, x_arg_latex="", y_arg_latex="",
                   deg=5, reg_type="ols"):
    new_data = filter_data(
        data, x_arg, {"degree": 5, "method": "manual", "reg_type": reg_type})
    data_dict_array = {
        reg: select_data(
            new_data, x_arg, data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }
    x_arg_values, _ = select_data(
        new_data, x_arg, data_type="regression",
        stats_to_select=["r2", "mse", "bias", "var"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_arg_values, data_dict_array["regression"]
            [1][y_arg], label=r"OLS Regression")
    ax.plot(x_arg_values,
            data_dict_array["kfoldcv"][1][y_arg], label=r"$k$-fold CV")
    ax.plot(x_arg_values, data_dict_array["mccv"][1][y_arg], label=r"MCCV")
    ax.plot(x_arg_values,
            data_dict_array["bootstrap"][1][y_arg], label=r"Bootstrap")

    if x_arg_latex == "":
        ax.set_xlabel(x_arg)
    else:
        ax.set_xlabel(x_arg_latex)
    if y_arg_latex == "":
        ax.set_xlabel(y_arg)
    # else:
        ax.set_ylabel(y_arg_latex)

    # ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    figname = "../fig/{0:s}_vs_{1:s}_deg{2:d}_{3:s}.pdf".format(
        x_arg, y_arg, deg, reg_type)
    fig.savefig(figname)
    print("Figure saved at {}".format(figname))

    # plt.show()


def plot_R2_noise(data, deg=5, reg_type="ols"):
    new_data = filter_data(
        data, "noise", {"degree": 5, "method": "manual", "reg_type": reg_type})
    data_dict_array = {
        reg: select_data(
            new_data, "noise", data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }
    noise, _ = select_data(
        new_data, "noise", data_type="regression",
        stats_to_select=["r2", "mse", "bias", "var"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(noise, data_dict_array["regression"]
            [1]["r2"], label=r"OLS Regression")
    ax.plot(noise, data_dict_array["kfoldcv"][1]["r2"], label=r"$k$-fold CV")
    ax.plot(noise, data_dict_array["mccv"][1]["r2"], label=r"MCCV")
    ax.plot(noise, data_dict_array["bootstrap"][1]["r2"], label=r"Bootstrap")
    ax.set_xlabel(r"Noise($\mathcal{N}(x,\infty))$)")
    ax.set_ylabel(r"$R^2$")
    # ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    figname = "../fig/noise_vs_r2_deg{0:d}_{1:s}.pdf".format(deg, reg_type)
    fig.savefig(figname)
    print("Figure saved at {}".format(figname))

    # plt.show()


def plot_bias_variance_all(data_, data_type, data_type_header="",
                           tick_param_fs=None):

    ols_data = filter_data(
        data_, sort_by="degree", data_type="regression",
        property_dict={
            "method": "manual", "reg_type": "ols", "noise": 0.0})

    ridge_data = filter_data(
        data_, sort_by="degree", data_type="regression",
        property_dict={
            "alpha": 1.0, "method": "manual",
            "reg_type": "ridge", "noise": 0.0})

    lasso_data = filter_data(
        data_, sort_by="degree", data_type="regression",
        property_dict={
            "alpha": 1.0, "method": "manual",
            "reg_type": "lasso", "noise": 0.0})

    max_degree = 5

    degree_values, _ = select_data(
        ols_data, "degree", data_type=data_type,
        stats_to_select=["r2", "mse", "bias", "var"])

    degree_values = sorted(list(set(degree_values)))

    ols_values = np.empty((len(degree_values), 3))
    ridge_values = np.empty((len(degree_values), 3))
    lasso_values = np.empty((len(degree_values), 3))

    for i, degree in enumerate(degree_values):
        for j, stat in enumerate(["mse", "bias", "var"]):
            ols_values[i, j] = ols_data[i]["data"][data_type][stat]
            ridge_values[i, j] = ridge_data[i]["data"][data_type][stat]
            lasso_values[i, j] = lasso_data[i]["data"][data_type][stat]

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(degree_values[:max_degree],
             ols_values[:max_degree, 0], "-o", label=r"MSE")
    ax1.plot(degree_values[:max_degree],
             ols_values[:max_degree, 1], "-.", label=r"Bias$^2$")
    ax1.plot(degree_values[:max_degree],
             ols_values[:max_degree, 2], "-x", label=r"Var")
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(312)
    ax2.plot(degree_values[:max_degree],
             ridge_values[:max_degree, 0], "-o", label=r"MSE")
    ax2.plot(degree_values[:max_degree],
             ridge_values[:max_degree, 1], "-.", label=r"Bias$^2$")
    ax2.plot(degree_values[:max_degree],
             ridge_values[:max_degree, 2], "-x", label=r"Var")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(313)
    ax3.plot(degree_values[:max_degree],
             lasso_values[:max_degree, 0], "-o", label=r"MSE")
    ax3.plot(degree_values[:max_degree],
             lasso_values[:max_degree, 1], "-.", label=r"Bias$^2$")
    ax3.plot(degree_values[:max_degree],
             lasso_values[:max_degree, 2], "-x", label=r"Var")
    ax3.legend()
    ax3.grid(True)

    ax1.set_title(r"{0:s}".format(data_type_header))
    ax1.set_ylabel(r"OLS")
    ax2.set_ylabel(r"Ridge")
    ax3.set_ylabel(r"Lasso")

    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")

    ax3.set_xlabel(r"Polynomial degree")

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels(degree_values[:max_degree], fontsize=tick_param_fs)

    figname = "../fig/bias_variance_tradeoff_all_{0:s}_{1:s}.pdf".format(
        data_type, stat)
    fig.savefig(figname)
    print("Figure saved at {}".format(figname))
    # plt.show()


def plot_bias_variance(data_, regression_type, data_type,
                       data_type_header="", tick_param_fs=None):

    data = filter_data(
        data_, sort_by="degree", data_type="regression",
        property_dict={
            "method": "manual", "reg_type": regression_type, "noise": 0.0})

    max_degree = 5

    degree_values, _ = select_data(
        data, "degree", data_type=data_type,
        stats_to_select=["r2", "mse", "bias", "var"])

    degree_values = sorted(list(set(degree_values)))

    reg_values = np.empty((len(degree_values), 3))

    for i, degree in enumerate(degree_values):
        for j, stat in enumerate(["mse", "bias", "var"]):
            reg_values[i, j] = data[i]["data"][data_type][stat]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(degree_values[:max_degree],
             reg_values[:max_degree, 0], "-o", label=r"MSE")
    ax1.plot(degree_values[:max_degree],
             reg_values[:max_degree, 1], "-.", label=r"Bias$^2$")
    ax1.plot(degree_values[:max_degree],
             reg_values[:max_degree, 2], "-x", label=r"Var")
    ax1.legend()
    ax1.grid(True)

    ax1.set_title(r"{0:s}".format(data_type_header))
    ax1.set_xlabel(r"Polynomial degree")
    ax1.set_xticklabels(degree_values[:max_degree], fontsize=tick_param_fs)
    # ax1.set_yscale("log")

    figname = "../fig/bias_variance_tradeoff_{0:s}_{1:s}_{2:}.pdf".format(
        data_type, stat, regression_type)
    fig.savefig(figname)
    print("Figure saved at {}".format(figname))
    # plt.show()
    # exit(1)


def heat_map(data_, reg_type, degree, data_type="regression",
             stat="r2", stat_latex=r"$R^2$"):

    if data_type == "regression" and stat == "var":
        print("Stat var not available for regression.")
        return 1

    data = []
    available_stats = ["r2", "mse", "bias", "var"]
    # Gets noise values
    new_data = filter_data(
        data_, sort_by="noise", data_type=data_type,
        property_dict={"degree": 5, "method": "manual", "reg_type": reg_type})

    # for i, d in enumerate(new_data):
    #     print(i, d.keys(), d["method"], d["degree"], d["noise"],
    #           d["reg_type"], d["alpha"], d["data"].keys())

    noise_values, _ = select_data(new_data, "noise", data_type=data_type,
                                  stats_to_select=available_stats)

    alpha_values, _ = select_data(new_data, "alpha", data_type=data_type,
                                  stats_to_select=available_stats)

    # filter_data2(data_, degree_values=[5])

    alpha_values = sorted(list(set(alpha_values)))
    noise_values = sorted(list(set(noise_values)))
    plot_data = np.empty((len(alpha_values), len(noise_values)))

    for i, alpha in enumerate(alpha_values):
        for j, noise in enumerate(noise_values):
            for d in new_data:
                if d["noise"] == noise and d["alpha"] == alpha:
                    # print(d["data"]["regression"][stat])
                    plot_data[i, j] = d["data"]["regression"][stat]

    heatmap_plotter(alpha_values, noise_values, plot_data,
                    "../fig/{0:s}_{1:s}_heatmap.pdf".format(reg_type, stat),
                    xlabel=r"$\lambda$",
                    ylabel=r"Noise$(\mathcal{N}(',\infty))$",
                    cbartitle=stat_latex)


def heatmap_plotter(x, y, z, figname, tick_param_fs=None, label_fs=None,
                    vmin=None, vmax=None, xlabel=None, ylabel=None,
                    cbartitle=None):

    fig, ax = plt.subplots()

    yheaders = ['%1.2f' % i for i in y]
    xheaders = ['%1.2e' % i for i in x]

    heatmap = ax.pcolor(z, edgecolors="k", linewidth=2, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.ax.tick_params(labelsize=tick_param_fs)
    cbar.ax.set_title(cbartitle, fontsize=label_fs)

    # ax.set_title(method, fontsize=fs1)
    ax.set_xticks(np.arange(z.shape[1]) + .5, minor=False)
    ax.set_yticks(np.arange(z.shape[0]) + .5, minor=False)

    ax.set_xticklabels(xheaders, rotation=90, fontsize=tick_param_fs)
    ax.set_yticklabels(yheaders, fontsize=tick_param_fs)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    plt.tight_layout()

    fig.savefig(figname)
    print("Figure saved at {}".format(figname))
    # plt.show()


def terrain_analysis(data):
    pass


def main():
    franke_analysis(
        # load_pickle("franke_func.pickle"),
        load_pickle("franke_func_ols_final3.pickle")
    )
    # terrain_analysis(load_pickle("franke_func.pickle"))


if __name__ == '__main__':
    main()
