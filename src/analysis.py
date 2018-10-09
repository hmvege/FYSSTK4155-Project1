import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import copy as cp


from lib.table_printer import TablePrinter
from lib.sciprint import sciprint

# Proper LaTeX font
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

    noise_values, _ = select_data(ridge_data, "noise", data_type="regression",
                                  stats_to_select="r2")
    alpha_values, _ = select_data(lasso_data, "alpha", data_type="bootstrap",
                                  stats_to_select="r2")
    noise_values = sorted(list(set(noise_values)))
    alpha_values = sorted(list(set(alpha_values)))

    # create_beta_table(ols_data)

    # plot_beta_values(data, noise=0.0, deg=5, data_type="bootstrap", reg_type="ols")
    for alpha_ in alpha_values:
        plot_beta_values(data, noise=0.0, deg=5, alpha=alpha_, data_type="bootstrap",
                         noise_values=noise_values, alpha_values=alpha_values)

    # plot_beta_values(data, noise=0.0, deg=5, data_type="bootstrap", noise_values=noise_values, alpha_values=alpha_values)

    # plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="ols")
    # plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="ridge")
    # plot_R2_noise(cp.deepcopy(data), deg=5, reg_type="lasso")

    # plot_argx_argy(cp.deepcopy(data), "noise", "r2",
    #                x_arg_latex=r"Noise($\mathcal{N}(',\infty)$)",
    #                y_arg_latex=r"$R^2$", deg=5, reg_type="lasso")

    # plot_bias_variance_all(cp.deepcopy(data), "mccv",
    #                        data_type_header=r"MC-CV")

    # plot_bias_variance(cp.deepcopy(data), "ols", "mccv",
    #                    data_type_header=r"MC-CV")
    # plot_bias_variance(cp.deepcopy(data), "ridge", "kfoldcv",
    #                    data_type_header=r"$k$-fold CV")
    # plot_bias_variance(cp.deepcopy(data), "lasso", "bootstrap",
    #                    data_type_header=r"Bootstrap")

    # heat_map(cp.deepcopy(ridge_data), "ridge",
    #          5, stat="r2", stat_latex=r"$R^2$")
    # heat_map(cp.deepcopy(ridge_data), "ridge",
    #          5, stat="mse", stat_latex=r"MSE")
    # heat_map(cp.deepcopy(ridge_data), "ridge",
    #          5, stat="bias", stat_latex=r"Bias$^2$")
    # heat_map(cp.deepcopy(ridge_data), "ridge",
    #          5, stat="var", stat_latex=r"Var")

    # heat_map(cp.deepcopy(lasso_data), "lasso",
    #          5, stat="r2", stat_latex=r"$R^2$")
    # heat_map(cp.deepcopy(lasso_data), "lasso",
    #          5, stat="mse", stat_latex=r"MSE")
    # heat_map(cp.deepcopy(lasso_data), "lasso",
    #          5, stat="bias", stat_latex=r"Bias$^2$")
    # heat_map(cp.deepcopy(lasso_data), "lasso",
    #          5, stat="var", stat_latex=r"Var")

    find_optimal_parameters(data)


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


def plot_beta_values(data_, noise=0.0, alpha=0.1, deg=5, data_type="",
                     noise_values=[], alpha_values=[]):

    data = []
    available_stats = ["r2", "mse", "bias", "var"]

    ols_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "ols", "noise": noise})

    ridge_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "ridge", "noise": noise,
                       "alpha": alpha})

    lasso_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "lasso", "noise": noise,
                       "alpha": alpha})

    if len(noise_values) == 0:
        noise_values, _ = select_data(lasso_data, "noise", data_type=data_type,
                                      stats_to_select=available_stats)

    if len(alpha_values) == 0:
        alpha_values, _ = select_data(lasso_data, "alpha", data_type=data_type,
                                      stats_to_select=available_stats)

    alpha_values = sorted(list(set(alpha_values)))
    noise_values = sorted(list(set(noise_values)))

    def select_deg(data_dict_, deg_):
        for d_ in data_dict_:
            if d_["degree"] == deg_:
                return d_

    print(ols_data[0].keys(), ols_data[0]["data"].keys(),
          ols_data[0]["data"]["regression"].keys())

    ols_data_arr = np.array([
        select_deg(ols_data, deg)["data"][data_type]["beta_coefs"],
        np.sqrt(select_deg(ols_data, deg)[
                "data"][data_type]["beta_coefs_var"]),
        select_deg(ols_data, deg)["data"][data_type]["beta_95c"]])

    # for i, deg in enumerate(alpha_values):
    ridge_data_arr = np.array([
        select_deg(ridge_data, deg)["data"][data_type]["beta_coefs"],
        np.sqrt(select_deg(ridge_data, deg)[
                "data"][data_type]["beta_coefs_var"]),
        select_deg(ridge_data, deg)["data"][data_type]["beta_95c"]])

    lasso_data_arr = np.array([
        select_deg(lasso_data, deg)["data"][data_type]["beta_coefs"],
        np.sqrt(select_deg(lasso_data, deg)[
                "data"][data_type]["beta_coefs_var"]),
        select_deg(lasso_data, deg)["data"][data_type]["beta_95c"]])

    x = np.arange(ols_data_arr.shape[1])

    # What type of error to plot. 1=variance, 2=95 conf.interval
    err_index = 2

    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    ax1.errorbar(x, ols_data_arr[0], yerr=ols_data_arr[err_index],  fmt=".",
                 capsize=4, elinewidth=1, markeredgewidth=1, label=r"OLS")
    ax1.errorbar(x, ridge_data_arr[0], yerr=ridge_data_arr[err_index], fmt="o",
                 capsize=4, elinewidth=1, markeredgewidth=1,
                 label=r"Ridge $\lambda={:.1e}$".format(alpha))
    ax1.errorbar(x, lasso_data_arr[0], yerr=lasso_data_arr[err_index], fmt="x",
                 capsize=4, elinewidth=1, markeredgewidth=1,
                 label=r"Lasso $\lambda={:.1e}$".format(alpha))
    ax1.grid(True)
    ax1.legend(loc="upper right")
    # ax1.set_xticks(x)
    ax1.set_xticklabels([])
    # ax1.set_xticklabels([r"$\beta_{:d}$".format(i) for i in range(x.shape[0])])
    # ax1.set_yscale("log")
    ax1.set_ylabel(r"$\beta$")
    ax1.axhline(0, linestyle="-", color="0", alpha=0.3)
    ax1.set_ylim([-100, 100])

    ax2 = fig.add_subplot(212)
    ax2.errorbar(x, ols_data_arr[0], yerr=ols_data_arr[err_index],  fmt=".",
                 capsize=4, elinewidth=1, markeredgewidth=1, label=r"OLS")
    ax2.errorbar(x, ridge_data_arr[0], yerr=ridge_data_arr[err_index], fmt="o",
                 capsize=4, elinewidth=1, markeredgewidth=1,
                 label=r"Ridge $\lambda={:.1e}$".format(alpha))
    ax2.errorbar(x, lasso_data_arr[0], yerr=lasso_data_arr[err_index], fmt="x",
                 capsize=4, elinewidth=1, markeredgewidth=1,
                 label=r"Lasso $\lambda={:.1e}$".format(alpha))
    ax2.grid(True)
    # ax2.legend(loc="upper right")
    # ax1.set_xticks(x)
    ax2.set_xticklabels([])

    # ax1.set_yscale("log")
    ax2.set_xlabel(r"$\beta_i$")
    ax2.set_ylabel(r"$\beta$")
    # ax1.set_xlabel(r"$\beta_i$")
    ax2.axhline(0, linestyle="-", color="0", alpha=0.3)
    ax2.set_ylim([-5, 5])

    fig.align_ylabels([ax1, ax2])

    figure_name = ("../fig/beta_values_d{:d}_noise{:.4f}_alpha{:.4f}_{:s}.pdf".format(
        deg, float(noise), float(alpha), data_type))
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


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

    figure_name = "../fig/{0:s}_vs_{1:s}_deg{2:d}_{3:s}.pdf".format(
        x_arg, y_arg, deg, reg_type)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


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

    figure_name = "../fig/noise_vs_r2_deg{0:d}_{1:s}.pdf".format(deg, reg_type)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


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
    print(degree_values)

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
    ax1.plot(degree_values[:max_degree+1],
             ols_values[:max_degree+1, 0], "-o", label=r"MSE")
    ax1.plot(degree_values[:max_degree+1],
             ols_values[:max_degree+1, 1], "-.", label=r"Bias$^2$")
    ax1.plot(degree_values[:max_degree+1],
             ols_values[:max_degree+1, 2], "-x", label=r"Var")
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(312)
    ax2.plot(degree_values[:max_degree+1],
             ridge_values[:max_degree+1, 0], "-o", label=r"MSE")
    ax2.plot(degree_values[:max_degree+1],
             ridge_values[:max_degree+1, 1], "-.", label=r"Bias$^2$")
    ax2.plot(degree_values[:max_degree+1],
             ridge_values[:max_degree+1, 2], "-x", label=r"Var")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(313)
    ax3.plot(degree_values[:max_degree+1],
             lasso_values[:max_degree+1, 0], "-o", label=r"MSE")
    ax3.plot(degree_values[:max_degree+1],
             lasso_values[:max_degree+1, 1], "-.", label=r"Bias$^2$")
    ax3.plot(degree_values[:max_degree+1],
             lasso_values[:max_degree+1, 2], "-x", label=r"Var")
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

    ax1.set_xticks(degree_values[:max_degree+1])
    ax1.set_xticklabels([])
    ax2.set_xticks(degree_values[:max_degree+1])
    ax2.set_xticklabels([])
    ax3.set_xticks(degree_values[:max_degree+1])
    ax3.set_xticklabels([0] + degree_values[:max_degree+1],
                        fontsize=tick_param_fs)

    fig.align_ylabels([ax1, ax2, ax3])

    figure_name = "../fig/bias_variance_tradeoff_all_{0:s}_{1:s}.pdf".format(
        data_type, stat)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


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
    ax1.plot(degree_values[:max_degree+1],
             reg_values[:max_degree+1, 0], "-o", label=r"MSE")
    ax1.plot(degree_values[:max_degree+1],
             reg_values[:max_degree+1, 1], "-.", label=r"Bias$^2$")
    ax1.plot(degree_values[:max_degree+1],
             reg_values[:max_degree+1, 2], "-x", label=r"Var")
    ax1.legend()
    ax1.grid(True)

    ax1.set_title(r"{0:s}".format(data_type_header))
    ax1.set_xlabel(r"Polynomial degree")

    # ax1.set_xticks(np.arange(z.shape[1]) + .5)
    ax1.set_xticks(degree_values[:max_degree+1])
    ax1.set_xticklabels([0] + degree_values[:max_degree+1],
                        fontsize=tick_param_fs)
    # ax1.set_yscale("log")

    figure_name = "../fig/bias_var iance_tradeoff_{0:s}_{1:s}_{2:}.pdf".format(
        data_type, stat, regression_type)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


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


def heatmap_plotter(x, y, z, figure_name, tick_param_fs=None, label_fs=None,
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

    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def find_optimal_parameters(data)



def main():
    franke_analysis(
        # load_pickle("franke_func.pickle"),
        load_pickle("franke_func_ols_final4.pickle")
    )
    # terrain_analysis(load_pickle("franke_func.pickle"))


if __name__ == '__main__':
    main()
