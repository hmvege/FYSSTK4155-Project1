import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import copy as cp
import os

from lib.table_printer import TablePrinter
from lib.sciprint import sciprint

import lib.metrics as metrics
import lib.bootstrap as bs
import lib.regression as reg
import lib.cross_validation as cv
import lib.scikit_resampling as sk_resampling
import sklearn.model_selection as sk_modsel
import sklearn.preprocessing as sk_preproc
import sklearn.linear_model as sk_model
import sklearn.metrics as sk_metrics
import sklearn.utils as sk_utils


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# Proper LaTeX font
import matplotlib as mpl
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
mpl.rcParams["font.family"] += ["serif"]

def main():
    # analysis(load_pickle("franke_func_data.pickle"), "franke1")
    # analysis(load_pickle("franke_func_data_final11.pickle"), "franke2")
    analysis(load_pickle("franke_func_data_final12.pickle"), "franke2")
    # analysis(load_pickle("_ff_data_bak.pickle"), "franke3")
    # analysis(load_pickle("terrain_data1.pickle"), "terrain1")


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


def analysis(data, analysis_name):

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

    print (len(data))
    # print(lasso_data[0].keys(), lasso_data[0]["data"].keys(),
    #       lasso_data[0]["data"]["regression"].keys())

    noise_values, _ = select_data(ridge_data, "noise", data_type="regression",
                                  stats_to_select="r2")
    alpha_values, _ = select_data(lasso_data, "alpha", data_type="bootstrap",
                                  stats_to_select="r2")
    degree_values, _ = select_data(ridge_data, "degree",
                                   data_type="regression",
                                   stats_to_select="r2")
    noise_values = sorted(list(set(noise_values)))
    alpha_values = sorted(list(set(alpha_values)))
    degree_values = sorted(list(set(degree_values)))

    stats = [
        ("r2", r"$R^2$"),
        ("mse", r"MSE"),
        ("bias", r"Bias$^2$"),
        ("var", r"Var"),
    ]

    data_type_values = [
        ("regression", "Regression"),
        ("kfoldcv", r"$k$-fold CV"),
        ("mccv", r"MC-CV"),
        ("bootstrap", r"Bootstrap")
    ]

    regression_types = ["ols", "ridge", "lasso"]

    # create_beta_table(ols_data)

    # for alpha_ in alpha_values:
    #     for noise_ in noise_values:
    #         for deg_ in degree_values:
    #             plot_beta_values(data, noise=noise_, deg=deg_,
    #                              alpha=alpha_, data_type="bootstrap",
    #                              noise_values=noise_values,
    #                              alpha_values=alpha_values,
    #                              aname=analysis_name)

    # for deg_ in degree_values[4:]:
    #     plot_R2_noise(cp.deepcopy(data), deg=deg_,
    #                   reg_type="ols", aname=analysis_name)
    #     for alpha_ in alpha_values:
    #         plot_R2_noise(cp.deepcopy(data), deg=deg_, alpha=alpha_,
    #                       reg_type="ridge", aname=analysis_name)
    #         plot_R2_noise(cp.deepcopy(data), deg=deg_, alpha=alpha_,
    #                       reg_type="lasso", aname=analysis_name)

    # plot_argx_argy(cp.deepcopy(data), "noise", "r2",
    #                x_arg_latex=r"Noise($\mathcal{N}(',\infty)$)",
    #                y_arg_latex=r"$R^2$", deg=5, reg_type="lasso",
    #                aname=analysis_name)

    # for dtype_ in data_type_values:
    #     if dtype_[0]=="regression":
    #         continue
    #     plot_bias_variance_all(cp.deepcopy(data), "mccv",
    #                            data_type_header=r"MC-CV",
    #                            aname=analysis_name)
    #     for reg_type_ in regression_types:
    #         plot_bias_variance(cp.deepcopy(data), reg_type_, dtype_[0],
    #                            data_type_header=dtype_[1],
    #                            aname=analysis_name)

    # for deg_ in degree_values:
    #     for dtype_ in data_type_values:
    #         for stat_ in stats:
    #             if stat_[0] == "var" and dtype_[0] == "regression":
    #                 continue
    #             else:
    #                 heat_map(cp.deepcopy(ridge_data), "ridge",
    #                          deg_, stat=stat_[0], stat_latex=stat_[1],
    #                          data_type=dtype_[0], aname=analysis_name)
    #                 heat_map(cp.deepcopy(lasso_data), "lasso",
    #                          deg_, stat=stat_[0], stat_latex=stat_[1],
    #                          data_type=dtype_[0], aname=analysis_name)


    # find_optimal_parameters(data, aname=analysis_name)

    minvals = []# min of mse and r2

    for i, d_ in enumerate(data):
        dd = {}
        r2 = d_["data"]["regression"]["r2"]
        mse = d_["data"]["bootstrap"]["mse"]
        score = np.sqrt((1-r2)**2 + mse**2)
        dd["score"] = score
        dd["r2"] = r2
        dd["mse"] = mse
        dd["bias"] = d_["data"]["bootstrap"]["bias"]
        dd["var"] = d_["data"]["bootstrap"]["var"]
        if "alpha" in d_:
            dd["alpha"] = d_["alpha"]
        if "noise" in d_:
            dd["noise"] = d_["data"]["bootstrap"]["bias"]

        dd["reg_type"] = d_["reg_type"]
        dd["degree"] = d_["degree"]

        dd["beta"] = d_["data"]["regression"]["beta_coefs"]

        minvals.append(dd)

    print("BEST ESTIMATE:")
    opt_val = min(minvals, key=lambda d: d["score"])
    print(opt_val)

    if "franke" in analysis_name:
        ### Franke func
        Nrows, Ncols = 100, 100
        ax_rows = np.linspace(0,1,Nrows)
        ax_cols = np.linspace(0,1,Ncols)
    else:        
        ### Terrain data
        Nrows, Ncols = 100, 250
        ax_rows = np.arange(Nrows)
        ax_cols = np.arange(Ncols)

    [x, y] = np.meshgrid(ax_cols, ax_rows)

    poly = sk_preproc.PolynomialFeatures(degree=opt_val["degree"], include_bias=True)
    X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]))
    opt_beta = opt_val["beta"]
    z = X @ opt_beta

    if "alpha" in opt_val:
        figure_name = "../fig/{}_{}_{}".format(
            opt_val["reg_type"],opt_val["degree"],opt_val["alpha"])
    else:
        figure_name = "../fig/{}_{}".format(
            opt_val["reg_type"],opt_val["degree"])

    plot_simple_surface(x, y, z.reshape(Nrows,Ncols), figure_name)



    # for noise_ in noise_values:
    #     minvals = []# min of mse and r2
    #     print ("noise:", noise_)
    #     for i, d_ in enumerate(data):
    #         if noise_ != d_["noise"]: continue

    #         dd = {}
    #         r2 = d_["data"]["regression"]["r2"]
    #         mse = d_["data"]["bootstrap"]["mse"]
    #         score = np.sqrt((1-r2)**2 + mse**2)
    #         dd["score"] = score
    #         dd["r2"] = r2
    #         dd["mse"] = mse
    #         dd["bias"] = d_["data"]["bootstrap"]["bias"]
    #         dd["var"] = d_["data"]["bootstrap"]["var"]
    #         if "alpha" in d_:
    #             dd["alpha"] = d_["alpha"]
    #         if "noise" in d_:
    #             dd["noise"] = d_["noise"]

    #         dd["reg_type"] = d_["reg_type"]
    #         dd["degree"] = d_["degree"]

    #         dd["beta"] = d_["data"]["regression"]["beta_coefs"]

    #         minvals.append(dd)

    #     print("BEST ESTIMATE:")
    #     opt_val = min(minvals, key=lambda d: d["score"])
    #     print(opt_val)

    #     if "franke" in analysis_name:
    #         print ("franke")
    #         ### Franke func
    #         Nrows, Ncols = 100, 100
    #         ax_rows = np.linspace(0,1,Nrows)
    #         ax_cols = np.linspace(0,1,Ncols)
    #     else:        
    #         ### Terrain data
    #         Nrows, Ncols = 100, 250
    #         ax_rows = np.arange(Nrows)
    #         ax_cols = np.arange(Ncols)

    #     [x, y] = np.meshgrid(ax_cols, ax_rows)

    #     poly = sk_preproc.PolynomialFeatures(degree=opt_val["degree"], include_bias=True)
    #     X = poly.fit_transform(cp.deepcopy(np.c_[x.ravel(), y.ravel()]))
    #     opt_beta = opt_val["beta"]
    #     z = X @ opt_beta

    #     if "alpha" in opt_val:
    #         figure_name = "../fig/{}_{}_{}".format(
    #             opt_val["reg_type"],opt_val["degree"],opt_val["alpha"])
    #     else:
    #         figure_name = "../fig/{}_{}".format(
    #             opt_val["reg_type"],opt_val["degree"])

    #     plot_simple_surface(x, y, z.reshape(Nrows,Ncols), figure_name)



    # for deg_ in degree_values:
    #     for dtype_ in data_type_values:
    #         for regtype_ in regression_types:


            


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
            # elif type(val) == np.int_:  # Degrees filtering
            #     if val != d[key]:
            #         break
            elif type(d[key]) == np.float_:
                if val != d[key]:
                    break
            else:
                # print (key, d[key], val, type(val), type(d[key]))
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
                     noise_values=[], alpha_values=[], aname=""):

    data = []
    available_stats = ["r2", "mse", "bias", "var"]

    ols_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "ols", "noise": int(noise)})

    ridge_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "ridge", "noise": int(noise),
                       "alpha": alpha})

    lasso_data = filter_data(
        data_, sort_by="degree", data_type="",
        property_dict={"method": "manual", "reg_type": "lasso", "noise": int(noise),
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

    figure_name = (
        "../fig/{:s}_beta_values_d{:d}_noise{:.4f}_alpha{:.4f}_"
        "{:s}.pdf".format(aname, deg, float(noise), float(alpha), data_type))
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def plot_argx_argy(data, x_arg, y_arg, x_arg_latex="", y_arg_latex="",
                   deg=5, reg_type="ols", aname=""):
    new_data = filter_data(
        data, x_arg, {"degree": deg, "method": "manual", "reg_type": reg_type})
    data_dict_array = {
        reg: select_data(
            new_data, x_arg, data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }
    x_arg_values, _ = select_data(
        new_data, x_arg, data_type="regression",
        stats_to_select=["r2", "mse", "bias", "var"])

    x_arg_values = sorted(x_arg_values)


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

    figure_name = "../fig/{4:s}_{0:s}_vs_{1:s}_deg{2:d}_{3:s}.pdf".format(
        x_arg, y_arg, deg, reg_type, aname)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def plot_R2_noise(data_, deg=5, reg_type="ols", alpha="", aname=""):

    if reg_type != "ols":
        if alpha=="":
            exit("please provide an alpha value")
        new_data = filter_data(
            data_, sort_by="noise",
            property_dict={
                "method": "manual", "degree": int(deg), 
                "alpha": alpha, "reg_type": reg_type})
    else:
        new_data = filter_data(
            data_, sort_by="noise",
            property_dict={
                "method": "manual", "degree": int(deg), "reg_type": reg_type})

    noise_values, _ = select_data(
        data_, "noise", data_type="regression",
        stats_to_select=["r2", "mse", "bias", "var"])

    noise_values = list(set(noise_values))

    reg_values = np.empty(len(noise_values))
    kfcv_values = np.empty(len(noise_values))
    mccv_values = np.empty(len(noise_values))
    bs_values = np.empty(len(noise_values))

    for i, d_ in enumerate(new_data):
        reg_values[i] = d_["data"]["regression"]["r2"]
        kfcv_values[i] = d_["data"]["kfoldcv"]["r2"]
        mccv_values[i] = d_["data"]["mccv"]["r2"]
        bs_values[i] = d_["data"]["bootstrap"]["r2"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(noise_values, reg_values, "-o", label=r"Regression")
    ax.plot(noise_values, kfcv_values, "-x", label=r"$k$-fold CV")
    ax.plot(noise_values, mccv_values, "-^", label=r"MCCV")
    ax.plot(noise_values, bs_values, "-v", label=r"Bootstrap")
    ax.set_xlabel(r"Noise($\mathcal{N}(x,\infty))$)")
    ax.set_ylabel(r"$R^2$")
    # ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    if alpha!="":
        alpha_str = "_alpha{:.2f}".format(alpha)
    else:
        alpha_str = ""

    figure_name = "../fig/{2:s}_noise_vs_r2_deg{0:d}{3:s}_{1:s}.pdf".format(
        deg, reg_type, aname, alpha_str)
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def plot_bias_variance_all(data_, data_type, data_type_header="",
                           tick_param_fs=None, aname=""):

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

    figure_name = ("../fig/{2:s}_bias_variance_tradeoff_all_{0:s}_"
        "{1:s}.pdf".format(data_type, stat, aname))
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def plot_bias_variance(data_, regression_type, data_type,
                       data_type_header="", tick_param_fs=None,
                       max_degree=0, aname=""):

    data = filter_data(
        data_, sort_by="degree", data_type="regression",
        property_dict={
            "method": "manual", "reg_type": regression_type, "noise": 0.0})

    degree_values, _ = select_data(
        data, "degree", data_type=data_type,
        stats_to_select=["r2", "mse", "bias", "var"])

    degree_values = sorted(list(set(degree_values)))

    if max_degree == 0:
        max_degree = degree_values[-1]

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

    figure_name = ("../fig/{3:s}_bias_variance_tradeoff_{0:s}_{1:s}"
        "_{2:}.pdf".format(data_type, stat, regression_type, aname))
    fig.savefig(figure_name)
    print("Figure saved at {}".format(figure_name))
    plt.close(fig)


def heat_map(data_, reg_type, degree, data_type="regression",
             stat="r2", stat_latex=r"$R^2$", aname=""):

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
                    try:
                        plot_data[i, j] = d["data"][data_type][stat]
                        # print(d["data"]["regression"][stat])
                    except KeyError:
                        print(d["reg_type"], d["alpha"], d["noise"], d["degree"],data_type, d.keys(), d["data"].keys(), d["data"][data_type].keys())
                        exit()

    heatmap_plotter(alpha_values, noise_values, plot_data,
                    "../fig/{2:s}_{0:s}_{1:s}_deg{3:d}_dtype{4:s}_heatmap.pdf".format(reg_type,
                                                                  stat, aname, degree, data_type),
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

def plot_simple_surface(x, y, z, filename="../../fig/simple_surface"):
    """Surface plotter from project notes."""

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plotting the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    # Customize the z axis
    ax.set_zlim(-01.0, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # print(os.getcwd())
    figpath = os.path.abspath(filename + ".pdf")
    # print(figpath)
    # plt.savefig(figpath, dpi=350)
    plt.show()



def find_optimal_parameters(data):
    pass


if __name__ == '__main__':
    main()
