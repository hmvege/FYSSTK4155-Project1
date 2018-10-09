import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import copy as cp

from lib.table_printer import TablePrinter
from lib.sciprint import sciprint


# from matplotlib import rc, rcParams
# rc("text", usetex=True)
# # rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
# # rcParams["font.family"] += ["serif"]


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
    # plot_R2_noise(cp.deepcopy(data))

    # plot_bias_variance(cp.deepcopy(data))

    print(len(data))
    heat_map(cp.deepcopy(ridge_data), "ridge", 5)
    # heat_map(cp.deepcopy(lasso_data), "lasso", 5)


def create_beta_table(data):
    header = [r"$R^2$", "MSE", r"Bias$^2$", '$\beta_0$',
              '$\beta_1$', '$\beta_2$', '$\beta_3$', '$\beta_4$']
    print(header)
    table = []
    print(data[0].keys())
    ptab = TablePrinter(header, table)
    print(ptab)
    ptab.print_table()


def filter_data(data, sort_by="", data_type="", property_dict={}):
    """Only selects items with what provided as arguments."""
    new_data = []
    for d in sorted(data, key=lambda s: s[sort_by]):
        for key, val in property_dict.items():

            if val == "manual" and d["reg_type"] == "lasso":
                continue

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
            # If we provide a data type, we will filter out based on that
            new_data.append(d)
            # if data_type == "":
            # else:
            #     if data_type in d["data"]:
            #         new_data.append(d)

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

    return np.asarray(sorted_data), \
        {k: np.asarray(v) for k, v in new_data.items()}


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
    ax.set_xlabel(r"$\mathcal{N}(x,\infty))$")
    ax.set_ylabel(r"$R^2$")
    # ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_bias_variance(data):
    ols_data = filter_data(
        data, "degree", {"method": "manual", "noise": 0, "reg_type": "ols"})
    ols_data_dict_array = {
        reg: select_data(
            ols_data, "degree", data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }

    ridge_data = filter_data(
        data, "degree",
        {"method": "manual", "noise": 0, "reg_type": "ridge", "alpha": 1e-05})

    ridge_data_dict_array = {
        reg: select_data(
            ridge_data, "degree", data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }

    lasso_data = filter_data(
        data, "degree",
        {"method": "manual", "noise": 0, "reg_type": "lasso", "alpha": 0.1})
    lasso_data_dict_array = {
        reg: select_data(
            lasso_data, "degree", data_type=reg,
            stats_to_select=["r2", "mse", "bias", "var"])
        for reg in data[0]["data"].keys()
    }

    degree, _ = select_data(
        ols_data, "degree", data_type="regression",
        stats_to_select=["r2", "mse", "bias", "var"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(degree, ols_data_dict_array["regression"][1]["mse"], label=r"OLS")
    ax.plot(
        degree, ridge_data_dict_array["regression"][1]["mse"], label=r"Lasso")
    ax.plot(
        degree, lasso_data_dict_array["regression"][1]["mse"], label=r"Ridge")
    ax.set_xlabel(r"Polynomial degree")
    ax.set_ylabel(r"$MSE$")
    # ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    plt.show()


def heat_map(data_, reg_type, degree,
             stat_vars=["r2", "mse", "bias", "var"]):
    data = []
    # Gets noise values
    new_data = filter_data(
        data_, sort_by="noise", data_type="regression", property_dict={"degree": 5, "method": "manual", "reg_type": reg_type})

    for i, d in enumerate(new_data):
        print(i, d.keys(), d["method"], d["degree"], d["noise"], d["reg_type"], d["alpha"], d["data"].keys(), d["data"]["alpha"])

    noise_values, _ = select_data(new_data, "noise", data_type="regression",
                           stats_to_select=["r2", "mse", "bias", "var"])

    alpha_values, _ = select_data(new_data, "alpha", data_type="regression",
                           stats_to_select=["r2", "mse", "bias", "var"])

    print(new_data[100])

    print("ddd")
    print(new_data[108])

    print (len(set(noise_values)), len(set(alpha_values)))
    exit("to")

    data_dict_array = {}
    for noise in list(set(noise_values)):
        # Gets alpha values

        for elem in data_:
            # print_elem(elem)
            if elem["reg_type"] == reg_type:
                data.append(elem)

        filtered_data = filter_data(
            new_data, "alpha",
            {"method": "manual", "degree": degree, "reg_type": reg_type, "noise": noise})


        exit(1)
        data_dict_array = {}
        for reg in data[0]["data"].keys():
            if reg == "alpha":
                continue

            data_dict_array[reg] = select_data(
                filtered_data, "alpha", data_type="regression",
                stats_to_select=["r2", "mse", "bias", "var"])

        # Get alphas
        alpha, _ = select_data(
            filtered_data, "alpha", data_type="regression",
            stats_to_select=stat_vars)


        # print (alpha)
        exit(1)

    print(alpha)
    print("lasso", data_dict_array["regression"][1])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(degree, ols_data_dict_array["regression"][1]["mse"], label=r"OLS")
    # ax.plot(degree, ridge_data_dict_array["regression"][1]["mse"], label=r"Lasso")
    # ax.plot(degree, lasso_data_dict_array["regression"][1]["mse"], label=r"Ridge")
    # ax.set_xlabel(r"Polynomial degree")
    # ax.set_ylabel(r"$MSE$")
    # # ax.set_yscale("log")
    # ax.legend()
    # ax.grid(True)
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
