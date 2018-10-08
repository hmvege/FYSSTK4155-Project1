import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

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

    ols_beta = []
    ols_beta_variance = []
    ols_noise = []
    ols_r2 = []

    print(ols_data[0].keys())
    print(ols_data[0]["data"].keys())

    for val in sorted(ols_data, key=lambda s: s["noise"]):
        # print(val["data"].keys())
        # ols_r2.append(val["data"]["regression"]["r2"])
        ols_noise.append(val["noise"])

    plot_bias_variance(ols_data, ridge_data, lasso_data)

    # create_beta_table(ols_data)
    plot_R2_noise(ols_data)


def create_beta_table(data):
    header = [r"$R^2$", "MSE", r"Bias$^2$", '$\beta_0$',
              '$\beta_1$', '$\beta_2$', '$\beta_3$', '$\beta_4$']
    print(header)
    table = []
    print(data[0].keys())
    ptab = TablePrinter(header, table)
    print(ptab)
    ptab.print_table()


def filter_data(data, sort_by="", property_dict={}):
    """Only selects items with what provided as arguments."""
    new_data = []
    for d in sorted(data, key=lambda s: s[sort_by]):
        for key, val in property_dict.items():
            if isinstance(val, int):
                if val != d[key]:
                    break
            else:
                if not val in d[key]:
                    break
        else:
            new_data.append(d)
    print("Filtered {} items.".format(len(data)-len(new_data)))
    return new_data


def select_data(data, sort_by="", data_types="all", stats_to_select=[]):
    """Selects data provided in dict and sorts by given item type. 
    Returns in matrix on order of select data, together with a list of what
    type we are looking at."""

    sorted_data = []
    new_data = []
    new_data_types = []
    for d in sorted(data, key=lambda s: s[sort_by]):
        # print (d.keys(), d["reg_type"],d["degree"], d["method"], d["noise"], d["data"].keys())
        sorted_data.append(d[sort_by])
        if data_types == "all":

            if new_data_types != d["data"].keys():
                new_data_types = d["data"].keys()

            for dtype_key, dtype_values in d["data"].items():

                new_data = []

                for key in stats_to_select:
                    if not key in dtype_values.keys():
                        print("Missing key:", key)
                    else:
                        new_data.append(dtype_values[key])
        else:
            for dtype_key, dtype_values in data_types:
                for key in stats_to_select:
                    if not key in dtype_values.keys():
                        print("Missing key:", key)
                    else:
                        new_data.append(dtype_values[key])

    return np.asarray(sorted_data), np.asarray(new_data), stats_to_select, data_types


def plot_R2_noise(data, deg=5):
    noise = np.asarray([d["noise"]
                        for d in sorted(data, key=lambda s: s["noise"])])

    new_data = filter_data(
        data, "noise", {"data": "regression", "degree": 5, "method": "manual"})
    noise, data_array, data_array_names, data_types = select_data(
        new_data, "noise", data_types="all", 
        stats_to_select=["r2", "mse", "bias", "var"])
    print(data_array_names, data_types)
    print (noise)
    print (data_array)

    # def get_r2_data(data_, method, pol_degree):
    #     new_data_list = []
    #     for d in sorted(data, key=lambda s: s["noise"]):
    #         if d["degree"] == pol_degree:
    #             new_data_list.append(d)
    #     # for d in new_data_list:

    exit(1)
    # return np.asarray(new_data_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(get_r2_data(data, "regression", deg))
    ax.plot(noise, get_r2_data(data, "regression", deg), label=r"OLS Regression")
    ax.plot(noise, get_r2_data(data, "kfoldcv", deg), label=r"$k$-fold CV")
    ax.plot(noise, get_r2_data(data, "mccv", deg), label=r"MCCV")
    ax.plot(noise, get_r2_data(data, "bootstrap", deg), label=r"Bootstrap")
    ax.set_xlabel(r"$\mathcal{N}(x,\infty))$")
    ax.set_ylabel(r"$R^2$")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_bias_variance(ols_data, ridge_data, lasso_data):
    degree = [d["degree"] for d in ols_data]


def terrain_analysis(data):
    pass


def main():
    franke_analysis(
        # load_pickle("franke_func.pickle"),
        load_pickle("franke_func_ols_test.pickle")
    )
    # terrain_analysis(load_pickle("franke_func.pickle"))


if __name__ == '__main__':
    main()
