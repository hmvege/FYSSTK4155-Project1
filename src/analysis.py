import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

from lib.table_printer import TablePrinter
from lib.sciprint import sciprint 

def franke_analysis(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)

    ols_data = []
    ridge_data = []
    lasso_data = []

    ols_beta_values = []

    for elem in data:
        if elem["reg_type"] == "ols":
            ols_data.append(elem)
        if elem["reg_type"] == "ridge":
            ridge_data.append(elem)
        if elem["reg_type"] == "lasso":
            lasso_data.append(elem)


    # assert len(lasso_data) + len(ridge_data) + len(ols_data) == len(data)




def terrain_analysis(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)

        

def main():
    franke_analysis("franke_func.pickle")
    # terrain_analysis("real_data.pickle")

if __name__ == '__main__':
    main()