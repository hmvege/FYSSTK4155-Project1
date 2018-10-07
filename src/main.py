#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

import lib.regression as libreg
import lib.metrics as metrics

# Task a)-b), Franke function testing
import tasks.franke_func_reg as ff_tasks

def main():
    folder_path = "../../MachineLearning/doc/Projects/2018/Project1/DataFiles/"
    abs_folder_path = os.path.abspath(folder_path)

    # print(os.listdir(abs_folder_path))
    # franke_function_example()
    ff_tasks.franke_func_tasks()
    # plt.show()

if __name__ == '__main__':
    main()
