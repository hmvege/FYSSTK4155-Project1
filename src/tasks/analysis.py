import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

def franke_analysis(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)

    for elem in data:
        print (elem.keys())

def terrain_analysis(picke_file_name):
    with open(picke_file_name, "rb") as f:
        data = pickle.load(f)

    

def main():
    franke_analysis("franke_func.pickle")
    # terrain_analysis("real_data.pickle")

if __name__ == '__main__':
    main()