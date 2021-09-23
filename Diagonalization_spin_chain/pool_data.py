import sys
import numpy as np
import os
from configparser import ConfigParser
from support_functions import prepend_line

if len(sys.argv) > 1:
    path_to_look = sys.argv[1]
else:
    path_to_look = "./Plots/"


def pool_data_files(root):
    for entry in os.scandir(root):
        if entry.name[-11:] == "_config.ini":
            config_name = entry.name[:-11]
    config_object = ConfigParser()
    # read original config
    config_object.read(root + config_name + "_config.ini")
    outputtype = config_object.get("Output", "outputtype")
    samples = config_object.getint("Output", "samples")
    filename = config_object.get("Output", "filename")
    real_filename = filename.split('/')[-1]
    do_pooling = True
    for i in range(samples):
        if not os.path.isfile(root + real_filename + f"_{i}.npz"):
            do_pooling = False
            break
    if do_pooling:
        # read config of first realization (to get the time passed) and write it into original one.
        with open(root + config_name + f"_0.ini", 'r') as config_0:
            time_passed = config_0.readline()
        prepend_line(root + config_name + "_config.ini", time_passed)
        if outputtype == "calc_eigvals_eigvecs":
            # data : eigenvalues [dim], eigenvectors [dim, dim]
            data0 = np.load(root + real_filename + "_0.npz")
            dim = data0["arr_0"].size
            pooled_eigenvalues = np.empty((samples, dim))
            pooled_eigenvectors = np.empty((samples, dim, dim))
            for i in range(samples):
                data = np.load(root + real_filename + f"_{i}.npz")
                pooled_eigenvalues[i] = data["arr_0"]
                pooled_eigenvectors[i] = data["arr_1"]
            np.savez(root + real_filename + ".npz", eigenvalues=np.array(pooled_eigenvalues),
                     eigenvectors=np.array(pooled_eigenvectors))
            os.rename(root + config_name + "_config.ini", root + config_name + ".ini")
            for i in range(samples):
                os.remove(root + real_filename + f"_{i}.npz")
                os.remove(root + config_name + f"_{i}.ini")
        else:
            raise NotImplementedError(
                f"Error for {config_name}: Pooling of {outputtype} isn't implemented yet.")
    else:
        print("Not all npz files are generated yet")


for root, dirs, files in os.walk(path_to_look):
    if "ToPool" in files:
        pool_data_files(root + "/")
        os.remove(root + "/ToPool")

