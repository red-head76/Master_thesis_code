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
    config_names = []
    for entry in os.scandir(root):
        if entry.name[-4:] == ".ini":
            config_names.append(entry.name)
    for config_name in config_names:
        config_object = ConfigParser()
        # read original config
        config_object.read(root + config_name)
        outputtype = config_object.get("Output", "outputtype")
        samples = config_object.getint("Output", "samples")
        filename = config_object.get("Output", "filename")
        real_filename = filename.split('/')[-1]
        # in case root has changed
        filename = root + real_filename
        do_pooling = True
        for i in range(samples):
            if not os.path.isfile(filename + f"_{i}.npz"):
                do_pooling = False
                break
        if do_pooling:
            if outputtype == "calc_eigvals_eigvecs":
                pool_eigvals_eigvecs(config_object, filename, samples)
            elif outputtype == "plot_occupation_imbalance":
                pool_occupation_imbalance(config_object, filename, samples)
            else:
                raise NotImplementedError(
                    f"Error for {config_name}: Pooling of {outputtype} isn't implemented (yet).")
        else:
            print("Not all npz files are generated yet")


def pool_eigvals_eigvecs(config_object, filename, samples):
    # data : eigenvalues [dim], eigenvectors [dim, dim]
    data0 = np.load(filename + "_0.npz")
    dim = data0["arr_0"].size
    pooled_eigenvalues = np.empty((samples, dim))
    pooled_eigenvectors = np.empty((samples, dim, dim))
    for i in range(samples):
        data = np.load(filename + f"_{i}.npz")
        pooled_eigenvalues[i] = data["arr_0"]
        pooled_eigenvectors[i] = data["arr_1"]
    np.savez(filename + ".npz", eigenvalues=pooled_eigenvalues, eigenvectors=pooled_eigenvectors)
    for i in range(samples):
        os.remove(f"{filename}_{i}.npz")


def pool_occupation_imbalance(config_object, filename, samples):
    # data : [times, occupation_imbalance_means, occupation_imbalance_stds]
    times_size = config_object.getint("Other", "timesteps") + 1
    occupation_imbalances = np.empty([samples, times_size])
    for i in range(samples):
        data = np.load(f"{filename}_{i}.npz")
        # this is only needed once
        if i == 0:
            times = data["arr_0"]
        occupation_imbalances[i] = data["arr_1"]
    np.savez(f"{filename}.npz", times=times, occupation_imbalances=occupation_imbalances)
    for i in range(samples):
        os.remove(f"{filename}_{i}.npz")


for root, dirs, files in os.walk(path_to_look):
    if "ToPool" in files:
        pool_data_files(root + "/")
        os.remove(root + "/ToPool")
