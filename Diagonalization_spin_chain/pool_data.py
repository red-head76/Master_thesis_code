import sys
import numpy as np
import os
from configparser import ConfigParser
from support_functions import prepend_line
from re import search

if len(sys.argv) > 1:
    path_to_look = sys.argv[1]
else:
    path_to_look = "./Plots/"


def pool_data_files(root):
    config_names = []
    for entry in os.scandir(root):
        if entry.name[-4:] == ".ini" and not search("_\d{0,3}.ini", entry.name):
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
            elif outputtype == "plot_exp_sig_z_central_spin":
                pool_exp_sig_z_central_spin(config_object, filename, samples)
            else:
                raise NotImplementedError(
                    f"Error for {config_name}: Pooling of {outputtype} isn't implemented (yet).")
            for i in range(samples):
                os.remove(f"{filename}_{i}.npz")
            for i in range(samples):
                if os.path.isfile(f"{root}/{config_name[:-4]}_{i}.ini"):
                    os.remove(f"{root}/{config_name[:-4]}_{i}.ini")
        else:
            print(f"Not all npz files are generated in {root}{config_name}")


def pool_eigvals_eigvecs(config_object, filename, samples):
    # data: eigenvalues [dim], eigenvectors [dim, dim]
    data0 = np.load(filename + "_0.npz")
    dim = data0["arr_0"].size
    pooled_eigenvalues = np.empty((samples, dim))
    pooled_eigenvectors = np.empty((samples, dim, dim))
    for i in range(samples):
        data = np.load(filename + f"_{i}.npz")
        pooled_eigenvalues[i] = data["arr_0"]
        pooled_eigenvectors[i] = data["arr_1"]
    np.savez(filename + ".npz", pooled_eigenvalues, pooled_eigenvectors)


def pool_occupation_imbalance(config_object, filename, samples):
    # data: [times, occupation_imbalance_means, occupation_imbalance_stds]
    times_size = config_object.getint("Other", "timesteps") + 1
    occupation_imbalances = np.empty([samples, times_size])
    for i in range(samples):
        data = np.load(f"{filename}_{i}.npz")
        # this is only needed once
        if i == 0:
            times = data["arr_0"]
        occupation_imbalances[i] = data["arr_1"]
    np.savez(f"{filename}.npz", times,
             occupation_imbalances.mean(axis=0), occupation_imbalances.std(axis=0))


def pool_half_chain_entropy(config_object, filename, samples):
    # data: [times, hce_means, hce_stds]
    time_size = config_object.getint("Other", "timesteps") + 1
    hces = np.empty((samples, time_size))
    for i in range(samples):
        data = np.load(f"{filename}_{i}.npz")
        # this is only needed once
        if i == 0:
            times = data["arr_0"]
        hces[i] = data["arr_1"]
    np.savez(f"{filename}.npz", times, hces.mean(axis=0), hces.std(axis=0))


def pool_exp_sig_z_central_spin(config_object, filename, samples):
    # data: [times, exp_sig_z_means, exp_sig_z_stds]
    times_size = config_object.getint("Other", "timesteps") + 1
    exp_sig_zs = np.empty((samples, times_size))
    for i in range(samples):
        data = np.load(f"{filename}_{i}.npz")
        if i == 0:
            times = data["arr_0"]
        exp_sig_zs[i] = data["arr_1"]
    np.savez(f"{filename}.npz", times,
             exp_sig_zs.mean(axis=0), exp_sig_zs.std(axis=0))


for root, dirs, files in os.walk(path_to_look):
    if "ToPool" in files:
        pool_data_files(root + "/")
        os.remove(root + "/ToPool")
