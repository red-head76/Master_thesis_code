from shutil import copy
import sys
import os
from configparser import ConfigParser
import numpy as np
import support_functions as sf
from time_evo import time_evo_subspace


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


if len(sys.argv) > 1:
    ofee_config_file = sys.argv[1]
else:
    raise FileNotFoundError("No ofee config given.")

# Read ofee_config_file with a config_object
ofee_config_object = ConfigParser(converters={"list": convert_list})
ofee_config_object.read(ofee_config_file)

# Output option
Output = ofee_config_object["Output"]
outputtype = Output["outputtype"]
data_paths = ofee_config_object.getlist("Output", "data_paths")
save_path = Output["filename"]
show = ofee_config_object.getboolean("Output", "show")
save_plot = ofee_config_object.getboolean("Output", "save_plot")

# Other setup
Other = ofee_config_object["Other"]
idx_psi_0 = int(Other["idx_psi_0"])
timestart = float(Other["timestart"])
timeend = float(Other["timeend"])
timesteps = int(Other["timesteps"])

if not os.path.isdir(save_path):
    os.mkdir(save_path)
copy(ofee_config_file, save_path + '/' + ofee_config_file.rstrip('/').split('/')[-1])

data_configs = []
for path in data_paths:
    for entry in os.scandir(path):
        if entry.name[-11:] == "_config.ini":
            data_configs.append(path + '/' + entry.name)

for data_config_file in data_configs:
    # Read data_config_file with a config_object
    data_config_object = ConfigParser(converters={"list": convert_list})
    data_config_object.read(data_config_file)

    # System
    chain_length = data_config_object.getint("System", "chain_length")
    central_spin = data_config_object.getboolean("System", "central_spin")
    total_spins = int(chain_length + central_spin)
    dim = int(2**total_spins)

    # Coupling Constants
    Constants = data_config_object["Constants"]
    J = float(Constants["J"])
    B0 = data_config_object.getfloat("Constants", "B0")
    A = data_config_object.getfloat("Constants", "A")

    # Output option
    Output = data_config_object["Output"]
    filename = Output["filename"].rstrip('/')
    samples = data_config_object.getint("Output", "samples")

    # Other setup
    Other = data_config_object["Other"]
    seed = int(Other["seed"])

    if outputtype in ["plot_half_chain_entropy", "plot_occupation_imbalance",
                      "plot_exp_sig_z_central_spin", "plot_correlation", "calc_psi_t"]:
        times = np.logspace(np.log10(timestart), np.log10(timeend), timesteps + 1)

    def read_eigvals_evecs(filename, idx):
        data = np.load(filename + f"_{idx}.npz")
        return data["arr_0"], data["arr_1"]

    copy(data_config_file, save_path + '/' + filename.rstrip('/').split('/')[-1] + ".ini")

    if outputtype == "plot_occupation_imbalance":
        occ_imbalance = np.empty((samples, times.size))
        for idx in range(samples):
            eigenvalues, eigenvectors = read_eigvals_evecs(filename, idx)
            psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
            # This mask filters out the states of the biggest subspace
            subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(
                np.arange(dim), total_spins), axis=1) - total_spins//2))[0]
            psi_z = np.arange(0, int(2**(total_spins)))[subspace_mask]
            # discard central_spin
            sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, :chain_length]
            # discard central spin in exp_sig_z
            exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
            # occupation imbalance mask: even minus odd sites (normed by chain length)
            occ_imbalance[idx] = (np.where(np.arange(chain_length) % 2, exp_sig_z,
                                           -exp_sig_z).sum(axis=1) / (chain_length / 2))
        occ_imbalance_mean = occ_imbalance.mean(axis=0)
        occ_imbalance_std = occ_imbalance.std(axis=0)
        np.savez(save_path + '/' + filename.rstrip('/').split('/')[-1] + ".npz",
                 times, occ_imbalance_mean, occ_imbalance_std)

    elif outputtype == "plot_exp_sig_z_central_spin":
        exp_sig_z = np.empty((samples, times.size))
        for idx in range(samples):
            eigenvalues, eigenvectors = read_eigvals_evecs(filename, idx)
            psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
            subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(
                np.arange(dim), total_spins), axis=1) - total_spins//2))[0]
            psi_z = np.arange(0, int(2**(total_spins)))[subspace_mask]
            # only the last spin
            sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, -1]
            exp_sig_z[idx] = (np.abs(psi_t)**2 @ sigma_z)
        exp_sig_z_mean = exp_sig_z.mean(axis=0)
        exp_sig_z_std = exp_sig_z.std(axis=0)
        np.savez(save_path + '/' + filename.rstrip('/').split('/')[-1] + ".npz",
                 times, exp_sig_z_mean, exp_sig_z_std)

    elif outputtype == "plot_half_chain_entropy":
        hce = np.empty((samples, times.size))
        for idx in range(samples):
            eigenvalues, eigenvectors = read_eigvals_evecs(filename, idx)
            psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
            # Outer product along axis 1
            rho_t = psi_t[:, :, np.newaxis] * psi_t.conj()[:, np.newaxis, :]
            # Partial trace
            subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(
                np.arange(dim), total_spins), axis=1) - total_spins//2))[0]
            rho_a_t = sf.partial_trace_subspace(rho_t, subspace_mask, total_spins//2, True)
            # hce = -tr(rho_a ln(rho_a))
            #    = -tr(rho ln(rho)) = tr(D ln(D)), where D is the diagonalized matrix
            eigvals = np.linalg.eigvalsh(rho_a_t)
            # ugly, but cuts out the the Runtime warning caused by of 0 values in log
            hce[idx] = -np.sum(eigvals * np.log(eigvals, where=eigvals > 0,
                                                out=np.zeros(eigvals.shape)), axis=1)
        hce_mean = hce.mean(axis=0)
        hce_std = hce.std(axis=0)
        np.savez(save_path + '/' + filename.rstrip('/').split('/')[-1] + ".npz",
                 times, hce_mean, hce_std)

    else:
        raise NotImplementedError()
