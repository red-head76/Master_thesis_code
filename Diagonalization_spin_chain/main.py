import pdb
import sys
import numpy as np
import output
import output_trash
import time
from os.path import isfile
from configparser import ConfigParser
from matplotlib.pyplot import show as show_plot
from create_config import create_config
from support_functions import save_data

# measure the time a script needs
time0 = time.time()


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    # if there is no config file, create one with default values
    if not isfile("config.ini"):
        create_config()
    config_file = "config.ini"

# Read config_file with a config_object
config_object = ConfigParser(converters={"list": convert_list})
config_object.read(config_file)

# System setup
System = config_object["System"]
central_spin = config_object.getboolean("System", "central_spin")
chain_length = str_to_int(config_object.getlist("System", "chain_length"))
total_spins = central_spin + np.array(chain_length)
dim = np.array(2**total_spins, dtype=int)
periodic_boundaries = config_object.getboolean("System", "periodic_boundaries")
spin_constant = config_object.getboolean("System", "spin_constant")

# Coupling Constants
Constants = config_object["Constants"]
J = float(Constants["J"])
B0 = str_to_float(config_object.getlist("Constants", "B0"))
A = str_to_float(config_object.getlist("Constants", "A"))
scaling = Constants["scaling"]

# Output option
Output = config_object["Output"]
outputtype = Output["outputtype"]
save = Output["filename"]
samples = str_to_int(config_object.getlist("Output", "samples"))
show = config_object.getboolean("Output", "show")
save_plot = config_object.getboolean("Output", "save_plot")

# Other setup
Other = config_object["Other"]
seed = int(Other["seed"])

# Set to true in the following, if an animation is called
anim = False

if outputtype in ["plot_time_evo", "animate_time_evo"]:
    # Initial state
    idx_psi_0 = int(Other["idx_psi_0"])
    # Time array
    t = np.linspace(0, float(Other["timeend"]), int(Other["timesteps"]))

if outputtype in ["plot_g"]:
    # Initial state
    rho0 = [np.eye(d) / d for d in dim]
    # Time array
    t = np.linspace(0, float(Other["timeend"]), int(Other["timesteps"]) + 1)

if outputtype in ["plot_half_chain_entropy", "plot_occupation_imbalance",
                  "plot_exp_sig_z_central_spin", "plot_correlation", "calc_psi_t"]:
    t = np.logspace(np.log10(float(Other["timestart"])), np.log10(float(Other["timeend"])),
                    int(Other["timesteps"]) + 1)

if outputtype == "plot_time_evo":
    data = output.plot_time_evo(t, idx_psi_0, chain_length[0], J, B0[0], A[0], spin_constant,
                                periodic_boundaries, central_spin, save)

elif outputtype == "animate_time_evo":
    data = output.animate_time_evo(t, idx_psi_0, chain_length[0], J, B0[0], A[0], spin_constant,
                                   periodic_boundaries, central_spin, save)
    anim = True

elif outputtype == "plot_r":
    data = output.plot_r_values(chain_length[0], J, B0[0], A[0], periodic_boundaries, central_spin,
                                spin_constant, samples[0], save)

elif outputtype == "plot_r_fig3":
    data = output.plot_r_fig3(chain_length, J, B0, periodic_boundaries, samples, save)

elif outputtype == "plot_half_chain_entropy":
    data = output.plot_half_chain_entropy(t, chain_length, J, B0, A, periodic_boundaries,
                                          central_spin, samples, seed, scaling, save)

elif outputtype == "plot_occupation_imbalance":
    data = output.plot_occupation_imbalance(t, chain_length, J, B0, A, periodic_boundaries,
                                            central_spin, samples, seed, scaling, save)

elif outputtype == "plot_exp_sig_z_central_spin":
    data = output.plot_exp_sig_z_central_spin(t, chain_length, J, B0, A, periodic_boundaries,
                                              samples, seed, scaling, save)

elif outputtype == "plot_correlation":
    data = output.plot_correlation(t, chain_length, J, B0, A, periodic_boundaries, samples,
                                   seed, scaling, save)

# Old stuff
elif outputtype == "plot_f_fig2":
    data = output_trash.plot_f_fig2(chain_length, J, B0, periodic_boundaries, samples, save)

elif outputtype == "plot_g":
    data = output_trash.plot_g_value(rho0, t, chain_length, J, B0,
                                     periodic_boundaries, samples, save)

elif outputtype == "plot_fa":
    data = output_trash.plot_fa_values(chain_length, J, B0, A[0],
                                       periodic_boundaries, central_spin, samples, save)

elif outputtype == "calc_eigvals_eigvecs":
    data = output.calc_eigvals_eigvecs_biggest_subspace(chain_length, J, B0, A,
                                                        periodic_boundaries, central_spin, seed,
                                                        scaling)
    anim = None
elif outputtype == "calc_psi_t":
    data = output.calc_psi_t(t, chain_length, J, B0, A, periodic_boundaries, central_spin,
                             seed, scaling)
    anim = None

else:
    raise ValueError(f"Option '{outputtype}' unknown")

if save:
    time_passed = time.time() - time0
    save_data(save, data, config_file, time_passed, save_plot, anim)

if show:
    show_plot()
