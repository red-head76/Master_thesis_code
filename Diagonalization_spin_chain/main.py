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

# Coupling Constants
Constants = config_object["Constants"]
J = float(Constants["J"])
J_xy = float(Constants["J_xy"])
B0 = str_to_float(config_object.getlist("Constants", "B0"))
A = str_to_float(config_object.getlist("Constants", "A"))
scaling = Constants["scaling"]

# Output option
Output = config_object["Output"]
outputtype = Output["outputtype"]
save_path = Output["filename"]
samples = str_to_int(config_object.getlist("Output", "samples"))
show = config_object.getboolean("Output", "show")
save_plot = config_object.getboolean("Output", "save_plot")
parallelized = config_object.getboolean("Output", "parallelized")
picture_format = config_object.get("Output", "picture_format")

# Other setup
Other = config_object["Other"]
seed = int(Other["seed"])
timestart = float(Other["timestart"])
timeend = float(Other["timeend"])
timesteps = int(Other["timesteps"]) + 1
initial_state = Other["initial_state"]

# Set to true in the following, if an animation is called
anim = False

if outputtype in ["plot_time_evo", "animate_time_evo", "animate_barplot", "plot_light_cone",
                  "plot_deff"]:
    # Time array
    t = np.linspace(timestart, timeend, timesteps)

if outputtype in ["plot_g"]:
    # Initial state
    rho0 = [np.eye(d) / d for d in dim]
    # Time array
    t = np.linspace(timestart, timeend, timesteps)

if outputtype in ["plot_half_chain_entropy", "plot_single_shot_half_chain_entropy",
                  "plot_occupation_imbalance", "plot_single_shot_occupation_imbalance",
                  "plot_exp_sig_z_central_spin", "plot_single_shot_exp_sig_z_central_spin",
                  "plot_correlation", "calc_psi_t", "plot_2_spin_up",
                  "plot_occupation_imbalance_plateau", "plot_occupation_imbalance_plateau_linfit",
                  "plot_exp_sig_z_single_spin", "plot_single_spin_entropy"]:
    t = np.logspace(np.log10(timestart), np.log10(timeend), timesteps)

if outputtype == "plot_time_evo":
    data = output.plot_time_evo(t, chain_length[0], J, J_xy, B0[0], A[0],
                                periodic_boundaries, central_spin, seed, scaling,
                                save_path, initial_state)

elif outputtype == "plot_light_cone":
    t = np.logspace(np.log10(timestart), np.log10(timeend), timesteps)
    data = output.plot_light_cone(t, chain_length[0], J, J_xy, B0[0], A[0],
                                  periodic_boundaries, central_spin, seed, scaling,
                                  save_path, initial_state)

elif outputtype == "animate_time_evo":
    data = output.animate_time_evo(t, chain_length[0], J, J_xy, B0[0], A[0],
                                   periodic_boundaries, central_spin, seed, scaling,
                                   save_path, initial_state)
    anim = data[-1]
    data = data[:-1]

elif outputtype == "animate_barplot":
    data = output.animate_barplot(t, chain_length[0], J, J_xy, B0[0], A[0],
                                  periodic_boundaries, central_spin, seed, scaling,
                                  save_path, initial_state)
    anim = data[-1]
    data = data[:-1]

elif outputtype == "plot_r":
    data = output.plot_r_values(chain_length[0], J, J_xy, B0[0], A[0], periodic_boundaries,
                                central_spin, samples[0], initial_state, scaling, save_path)

elif outputtype == "plot_r_fig3":
    data = output.plot_r_fig3(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                              initial_state, samples, scaling, save_path)

elif outputtype == "plot_half_chain_entropy":
    data = output.plot_half_chain_entropy(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_single_shot_half_chain_entropy":
    data = output.plot_single_shot_half_chain_entropy(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples,
        scaling, save_path, initial_state)

elif outputtype == "plot_occupation_imbalance":
    data = output.plot_occupation_imbalance(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_single_shot_occupation_imbalance":
    data = output.plot_single_shot_occupation_imbalance(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_occupation_imbalance_plateau":
    data = output.plot_occupation_imbalance_plateau(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_occupation_imbalance_plateau_linfit":
    data = output.plot_occupation_imbalance_plateau_linfit(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_exp_sig_z_central_spin":
    data = output.plot_exp_sig_z_central_spin(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_single_shot_exp_sig_z_central_spin":
    data = output.plot_single_shot_exp_sig_z_central_spin(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, samples, seed, scaling,
        save_path, initial_state)

elif outputtype == "plot_exp_sig_z_single_spin":
    data = output.plot_exp_sig_z_single_spin(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_single_spin_entropy":
    data = output.plot_single_spin_entropy(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

elif outputtype == "plot_correlation":
    data = output.plot_correlation(
        t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples, seed,
        scaling, save_path, initial_state)

# Old stuff
elif outputtype == "plot_f_fig2":
    data = output_trash.plot_f_fig2(chain_length, J, J_xy, B0, periodic_boundaries, samples,
                                    save_path)

elif outputtype == "plot_g":
    data = output_trash.plot_g_value(rho0, t, chain_length, J, J_xy, B0, periodic_boundaries,
                                     samples, save_path)

elif outputtype == "plot_fa":
    data = output_trash.plot_fa_values(chain_length, J, J_xy, B0, A[0], periodic_boundaries,
                                       central_spin, samples, save_path)

elif outputtype == "calc_eigvals_eigvecs":
    data = output.calc_eigvals_eigvecs_biggest_subspace(chain_length, J, J_xy, B0, A,
                                                        periodic_boundaries, central_spin, seed,
                                                        scaling)
    anim = None
elif outputtype == "calc_psi_t":
    data = output.calc_psi_t(t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                             seed, scaling)
    anim = None

elif outputtype == "plot_2_spin_up":
    data = output.plot_2_spin_up(t, chain_length[0], J, J_xy, B0[0], A[0],
                                 periodic_boundaries, central_spin, seed, scaling, save_path)

elif outputtype == "plot_deff":
    data = output.plot_deff(chain_length[0], J, J_xy, B0, A[0], periodic_boundaries,
                            central_spin, samples[0], scaling, save_path, initial_state)

elif outputtype == "plot_eigenstates":
    data = output.plot_eigenstates(chain_length[0], J, J_xy, B0[0], A[0], periodic_boundaries,
                                   central_spin, seed, scaling, save_path, initial_state)

else:
    raise ValueError(f"Option '{outputtype}' unknown")

if save_path:
    time_passed = time.time() - time0
    save_data(save_path, data, config_file, time_passed, save_plot, picture_format,
              parallelized, anim)

if show:
    show_plot()
