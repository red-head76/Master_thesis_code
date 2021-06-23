import pdb
from os.path import isfile
from configparser import ConfigParser
import numpy as np
from matplotlib.pyplot import show as show_plot
import output
from create_config import create_config


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


# if there is no config file, create one with default values
if not isfile("config.ini"):
    create_config()

# Read config.ini file with a config_object
config_object = ConfigParser(converters={"list": convert_list})
config_object.read("config.ini")

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

# Output option
Output = config_object["Output"]
outputtype = Output["outputtype"]
save = Output["filename"]
samples = str_to_int(config_object.getlist("Output", "samples"))
show = config_object.getboolean("Output", "show")

# Other setup
Other = config_object["Other"]
seed = int(Other["seed"])

if outputtype in ["plot_time_evo", "animate_time_evo"]:
    # Initial state
    idx_psi_0 = int(Other["idx_psi_0"])
    # Time array
    t = np.linspace(0, float(Other["timespan"]), int(Other["timesteps"]))

if outputtype in ["plot_g"]:
    # Initial state
    rho0 = [np.eye(d) / d for d in dim]
    # Time array
    t = np.linspace(0, float(Other["timespan"]), int(Other["timesteps"]) + 1)

if outputtype in ["plot_sa", "plot_occupation_imbalance", "plot_exp_sig_z_central_spin"]:
    t = np.logspace(np.log10(float(Other["timestart"])), np.log10(float(Other["timespan"])),
                    int(Other["timesteps"]) + 1)

if outputtype == "plot_time_evo":
    output.plot_time_evo(t, idx_psi_0, chain_length[0], J, B0[0], A[0], spin_constant,
                         periodic_boundaries, central_spin, save)

if outputtype == "animate_time_evo":
    output.animate_time_evo(t, idx_psi_0, chain_length[0], J, B0[0], A[0], spin_constant,
                            periodic_boundaries, central_spin, save)

if outputtype == "plot_r":
    output.plot_r_values(chain_length[0], J, B0[0], A[0], periodic_boundaries, central_spin,
                         spin_constant, samples[0], save)

if outputtype == "plot_r_fig3":
    output.plot_r_fig3(chain_length, J, B0, periodic_boundaries, samples, save)

if outputtype == "plot_f_fig2":
    output.plot_f_fig2(chain_length, J, B0, periodic_boundaries, samples, save)

if outputtype == "plot_g":
    output.plot_g_value(rho0, t, chain_length, J, B0,
                        periodic_boundaries, samples)

if outputtype == "plot_fa":
    output.plot_fa_values(chain_length, J, B0, A[0],
                          periodic_boundaries, central_spin, samples, save)

if outputtype == "plot_sa":
    output.plot_Sa_values(t, chain_length, J, B0, A,
                          periodic_boundaries, samples, save)

if outputtype == "plot_occupation_imbalance":
    pdb.set_trace()
    output.plot_occupation_imbalance(
        t, chain_length, J, B0, A, periodic_boundaries, central_spin, samples, seed, save)

if outputtype == "plot_exp_sig_z_central_spin":
    output.plot_exp_sig_z_central_spin(
        t, chain_length, J, B0, A, periodic_boundaries, samples, seed, save)


if show:
    show_plot()
