import numpy as np
import pdb
from os.path import isfile
from configparser import ConfigParser
from create_config import create_config
from output import plot_time_evo, animate_time_evo, plot_r_values
def convert_list(string):
    # Converts a string to a list of floats
    return np.array([i.strip() for i in string.split(',')])


# if there is no config file, create one with default values
if not isfile("config.ini"):
    create_config()

# Read config.ini file with a config_object
config_object = ConfigParser(converters={"list": convert_list})
config_object.read("config.ini")

# System setup
System = config_object["System"]
central_spin = config_object.getboolean("System", "central_spin")
chain_length = config_object.getlist("System", "chain_length").astype(np.int)
total_spins = central_spin + chain_length
dim = np.array(2**total_spins, dtype=np.int)
periodic_boundaries = config_object.getboolean("System", "periodic_boundaries")
spin_constant = config_object.getboolean("System", "spin_constant")
# Coupling Constants
Constants = config_object["Constants"]
J = float(Constants["J"])
B0 = config_object.getlist("Constants", "B0").astype(np.float)
# B0 = np.arange(0.5, 13)
A = float(Constants["A"])

# Output option
Output = config_object["Output"]
outputtype = Output["outputtype"]
save = Output["filename"]
sampling = int(Output["sampling"])

# Other setup
Other = config_object["Other"]
# Initial state
psi0 = np.zeros(dim)
psi0[int(Other["idx_psi0"])] = 1
# Time array
t = np.linspace(0, float(Other["timespan"]), int(Other["timesteps"]))

if outputtype == "plot":
    plot_time_evo(t, psi0, chain_length, J, B0, A, spin_constant, periodic_boundaries,
                  central_spin, save)

if outputtype == "animate":
    animate_time_evo(t, psi0, chain_length, J, B0, A, spin_constant, periodic_boundaries,
                     central_spin, save)

if outputtype == "plot_r_values":
    plot_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                  spin_constant, sampling)
