import pdb
import numpy as np
import matplotlib.pyplot as plt
from read_output import read_output
from configparser import ConfigParser

filenames = ["./20-5sp_20wf_delocalized", "./20-5sp_20wf_localized"]
# ["./run20", "./run40", "./run80"]
data = [read_output(filename + "/output") for filename in filenames]


def read_config(config_file):
    config_object = ConfigParser()
    config_object.read(config_file)
    return config_object


configs = [read_config(filename + "/config") for filename in filenames]

wfs = [40]
descriptions = ["delocalized", "localized"]
for dataset, config,  desc in zip(data, configs, descriptions):
    # Sz
    exp_sig_z = dataset["exp_q"]
    chain_length = config.getint("System", "chain_length")
    # cut central spin if its there
    exp_sig_z = exp_sig_z[:, :chain_length]
    occ_imbalance = np.where(np.arange(chain_length) % 2, exp_sig_z, -exp_sig_z).sum(axis=1)
    plt.plot(dataset["time"], occ_imbalance, label=desc)
    plt.xlabel("Time")
    plt.ylabel("Occupation imbalance")
    plt.legend()
    plt.semilogx()
plt.show()
