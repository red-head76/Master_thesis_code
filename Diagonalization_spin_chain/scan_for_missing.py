import os
import sys
from configparser import ConfigParser

path = sys.argv[1]

config_object = ConfigParser()
config_object.read(f"{path}/{path}_config.ini")

samples = config_object.getint("Output", "samples")

for i in range(samples):
    if not os.path.isfile(f"{path}/{path}_{i}.npz"):
        print(f"{path}_{i}.npz")
