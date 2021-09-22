import os
import sys
from configparser import ConfigParser

path = sys.argv[1]
names = path.split('/')[-1]

config_object = ConfigParser()
config_object.read(f"{path}/{names}_config.ini")

samples = config_object.getint("Output", "samples")

for i in range(samples):
    if not os.path.isfile(f"{path}/{names}_{i}.npz"):
        print(f"{names}_{i}.npz")
