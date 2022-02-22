import os
import sys
from configparser import ConfigParser
from re import search

path = sys.argv[1]

for entry in os.scandir(path):
    # Check 1. If it is a .ini, 2. if it is a config (not a sub_config with _digit)
    if (entry.name[-4:] == ".ini" and not search("_\d+.ini", entry.name)):
        config_object = ConfigParser()
        config_object.read(f"{path}/{entry.name}")
        samples = config_object.getint("Output", "samples")
        filename = config_object.get("Output", "filename")
        for i in range(samples):
            if not os.path.isfile(f"{filename}_{i}.npz"):
                print(f"{filename}_{i}.npz")
