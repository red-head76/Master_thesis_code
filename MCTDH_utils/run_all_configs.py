import os
import pdb

"""
Runs all config files that are placed in the folder 'config_files'.
Which means the 'standard' config.ini is neglected
"""
entries = os.scandir("./config_files")
config_files = []
for entry in entries:
    if entry.name[-4:] == ".ini":
        config_files.append(entry.name)

for config_file in config_files:
    os.system("python3 generate_input_file.py " +
              "./config_files/" + config_file)
