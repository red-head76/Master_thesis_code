import os
import sys
import shutil
"""
Removes all slurm, input and data files if no option is given
If any options are given, it only removes these files.
'-slurm' for slurm files
'-data' for the subdirectories inside ./input_files/
'-input' for the input files (and operator) files inside ./input_files
"""


def del_slurm():
    entries = os.scandir("./")
    for entry in entries:
        if entry.name[:5] == "slurm":
            os.remove(entry.name)


def del_input():
    # input and operator files
    entries = os.scandir("./input_files")
    for entry in entries:
        if entry.name[-3:] == ".op" or entry.name[-4:] == ".inp":
            os.remove("./input_files/" + entry.name)


def del_data():
    entries = os.scandir("./input_files")
    for entry in entries:
        if entry.is_dir:
            shutil.rmtree("./input_files/" + entry.name)


if len(sys.argv) == 1:
    print("Removing slurm, input and data...")
    del_slurm()
    del_input()
    del_data()
else:
    if "-slurm" in sys.argv:
        del_slurm()
    elif "-data" in sys.argv:
        del_data()
    elif "-input" in sys.argv:
        del_input()
    else:
        print("Only available options: '-slurm', '-data', '-input or no argument.")
