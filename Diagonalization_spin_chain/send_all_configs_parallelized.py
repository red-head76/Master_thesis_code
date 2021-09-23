import numpy as np
import os
import sys
import shutil
from configparser import ConfigParser
import fileinput
import pdb


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def replace_text(filename, text, replacement):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(text, replacement), end='')


# Make "scan_ids" optional because i don't want to rewrite all the configs
def send_single_config(config_name):
    """
    This function creates a directory named config_name in "./Plots/" and places all
    sub-calculations for each sample in it while parallelizing the job by dividing the full job
    for all samples for one job for each sample.
    """
    # Read config_file with a config_object
    config_object = ConfigParser(converters={"list": convert_list})
    config_object.read("config_files/" + config_name)
    samples = config_object.getint("Output", "samples")
    # path + filename
    filename = config_object.get("Output", "filename").rstrip('/')
    real_filename = filename.split('/')[-1]
    path = filename[:-len(real_filename)]
    if not os.path.isdir(path):
        os.mkdir(path)
    # Copy original config
    shutil.copyfile(f"./config_files/{config_name}",
                    f"{path}/{config_name[:-4]}_config.ini")
    for i in range(samples):
        new_config_name = f"{path}/{config_name[:-4]}_{i}.ini"
        shutil.copyfile("./config_files/" + config_name, new_config_name)
        # Set samples to one
        replace_text(new_config_name, f"samples = {samples}", "samples = 1")
        # Set new filename
        replace_text(new_config_name, f"filename = {filename}", f"filename = {filename}_{i}")
        # Set boolean parallelized to True (if it isn't the case yet)
        replace_text(new_config_name, "parallelized = False", "parallelized = True")
        if not os.path.isfile(f"{filename}_{i}.npz"):
            # sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
            os.system(
                f"sbatch --export=ALL,input={new_config_name}, -J {filename}_{i} start_job.sh")
            # os.system(f"python3 main.py {new_config_name}")
        else:
            print(f"{filename}_{i}.npz does already exist")


if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./config_files")
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini":
            config_files.append(entry.name)
    for config_name in config_files:
        send_single_config(config_name)
        # pool_data_files(config_name)

else:
    for config_name in sys.argv[1:]:
        send_single_config(config_name)
        # os.system(
        #     f"sbatch --export=ALL,input=config_files/{config_name}, -J {config_name} start_job.sh")
